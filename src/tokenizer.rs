//! Byte-level BPE tokenizer built from the GGUF vocabulary, matching Hugging Face `tokenizers` for LFM2:
//! added tokens are split out first (leftmost-longest), the rest goes through the LFM2 pre-tokenizer regex
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! (emulated by hand), then GPT-2 byte-to-unicode mapping and rank-ordered merges.

use crate::gguf::Gguf;
use crate::unicode_tables::{LETTER, NUMBER};
use std::collections::HashMap;

fn in_table(t: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    t.binary_search_by(|&(a, b)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

#[inline]
fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_table(&LETTER, c)
}
#[inline]
fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_table(&NUMBER, c)
}
#[inline]
fn is_space(c: char) -> bool {
    matches!(c as u32, 0x09..=0x0d | 0x20 | 0x85 | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000)
}
#[inline]
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}
#[inline]
fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// End (exclusive) of the regex match starting at `i` (every position matches at least one char).
fn match_at(s: &[char], i: usize) -> usize {
    let n = s.len();
    let c0 = s[i];
    // 1. contractions
    if c0 == '\'' && i + 1 < n {
        let l = |k: usize| if i + k < n { s[i + k].to_ascii_lowercase() } else { '\0' };
        match l(1) {
            's' | 't' | 'm' | 'd' => return i + 2,
            'r' if l(2) == 'e' => return i + 3,
            'v' if l(2) == 'e' => return i + 3,
            'l' if l(2) == 'l' => return i + 3,
            _ => {}
        }
    }
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    {
        let mut j = i;
        if !is_letter(c0) && !is_nl(c0) && !is_number(c0) && i + 1 < n && is_letter(s[i + 1]) {
            j = i + 1;
        }
        if is_letter(s[j]) {
            while j < n && is_letter(s[j]) {
                j += 1;
            }
            return j;
        }
    }
    // 3. \p{N}{1,3}
    if is_number(c0) {
        let mut j = i;
        while j < n && j < i + 3 && is_number(s[j]) {
            j += 1;
        }
        return j;
    }
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        let j0 = if c0 == ' ' && i + 1 < n && is_other(s[i + 1]) { i + 1 } else { i };
        if is_other(s[j0]) {
            let mut j = j0;
            while j < n && is_other(s[j]) {
                j += 1;
            }
            while j < n && is_nl(s[j]) {
                j += 1;
            }
            return j;
        }
    }
    // whitespace run
    let mut e = i;
    while e < n && is_space(s[e]) {
        e += 1;
    }
    debug_assert!(e > i);
    // 5. \s*[\r\n]+  -> up to and including the last newline of the run
    if let Some(k) = (i..e).rev().find(|&k| is_nl(s[k])) {
        return k + 1;
    }
    // 6. \s+(?!\S)
    if e == n {
        return e;
    }
    if e - 1 > i {
        return e - 1;
    }
    // 7. \s+
    e
}

pub fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let e = match_at(&chars, i);
        out.push(chars[i..e].iter().collect());
        i = e;
    }
    out
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xa1..=0xac).chain(0xae..=0xff).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        map[*b as usize] = char::from_u32(c).unwrap();
    }
    map
}

#[derive(Default)]
struct TrieNode {
    next: HashMap<u8, usize>,
    id: Option<u32>,
}

#[allow(dead_code)]
pub struct Tokenizer {
    pub tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>, // (left, right) -> (rank, merged)
    byte_tok: [u32; 256],
    trie: Vec<TrieNode>,
    pub bos: u32,
    pub mask: u32,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let toks = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_arr()).ok_or("missing tokenizer.ggml.tokens")?;
        let tokens: Vec<String> = toks.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let types: Vec<u64> = g
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            vocab.entry(t.clone()).or_insert(i as u32);
        }
        let merges_v = g.get("tokenizer.ggml.merges").and_then(|v| v.as_arr()).ok_or("missing tokenizer.ggml.merges")?;
        let mut merges = HashMap::with_capacity(merges_v.len());
        for (rank, m) in merges_v.iter().enumerate() {
            let m = m.as_str().unwrap_or("");
            let Some((a, b)) = m.split_once(' ') else { continue };
            let (Some(&ia), Some(&ib), Some(&im)) = (vocab.get(a), vocab.get(b), vocab.get(&format!("{a}{b}"))) else {
                continue;
            };
            merges.entry((ia, ib)).or_insert((rank as u32, im));
        }
        let b2u = bytes_to_unicode();
        let mut byte_tok = [0u32; 256];
        for b in 0..256 {
            byte_tok[b] = *vocab.get(&b2u[b].to_string()).ok_or_else(|| format!("byte {b} missing from vocab"))?;
        }
        // added tokens: everything that is not a normal token (control = 3, user-defined = 4)
        let mut trie = vec![TrieNode::default()];
        for (i, t) in tokens.iter().enumerate() {
            if types.get(i).copied().unwrap_or(1) == 1 || t.is_empty() {
                continue;
            }
            let mut node = 0;
            for &b in t.as_bytes() {
                node = match trie[node].next.get(&b) {
                    Some(&n) => n,
                    None => {
                        trie.push(TrieNode::default());
                        let n = trie.len() - 1;
                        trie[node].next.insert(b, n);
                        n
                    }
                };
            }
            trie[node].id.get_or_insert(i as u32);
        }
        let bos = g.u("tokenizer.ggml.bos_token_id").unwrap_or(1) as u32;
        let mask = g.u("tokenizer.ggml.mask_token_id").map(|v| v as u32).or_else(|| vocab.get("<|mask|>").copied()).ok_or("no mask token")?;
        Ok(Tokenizer { tokens, vocab, merges, byte_tok, trie, bos, mask })
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    /// Longest added token starting at byte `i`.
    fn added_at(&self, b: &[u8], i: usize) -> Option<(usize, u32)> {
        let mut node = 0;
        let mut best = None;
        let mut j = i;
        while j < b.len() {
            match self.trie[node].next.get(&b[j]) {
                Some(&n) => node = n,
                None => break,
            }
            j += 1;
            if let Some(id) = self.trie[node].id {
                best = Some((j, id));
            }
        }
        best
    }

    /// Token ids of `text` (no special tokens added).
    pub fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let b = text.as_bytes();
        let mut seg = 0;
        let mut i = 0;
        while i < b.len() {
            if self.trie[0].next.contains_key(&b[i]) {
                if let Some((e, id)) = self.added_at(b, i) {
                    self.encode_plain(&text[seg..i], out);
                    out.push(id);
                    i = e;
                    seg = e;
                    continue;
                }
            }
            i += 1;
        }
        self.encode_plain(&text[seg..], out);
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        for piece in pretokenize(text) {
            self.bpe(piece.as_bytes(), out);
        }
    }

    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = bytes.iter().map(|&b| self.byte_tok[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            let Some((rank, _, _)) = best else { break };
            // apply every occurrence of this pair left to right (same result as one-at-a-time for a single rank)
            let mut j = 0;
            let mut next = Vec::with_capacity(syms.len());
            while j < syms.len() {
                if j + 1 < syms.len() {
                    if let Some(&(r, m)) = self.merges.get(&(syms[j], syms[j + 1])) {
                        if r == rank {
                            next.push(m);
                            j += 2;
                            continue;
                        }
                    }
                }
                next.push(syms[j]);
                j += 1;
            }
            syms = next;
        }
        out.extend_from_slice(&syms);
    }
}
