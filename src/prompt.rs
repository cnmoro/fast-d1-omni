//! Questions in, token sequences out, answers back (a port of the reference `prompt.py`).
//!
//!   <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> <opt> <mask> option_1 </opt> ... <decide>

use crate::json::{self, Json};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

#[derive(Clone, Debug)]
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    pub criteria: Json,
}

impl Question {
    pub fn options(&self) -> usize {
        match (&self.qtype, &self.criteria) {
            (QType::Noul, _) => 2,
            (_, Json::Obj(o)) => o.len(),
            (_, Json::Arr(a)) => a.len(),
            _ => 0,
        }
    }
    pub fn type_name(&self) -> &'static str {
        match self.qtype {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
    /// Calibration temperature key, e.g. "choice:3-5".
    pub fn temperature_key(&self) -> String {
        let k = self.options();
        let b = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
        format!("{}:{b}", self.type_name())
    }
}

fn py_str(v: &Json) -> String {
    match v {
        Json::Str(s) => s.clone(),
        other => json::to_string(other),
    }
}

/// Validate one question (the reference `as_question` + `Question.__post_init__`).
pub fn as_question(q: &Json) -> Result<Question, String> {
    let (Some(t), Some(ins)) = (q.get("type"), q.get("instructions")) else {
        return Err("a question is a dict with `type`, `instructions` and, for choice and score, `criteria`".into());
    };
    if !matches!(q, Json::Obj(_)) {
        return Err("a question is a dict".into());
    }
    let qtype = match t.as_str() {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        Some("noul") => QType::Noul,
        _ => return Err(format!("question type must be one of ['choice', 'noul', 'score'], got {}", json::to_string(t))),
    };
    let criteria = q.get("criteria").cloned().unwrap_or(Json::Null);
    match qtype {
        QType::Choice => match &criteria {
            Json::Obj(o) if o.len() >= 2 => {}
            _ => return Err("a choice needs criteria {name: description} with at least two options".into()),
        },
        QType::Score => match &criteria {
            Json::Arr(a) if (2..=10).contains(&a.len()) => {}
            _ => return Err("a score needs criteria: a list of 2 to 10 level descriptions, lowest first".into()),
        },
        QType::Noul => match &criteria {
            Json::Null | Json::Obj(_) => {}
            _ => return Err(r#"noul criteria are optional: {"true": "...", "false": "..."} (or "yes", "no")"#.into()),
        },
    }
    Ok(Question { qtype, instructions: py_str(ins), criteria })
}

/// `<|name|>` -> `<¦name¦>`, so caller text cannot emit a delimiter or marker token.
pub fn escape(text: &str) -> String {
    if !text.contains("<|") {
        return text.to_string();
    }
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 8);
    let mut i = 0;
    let mut last = 0;
    while i + 1 < b.len() {
        if b[i] == b'<' && b[i + 1] == b'|' {
            let mut j = i + 2;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            if j > i + 2 && j + 1 < b.len() && b[j] == b'|' && b[j + 1] == b'>' {
                out.push_str(&text[last..i]);
                out.push_str("<¦");
                out.push_str(&text[i + 2..j]);
                out.push_str("¦>");
                i = j + 2;
                last = i;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    out
}

pub fn serialize(state: &Json) -> String {
    match state {
        Json::Str(s) => s.clone(),
        other => json::to_string(other),
    }
}

fn criterion(v: &Json) -> String {
    py_str(v)
}

fn empty(v: &Json) -> bool {
    matches!(v, Json::Null) || matches!(v, Json::Str(s) if s.is_empty())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Text,
    Image,
    Audio,
}

/// Option texts in the model's order (a noul is read as [false, true]).
pub fn render_options(q: &Question, mode: Mode) -> Vec<String> {
    match q.qtype {
        QType::Choice => {
            let Json::Obj(o) = &q.criteria else { return vec![] };
            o.iter()
                .enumerate()
                .map(|(i, (k, v))| {
                    if mode == Mode::Audio {
                        let t = if empty(v) { k.clone() } else { criterion(v) };
                        format!("option_{i:03}: {t}")
                    } else if empty(v) {
                        k.clone()
                    } else {
                        format!("{k}: {}", criterion(v))
                    }
                })
                .collect()
        }
        QType::Score => {
            let Json::Arr(a) = &q.criteria else { return vec![] };
            a.iter().enumerate().map(|(i, c)| format!("level {i}: {}", criterion(c))).collect()
        }
        QType::Noul => {
            if mode == Mode::Audio {
                return vec!["false: no".into(), "true: yes".into()];
            }
            let given = match &q.criteria {
                Json::Obj(o) if !o.is_empty() => Some(&q.criteria),
                _ => None,
            };
            let yes_no = Json::Obj(vec![("false".into(), Json::str("no")), ("true".into(), Json::str("yes"))]);
            let crit = match (given, mode) {
                (Some(c), _) => c.clone(),
                (None, Mode::Image) => yes_no,
                (None, _) => Json::Obj(vec![]),
            };
            let pick = |a: &str, b: &str| -> Json {
                match crit.get(a) {
                    Some(v) => v.clone(),
                    None => crit.get(b).cloned().unwrap_or(Json::Null),
                }
            };
            let f = pick("false", "no");
            let t = pick("true", "yes");
            vec![
                format!("false: {}", if empty(&f) { "no, the statement does not hold".to_string() } else { criterion(&f) }),
                format!("true: {}", if empty(&t) { "yes, the statement holds".to_string() } else { criterion(&t) }),
            ]
        }
    }
}

pub struct Delims {
    pub bos: u32,
    pub mask: u32,
    pub state: u32,
    pub q: u32,
    pub opt: u32,
    pub opt_end: u32,
    pub decide: u32,
}

impl Delims {
    pub fn new(tok: &Tokenizer) -> Result<Delims, String> {
        let id = |s: &str| tok.token_id(s).ok_or_else(|| format!("token {s} missing from vocab"));
        Ok(Delims {
            bos: tok.bos,
            mask: tok.mask,
            state: id("<|reserved_7|>")?,
            q: id("<|reserved_8|>")?,
            opt: id("<|reserved_9|>")?,
            opt_end: id("<|reserved_10|>")?,
            decide: id("<|reserved_11|>")?,
        })
    }
}

fn enc(tok: &Tokenizer, s: &str) -> Vec<u32> {
    let mut v = Vec::new();
    tok.encode(&escape(s), &mut v);
    v
}

/// Token ids of one question over one (serialized) state, and the position of each option's marker.
pub fn encode(tok: &Tokenizer, dl: &Delims, state_ids: &[u32], q: &Question, max_len: usize, mode: Mode)
    -> Result<(Vec<i32>, Vec<i32>), String> {
    const PER_OPTION: i64 = 24;
    let opts = render_options(q, mode);
    let n = opts.len() as i64;
    let budget = 96i64.max((n * PER_OPTION + 32).min(max_len as i64 / 2));
    let per = 2i64.max((budget - 3 * n).div_euclid(n)) as usize;
    let mut question: Vec<u32> = vec![dl.q];
    question.extend(enc(tok, &q.instructions));
    question.truncate(16i64.max(budget) as usize);
    let mut markers = Vec::with_capacity(opts.len());
    for text in &opts {
        markers.push(question.len() + 1);
        question.push(dl.opt);
        question.push(dl.mask);
        let t = enc(tok, &format!(" {text}"));
        question.extend_from_slice(&t[..t.len().min(per)]);
        question.push(dl.opt_end);
    }
    question.push(dl.decide);
    let room = (max_len as i64 - question.len() as i64 - 2).max(0) as usize;
    let n_state = 1 + state_ids.len().min(room);
    let mut ids: Vec<i32> = Vec::with_capacity(1 + n_state + question.len());
    ids.push(dl.bos as i32);
    ids.push(dl.state as i32);
    ids.extend(state_ids[..n_state - 1].iter().map(|&x| x as i32));
    ids.extend(question.iter().map(|&x| x as i32));
    ids.truncate(max_len);
    let markers: Vec<i32> = markers.iter().map(|&m| (m + 1 + n_state) as i32).collect();
    if *markers.last().unwrap() as usize >= max_len {
        return Err("the options do not fit in the context".into());
    }
    Ok((ids, markers))
}

/// State tokens (the expensive part), shared by every question of a request.
pub fn state_tokens(tok: &Tokenizer, state: &Json) -> Vec<u32> {
    enc(tok, &serialize(state))
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

/// Typed answer from a question's probabilities in option order (noul: [yes, no]).
pub fn answer(q: &Question, probs: &[f32]) -> Json {
    let p64: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
    let typ = ("type".to_string(), Json::str(q.type_name()));
    if q.qtype == QType::Noul {
        return Json::Obj(vec![typ, ("noul".into(), Json::num(p64[0]))]);
    }
    let mut best = 0;
    for i in 1..p64.len() {
        if p64[i] > p64[best] {
            best = i;
        }
    }
    match q.qtype {
        QType::Choice => {
            let Json::Obj(o) = &q.criteria else { unreachable!() };
            let names: Vec<&String> = o.iter().map(|(k, _)| k).collect();
            Json::Obj(vec![
                typ,
                ("choice".into(), Json::str(names[best].clone())),
                ("confidence".into(), Json::num(p64[best])),
                ("probabilities".into(), Json::Obj(names.iter().zip(&p64).map(|(n, &p)| ((*n).clone(), Json::num(p))).collect())),
            ])
        }
        _ => {
            let Json::Arr(a) = &q.criteria else { unreachable!() };
            let score: f64 = p64.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
            Json::Obj(vec![
                typ,
                ("score".into(), Json::num(score)),
                ("confidence".into(), Json::num(p64[best])),
                ("probabilities".into(), Json::Obj(p64.iter().enumerate().map(|(i, &p)| (i.to_string(), Json::num(p))).collect())),
                ("legend".into(), Json::Obj(a.iter().enumerate().map(|(i, c)| (i.to_string(), Json::str(criterion(c)))).collect())),
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn esc() {
        assert_eq!(escape("a<|mask|>b<|x y|><||>"), "a<¦mask¦>b<|x y|><||>");
    }
}
