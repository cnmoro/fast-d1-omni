//! The LFM2.5 bidirectional trunk and the decision head, run over packed variable-length batches.
//!
//! A text batch is a concatenation of question sequences (no padding). A sequence may follow a cached media prefix:
//! the prefix was run through the trunk once (prefix positions never read text), its per-layer K/V and its last
//! conv state are kept on the GPU, and every question over the same media reuses them.

use crate::cuda::{self, k, AttnSeq, Blas, DevBuf, HostBuf, Out, SeqInfo, Stream, F16};
use crate::gguf::Gguf;
use std::ffi::c_void;
use std::ptr;

pub const D: usize = 1024;
pub const HEADS: usize = 16;
pub const KV_HEADS: usize = 8;
pub const HD: usize = 64;
pub const KV_DIM: usize = KV_HEADS * HD;
pub const MAX_POS: usize = 16384 + 64;

pub enum Mixer {
    Conv { in_proj: DevBuf, out_proj: DevBuf, convw: DevBuf, idx: usize },
    Attn { qkv: DevBuf, out: DevBuf, qn: DevBuf, kn: DevBuf, idx: usize },
}

pub struct TrunkLayer {
    pub mixer: Mixer,
    pub op_norm: DevBuf,
    pub ffn_norm: DevBuf,
    pub gate_up: DevBuf, // [2F, D]
    pub down: DevBuf,    // [D, F]
    pub ffn: usize,
}

pub struct HeadLayer {
    pub n1_w: DevBuf,
    pub n1_b: DevBuf,
    pub qkv: DevBuf,
    pub qkv_b: DevBuf,
    pub out: DevBuf,
    pub out_b: DevBuf,
    pub n2_w: DevBuf,
    pub n2_b: DevBuf,
    pub up: DevBuf,
    pub up_b: DevBuf,
    pub down: DevBuf,
    pub down_b: DevBuf,
    pub ffn: usize,
}

#[allow(dead_code)]
pub struct TextModel {
    pub vocab: usize,
    pub embed: DevBuf,
    pub final_norm: DevBuf,
    pub layers: Vec<TrunkLayer>,
    pub head: Vec<HeadLayer>,
    pub type_emb: DevBuf,
    pub cls_norm_w: DevBuf,
    pub cls_norm_b: DevBuf,
    pub cls_w: DevBuf,
    pub cls_b: DevBuf,
    pub out_w: DevBuf,
    pub out_b: f32,
    pub rope: DevBuf,
    pub n_attn: usize,
    pub n_conv: usize,
    pub eps: f32,
    pub max_ffn: usize,
}

/// A media prefix that went through the trunk: per attention layer K/V, and the last position's conv state.
/// Allocated and freed in stream order (no device-wide syncs).
pub struct PrefixCache {
    pub len: usize,
    pub k: *mut F16,  // [n_attn][len][KV_DIM]
    pub v: *mut F16,  // [n_attn][len][KV_DIM]
    pub bx: *mut F16, // [n_conv][D]
    pub bytes: usize,
}
unsafe impl Send for PrefixCache {}

impl PrefixCache {
    pub fn bytes_for(m: &TextModel, len: usize) -> usize {
        2 * m.n_attn * len * KV_DIM * 2 + m.n_conv * D * 2
    }
    pub fn alloc(m: &TextModel, len: usize, st: Stream) -> PrefixCache {
        let kv = m.n_attn * len * KV_DIM * 2;
        unsafe {
            PrefixCache {
                len,
                k: cuda::malloc_async(kv, st) as *mut F16,
                v: cuda::malloc_async(kv, st) as *mut F16,
                bx: cuda::malloc_async(m.n_conv * D * 2, st) as *mut F16,
                bytes: Self::bytes_for(m, len),
            }
        }
    }
    pub fn free(self, st: Stream) {
        unsafe {
            cuda::free_async(self.k as *mut c_void, st);
            cuda::free_async(self.v as *mut c_void, st);
            cuda::free_async(self.bx as *mut c_void, st);
        }
    }
}

fn up32(g: &Gguf, n: &str) -> Result<DevBuf, String> {
    Ok(DevBuf::from_slice(&g.f32(n)?))
}
fn up16(g: &Gguf, n: &str) -> Result<DevBuf, String> {
    Ok(DevBuf::from_slice(&g.f16(n)?))
}
fn cat16(g: &Gguf, names: &[&str]) -> Result<DevBuf, String> {
    let mut v = Vec::new();
    for n in names {
        v.extend(g.f16(n)?);
    }
    Ok(DevBuf::from_slice(&v))
}

/// RoPE table [pos][32] of (cos, sin), computed as HF builds it (fp32 angle, then cos/sin).
fn rope_table(theta: f32) -> Vec<f32> {
    let mut inv = [0f32; 32];
    for (i, f) in inv.iter_mut().enumerate() {
        let e = (2 * i) as f32 / HD as f32;
        *f = 1.0 / theta.powf(e);
    }
    let mut t = Vec::with_capacity(MAX_POS * 64);
    for p in 0..MAX_POS {
        for &f in &inv {
            let a = (p as f32 * f) as f64;
            t.push(a.cos() as f32);
            t.push(a.sin() as f32);
        }
    }
    t
}

impl TextModel {
    pub fn load(g: &Gguf) -> Result<TextModel, String> {
        let arch = g.s("general.architecture").unwrap_or("");
        if arch != "lfm2" {
            return Err(format!("unexpected architecture {arch:?} (want lfm2)"));
        }
        let n_layers = g.u("lfm2.block_count").ok_or("missing lfm2.block_count")? as usize;
        let n_head_layers = g.u("lfm2.decision.block_count").unwrap_or(2) as usize;
        let n_trunk = n_layers - n_head_layers;
        let eps = g.f("lfm2.attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32;
        let theta = g.f("lfm2.rope.freq_base").unwrap_or(1e6) as f32;
        let emb_dims = g.dims("token_embd.weight")?;
        if emb_dims[1] != D {
            return Err(format!("hidden size {} unsupported (want {D})", emb_dims[1]));
        }
        let mut layers = Vec::new();
        let (mut n_attn, mut n_conv, mut max_ffn) = (0, 0, 0);
        for i in 0..n_trunk {
            let p = format!("blk.{i}.");
            let t = |s: &str| format!("{p}{s}");
            let mixer = if g.has(&t("attn_q.weight")) {
                let m = Mixer::Attn {
                    qkv: cat16(g, &[&t("attn_q.weight"), &t("attn_k.weight"), &t("attn_v.weight")])?,
                    out: up16(g, &t("attn_output.weight"))?,
                    qn: up32(g, &t("attn_q_norm.weight"))?,
                    kn: up32(g, &t("attn_k_norm.weight"))?,
                    idx: n_attn,
                };
                n_attn += 1;
                m
            } else {
                let m = Mixer::Conv {
                    in_proj: up16(g, &t("shortconv.in_proj.weight"))?,
                    out_proj: up16(g, &t("shortconv.out_proj.weight"))?,
                    convw: up32(g, &t("shortconv.conv.weight"))?,
                    idx: n_conv,
                };
                n_conv += 1;
                m
            };
            let ffn = g.dims(&t("ffn_gate.weight"))?[0];
            max_ffn = max_ffn.max(ffn);
            layers.push(TrunkLayer {
                mixer,
                op_norm: up32(g, &t("attn_norm.weight"))?,
                ffn_norm: up32(g, &t("ffn_norm.weight"))?,
                gate_up: cat16(g, &[&t("ffn_gate.weight"), &t("ffn_up.weight")])?,
                down: up16(g, &t("ffn_down.weight"))?,
                ffn,
            });
        }
        let mut head = Vec::new();
        for i in n_trunk..n_layers {
            let p = format!("blk.{i}.");
            let t = |s: &str| format!("{p}{s}");
            let ffn = g.dims(&t("ffn_up.weight"))?[0];
            head.push(HeadLayer {
                n1_w: up32(g, &t("attn_norm.weight"))?,
                n1_b: up32(g, &t("attn_norm.bias"))?,
                qkv: up16(g, &t("attn_qkv.weight"))?,
                qkv_b: up32(g, &t("attn_qkv.bias"))?,
                out: up16(g, &t("attn_output.weight"))?,
                out_b: up32(g, &t("attn_output.bias"))?,
                n2_w: up32(g, &t("ffn_norm.weight"))?,
                n2_b: up32(g, &t("ffn_norm.bias"))?,
                up: up16(g, &t("ffn_up.weight"))?,
                up_b: up32(g, &t("ffn_up.bias"))?,
                down: up16(g, &t("ffn_down.weight"))?,
                down_b: up32(g, &t("ffn_down.bias"))?,
                ffn,
            });
            max_ffn = max_ffn.max(ffn);
        }
        let out_b = g.f32("cls.output.bias")?[0];
        Ok(TextModel {
            vocab: emb_dims[0],
            embed: up16(g, "token_embd.weight")?,
            final_norm: up32(g, "token_embd_norm.weight")?,
            layers,
            head,
            type_emb: up32(g, "token_types.weight")?,
            cls_norm_w: up32(g, "cls.norm.weight")?,
            cls_norm_b: up32(g, "cls.norm.bias")?,
            cls_w: up16(g, "cls.weight")?,
            cls_b: up32(g, "cls.bias")?,
            out_w: up16(g, "cls.output.weight")?,
            out_b,
            rope: DevBuf::from_slice(&rope_table(theta)),
            n_attn,
            n_conv,
            eps,
            max_ffn,
        })
    }
}

// ------------------------------------------------------------------------------------------------ batches

/// One question sequence of a text batch.
pub struct TextSeq<'a> {
    pub ids: &'a [i32],
    pub markers: &'a [i32],
    pub qtype: i32,
    pub prefix: Option<&'a PrefixCache>,
}

/// One media prefix to push through the trunk: fp32 embeddings [len, D] on the device -> `cache`.
pub struct PrefixSeq<'a> {
    pub emb: *const f32,
    pub cache: &'a PrefixCache,
}

const META_BYTES: usize = 8 << 20;
pub const SLOTS: usize = 2;

/// Scratch for one in-flight batch pipeline (activations are shared, host staging is double-buffered).
pub struct Workspace {
    pub cap_tokens: usize,
    pub cap_markers: usize,
    pub h: DevBuf,
    pub xn: DevBuf,
    pub big: DevBuf,
    pub act: DevBuf,
    pub mq: DevBuf,   // [M, 3*D] gathered head queries
    pub mo: DevBuf,   // [M, D]
    pub mh: DevBuf,   // [M, D] fp32
    pub mx: DevBuf,   // [M, D]
    pub mbig: DevBuf, // [M, ffn]
    pub logits: DevBuf,
    pub meta: DevBuf,
    pub host_meta: Vec<HostBuf>,
    pub host_logits: Vec<HostBuf>,
    pub events: Vec<cuda::Event>,
    pub busy: Vec<bool>,
}
unsafe impl Send for Workspace {}

#[allow(dead_code)]
impl Workspace {
    pub fn new(m: &TextModel, cap_tokens: usize) -> Workspace {
        let cap_markers = cap_tokens / 2 + 64;
        let big_w = (2 * m.max_ffn).max(3 * D);
        Workspace {
            cap_tokens,
            cap_markers,
            h: DevBuf::new(cap_tokens * D * 4),
            xn: DevBuf::new(cap_tokens * D * 2),
            big: DevBuf::new(cap_tokens * big_w * 2),
            act: DevBuf::new(cap_tokens * m.max_ffn.max(D) * 2),
            mq: DevBuf::new(cap_markers * 3 * D * 2),
            mo: DevBuf::new(cap_markers * D * 2),
            mh: DevBuf::new(cap_markers * D * 4),
            mx: DevBuf::new(cap_markers * D * 2),
            mbig: DevBuf::new(cap_markers * m.max_ffn * 2),
            logits: DevBuf::new(cap_markers * 4),
            meta: DevBuf::new(META_BYTES),
            host_meta: (0..SLOTS).map(|_| HostBuf::new(META_BYTES)).collect(),
            host_logits: (0..SLOTS).map(|_| HostBuf::new(cap_markers * 4)).collect(),
            events: (0..SLOTS).map(|_| cuda::new_event(false)).collect(),
            busy: vec![false; SLOTS],
        }
    }
    pub fn bytes(&self) -> usize {
        [&self.h, &self.xn, &self.big, &self.act, &self.mq, &self.mo, &self.mh, &self.mx, &self.mbig, &self.logits, &self.meta]
            .iter()
            .map(|b| b.bytes)
            .sum()
    }
}

/// Bump writer over a pinned host buffer; returns device-side addresses for each section.
struct MetaWriter {
    host: *mut u8,
    dev: *mut u8,
    off: usize,
}
impl MetaWriter {
    fn put<T: Copy>(&mut self, items: &[T]) -> *const T {
        self.off = self.off.div_ceil(16) * 16;
        let bytes = std::mem::size_of_val(items);
        assert!(self.off + bytes <= META_BYTES, "batch metadata exceeds {META_BYTES} bytes");
        unsafe {
            ptr::copy_nonoverlapping(items.as_ptr() as *const u8, self.host.add(self.off), bytes);
        }
        let d = unsafe { self.dev.add(self.off) } as *const T;
        self.off += bytes;
        d
    }
}

fn work_items(seqs: &[(usize, usize)]) -> Vec<[i32; 2]> {
    // (seq index, query tile start); longest sequences first for better tail behaviour
    let mut w = Vec::new();
    for (i, &(_, len)) in seqs.iter().enumerate() {
        let mut q = 0;
        while q < len {
            w.push([i as i32, q as i32]);
            q += 64;
        }
    }
    w
}

/// A launched batch: wait on `event`, then read `n` logits from the slot's host buffer.
pub struct Pending {
    pub slot: usize,
    pub n_logits: usize,
}

pub struct Runner<'a> {
    pub m: &'a TextModel,
    pub blas: &'a Blas,
    pub st: Stream,
}

impl<'a> Runner<'a> {
    #[allow(clippy::too_many_arguments)]
    fn trunk(&self, ws: &Workspace, t: usize, tok_seq: *const i32, seqs: *const SeqInfo, attn: &[*const AttnSeq],
             work: *const i32, n_work: usize, prefix_pass: bool) {
        let m = self.m;
        let st = self.st;
        let b = self.blas;
        let (h, xn, big, act) = (ws.h.f32(), ws.xn.f16(), ws.big.f16(), ws.act.f16());
        let nl = m.layers.len();
        unsafe {
            for (li, l) in m.layers.iter().enumerate() {
                let last = li + 1 == nl;
                cuda::check(k::d1_rmsnorm(h, l.op_norm.f32(), xn, t as i32, D as i32, m.eps, st));
                match &l.mixer {
                    Mixer::Conv { in_proj, out_proj, convw, idx } => {
                        b.gemm(t, 3 * D, D, xn, D, in_proj.f16(), D, false, Out::H(big), 3 * D, 1.0, 0.0);
                        cuda::check(k::d1_shortconv(big, convw.f32(), tok_seq, seqs, act, t as i32, D as i32, *idx as i32, st));
                        if prefix_pass && last {
                            return; // only the conv state of the last layer is needed
                        }
                        b.gemm(t, D, D, act, D, out_proj.f16(), D, false, Out::F(h), D, 1.0, 1.0);
                    }
                    Mixer::Attn { qkv, out, qn, kn, idx } => {
                        let ld = D + 2 * KV_DIM;
                        b.gemm(t, ld, D, xn, D, qkv.f16(), D, false, Out::H(big), ld, 1.0, 0.0);
                        cuda::check(k::d1_qk_norm_rope(
                            big, ld as i32, HEADS as i32, KV_HEADS as i32, qn.f32(), kn.f32(), m.rope.f32(), tok_seq, seqs,
                            *idx as i32, t as i32, m.eps, st,
                        ));
                        cuda::check(k::d1_flash_attn(
                            big, ld as i32, big.add(D), big.add(D + KV_DIM), ld as i32, KV_DIM as i32, ptr::null(),
                            ptr::null(), ptr::null(), act, D as i32, attn[*idx], work, n_work as i32, HEADS as i32,
                            (HEADS / KV_HEADS) as i32, 0.125, st,
                        ));
                        if prefix_pass && last {
                            return;
                        }
                        b.gemm(t, D, D, act, D, out.f16(), D, false, Out::F(h), D, 1.0, 1.0);
                    }
                }
                cuda::check(k::d1_rmsnorm(h, l.ffn_norm.f32(), xn, t as i32, D as i32, m.eps, st));
                b.gemm(t, 2 * l.ffn, D, xn, D, l.gate_up.f16(), D, false, Out::H(big), 2 * l.ffn, 1.0, 0.0);
                cuda::check(k::d1_swiglu(big, act, t as i32, l.ffn as i32, st));
                b.gemm(t, D, l.ffn, act, l.ffn, l.down.f16(), l.ffn, false, Out::F(h), D, 1.0, 1.0);
            }
        }
    }

    /// Enqueue a prefix pass: media embeddings -> trunk -> per-layer K/V + conv state saved into each cache.
    pub fn launch_prefix(&self, ws: &mut Workspace, slot: usize, items: &[PrefixSeq]) {
        let m = self.m;
        let st = self.st;
        let t: usize = items.iter().map(|p| p.cache.len).sum();
        assert!(t <= ws.cap_tokens, "prefix batch of {t} tokens exceeds workspace capacity {}", ws.cap_tokens);
        self.wait_slot(ws, slot);
        let mut w = MetaWriter { host: ws.host_meta[slot].ptr, dev: ws.meta.ptr as *mut u8, off: 0 };
        let mut tok_seq = Vec::with_capacity(t);
        let mut seqs = Vec::with_capacity(items.len());
        let mut spans = Vec::new();
        let mut start = 0;
        for (i, p) in items.iter().enumerate() {
            let n = p.cache.len;
            tok_seq.extend(std::iter::repeat_n(i as i32, n));
            seqs.push(SeqInfo {
                start: start as i32,
                len: n as i32,
                pos_off: 0,
                qtype: 0,
                ext_bx: ptr::null(),
                save_bx: p.cache.bx,
                save_k: p.cache.k,
                save_v: p.cache.v,
            });
            spans.push((start, n));
            start += n;
        }
        let d_tok = w.put(&tok_seq);
        let d_seqs = w.put(&seqs);
        let mut attn = Vec::new();
        for _ in 0..m.n_attn {
            let a: Vec<AttnSeq> = spans
                .iter()
                .map(|&(s, n)| AttnSeq {
                    q_start: s as i32,
                    q_len: n as i32,
                    kv_start: s as i32,
                    kv_len: n as i32,
                    ext_len: 0,
                    _pad: 0,
                    ext_k: ptr::null(),
                    ext_v: ptr::null(),
                })
                .collect();
            attn.push(w.put(&a));
        }
        let work = work_items(&spans);
        let d_work = w.put(&work) as *const i32;
        unsafe {
            cuda::h2d_async(ws.meta.ptr, ws.host_meta[slot].ptr as *const c_void, w.off, st);
            for (p, &(s, n)) in items.iter().zip(&spans) {
                cuda::d2d_async(ws.h.f32().add(s * D) as *mut c_void, p.emb as *const c_void, n * D * 4, st);
            }
        }
        self.trunk(ws, t, d_tok, d_seqs, &attn, d_work, work.len(), true);
        cuda::event_record(ws.events[slot], st);
        ws.busy[slot] = true;
    }

    fn wait_slot(&self, ws: &mut Workspace, slot: usize) {
        if ws.busy[slot] {
            cuda::event_sync(ws.events[slot]);
            ws.busy[slot] = false;
        }
    }

    /// Enqueue a text batch. Logits (one per option marker, in sequence then marker order) land in the slot's
    /// pinned buffer once `ws.events[slot]` completes.
    pub fn launch_text(&self, ws: &mut Workspace, slot: usize, seqs_in: &[TextSeq]) -> Pending {
        let m = self.m;
        let st = self.st;
        let b = self.blas;
        let t: usize = seqs_in.iter().map(|s| s.ids.len()).sum();
        let n_mark: usize = seqs_in.iter().map(|s| s.markers.len()).sum();
        assert!(t <= ws.cap_tokens, "text batch of {t} tokens exceeds workspace capacity {}", ws.cap_tokens);
        assert!(n_mark <= ws.cap_markers);
        self.wait_slot(ws, slot);
        let mut w = MetaWriter { host: ws.host_meta[slot].ptr, dev: ws.meta.ptr as *mut u8, off: 0 };

        let mut ids = Vec::with_capacity(t);
        let mut tok_seq = Vec::with_capacity(t);
        let mut seqs = Vec::with_capacity(seqs_in.len());
        let mut spans = Vec::with_capacity(seqs_in.len());
        let mut marker_rows = Vec::with_capacity(n_mark);
        let mut mspans = Vec::with_capacity(seqs_in.len());
        let mut start = 0;
        for (i, s) in seqs_in.iter().enumerate() {
            let n = s.ids.len();
            ids.extend_from_slice(s.ids);
            tok_seq.extend(std::iter::repeat_n(i as i32, n));
            seqs.push(SeqInfo {
                start: start as i32,
                len: n as i32,
                pos_off: s.prefix.map_or(0, |p| p.len as i32),
                qtype: s.qtype,
                ext_bx: s.prefix.map_or(ptr::null(), |p| p.bx as *const F16),
                ..Default::default()
            });
            mspans.push((marker_rows.len(), s.markers.len()));
            for &mk in s.markers {
                marker_rows.push((start + mk as usize) as i32);
            }
            spans.push((start, n));
            start += n;
        }
        let d_ids = w.put(&ids);
        let d_tok = w.put(&tok_seq);
        let d_seqs = w.put(&seqs);
        let mut attn = Vec::new();
        for l in 0..m.n_attn {
            let a: Vec<AttnSeq> = seqs_in
                .iter()
                .zip(&spans)
                .map(|(s, &(st0, n))| {
                    let (el, ek, ev) = match s.prefix {
                        Some(p) => (
                            p.len as i32,
                            unsafe { p.k.add(l * p.len * KV_DIM) } as *const F16,
                            unsafe { p.v.add(l * p.len * KV_DIM) } as *const F16,
                        ),
                        None => (0, ptr::null(), ptr::null()),
                    };
                    AttnSeq { q_start: st0 as i32, q_len: n as i32, kv_start: st0 as i32, kv_len: n as i32, ext_len: el, _pad: 0, ext_k: ek, ext_v: ev }
                })
                .collect();
            attn.push(w.put(&a));
        }
        let head_attn: Vec<AttnSeq> = spans
            .iter()
            .map(|&(s, n)| AttnSeq { q_start: s as i32, q_len: n as i32, kv_start: s as i32, kv_len: n as i32, ext_len: 0, _pad: 0, ext_k: ptr::null(), ext_v: ptr::null() })
            .collect();
        let d_head_attn = w.put(&head_attn);
        let last_attn: Vec<AttnSeq> = spans
            .iter()
            .zip(&mspans)
            .map(|(&(s, n), &(ms, mn))| AttnSeq { q_start: ms as i32, q_len: mn as i32, kv_start: s as i32, kv_len: n as i32, ext_len: 0, _pad: 0, ext_k: ptr::null(), ext_v: ptr::null() })
            .collect();
        let d_last_attn = w.put(&last_attn);
        let work = work_items(&spans);
        let d_work = w.put(&work) as *const i32;
        let mwork = work_items(&mspans);
        let d_mwork = w.put(&mwork) as *const i32;
        let d_mrows = w.put(&marker_rows);

        let (h, xn, big, act) = (ws.h.f32(), ws.xn.f16(), ws.big.f16(), ws.act.f16());
        unsafe {
            cuda::h2d_async(ws.meta.ptr, ws.host_meta[slot].ptr as *const c_void, w.off, st);
            cuda::check(k::d1_embed(m.embed.f16(), d_ids, h, t as i32, D as i32, st));
        }
        self.trunk(ws, t, d_tok, d_seqs, &attn, d_work, work.len(), false);
        unsafe {
            cuda::check(k::d1_head_init(h, m.final_norm.f32(), m.type_emb.f32(), d_tok, d_seqs, h, t as i32, m.eps, st));
            let mut pend: *const f32 = ptr::null();
            let nh = m.head.len();
            let qkv_ld = 3 * D;
            for (li, l) in m.head.iter().enumerate() {
                cuda::check(k::d1_layernorm(h, pend, 1.0, l.n1_w.f32(), l.n1_b.f32(), xn, ptr::null_mut(), t as i32, D as i32, 1e-5, st));
                b.gemm(t, qkv_ld, D, xn, D, l.qkv.f16(), D, false, Out::H(big), qkv_ld, 1.0, 0.0);
                let qb = l.qkv_b.f32();
                if li + 1 < nh {
                    cuda::check(k::d1_flash_attn(
                        big, qkv_ld as i32, big.add(D), big.add(2 * D), qkv_ld as i32, 0, qb, qb.add(D), qb.add(2 * D),
                        act, D as i32, d_head_attn, d_work, work.len() as i32, HEADS as i32, 1, 0.125, st,
                    ));
                    b.gemm(t, D, D, act, D, l.out.f16(), D, false, Out::F(h), D, 1.0, 1.0);
                    cuda::check(k::d1_layernorm(h, l.out_b.f32(), 1.0, l.n2_w.f32(), l.n2_b.f32(), xn, ptr::null_mut(), t as i32, D as i32, 1e-5, st));
                    b.gemm(t, l.ffn, D, xn, D, l.up.f16(), D, false, Out::H(big), l.ffn, 1.0, 0.0);
                    cuda::check(k::d1_bias_act_h(big, l.up_b.f32(), t as i32, l.ffn as i32, 1, st));
                    b.gemm(t, D, l.ffn, big, l.ffn, l.down.f16(), l.ffn, false, Out::F(h), D, 1.0, 1.0);
                    pend = l.down_b.f32();
                } else {
                    // Last head layer: only the option markers are scored, so queries, the output projection and
                    // the FFN run on marker rows only (keys/values still cover every text position).
                    let (mq, mo, mh, mx, mbig) = (ws.mq.f16(), ws.mo.f16(), ws.mh.f32(), ws.mx.f16(), ws.mbig.f16());
                    let nm = n_mark as i32;
                    cuda::check(k::d1_gather_rows(big as *const c_void, mq as *mut c_void, d_mrows, nm, (qkv_ld * 2) as i32, st));
                    cuda::check(k::d1_gather_rows(h as *const c_void, mh as *mut c_void, d_mrows, nm, (D * 4) as i32, st));
                    cuda::check(k::d1_flash_attn(
                        mq, qkv_ld as i32, big.add(D), big.add(2 * D), qkv_ld as i32, 0, qb, qb.add(D), qb.add(2 * D),
                        mo, D as i32, d_last_attn, d_mwork, mwork.len() as i32, HEADS as i32, 1, 0.125, st,
                    ));
                    b.gemm(n_mark, D, D, mo, D, l.out.f16(), D, false, Out::F(mh), D, 1.0, 1.0);
                    cuda::check(k::d1_layernorm(mh, l.out_b.f32(), 1.0, l.n2_w.f32(), l.n2_b.f32(), mx, ptr::null_mut(), nm, D as i32, 1e-5, st));
                    b.gemm(n_mark, l.ffn, D, mx, D, l.up.f16(), D, false, Out::H(mbig), l.ffn, 1.0, 0.0);
                    cuda::check(k::d1_bias_act_h(mbig, l.up_b.f32(), nm, l.ffn as i32, 1, st));
                    b.gemm(n_mark, D, l.ffn, mbig, l.ffn, l.down.f16(), l.ffn, false, Out::F(mh), D, 1.0, 1.0);
                    // scorer: LN -> Linear -> GELU -> Linear(1)
                    cuda::check(k::d1_layernorm(mh, l.down_b.f32(), 1.0, m.cls_norm_w.f32(), m.cls_norm_b.f32(), mx, ptr::null_mut(), nm, D as i32, 1e-5, st));
                    b.gemm(n_mark, D, D, mx, D, m.cls_w.f16(), D, false, Out::H(mo), D, 1.0, 0.0);
                    cuda::check(k::d1_scorer_out(mo, m.cls_b.f32(), m.out_w.f16(), m.out_b, ws.logits.f32(), nm, D as i32, st));
                }
            }
            cuda::d2h_async(ws.host_logits[slot].ptr as *mut c_void, ws.logits.ptr, n_mark * 4, st);
        }
        cuda::event_record(ws.events[slot], st);
        ws.busy[slot] = true;
        Pending { slot, n_logits: n_mark }
    }

    /// Block until a launched text batch is done and return its logits.
    pub fn finish(&self, ws: &mut Workspace, p: &Pending) -> Vec<f32> {
        cuda::event_sync(ws.events[p.slot]);
        ws.busy[p.slot] = false;
        let src = ws.host_logits[p.slot].ptr as *const f32;
        unsafe { std::slice::from_raw_parts(src, p.n_logits).to_vec() }
    }
}
