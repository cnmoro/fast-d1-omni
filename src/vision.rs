//! Images to prefix embeddings: LFM2-VL tiling, a SigLIP2 NaFlex tower and the 2x2 pixel-unshuffle projector.

use crate::cuda::{self, k, AttnSeq, Blas, DevBuf, Out, Stream, F16};
use crate::gguf::Gguf;
use crate::image::{self, Image};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

const TILE: usize = 512;
const PATCH: usize = 16;
const VD: usize = 768;
const VHEADS: usize = 12;
const VFF: usize = 3072;
const OUT_D: usize = 1024;

/// Ratios in the order the reference's `sorted(set(...), key=x*y)` yields them (tie order matters).
const RATIOS: [(usize, usize); 26] = [
    (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1), (3, 2), (2, 3), (7, 1),
    (1, 7), (4, 2), (2, 4), (1, 8), (8, 1), (1, 9), (3, 3), (9, 1), (2, 5), (5, 2), (10, 1), (1, 10),
];

fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

pub struct Plan {
    pub grid: (usize, usize), // (cols, rows)
    pub thumb: (usize, usize), // (h, w)
    pub tiled: bool,
}

/// LFM2-VL smart resize, tile grid and thumbnail.
pub fn layout(width: usize, height: usize) -> Plan {
    let (fw, fh) = (width as f64, height as f64);
    let factor = 32.0f64;
    let (maximum, minimum) = (256.0 * 1024.0, 64.0 * 1024.0);
    let mut h = factor.max(py_round(fh / factor) * factor);
    let mut w = factor.max(py_round(fw / factor) * factor);
    if h * w > maximum {
        let beta = (fh * fw / maximum).sqrt();
        h = factor.max((fh / beta / factor).floor() * factor);
        w = factor.max((fw / beta / factor).floor() * factor);
    } else if h * w < minimum {
        let beta = (minimum / (fh * fw)).sqrt();
        h = (fh * beta / factor).ceil() * factor;
        w = (fw * beta / factor).ceil() * factor;
    }
    let large = 16f64.max(py_round(fh / factor) * factor) * 16f64.max(py_round(fw / factor) * factor) > maximum * 2.0;
    let mut grid = (1, 1);
    if large {
        let mut best = f64::INFINITY;
        for &(x, y) in &RATIOS {
            let diff = (fw / fh - x as f64 / y as f64).abs();
            if diff < best || (diff == best && fw * fh > 0.5 * (TILE * TILE * x * y) as f64) {
                grid = (x, y);
                best = diff;
            }
        }
    }
    Plan { grid, thumb: (h as usize, w as usize), tiled: large }
}

/// Patchified crops of one image: fp16 patch rows [n, 768] in (y, x, c) order, and each crop's (ph, pw).
pub struct Patches {
    pub data: Vec<F16>,
    pub shapes: Vec<(usize, usize)>,
}

impl Patches {
    pub fn tokens(&self) -> usize {
        self.shapes.iter().map(|(h, w)| h * w / 4).sum()
    }
}

fn patchify(crop: &Image, out: &mut Vec<F16>) -> (usize, usize) {
    static LUT: std::sync::OnceLock<[F16; 256]> = std::sync::OnceLock::new();
    let lut = LUT.get_or_init(|| std::array::from_fn(|v| cuda::f32_to_f16((v as f32 - 127.5) / 127.5)));
    let (ph, pw) = (crop.h / PATCH, crop.w / PATCH);
    let base = out.len();
    out.resize(base + ph * pw * PATCH * PATCH * 3, 0);
    let dst = &mut out[base..];
    let prow = PATCH * PATCH * 3;
    for py in 0..ph {
        for px in 0..pw {
            let d = &mut dst[(py * pw + px) * prow..(py * pw + px + 1) * prow];
            for y in 0..PATCH {
                let s = ((py * PATCH + y) * crop.w + px * PATCH) * 3;
                for (o, &v) in d[y * PATCH * 3..(y + 1) * PATCH * 3].iter_mut().zip(&crop.rgb[s..s + PATCH * 3]) {
                    *o = lut[v as usize];
                }
            }
        }
    }
    (ph, pw)
}

/// Decode-independent preprocessing: tiles (if large) then the thumbnail.
pub fn preprocess(img: &Image) -> Patches {
    let plan = layout(img.w, img.h);
    let mut data = Vec::new();
    let mut shapes = Vec::new();
    if plan.tiled {
        let (gw, gh) = plan.grid;
        let big = image::resize(img, gh * TILE, gw * TILE);
        for r in 0..gh {
            for c in 0..gw {
                let mut rgb = Vec::with_capacity(TILE * TILE * 3);
                for y in 0..TILE {
                    let s = ((r * TILE + y) * big.w + c * TILE) * 3;
                    rgb.extend_from_slice(&big.rgb[s..s + TILE * 3]);
                }
                shapes.push(patchify(&Image { w: TILE, h: TILE, rgb }, &mut data));
            }
        }
    }
    let thumb = image::resize(img, plan.thumb.0, plan.thumb.1);
    shapes.push(patchify(&thumb, &mut data));
    Patches { data, shapes }
}

struct VLayer {
    ln1_w: DevBuf,
    ln1_b: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    out: DevBuf,
    out_b: DevBuf,
    ln2_w: DevBuf,
    ln2_b: DevBuf,
    fc1: DevBuf,
    fc1_b: DevBuf,
    fc2: DevBuf,
    fc2_b: DevBuf,
}

pub struct VisionModel {
    patch_w: DevBuf,
    patch_b: Vec<f32>,
    pos: Vec<f32>, // [16*16][768]
    pos_cache: Mutex<HashMap<(usize, usize), DevBuf>>,
    layers: Vec<VLayer>,
    post_w: DevBuf,
    post_b: DevBuf,
    proj1: DevBuf,
    proj1_b: DevBuf,
    proj2: DevBuf,
    proj2_b: DevBuf,
    eps: f32,
    pub max_patches: usize,
    // scratch
    h: DevBuf,
    xn: DevBuf,
    big: DevBuf,
    act: DevBuf,
    pix: DevBuf,
    meta: DevBuf,
}

fn f32v(g: &Gguf, n: &str) -> Result<DevBuf, String> {
    Ok(DevBuf::from_slice(&g.f32(n)?))
}
fn f16v(g: &Gguf, n: &str) -> Result<DevBuf, String> {
    Ok(DevBuf::from_slice(&g.f16(n)?))
}

/// Separable antialiased bilinear (float) like F.interpolate(antialias=True, align_corners=False).
fn aa_weights(inp: usize, out: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = inp as f64 / out as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..out)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(inp as i64) as usize;
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|j| {
                    let x = (j as f64 - center + 0.5) * invscale;
                    if x.abs() < 1.0 { 1.0 - x.abs() } else { 0.0 }
                })
                .collect();
            let t: f64 = w.iter().sum();
            if t != 0.0 {
                w.iter_mut().for_each(|v| *v /= t);
            }
            (xmin, w.into_iter().map(|v| v as f32).collect())
        })
        .collect()
}

impl VisionModel {
    pub fn load(g: &Gguf, max_patches: usize) -> Result<VisionModel, String> {
        let nl = g.u("clip.vision.block_count").unwrap_or(12) as usize;
        let eps = g.f("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6) as f32;
        // patch embedding: GGUF [768][3][16][16] -> [768][(y*16+x)*3+c]
        let pw = g.f32("v.patch_embd.weight")?;
        let mut w = vec![0u16; VD * VD];
        for o in 0..VD {
            for c in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        w[o * VD + (y * PATCH + x) * 3 + c] = cuda::f32_to_f16(pw[o * VD + c * 256 + y * PATCH + x]);
                    }
                }
            }
        }
        let mut layers = Vec::new();
        for i in 0..nl {
            let t = |s: &str| format!("v.blk.{i}.{s}");
            let mut qkv = g.f16(&t("attn_q.weight"))?;
            qkv.extend(g.f16(&t("attn_k.weight"))?);
            qkv.extend(g.f16(&t("attn_v.weight"))?);
            let mut qkv_b = g.f32(&t("attn_q.bias"))?;
            qkv_b.extend(g.f32(&t("attn_k.bias"))?);
            qkv_b.extend(g.f32(&t("attn_v.bias"))?);
            layers.push(VLayer {
                ln1_w: f32v(g, &t("ln1.weight"))?,
                ln1_b: f32v(g, &t("ln1.bias"))?,
                qkv: DevBuf::from_slice(&qkv),
                qkv_b: DevBuf::from_slice(&qkv_b),
                out: f16v(g, &t("attn_out.weight"))?,
                out_b: f32v(g, &t("attn_out.bias"))?,
                ln2_w: f32v(g, &t("ln2.weight"))?,
                ln2_b: f32v(g, &t("ln2.bias"))?,
                fc1: f16v(g, &t("ffn_up.weight"))?,
                fc1_b: f32v(g, &t("ffn_up.bias"))?,
                fc2: f16v(g, &t("ffn_down.weight"))?,
                fc2_b: f32v(g, &t("ffn_down.bias"))?,
            });
        }
        let cap = max_patches;
        Ok(VisionModel {
            patch_w: DevBuf::from_slice(&w),
            patch_b: g.f32("v.patch_embd.bias")?,
            pos: g.f32("v.position_embd.weight")?,
            pos_cache: Mutex::new(HashMap::new()),
            layers,
            post_w: f32v(g, "v.post_ln.weight")?,
            post_b: f32v(g, "v.post_ln.bias")?,
            proj1: f16v(g, "mm.1.weight")?,
            proj1_b: f32v(g, "mm.1.bias")?,
            proj2: f16v(g, "mm.2.weight")?,
            proj2_b: f32v(g, "mm.2.bias")?,
            eps,
            max_patches: cap,
            h: DevBuf::new(cap * VD * 4),
            xn: DevBuf::new(cap * VD * 2),
            big: DevBuf::new(cap * VFF * 2),
            act: DevBuf::new(cap * VD * 2),
            pix: DevBuf::new(cap * VD * 2),
            meta: DevBuf::new(1 << 20),
        })
    }

    /// Resized position embeddings + patch bias for a (ph, pw) grid, cached on the GPU.
    fn pos_for(&self, ph: usize, pw: usize) -> *const f32 {
        let mut c = self.pos_cache.lock().unwrap();
        if let Some(b) = c.get(&(ph, pw)) {
            return b.f32();
        }
        let wy = aa_weights(16, ph);
        let wx = aa_weights(16, pw);
        // horizontal pass: [16][pw][768]
        let mut tmp = vec![0f32; 16 * pw * VD];
        for y in 0..16 {
            for (x, (s, ws)) in wx.iter().enumerate() {
                for (j, &wt) in ws.iter().enumerate() {
                    let src = &self.pos[((y * 16) + s + j) * VD..((y * 16) + s + j + 1) * VD];
                    let dst = &mut tmp[(y * pw + x) * VD..(y * pw + x + 1) * VD];
                    for c in 0..VD {
                        dst[c] += wt * src[c];
                    }
                }
            }
        }
        let mut out = vec![0f32; ph * pw * VD];
        for (y, (s, ws)) in wy.iter().enumerate() {
            for x in 0..pw {
                let dst = (y * pw + x) * VD;
                for (j, &wt) in ws.iter().enumerate() {
                    let src = ((s + j) * pw + x) * VD;
                    for c in 0..VD {
                        out[dst + c] += wt * tmp[src + c];
                    }
                }
                for c in 0..VD {
                    out[dst + c] += self.patch_b[c];
                }
            }
        }
        let b = DevBuf::from_slice(&out);
        let p = b.f32();
        c.insert((ph, pw), b);
        p
    }

    /// Run crops through the tower and projector. `out` receives fp32 [Σ ph*pw/4, 1024] in crop order.
    pub fn encode(&self, blas: &Blas, st: Stream, items: &[&Patches], out: *mut f32) {
        let shapes: Vec<(usize, usize)> = items.iter().flat_map(|p| p.shapes.iter().copied()).collect();
        let t: usize = shapes.iter().map(|(h, w)| h * w).sum();
        assert!(t <= self.max_patches, "{t} patches exceed the vision batch capacity {}", self.max_patches);
        // stage patches + metadata
        let mut seqs = Vec::new();
        let mut work: Vec<[i32; 2]> = Vec::new();
        let mut off = 0;
        for (i, &(ph, pw)) in shapes.iter().enumerate() {
            let n = ph * pw;
            seqs.push(AttnSeq { q_start: off as i32, q_len: n as i32, kv_start: off as i32, kv_len: n as i32, ext_len: 0, _pad: 0, ext_k: ptr::null(), ext_v: ptr::null() });
            let mut q = 0;
            while q < n {
                work.push([i as i32, q as i32]);
                q += 64;
            }
            off += n;
        }
        let sb = std::mem::size_of_val(seqs.as_slice());
        let wb = std::mem::size_of_val(work.as_slice());
        let woff = sb.div_ceil(16) * 16;
        assert!(woff + wb <= self.meta.bytes);
        // metadata and pixels are staged from pageable memory: such copies are consumed before the call returns,
        // so no host buffer outlives this function and no stream sync is needed
        let mut meta = vec![0u8; woff + wb];
        unsafe {
            ptr::copy_nonoverlapping(seqs.as_ptr() as *const u8, meta.as_mut_ptr(), sb);
            ptr::copy_nonoverlapping(work.as_ptr() as *const u8, meta.as_mut_ptr().add(woff), wb);
            cuda::h2d_async(self.meta.ptr, meta.as_ptr() as *const c_void, woff + wb, st);
            let mut po = 0;
            for p in items {
                cuda::h2d_async(self.pix.f16().add(po) as *mut c_void, p.data.as_ptr() as *const c_void, p.data.len() * 2, st);
                po += p.data.len();
            }
        }
        let d_seqs = self.meta.ptr as *const AttnSeq;
        let d_work = unsafe { (self.meta.ptr as *const u8).add(woff) } as *const i32;
        let (h, xn, big, act) = (self.h.f32(), self.xn.f16(), self.big.f16(), self.act.f16());
        unsafe {
            // embeddings: h = pos + bias (per crop), then += pixels · W^T
            let mut o = 0;
            for &(ph, pw) in &shapes {
                let p = self.pos_for(ph, pw);
                cuda::d2d_async(h.add(o * VD) as *mut c_void, p as *const c_void, ph * pw * VD * 4, st);
                o += ph * pw;
            }
            blas.gemm(t, VD, VD, self.pix.f16(), VD, self.patch_w.f16(), VD, false, Out::F(h), VD, 1.0, 1.0);
            let mut pend: *const f32 = ptr::null();
            let ti = t as i32;
            for l in &self.layers {
                cuda::check(k::d1_layernorm(h, pend, 1.0, l.ln1_w.f32(), l.ln1_b.f32(), xn, ptr::null_mut(), ti, VD as i32, self.eps, st));
                blas.gemm(t, 3 * VD, VD, xn, VD, l.qkv.f16(), VD, false, Out::H(big), 3 * VD, 1.0, 0.0);
                let qb = l.qkv_b.f32();
                cuda::check(k::d1_flash_attn(
                    big, (3 * VD) as i32, big.add(VD), big.add(2 * VD), (3 * VD) as i32, 0, qb, qb.add(VD), qb.add(2 * VD),
                    act, VD as i32, d_seqs, d_work, work.len() as i32, VHEADS as i32, 1, 0.125, st,
                ));
                blas.gemm(t, VD, VD, act, VD, l.out.f16(), VD, false, Out::F(h), VD, 1.0, 1.0);
                cuda::check(k::d1_layernorm(h, l.out_b.f32(), 1.0, l.ln2_w.f32(), l.ln2_b.f32(), xn, ptr::null_mut(), ti, VD as i32, self.eps, st));
                blas.gemm(t, VFF, VD, xn, VD, l.fc1.f16(), VD, false, Out::H(big), VFF, 1.0, 0.0);
                cuda::check(k::d1_bias_act_h(big, l.fc1_b.f32(), ti, VFF as i32, 3, st));
                blas.gemm(t, VD, VFF, big, VFF, l.fc2.f16(), VFF, false, Out::F(h), VD, 1.0, 1.0);
                pend = l.fc2_b.f32();
            }
            cuda::check(k::d1_layernorm(h, pend, 1.0, self.post_w.f32(), self.post_b.f32(), xn, ptr::null_mut(), ti, VD as i32, self.eps, st));
            // projector: unshuffle each crop -> big [P, 3072] -> linear_1 + GELU -> linear_2 (+bias) -> out
            let mut src = 0;
            let mut dst = 0;
            for &(ph, pw) in &shapes {
                cuda::check(k::d1_unshuffle(xn.add(src * VD), big.add(dst * 4 * VD), ph as i32, pw as i32, VD as i32, st));
                src += ph * pw;
                dst += ph * pw / 4;
            }
            let p = dst;
            let hid = 2048;
            blas.gemm(p, hid, 4 * VD, big, 4 * VD, self.proj1.f16(), 4 * VD, false, Out::H(act), hid, 1.0, 0.0);
            cuda::check(k::d1_bias_act_h(act, self.proj1_b.f32(), p as i32, hid as i32, 2, st));
            cuda::check(k::d1_fill_rows(out, self.proj2_b.f32(), ptr::null(), p as i32, OUT_D as i32, st));
            blas.gemm(p, OUT_D, hid, act, hid, self.proj2.f16(), hid, false, Out::F(out), OUT_D, 1.0, 1.0);
        }
    }
}
