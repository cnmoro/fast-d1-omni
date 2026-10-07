//! Speech to prefix embeddings: WAV decode (+resample to 16 kHz), NeMo log-mel front end (CPU), a 17-layer
//! FastConformer with 8x conv subsampling, an MLP adapter and a residual correction (GPU).

use crate::cuda::{self, k, Blas, DevBuf, Out, Stream, F16};
use crate::gguf::Gguf;
use std::ffi::c_void;
use std::ptr;

pub const SAMPLE_RATE: usize = 16000;
const MIN_SAMPLES: usize = 8000;
const MAX_SECONDS: usize = 30;
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
const DM: usize = 512;
const AHEADS: usize = 8;
const SUB_C: usize = 256;
const OUT_D: usize = 1024;

// ------------------------------------------------------------------------------------------------ WAV

/// Decode a WAV file to mono f32 samples at 16 kHz.
pub fn decode_wav(d: &[u8]) -> Result<Vec<f32>, String> {
    if d.len() < 12 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        return Err("audio must be a WAV file (RIFF/WAVE)".into());
    }
    let mut pos = 12;
    let (mut fmt, mut ch, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= d.len() {
        let id = &d[pos..pos + 4];
        let len = u32::from_le_bytes([d[pos + 4], d[pos + 5], d[pos + 6], d[pos + 7]]) as usize;
        let body = &d[pos + 8..(pos + 8 + len).min(d.len())];
        if id == b"fmt " && body.len() >= 16 {
            fmt = u16::from_le_bytes([body[0], body[1]]);
            ch = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
            if fmt == 0xfffe && body.len() >= 26 {
                fmt = u16::from_le_bytes([body[24], body[25]]);
            }
        } else if id == b"data" {
            data = Some(body);
        }
        pos += 8 + len + (len & 1);
    }
    let data = data.ok_or("WAV has no data chunk")?;
    if ch == 0 || rate == 0 {
        return Err("WAV has no valid fmt chunk".into());
    }
    let bps = bits as usize / 8;
    let frame = bps * ch as usize;
    if frame == 0 {
        return Err("bad WAV format".into());
    }
    let n = data.len() / frame;
    // cap decoding at 30 s of input
    let n = n.min(MAX_SECONDS * rate as usize + rate as usize);
    let sample = |b: &[u8]| -> Result<f32, String> {
        Ok(match (fmt, bits) {
            (1, 8) => (b[0] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8388608.0,
            (1, 32) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2147483648.0,
            (3, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (3, 64) => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
            _ => return Err(format!("unsupported WAV encoding (format {fmt}, {bits} bits)")),
        })
    };
    let mut mono = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = 0f32;
        for c in 0..ch as usize {
            let o = i * frame + c * bps;
            s += sample(&data[o..o + bps])?;
        }
        mono.push(s / ch as f32);
    }
    Ok(if rate as usize == SAMPLE_RATE { mono } else { resample(&mono, rate as usize, SAMPLE_RATE) })
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn bessel_i0(x: f64) -> f64 {
    let (mut s, mut t) = (1.0, 1.0);
    for k in 1..50 {
        t *= (x / (2.0 * k as f64)).powi(2);
        s += t;
    }
    s
}

/// Kaiser-windowed sinc resampler (polyphase: the filter for each output phase is precomputed).
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(from, to);
    let (up, down) = (to / g, from / g); // output n reads input position n * down / up
    let n_out = (x.len() as u64 * up as u64 / down as u64) as usize;
    let cutoff = (up as f64 / down as f64).min(1.0) * 0.97;
    let half = (32.0 / cutoff).ceil() as i64; // taps on each side, in input samples
    let beta = 8.6;
    let i0b = bessel_i0(beta);
    let taps = (2 * half + 1) as usize;
    let tap = |t: f64| -> f32 {
        let r = t / half as f64;
        if r.abs() >= 1.0 {
            return 0.0;
        }
        let sinc = if t == 0.0 { 1.0 } else { (std::f64::consts::PI * t * cutoff).sin() / (std::f64::consts::PI * t * cutoff) };
        (cutoff * sinc * bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b) as f32
    };
    // phase p = (n * down) % up: centre = base + p/up; taps at input base - half + j
    let table: Option<Vec<f32>> = (up <= 4096).then(|| {
        let mut t = vec![0f32; up * taps];
        for p in 0..up {
            let frac = p as f64 / up as f64;
            for j in 0..taps {
                t[p * taps + j] = tap((j as i64 - half) as f64 - frac);
            }
        }
        t
    });
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let pos = n as u64 * down as u64;
        let base = (pos / up as u64) as i64;
        let p = (pos % up as u64) as usize;
        let mut acc = 0f32;
        for j in 0..taps {
            let idx = base - half + j as i64;
            if idx < 0 || idx as usize >= x.len() {
                continue;
            }
            let w = match &table {
                Some(t) => t[p * taps + j],
                None => tap((j as i64 - half) as f64 - p as f64 / up as f64),
            };
            acc += w * x[idx as usize];
        }
        out.push(acc);
    }
    out
}

// ------------------------------------------------------------------------------------------------ mel

fn slaney_filterbank() -> Vec<f32> {
    let (f_sp, min_log_hz, logstep) = (200.0 / 3.0, 1000.0f64, (6.4f64).ln() / 27.0);
    let min_log_mel = min_log_hz / f_sp;
    let hz_to_mel = |f: f64| if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp };
    let mel_to_hz = |m: f64| if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m };
    let nb = N_FFT / 2 + 1;
    let (m0, m1) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let mel_f: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (N_MELS + 1) as f64)).collect();
    let fftfreqs: Vec<f64> = (0..nb).map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64).collect();
    let mut w = vec![0f32; N_MELS * nb];
    for i in 0..N_MELS {
        let fd0 = mel_f[i + 1] - mel_f[i];
        let fd1 = mel_f[i + 2] - mel_f[i + 1];
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = -(mel_f[i] - f) / fd0;
            let upper = (mel_f[i + 2] - f) / fd1;
            let v = (lower.min(upper)).max(0.0) as f32;
            w[i * nb + j] = v * enorm as f32;
        }
    }
    w
}

struct FftTables {
    tw: Vec<(f64, f64)>, // e^{-2 pi i k / N}, k < N/2
    rev: Vec<usize>,
    win: Vec<f64>,
    fb: Vec<f32>,
    fb_range: Vec<(usize, usize)>, // nonzero bins per mel filter
}

fn tables() -> &'static FftTables {
    static T: std::sync::OnceLock<FftTables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let n = N_FFT;
        let tw = (0..n / 2).map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (a.cos(), a.sin())
        }).collect();
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let mut win = vec![0f64; n];
        let off = (n - WIN) / 2;
        for i in 0..WIN {
            win[off + i] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64).cos();
        }
        let fb = slaney_filterbank();
        let nb = n / 2 + 1;
        let fb_range = (0..N_MELS)
            .map(|m| {
                let row = &fb[m * nb..(m + 1) * nb];
                let lo = row.iter().position(|&v| v != 0.0).unwrap_or(0);
                let hi = row.iter().rposition(|&v| v != 0.0).map_or(0, |x| x + 1);
                (lo, hi.max(lo))
            })
            .collect();
        FftTables { tw, rev, win, fb, fb_range }
    })
}

fn fft(re: &mut [f64], im: &mut [f64], t: &FftTables) {
    let n = re.len();
    for i in 0..n {
        let j = t.rev[i];
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = t.tw[k * step];
                let (a, b) = (s + k, s + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

/// Normalised log-mel features, time-major [frames][128], and the number of valid frames.
pub fn mel(samples: &[f32]) -> (Vec<f32>, usize) {
    let mut x: Vec<f32> = samples[..samples.len().min(MAX_SECONDS * SAMPLE_RATE)].to_vec();
    if x.len() < MIN_SAMPLES {
        x.resize(MIN_SAMPLES, 0.0);
    }
    let n = x.len();
    let valid = n / HOP;
    let frames = 1 + n / HOP;
    let mut y = vec![0f32; n];
    y[0] = x[0];
    for i in 1..n {
        y[i] = x[i] - 0.97 * x[i - 1];
    }
    let pad = N_FFT / 2;
    let t = tables();
    let (win, fb) = (&t.win, &t.fb);
    let nb = N_FFT / 2 + 1;
    let mut out = vec![0f32; frames * N_MELS];
    let mut re = vec![0f64; N_FFT];
    let mut im = vec![0f64; N_FFT];
    let mut pw = vec![0f32; nb];
    for f in 0..frames {
        for i in 0..N_FFT {
            let si = (f * HOP + i) as isize - pad as isize;
            re[i] = if si >= 0 && (si as usize) < n { y[si as usize] as f64 * win[i] } else { 0.0 };
            im[i] = 0.0;
        }
        fft(&mut re, &mut im, t);
        for kk in 0..nb {
            pw[kk] = (re[kk] * re[kk] + im[kk] * im[kk]) as f32;
        }
        for m in 0..N_MELS {
            let mut s = 0f32;
            let (lo, hi) = t.fb_range[m];
            for kk in lo..hi {
                s += fb[m * nb + kk] * pw[kk];
            }
            out[f * N_MELS + m] = (s + 2f32.powi(-24)).ln();
        }
    }
    for m in 0..N_MELS {
        let mut mean = 0f64;
        for f in 0..valid {
            mean += out[f * N_MELS + m] as f64;
        }
        mean /= valid as f64;
        let mut var = 0f64;
        for f in 0..valid {
            let d = out[f * N_MELS + m] as f64 - mean;
            var += d * d;
        }
        let std = (var / (valid as f64 - 1.0)).sqrt();
        let std = if std.is_nan() { 0.0 } else { std };
        for f in 0..frames {
            out[f * N_MELS + m] = if f < valid { ((out[f * N_MELS + m] as f64 - mean) / (std + 1e-5)) as f32 } else { 0.0 };
        }
    }
    (out, valid)
}

// ------------------------------------------------------------------------------------------------ GPU model

struct ALayer {
    ff1_n_w: DevBuf,
    ff1_n_b: DevBuf,
    ff1_up: DevBuf,
    ff1_up_b: DevBuf,
    ff1_down: DevBuf,
    ff1_down_b: DevBuf,
    att_n_w: DevBuf,
    att_n_b: DevBuf,
    qkv: DevBuf,
    bq: DevBuf,
    bk: DevBuf,
    bv: DevBuf,
    pu: DevBuf,
    pv: DevBuf,
    pos: DevBuf,
    out: DevBuf,
    out_b: DevBuf,
    conv_n_w: DevBuf,
    conv_n_b: DevBuf,
    pw1: DevBuf,
    pw1_b: DevBuf,
    dw: DevBuf,
    dw_b: DevBuf,
    bn_s: DevBuf,
    bn_b: DevBuf,
    pw2: DevBuf,
    pw2_b: DevBuf,
    ff2_n_w: DevBuf,
    ff2_n_b: DevBuf,
    ff2_up: DevBuf,
    ff2_up_b: DevBuf,
    ff2_down: DevBuf,
    ff2_down_b: DevBuf,
    out_n_w: DevBuf,
    out_n_b: DevBuf,
}

pub struct AudioModel {
    c0_w: DevBuf,
    c0_b: DevBuf,
    c2_w: DevBuf,
    c2_b: DevBuf,
    c3_w: DevBuf,
    c3_b: DevBuf,
    c5_w: DevBuf,
    c5_b: DevBuf,
    c6_w: DevBuf,
    c6_b: DevBuf,
    pre_out: DevBuf, // fp32 [512][16*256] columns permuted to (f, c)
    pre_out_b: DevBuf,
    layers: Vec<ALayer>,
    ad_n_w: DevBuf,
    ad_n_b: DevBuf,
    ad1: DevBuf,
    ad1_b: DevBuf,
    ad2: DevBuf,
    ad2_b: DevBuf,
    res_n_w: DevBuf,
    res_n_b: DevBuf,
    res_down: DevBuf,
    res_down_b: DevBuf,
    res_up: DevBuf,
    res_up_b: DevBuf,
    eps: f32,
    // scratch
    mel: DevBuf,
    s1: DevBuf,
    s2: DevBuf,
    s3: DevBuf,
    x: DevBuf,
    xn: DevBuf,
    big: DevBuf,
    qu: DevBuf,
    qv: DevBuf,
    kk: DevBuf,
    vv: DevBuf,
    pe: DevBuf,
    pp: DevBuf,
    ac: DevBuf,
    bd: DevBuf,
    probs: DevBuf,
    o: DevBuf,
}

fn f32v(g: &Gguf, n: &str) -> Result<DevBuf, String> {
    Ok(DevBuf::from_slice(&g.f32(n)?))
}
fn f16v(g: &Gguf, n: &str) -> Result<DevBuf, String> {
    Ok(DevBuf::from_slice(&g.f16(n)?))
}

const MAX_T: usize = 384;

impl AudioModel {
    pub fn load(g: &Gguf) -> Result<AudioModel, String> {
        let nl = g.u("clip.audio.block_count").unwrap_or(17) as usize;
        let eps = g.f("clip.audio.attention.layer_norm_epsilon").unwrap_or(1e-5) as f32;
        let mut layers = Vec::new();
        for i in 0..nl {
            let t = |s: &str| format!("a.blk.{i}.{s}");
            let mut qkv = g.f16(&t("attn_q.weight"))?;
            qkv.extend(g.f16(&t("attn_k.weight"))?);
            qkv.extend(g.f16(&t("attn_v.weight"))?);
            layers.push(ALayer {
                ff1_n_w: f32v(g, &t("ffn_norm.weight"))?,
                ff1_n_b: f32v(g, &t("ffn_norm.bias"))?,
                ff1_up: f16v(g, &t("ffn_up.weight"))?,
                ff1_up_b: f32v(g, &t("ffn_up.bias"))?,
                ff1_down: f16v(g, &t("ffn_down.weight"))?,
                ff1_down_b: f32v(g, &t("ffn_down.bias"))?,
                att_n_w: f32v(g, &t("ln1.weight"))?,
                att_n_b: f32v(g, &t("ln1.bias"))?,
                qkv: DevBuf::from_slice(&qkv),
                bq: f32v(g, &t("attn_q.bias"))?,
                bk: f32v(g, &t("attn_k.bias"))?,
                bv: f32v(g, &t("attn_v.bias"))?,
                pu: f32v(g, &t("pos_bias_u"))?,
                pv: f32v(g, &t("pos_bias_v"))?,
                pos: f16v(g, &t("linear_pos.weight"))?,
                out: f16v(g, &t("attn_out.weight"))?,
                out_b: f32v(g, &t("attn_out.bias"))?,
                conv_n_w: f32v(g, &t("norm_conv.weight"))?,
                conv_n_b: f32v(g, &t("norm_conv.bias"))?,
                pw1: f16v(g, &t("conv_pw1.weight"))?,
                pw1_b: f32v(g, &t("conv_pw1.bias"))?,
                dw: f32v(g, &t("conv_dw.weight"))?,
                dw_b: f32v(g, &t("conv_dw.bias"))?,
                bn_s: f32v(g, &t("conv_norm.weight"))?,
                bn_b: f32v(g, &t("conv_norm.bias"))?,
                pw2: f16v(g, &t("conv_pw2.weight"))?,
                pw2_b: f32v(g, &t("conv_pw2.bias"))?,
                ff2_n_w: f32v(g, &t("ffn_norm_1.weight"))?,
                ff2_n_b: f32v(g, &t("ffn_norm_1.bias"))?,
                ff2_up: f16v(g, &t("ffn_up_1.weight"))?,
                ff2_up_b: f32v(g, &t("ffn_up_1.bias"))?,
                ff2_down: f16v(g, &t("ffn_down_1.weight"))?,
                ff2_down_b: f32v(g, &t("ffn_down_1.bias"))?,
                out_n_w: f32v(g, &t("ln2.weight"))?,
                out_n_b: f32v(g, &t("ln2.bias"))?,
            });
        }
        // pre_encode.out: [512][c*16+f] -> [512][f*256+c]
        let po = g.f32("a.pre_encode.out.weight")?;
        let mut pp = vec![0f32; DM * 16 * SUB_C];
        for o in 0..DM {
            for c in 0..SUB_C {
                for f in 0..16 {
                    pp[o * 4096 + f * SUB_C + c] = po[o * 4096 + c * 16 + f];
                }
            }
        }
        let t0 = 3000 + 1;
        let (t1, t2, t3) = (1500, 750, MAX_T);
        Ok(AudioModel {
            c0_w: f32v(g, "a.conv1d.0.weight")?,
            c0_b: f32v(g, "a.conv1d.0.bias")?,
            c2_w: f32v(g, "a.conv1d.2.weight")?,
            c2_b: f32v(g, "a.conv1d.2.bias")?,
            c3_w: f32v(g, "a.conv1d.3.weight")?,
            c3_b: f32v(g, "a.conv1d.3.bias")?,
            c5_w: f32v(g, "a.conv1d.5.weight")?,
            c5_b: f32v(g, "a.conv1d.5.bias")?,
            c6_w: f32v(g, "a.conv1d.6.weight")?,
            c6_b: f32v(g, "a.conv1d.6.bias")?,
            pre_out: DevBuf::from_slice(&pp),
            pre_out_b: f32v(g, "a.pre_encode.out.bias")?,
            layers,
            ad_n_w: f32v(g, "mm.a.mlp.0.weight")?,
            ad_n_b: f32v(g, "mm.a.mlp.0.bias")?,
            ad1: f16v(g, "mm.a.mlp.1.weight")?,
            ad1_b: f32v(g, "mm.a.mlp.1.bias")?,
            ad2: f16v(g, "mm.a.mlp.3.weight")?,
            ad2_b: f32v(g, "mm.a.mlp.3.bias")?,
            res_n_w: f32v(g, "mm.a.mlp.4.weight")?,
            res_n_b: f32v(g, "mm.a.mlp.4.bias")?,
            res_down: f16v(g, "mm.a.mlp.5.weight")?,
            res_down_b: f32v(g, "mm.a.mlp.5.bias")?,
            res_up: f16v(g, "mm.a.mlp.6.weight")?,
            res_up_b: f32v(g, "mm.a.mlp.6.bias")?,
            eps,
            mel: DevBuf::new(t0 * N_MELS * 4),
            s1: DevBuf::new(t1 * 64 * SUB_C * 4),
            s2: DevBuf::new(t2 * 32 * SUB_C * 4),
            s3: DevBuf::new(t2 * 32 * SUB_C * 4),
            x: DevBuf::new(t3 * DM * 4),
            xn: DevBuf::new(t3 * OUT_D * 2),
            big: DevBuf::new(t3 * 4 * DM * 2),
            qu: DevBuf::new(t3 * DM * 2),
            qv: DevBuf::new(t3 * DM * 2),
            kk: DevBuf::new(t3 * DM * 2),
            vv: DevBuf::new(t3 * DM * 2),
            pe: DevBuf::new(2 * t3 * DM * 2),
            pp: DevBuf::new(2 * t3 * DM * 2),
            ac: DevBuf::new(AHEADS * t3 * t3 * 4),
            bd: DevBuf::new(AHEADS * t3 * 2 * t3 * 4),
            probs: DevBuf::new(AHEADS * t3 * t3 * 2),
            o: DevBuf::new(t3 * DM * 2),
        })
    }

    /// One clip's mel features -> fp32 prefix embeddings [T, 1024] written to `out`. Returns T.
    pub fn encode(&self, blas: &Blas, st: Stream, mel: &[f32], valid: usize, out: *mut f32) -> usize {
        let l1 = (valid - 1) / 2 + 1;
        let l2 = (l1 - 1) / 2 + 1;
        let t = (l2 - 1) / 2 + 1;
        assert!(t <= MAX_T);
        let ti = t as i32;
        unsafe {
            cuda::h2d_async(self.mel.ptr, mel.as_ptr() as *const c_void, valid * N_MELS * 4, st);
            let (s1, s2, s3) = (self.s1.f32(), self.s2.f32(), self.s3.f32());
            cuda::check(k::d1_sub_conv0(self.mel.f32(), valid as i32, N_MELS as i32, self.c0_w.f32(), self.c0_b.f32(), s1, l1 as i32, 64, st));
            cuda::check(k::d1_sub_dwconv(s1, l1 as i32, 64, self.c2_w.f32(), self.c2_b.f32(), s2, l2 as i32, 32, st));
            cuda::check(k::d1_fill_rows(s3, self.c3_b.f32(), ptr::null(), (l2 * 32) as i32, SUB_C as i32, st));
            blas.sgemm(l2 * 32, SUB_C, SUB_C, s2, SUB_C, self.c3_w.f32(), SUB_C, s3, SUB_C, 1.0);
            cuda::check(k::d1_bias_act_f(s3, ptr::null(), (l2 * 32) as i32, SUB_C as i32, 1, st));
            cuda::check(k::d1_sub_dwconv(s3, l2 as i32, 32, self.c5_w.f32(), self.c5_b.f32(), s2, t as i32, 16, st));
            cuda::check(k::d1_fill_rows(s1, self.c6_b.f32(), ptr::null(), (t * 16) as i32, SUB_C as i32, st));
            blas.sgemm(t * 16, SUB_C, SUB_C, s2, SUB_C, self.c6_w.f32(), SUB_C, s1, SUB_C, 1.0);
            cuda::check(k::d1_bias_act_f(s1, ptr::null(), (t * 16) as i32, SUB_C as i32, 1, st));
            let x = self.x.f32();
            cuda::check(k::d1_fill_rows(x, self.pre_out_b.f32(), ptr::null(), ti, DM as i32, st));
            blas.sgemm(t, DM, 16 * SUB_C, s1, 16 * SUB_C, self.pre_out.f32(), 16 * SUB_C, x, DM, 1.0);

            // relative positions t-1 .. -(t-1)
            let np = 2 * t - 1;
            let mut pe = vec![0u16; np * DM];
            let c = -(10000f64.ln() / DM as f64) as f32;
            for n in 0..np {
                let p = (t as i64 - 1 - n as i64) as f32;
                for i in 0..DM / 2 {
                    let div = ((2 * i) as f32 * c).exp();
                    let a = (p * div) as f64;
                    pe[n * DM + 2 * i] = cuda::f32_to_f16(a.sin() as f32);
                    pe[n * DM + 2 * i + 1] = cuda::f32_to_f16(a.cos() as f32);
                }
            }
            cuda::h2d_async(self.pe.ptr, pe.as_ptr() as *const c_void, pe.len() * 2, st);

            let (xn, big) = (self.xn.f16(), self.big.f16());
            let ln = |x: *mut f32, pb: *const f32, ps: f32, w: &DevBuf, b: &DevBuf, y: *mut F16, yf: *mut f32, d: usize| {
                cuda::check(k::d1_layernorm(x, pb, ps, w.f32(), b.f32(), y, yf, ti, d as i32, self.eps, st));
            };
            let ff = 4 * DM;
            let mut pend: (*const f32, f32) = (ptr::null(), 1.0);
            for l in &self.layers {
                // FF1 (half step)
                ln(x, pend.0, pend.1, &l.ff1_n_w, &l.ff1_n_b, xn, ptr::null_mut(), DM);
                blas.gemm(t, ff, DM, xn, DM, l.ff1_up.f16(), DM, false, Out::H(big), ff, 1.0, 0.0);
                cuda::check(k::d1_bias_act_h(big, l.ff1_up_b.f32(), ti, ff as i32, 4, st));
                blas.gemm(t, DM, ff, big, ff, l.ff1_down.f16(), ff, false, Out::F(x), DM, 0.5, 1.0);
                // rel-pos self attention
                ln(x, l.ff1_down_b.f32(), 0.5, &l.att_n_w, &l.att_n_b, xn, ptr::null_mut(), DM);
                blas.gemm(t, 3 * DM, DM, xn, DM, l.qkv.f16(), DM, false, Out::H(big), 3 * DM, 1.0, 0.0);
                cuda::check(k::d1_relpos_prep(big, l.bq.f32(), l.bk.f32(), l.bv.f32(), l.pu.f32(), l.pv.f32(), self.qu.f16(), self.qv.f16(), self.kk.f16(), self.vv.f16(), ti, DM as i32, st));
                blas.gemm(np, DM, DM, self.pe.f16(), DM, l.pos.f16(), DM, false, Out::H(self.pp.f16()), DM, 1.0, 0.0);
                blas.gemm_batched(AHEADS, t, t, 64, self.qu.f16(), DM, 64, self.kk.f16(), DM, 64, false, Out::F(self.ac.f32()), t, t * t, 1.0);
                blas.gemm_batched(AHEADS, t, np, 64, self.qv.f16(), DM, 64, self.pp.f16(), DM, 64, false, Out::F(self.bd.f32()), np, t * np, 1.0);
                cuda::check(k::d1_relpos_softmax(self.ac.f32(), self.bd.f32(), self.probs.f16(), ti, AHEADS as i32, 0.125, st));
                blas.gemm_batched(AHEADS, t, 64, t, self.probs.f16(), t, t * t, self.vv.f16(), DM, 64, true, Out::H(self.o.f16()), DM, 64, 1.0);
                blas.gemm(t, DM, DM, self.o.f16(), DM, l.out.f16(), DM, false, Out::F(x), DM, 1.0, 1.0);
                // conv module
                ln(x, l.out_b.f32(), 1.0, &l.conv_n_w, &l.conv_n_b, xn, ptr::null_mut(), DM);
                blas.gemm(t, 2 * DM, DM, xn, DM, l.pw1.f16(), DM, false, Out::H(big), 2 * DM, 1.0, 0.0);
                cuda::check(k::d1_conformer_dw(big, l.pw1_b.f32(), l.dw.f32(), l.dw_b.f32(), l.bn_s.f32(), l.bn_b.f32(), self.o.f16(), ti, DM as i32, 9, st));
                blas.gemm(t, DM, DM, self.o.f16(), DM, l.pw2.f16(), DM, false, Out::F(x), DM, 1.0, 1.0);
                // FF2 (half step)
                ln(x, l.pw2_b.f32(), 1.0, &l.ff2_n_w, &l.ff2_n_b, xn, ptr::null_mut(), DM);
                blas.gemm(t, ff, DM, xn, DM, l.ff2_up.f16(), DM, false, Out::H(big), ff, 1.0, 0.0);
                cuda::check(k::d1_bias_act_h(big, l.ff2_up_b.f32(), ti, ff as i32, 4, st));
                blas.gemm(t, DM, ff, big, ff, l.ff2_down.f16(), ff, false, Out::F(x), DM, 0.5, 1.0);
                // norm_out replaces the residual
                ln(x, l.ff2_down_b.f32(), 0.5, &l.out_n_w, &l.out_n_b, ptr::null_mut(), x, DM);
                pend = (ptr::null(), 1.0);
            }
            let _ = pend;
            // adapter: LN -> Linear -> GELU -> Linear
            ln(x, ptr::null(), 1.0, &self.ad_n_w, &self.ad_n_b, xn, ptr::null_mut(), DM);
            blas.gemm(t, OUT_D, DM, xn, DM, self.ad1.f16(), DM, false, Out::H(big), OUT_D, 1.0, 0.0);
            cuda::check(k::d1_bias_act_h(big, self.ad1_b.f32(), ti, OUT_D as i32, 2, st));
            cuda::check(k::d1_fill_rows(out, self.ad2_b.f32(), ptr::null(), ti, OUT_D as i32, st));
            blas.gemm(t, OUT_D, OUT_D, big, OUT_D, self.ad2.f16(), OUT_D, false, Out::F(out), OUT_D, 1.0, 1.0);
            // residual: a + up(GELU(down(LN(a))))
            ln(out, ptr::null(), 1.0, &self.res_n_w, &self.res_n_b, xn, ptr::null_mut(), OUT_D);
            blas.gemm(t, DM, OUT_D, xn, OUT_D, self.res_down.f16(), OUT_D, false, Out::H(big), DM, 1.0, 0.0);
            cuda::check(k::d1_bias_act_h(big, self.res_down_b.f32(), ti, DM as i32, 2, st));
            blas.gemm(t, OUT_D, DM, big, DM, self.res_up.f16(), DM, false, Out::F(out), OUT_D, 1.0, 1.0);
            cuda::check(k::d1_bias_act_f(out, self.res_up_b.f32(), ti, OUT_D as i32, 0, st));

        }
        t
    }
}
