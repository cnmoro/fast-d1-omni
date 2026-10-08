//! Hugging Face Hub downloads (no TLS stack of our own: HTTPS goes through the system `curl`, or `wget`).
//!
//! `--hf REPO[:QUANT]` (default `LiquidAI/d1-omni-600M-GGUF:F16`) resolves the model GGUF and its `mmproj` from the
//! repo's file listing, downloads them into `$D1_CACHE` (default `~/.cache/d1rs/models`), resumes partial downloads,
//! verifies the SHA-256 the Hub publishes for each LFS file, and reuses verified files on later starts. A private or
//! gated repo needs `HF_TOKEN`. `HF_ENDPOINT` points at a mirror.

use crate::json::{self, Json};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const DEFAULT_REPO: &str = "LiquidAI/d1-omni-600M-GGUF";
pub const DEFAULT_QUANT: &str = "F16";

pub struct Resolved {
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
}

struct Remote {
    path: String,
    size: u64,
    sha256: Option<String>,
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("D1_CACHE") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/d1rs/models")
}

fn have(cmd: &str) -> bool {
    Command::new(cmd).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

/// GET a small resource into memory.
fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let out = if have("curl") {
        let mut c = Command::new("curl");
        c.args(["-fsSL", "--retry", "3", "--connect-timeout", "20"]);
        if let Some(t) = &token {
            c.args(["-H", &format!("Authorization: Bearer {t}")]);
        }
        c.arg(url).output()
    } else if have("wget") {
        let mut c = Command::new("wget");
        c.args(["-q", "-O", "-", "--tries=3"]);
        if let Some(t) = &token {
            c.arg(format!("--header=Authorization: Bearer {t}"));
        }
        c.arg(url).output()
    } else {
        return Err("downloading needs `curl` or `wget` on PATH (or pass -m with a local GGUF file)".into());
    }
    .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("GET {url} failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout)
}

/// Download `url` to `dst` with resume and a progress bar on stderr.
fn download(url: &str, dst: &Path) -> Result<(), String> {
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let part = dst.with_extension("part");
    let status = if have("curl") {
        let mut c = Command::new("curl");
        c.args(["-fL", "--retry", "5", "--retry-delay", "2", "--connect-timeout", "20", "-C", "-", "--progress-bar", "-o"]).arg(&part);
        if let Some(t) = &token {
            c.args(["-H", &format!("Authorization: Bearer {t}")]);
        }
        c.arg(url).stdin(Stdio::null()).status()
    } else if have("wget") {
        let mut c = Command::new("wget");
        c.args(["-c", "--tries=5", "-q", "--show-progress", "-O"]).arg(&part);
        if let Some(t) = &token {
            c.arg(format!("--header=Authorization: Bearer {t}"));
        }
        c.arg(url).stdin(Stdio::null()).status()
    } else {
        return Err("downloading needs `curl` or `wget` on PATH (or pass -m with a local GGUF file)".into());
    }
    .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("download of {url} failed ({status}); rerun to resume"));
    }
    std::fs::rename(&part, dst).map_err(|e| e.to_string())
}

fn list(repo: &str, revision: &str) -> Result<Vec<Remote>, String> {
    let url = format!("{}/api/models/{repo}/tree/{revision}", endpoint());
    let body = fetch(&url)?;
    let j = json::parse(std::str::from_utf8(&body).map_err(|_| "bad listing")?).map_err(|e| format!("{url}: {e}"))?;
    let Json::Arr(items) = j else { return Err(format!("{url}: unexpected listing")) };
    Ok(items
        .iter()
        .filter(|f| f.get("type").and_then(|t| t.as_str()) == Some("file"))
        .filter_map(|f| {
            let path = f.get("path")?.as_str()?.to_string();
            let lfs = f.get("lfs");
            let size = lfs.and_then(|l| l.get("size")).or(f.get("size")).and_then(|s| s.as_f64()).unwrap_or(0.0) as u64;
            let sha256 = lfs.and_then(|l| l.get("oid")).and_then(|o| o.as_str()).map(|s| s.to_string());
            Some(Remote { path, size, sha256 })
        })
        .collect())
}

/// Pick `<name>-<QUANT>.gguf` and `mmproj-<name>-<QUANT>.gguf` (mmproj falls back to F16, then any).
fn pick(files: &[Remote], quant: &str) -> Result<(usize, Option<usize>), String> {
    let q = quant.to_ascii_lowercase();
    let gguf: Vec<(usize, String)> = files
        .iter()
        .enumerate()
        .filter(|(_, f)| f.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|(i, f)| (i, f.path.rsplit('/').next().unwrap_or(&f.path).to_ascii_lowercase()))
        .collect();
    let is_mm = |n: &str| n.starts_with("mmproj");
    let matches = |n: &str, q: &str| n.trim_end_matches(".gguf").ends_with(&format!("-{q}")) || n.trim_end_matches(".gguf").ends_with(&format!(".{q}"));
    let model = gguf
        .iter()
        .find(|(_, n)| !is_mm(n) && matches(n, &q))
        .map(|(i, _)| *i)
        .ok_or_else(|| {
            let avail: Vec<&str> = gguf.iter().filter(|(_, n)| !is_mm(n)).map(|(i, _)| files[*i].path.as_str()).collect();
            format!("no {quant} GGUF in the repo; available: {}", avail.join(", "))
        })?;
    let mm = gguf
        .iter()
        .find(|(_, n)| is_mm(n) && matches(n, &q))
        .or_else(|| gguf.iter().find(|(_, n)| is_mm(n) && matches(n, "f16")))
        .or_else(|| gguf.iter().find(|(_, n)| is_mm(n)))
        .map(|(i, _)| *i);
    Ok((model, mm))
}

fn ensure(repo: &str, revision: &str, f: &Remote) -> Result<PathBuf, String> {
    let dir = cache_dir().join(repo.replace('/', "--"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let dst = dir.join(f.path.rsplit('/').next().unwrap_or(&f.path));
    let ok_marker = dst.with_extension("gguf.sha256-ok");
    let current = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    if current(&dst) == f.size && f.size > 0 {
        // verified earlier (marker holds the hash) -> reuse without rehashing 700 MB
        if f.sha256.as_deref().is_none_or(|h| std::fs::read_to_string(&ok_marker).map(|m| m.trim() == h).unwrap_or(false)) {
            return Ok(dst);
        }
    }
    if current(&dst) > 0 {
        let _ = std::fs::rename(&dst, dst.with_extension("part")); // size/hash mismatch: resume or redo
    }
    let url = format!("{}/{repo}/resolve/{revision}/{}", endpoint(), f.path);
    eprintln!("downloading {repo}/{} ({:.0} MB) -> {}", f.path, f.size as f64 / 1e6, dst.display());
    download(&url, &dst)?;
    if f.size > 0 && current(&dst) != f.size {
        let _ = std::fs::remove_file(&dst);
        return Err(format!("{}: size mismatch after download", f.path));
    }
    if let Some(h) = &f.sha256 {
        eprint!("verifying sha256 ... ");
        let got = sha256_file(&dst)?;
        if &got != h {
            let _ = std::fs::remove_file(&dst);
            return Err(format!("{}: sha256 mismatch (expected {h}, got {got})", f.path));
        }
        eprintln!("ok");
        let _ = std::fs::write(&ok_marker, h);
    }
    Ok(dst)
}

/// Resolve `REPO[:QUANT][@REVISION]` to local files, downloading what is missing.
pub fn resolve(spec: &str, with_mmproj: bool) -> Result<Resolved, String> {
    let (spec, revision) = spec.split_once('@').unwrap_or((spec, "main"));
    let (repo, quant) = match spec.rsplit_once(':') {
        Some((r, q)) => (r, q),
        None => (spec, DEFAULT_QUANT),
    };
    let dir = cache_dir().join(repo.replace('/', "--"));
    let files = match list(repo, revision) {
        Ok(f) => f,
        Err(e) => {
            // offline: fall back to previously verified files in the cache
            let find = |mm: bool| -> Option<PathBuf> {
                std::fs::read_dir(&dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
                    let n = p.file_name().unwrap().to_string_lossy().to_ascii_lowercase();
                    n.ends_with(&format!("-{}.gguf", quant.to_ascii_lowercase())) && n.starts_with("mmproj") == mm
                })
            };
            if let Some(model) = find(false) {
                eprintln!("warning: {e}; using cached files");
                return Ok(Resolved { model, mmproj: if with_mmproj { find(true) } else { None } });
            }
            return Err(e);
        }
    };
    let (mi, mm) = pick(&files, quant)?;
    let model = ensure(repo, revision, &files[mi])?;
    let mmproj = match (with_mmproj, mm) {
        (true, Some(i)) => Some(ensure(repo, revision, &files[i])?),
        _ => None,
    };
    Ok(Resolved { model, mmproj })
}

// ------------------------------------------------------------------------------------------------ sha256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 {
            h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buf: Vec::with_capacity(64),
            len: 0,
        }
    }
    fn block(&mut self, b: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b_, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b_) ^ (a & c) ^ (b_ & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b_;
            b_ = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in self.h.iter_mut().zip([a, b_, c, d, e, f, g, h]) {
            *x = x.wrapping_add(y);
        }
    }
    pub fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if !self.buf.is_empty() {
            let n = (64 - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..n]);
            data = &data[n..];
            if self.buf.len() == 64 {
                let b = std::mem::take(&mut self.buf);
                self.block(&b);
                self.buf = b;
                self.buf.clear();
            }
        }
        let mut chunks = data.chunks_exact(64);
        for c in &mut chunks {
            self.block(c);
        }
        self.buf.extend_from_slice(chunks.remainder());
    }
    pub fn hex(mut self) -> String {
        let bits = self.len * 8;
        let mut pad = vec![0x80u8];
        while (self.buf.len() + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend_from_slice(&bits.to_be_bytes());
        let len = self.len;
        self.update(&pad);
        self.len = len;
        self.h.iter().map(|x| format!("{x:08x}")).collect()
    }
}

pub fn sha256_file(p: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(p).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha() {
        let mut h = Sha256::new();
        h.update(b"abc");
        assert_eq!(h.hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let mut h = Sha256::new();
        for _ in 0..1000 {
            h.update(&[b'a'; 1000][..]);
        }
        assert_eq!(h.hex(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }
}
