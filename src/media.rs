//! Media sniffing and optional format normalization.
//!
//! Every blob is classified by its magic bytes (not by the declared MIME type). JPEG/PNG/BMP/PPM images and PCM/float
//! WAV are decoded natively. Anything else (MP3, OGG/Opus, FLAC, M4A/AAC, WebM, AIFF, AMR, WebP, GIF, TIFF, HEIC/AVIF,
//! ...) is normalized with ffmpeg when a converter is configured: images to PPM, audio to 16 kHz mono s16le. The same
//! fallback covers files the native decoders reject (CMYK or 12-bit JPEG, ADPCM or mu-law WAV, ...).
//!
//! ffmpeg runs on untrusted input, so it may only read the one temporary input file (`-protocol_whitelist file`), has
//! no stdin, a wall-clock timeout and a cap on the output size.

use crate::audio;
use crate::image::{self, Image};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// natively decodable image
    Image,
    /// natively decodable audio (RIFF/WAVE)
    Wav,
    /// image that needs conversion
    OtherImage(&'static str),
    /// audio (or audio/video container) that needs conversion
    OtherAudio(&'static str),
    Unknown,
}

impl Kind {
    pub fn is_image(self) -> bool {
        matches!(self, Kind::Image | Kind::OtherImage(_))
    }
    pub fn is_audio(self) -> bool {
        matches!(self, Kind::Wav | Kind::OtherAudio(_))
    }
}

pub fn sniff(b: &[u8]) -> Kind {
    let at = |o: usize, m: &[u8]| b.len() >= o + m.len() && &b[o..o + m.len()] == m;
    if at(0, &[0xff, 0xd8, 0xff]) || at(0, b"\x89PNG\r\n\x1a\n") || at(0, b"BM") || (at(0, b"P6") || at(0, b"P5")) {
        return Kind::Image;
    }
    if at(0, b"RIFF") && at(8, b"WAVE") {
        return Kind::Wav;
    }
    if at(0, b"RIFF") && at(8, b"WEBP") {
        return Kind::OtherImage("webp");
    }
    if at(0, b"GIF87a") || at(0, b"GIF89a") {
        return Kind::OtherImage("gif");
    }
    if at(0, b"II*\0") || at(0, b"MM\0*") {
        return Kind::OtherImage("tiff");
    }
    if at(0, b"\0\0\0\x0cjP  ") || at(0, &[0xff, 0x4f, 0xff, 0x51]) {
        return Kind::OtherImage("jpeg2000");
    }
    if at(0, b"\0\0\x01\0") {
        return Kind::OtherImage("ico");
    }
    if at(0, b"qoif") {
        return Kind::OtherImage("qoi");
    }
    if at(4, b"ftyp") && b.len() >= 12 {
        let brand = &b[8..12];
        return match brand {
            b"heic" | b"heix" | b"hevc" | b"heim" | b"heis" | b"mif1" | b"msf1" => Kind::OtherImage("heic"),
            b"avif" | b"avis" => Kind::OtherImage("avif"),
            _ => Kind::OtherAudio("mp4/m4a"),
        };
    }
    if at(0, b"ID3") || (b.len() > 1 && b[0] == 0xff && (b[1] & 0xe0) == 0xe0 && (b[1] & 0x06) != 0) {
        // MPEG audio frame sync (layer bits != 0; AAC ADTS has layer 0)
        return Kind::OtherAudio("mp3");
    }
    if b.len() > 1 && b[0] == 0xff && (b[1] & 0xf6) == 0xf0 {
        return Kind::OtherAudio("aac");
    }
    if at(0, b"OggS") {
        return Kind::OtherAudio("ogg");
    }
    if at(0, b"fLaC") {
        return Kind::OtherAudio("flac");
    }
    if at(0, b"\x1a\x45\xdf\xa3") {
        return Kind::OtherAudio("webm/mkv");
    }
    if at(0, b"FORM") && (at(8, b"AIFF") || at(8, b"AIFC")) {
        return Kind::OtherAudio("aiff");
    }
    if at(0, b"#!AMR") {
        return Kind::OtherAudio("amr");
    }
    if at(0, b"RIFF") && at(8, b"AVI ") {
        return Kind::OtherAudio("avi");
    }
    if at(0, b".snd") {
        return Kind::OtherAudio("au");
    }
    if at(0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11") {
        return Kind::OtherAudio("wma/asf");
    }
    if at(0, b"caff") {
        return Kind::OtherAudio("caf");
    }
    Kind::Unknown
}

pub struct Converter {
    pub ffmpeg: Option<String>,
    pub timeout: Duration,
}

static SEQ: AtomicU64 = AtomicU64::new(0);

const MAX_AUDIO_OUT: usize = 31 * audio::SAMPLE_RATE * 2;
const MAX_IMAGE_OUT: usize = 100_000_000 * 3 + 64;

impl Converter {
    /// `spec`: "auto" (ffmpeg from PATH if present), "off", or a path to an ffmpeg binary.
    pub fn new(spec: &str) -> Converter {
        let ffmpeg = match spec {
            "off" | "none" | "false" => None,
            "auto" => {
                let ok = Command::new("ffmpeg").arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
                ok.then(|| "ffmpeg".to_string())
            }
            path => Some(path.to_string()),
        };
        Converter { ffmpeg, timeout: Duration::from_secs(30) }
    }

    pub fn enabled(&self) -> bool {
        self.ffmpeg.is_some()
    }

    fn run(&self, input: &[u8], args: &[&str], max_out: usize) -> Result<Vec<u8>, String> {
        let ff = self.ffmpeg.as_ref().ok_or("format conversion is disabled (start the server with --convert auto)")?;
        let path = std::env::temp_dir().join(format!(
            "d1rs-{}-{}-{}.bin",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
        ));
        std::fs::write(&path, input).map_err(|e| format!("temp file: {e}"))?;
        let res = (|| {
            let mut child = Command::new(ff)
                .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file", "-i"])
                .arg(format!("file:{}", path.display()))
                .args(args)
                .arg("pipe:1")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("cannot run {ff}: {e}"))?;
            let mut out = child.stdout.take().unwrap();
            let mut err = child.stderr.take().unwrap();
            let reader = std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 1 << 16];
                loop {
                    match out.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.len() > max_out {
                                break;
                            }
                        }
                    }
                }
                buf
            });
            let err_reader = std::thread::spawn(move || {
                let mut s = String::new();
                let _ = err.read_to_string(&mut s);
                s
            });
            let t0 = Instant::now();
            let status = loop {
                if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
                    break Some(st);
                }
                if t0.elapsed() > self.timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(2));
            };
            let buf = reader.join().unwrap_or_default();
            let msg = err_reader.join().unwrap_or_default();
            match status {
                None => Err("conversion timed out".to_string()),
                Some(st) if !st.success() || buf.is_empty() => {
                    Err(format!("conversion failed: {}", msg.lines().last().unwrap_or("unrecognised media").trim()))
                }
                Some(_) => Ok(buf),
            }
        })();
        let _ = std::fs::remove_file(&path);
        res
    }

    /// Any image ffmpeg can read -> RGB8 (first frame for animations).
    pub fn image(&self, input: &[u8]) -> Result<Image, String> {
        let ppm = self.run(input, &["-frames:v", "1", "-f", "image2pipe", "-c:v", "ppm", "-pix_fmt", "rgb24"], MAX_IMAGE_OUT)?;
        image::decode(&ppm)
    }

    /// Any audio (or audio track of a video) ffmpeg can read -> 16 kHz mono f32, first 30 s.
    pub fn audio(&self, input: &[u8]) -> Result<Vec<f32>, String> {
        let pcm = self.run(input, &["-vn", "-t", "30", "-ac", "1", "-ar", "16000", "-f", "s16le", "-c:a", "pcm_s16le"], MAX_AUDIO_OUT)?;
        Ok(pcm.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect())
    }

    /// Decode an image natively, falling back to conversion.
    pub fn decode_image(&self, b: &[u8]) -> Result<Image, String> {
        match sniff(b) {
            Kind::Image => match image::decode(b) {
                Ok(i) => Ok(i),
                Err(e) if self.enabled() => self.image(b).map_err(|e2| format!("{e}; {e2}")),
                Err(e) => Err(e),
            },
            Kind::OtherImage(f) if !self.enabled() => Err(format!("{f} images need conversion; start the server with --convert auto")),
            _ => self.image(b),
        }
    }

    /// Decode audio natively (WAV), falling back to conversion.
    pub fn decode_audio(&self, b: &[u8]) -> Result<Vec<f32>, String> {
        match sniff(b) {
            Kind::Wav => match audio::decode_wav(b) {
                Ok(a) => Ok(a),
                Err(e) if self.enabled() => self.audio(b).map_err(|e2| format!("{e}; {e2}")),
                Err(e) => Err(e),
            },
            Kind::OtherAudio(f) if !self.enabled() => Err(format!("{f} audio needs conversion; start the server with --convert auto")),
            _ => self.audio(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sniffing() {
        assert_eq!(sniff(b"\xff\xd8\xff\xe0...."), Kind::Image);
        assert_eq!(sniff(b"RIFF\0\0\0\0WAVEfmt "), Kind::Wav);
        assert_eq!(sniff(b"RIFF\0\0\0\0WEBPVP8 "), Kind::OtherImage("webp"));
        assert_eq!(sniff(b"ID3\x04\0\0\0\0"), Kind::OtherAudio("mp3"));
        assert_eq!(sniff(b"\xff\xfb\x90\x00"), Kind::OtherAudio("mp3"));
        assert_eq!(sniff(b"\xff\xf1\x50\x80"), Kind::OtherAudio("aac"));
        assert_eq!(sniff(b"OggS\0\x02"), Kind::OtherAudio("ogg"));
        assert_eq!(sniff(b"\0\0\0\x1cftypheic"), Kind::OtherImage("heic"));
        assert_eq!(sniff(b"\0\0\0\x20ftypM4A "), Kind::OtherAudio("mp4/m4a"));
    }
}
