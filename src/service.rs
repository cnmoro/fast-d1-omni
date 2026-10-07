//! The System One API: request JSON -> rows/media -> engine -> typed answers.

use crate::audio;
use crate::engine::{EngineHandle, Job, Media, Row};
use crate::gguf::Gguf;
use crate::media;
use crate::json::Json;
use crate::prompt::{self, Delims, Mode, QType, Question};
use crate::tokenizer::Tokenizer;
use crate::vision;
use std::collections::HashMap;
use std::sync::mpsc::channel;

#[derive(Debug)]
pub struct ApiError {
    pub code: u16,
    pub msg: String,
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError { code: 400, msg: msg.into() }
}

pub struct Limits {
    pub max_length: usize,
    pub image_text_length: usize,
    pub audio_text_length: usize,
    pub cap_tokens: usize,
}

pub struct Service {
    pub tok: Tokenizer,
    pub dl: Delims,
    pub engine: EngineHandle,
    pub temps: HashMap<String, f32>,
    pub limits: Limits,
    pub has_vision: bool,
    pub has_audio: bool,
    pub model_name: String,
    pub conv: media::Converter,
}

/// Calibration temperatures from GGUF metadata (`lfm2.decision.temperature.choice.3_5` -> `choice:3-5`).
pub fn temperatures(g: &Gguf) -> HashMap<String, f32> {
    let mut t = HashMap::new();
    for (k, v) in &g.meta {
        let Some(rest) = k.strip_prefix("lfm2.decision.temperature.") else { continue };
        let Some(f) = v.as_f64() else { continue };
        let key = match rest.split_once('.') {
            Some((ty, b)) => {
                let b = match b {
                    "11" => "11+".to_string(),
                    b => b.replace('_', "-"),
                };
                format!("{ty}:{b}")
            }
            None => rest.to_string(),
        };
        t.insert(key, f as f32);
    }
    t
}

pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut n = 0;
    for &c in s.as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err("invalid base64".into()),
        } as u32;
        acc = (acc << 6) | v;
        n += 6;
        if n >= 8 {
            n -= 8;
            out.push((acc >> n) as u8);
        }
    }
    Ok(out)
}

/// `data:<mime>;base64,<payload>` or bare base64 -> (mime, bytes)
fn data_uri(s: &str) -> Result<(String, Vec<u8>), ApiError> {
    if let Some(rest) = s.strip_prefix("data:") {
        let (meta, payload) = rest.split_once(',').ok_or_else(|| bad("malformed data URI"))?;
        let mime = meta.split(';').next().unwrap_or("").to_string();
        if !meta.ends_with(";base64") {
            return Err(bad("data URIs must be base64 encoded"));
        }
        return Ok((mime, base64_decode(payload).map_err(bad)?));
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return Err(bad("remote URLs are not fetched; send media as base64 data URIs"));
    }
    Ok((String::new(), base64_decode(s).map_err(bad)?))
}

pub struct Prepared {
    pub names: Vec<String>,
    pub questions: Vec<Question>,
    pub mode: Mode,
    pub input_tokens: usize,
    pub job_rows: Vec<Row>,
    pub media: Media,
}

impl Service {
    /// Parse, decode media, tokenize: all the CPU work, done on the request thread.
    pub fn prepare(&self, req: &Json) -> Result<Prepared, ApiError> {
        if !matches!(req, Json::Obj(_)) {
            return Err(bad("the request body must be a JSON object"));
        }
        let state = req.get("state").cloned().unwrap_or(Json::Null);
        let Some(Json::Obj(qs)) = req.get("questions") else {
            return Err(bad("`questions` must be an object {name: question}"));
        };
        if qs.is_empty() {
            return Err(bad("`questions` is empty"));
        }
        let mut names = Vec::new();
        let mut questions = Vec::new();
        for (n, q) in qs {
            names.push(n.clone());
            questions.push(prompt::as_question(q).map_err(|e| bad(format!("question {n:?}: {e}")))?);
        }
        // media
        let mut images: Vec<Vec<u8>> = Vec::new();
        let mut audios: Vec<Vec<u8>> = Vec::new();
        let list = |k: &str| -> Result<Vec<String>, ApiError> {
            match req.get(k) {
                None | Some(Json::Null) => Ok(vec![]),
                Some(Json::Arr(a)) => a.iter().map(|x| x.as_str().map(|s| s.to_string()).ok_or_else(|| bad(format!("`{k}` must be a list of strings")))).collect(),
                Some(Json::Str(s)) => Ok(vec![s.clone()]),
                _ => Err(bad(format!("`{k}` must be a list of strings"))),
            }
        };
        // route every blob by its content (magic bytes); the field name and MIME type are only hints for unknown data
        let mut put = |bytes: Vec<u8>, hint_image: bool| {
            let k = media::sniff(&bytes);
            if k.is_image() || (!k.is_audio() && hint_image) {
                images.push(bytes);
            } else {
                audios.push(bytes);
            }
        };
        for s in list("images")? {
            let (mime, bytes) = data_uri(&s)?;
            put(bytes, !mime.starts_with("audio/") && !mime.starts_with("video/"));
        }
        for k in ["files", "audio"] {
            for s in list(k)? {
                let (mime, bytes) = data_uri(&s)?;
                put(bytes, mime.starts_with("image/"));
            }
        }
        if !images.is_empty() && !audios.is_empty() {
            return Err(bad("a request carries images or audio, not both"));
        }
        if audios.len() > 1 {
            return Err(bad("a request carries one audio clip"));
        }
        let (media, mode) = if !images.is_empty() {
            if !self.has_vision {
                return Err(bad("this server has no vision encoder (start it with --mmproj)"));
            }
            let mut pats = Vec::new();
            for b in &images {
                let img = self.conv.decode_image(b).map_err(|e| bad(format!("image: {e}")))?;
                pats.push(vision::preprocess(&img));
            }
            (Media::Images(pats), Mode::Image)
        } else if let Some(a) = audios.first() {
            if !self.has_audio {
                return Err(bad("this server has no audio encoder (start it with --mmproj)"));
            }
            let samples = self.conv.decode_audio(a).map_err(|e| bad(format!("audio: {e}")))?;
            let (mel, valid) = audio::mel(&samples);
            (Media::Audio { mel, valid }, Mode::Audio)
        } else {
            (Media::None, Mode::Text)
        };
        let p = media.prefix_len();
        let lim = &self.limits;
        let max_len = match mode {
            Mode::Text => lim.max_length,
            Mode::Image => lim.image_text_length,
            Mode::Audio => lim.audio_text_length,
        }
        .min(lim.max_length.saturating_sub(p))
        .min(lim.cap_tokens);
        if max_len < 64 {
            return Err(bad(format!("the media take {p} of the {} positions; send fewer images", lim.max_length)));
        }
        let state = match (&state, mode) {
            (Json::Null, Mode::Audio) => Json::Obj(vec![]),
            (Json::Null, _) => Json::Str(String::new()),
            _ => state,
        };
        let st_ids = prompt::state_tokens(&self.tok, &state);
        let mut rows = Vec::with_capacity(questions.len());
        let mut input_tokens = 0;
        for (n, q) in names.iter().zip(&questions) {
            let (ids, markers) = prompt::encode(&self.tok, &self.dl, &st_ids, q, max_len, mode).map_err(|e| bad(format!("question {n:?}: {e}")))?;
            input_tokens += p + ids.len();
            rows.push(Row { ids, markers, qtype: q.qtype as i32 });
        }
        Ok(Prepared { names, questions, mode, input_tokens, job_rows: rows, media })
    }

    /// Run prepared rows and return per-question probabilities (option order; noul as [yes, no]).
    pub fn run(&self, p: Prepared) -> Result<(Prepared, Vec<Vec<f32>>), ApiError> {
        let (tx, rx) = channel();
        let mut p = p;
        let rows = std::mem::take(&mut p.job_rows);
        let media = std::mem::replace(&mut p.media, Media::None);
        self.engine
            .submit(Job { rows, media, reply: tx })
            .map_err(|e| ApiError { code: 503, msg: e })?;
        let logits = rx
            .recv()
            .map_err(|_| ApiError { code: 500, msg: "engine dropped the request".into() })?
            .map_err(|e| ApiError { code: 400, msg: e })?;
        let probs = p
            .questions
            .iter()
            .zip(&logits)
            .map(|(q, z)| {
                let n = q.options();
                let mut z: Vec<f32> = z[..n.min(z.len())].to_vec();
                if p.mode == Mode::Text {
                    let t = self.temps.get(&q.temperature_key()).or_else(|| self.temps.get(q.type_name())).copied().unwrap_or(1.0);
                    z.iter_mut().for_each(|v| *v /= t);
                }
                let mut pr = prompt::softmax(&z);
                if q.qtype == QType::Noul {
                    pr.reverse();
                }
                pr
            })
            .collect();
        Ok((p, probs))
    }

    pub fn systemone(&self, req: &Json) -> Result<Json, ApiError> {
        let p = self.prepare(req)?;
        let (p, probs) = self.run(p)?;
        let answers = p.names.iter().zip(&p.questions).zip(&probs).map(|((n, q), pr)| (n.clone(), prompt::answer(q, pr))).collect();
        Ok(Json::Obj(vec![
            ("answers".into(), Json::Obj(answers)),
            ("usage".into(), Json::Obj(vec![
                ("input_tokens".into(), Json::Num(p.input_tokens as f64, Some(p.input_tokens.to_string()))),
                ("output_tokens".into(), Json::Num(0.0, Some("0".into()))),
            ])),
        ]))
    }
}
