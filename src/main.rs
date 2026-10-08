//! d1: dependency-free CUDA inference server for LiquidAI d1-omni-600M (GGUF).

mod audio;
mod cuda;
mod engine;
mod gguf;
mod hub;
mod image;
mod json;
mod media;
mod model;
mod prompt;
mod server;
mod service;
mod tokenizer;
mod unicode_tables;
mod vision;

use json::Json;
use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const USAGE: &str = "d1 - CUDA inference for LiquidAI d1-omni-600M (GGUF)

USAGE:
  d1 serve    [MODEL] [--host 127.0.0.1] [--port 8080] [options]
  d1 run      [MODEL] REQUEST.json                         one request, prints the response
  d1 eval     [MODEL] CASES.jsonl                          probabilities per case (JSONL)
  d1 bench    [MODEL] [--clients 64] [--seconds 10] [--request REQ.json]
  d1 loadtest --url http://127.0.0.1:8080 [--clients 64] [--seconds 10] [--request A.json,B.json,...]
  d1 download [--hf REPO[:QUANT][@REV]] [--no-mmproj]      fetch the GGUFs into the cache, print their paths

MODEL (default: download LiquidAI/d1-omni-600M-GGUF:F16 with its mmproj on first use):
  -m MODEL.gguf [--mmproj MMPROJ.gguf]   local files
  --hf REPO[:QUANT][@REV]                a Hugging Face repo, e.g. LiquidAI/d1-omni-600M-GGUF:F16 (cached in
                                         $D1_CACHE or ~/.cache/d1rs/models, sha256-verified; HF_TOKEN for gated repos)
  --no-mmproj                            text only (skip the vision/audio encoders)

OPTIONS:
  --max-batch-tokens N   tokens per forward batch (default 8192); larger = more throughput, more latency
  --ctx N                workspace capacity in tokens, the longest single question (default 16384)
  --media-cache-mb N     GPU memory for cached media prefixes in flight (default 1536)
  --vision-patches N     vision tower batch capacity in patches (default 11264)
  --fast                 fp16 accumulation in fp16 GEMMs (~1.3-1.6x faster, slightly less exact)
  --convert MODE         media normalization: auto (ffmpeg from PATH if present, default), off, or /path/to/ffmpeg;
                         converts MP3/OGG/FLAC/M4A/WebM/... audio and WebP/GIF/TIFF/HEIC/AVIF/... images
  --threads N            HTTP worker threads (default 256)
  --no-tune              skip GEMM autotuning (first start tunes for ~1 min, cached in ~/.cache/d1rs)
  --tune-cache PATH      GEMM plan cache file
  --device N             CUDA device (default 0)
  -v                     log every request
";

struct Args {
    pos: Vec<String>,
    kv: HashMap<String, String>,
    flags: Vec<String>,
}

fn parse_args() -> Args {
    let mut a = Args { pos: vec![], kv: HashMap::new(), flags: vec![] };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let valued = ["-m", "--model", "--mmproj", "--host", "--port", "--max-batch-tokens", "--ctx", "--media-cache-mb", "--vision-patches",
                  "--threads", "--device", "--tune-cache", "--convert", "--hf", "--clients", "--seconds", "--request", "--url", "--requests"];
    let mut i = 0;
    while i < raw.len() {
        let s = &raw[i];
        if valued.contains(&s.as_str()) && i + 1 < raw.len() {
            let k = if s == "-m" { "--model".to_string() } else { s.clone() };
            a.kv.insert(k, raw[i + 1].clone());
            i += 2;
        } else if s.starts_with('-') {
            a.flags.push(s.clone());
            i += 1;
        } else {
            a.pos.push(s.clone());
            i += 1;
        }
    }
    a
}

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn num(&self, k: &str, d: usize) -> usize {
        self.get(k).map(|v| v.parse().unwrap_or_else(|_| die(&format!("{k} expects a number")))).unwrap_or(d)
    }
    fn flag(&self, f: &str) -> bool {
        self.flags.iter().any(|x| x == f)
    }
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// Local files from -m/--mmproj, or files resolved (and downloaded if needed) from the Hugging Face Hub.
fn model_paths(a: &Args) -> (String, Option<String>) {
    if let Some(m) = a.get("--model") {
        return (m.to_string(), a.get("--mmproj").map(|s| s.to_string()));
    }
    let spec = a.get("--hf").map(|s| s.to_string()).unwrap_or_else(|| format!("{}:{}", hub::DEFAULT_REPO, hub::DEFAULT_QUANT));
    let want_mm = !a.flag("--no-mmproj") && a.get("--mmproj").is_none();
    let r = hub::resolve(&spec, want_mm).unwrap_or_else(|e| die(&format!("{spec}: {e}")));
    let mm = a.get("--mmproj").map(|s| s.to_string()).or(r.mmproj.map(|p| p.display().to_string()));
    (r.model.display().to_string(), mm)
}

fn build_service(a: &Args) -> Arc<service::Service> {
    let (model_path, mmproj_path) = model_paths(a);
    let model = model_path.as_str();
    let dev = a.num("--device", 0) as i32;
    if cuda::device_count() <= dev {
        die("no CUDA device available");
    }
    cuda::set_device(dev);
    let (major, minor) = cuda::compute_capability(dev);
    if major * 10 + minor < 75 {
        die("a GPU with compute capability 7.5 (Turing) or newer is required");
    }
    let t0 = Instant::now();
    let g = gguf::Gguf::open(model).unwrap_or_else(|e| die(&e));
    let tok = tokenizer::Tokenizer::from_gguf(&g).unwrap_or_else(|e| die(&e));
    let dl = prompt::Delims::new(&tok).unwrap_or_else(|e| die(&e));
    let text = model::TextModel::load(&g).unwrap_or_else(|e| die(&e));
    let temps = service::temperatures(&g);
    let name = g.s("general.name").unwrap_or("d1").to_string();
    let (mut vision, mut audio) = (None, None);
    if let Some(mp) = mmproj_path.as_deref() {
        let mg = gguf::Gguf::open(mp).unwrap_or_else(|e| die(&e));
        if mg.get("clip.has_vision_encoder").is_some() {
            vision = Some(vision::VisionModel::load(&mg, a.num("--vision-patches", 11 * 1024)).unwrap_or_else(|e| die(&e)));
        }
        if mg.get("clip.has_audio_encoder").is_some() {
            audio = Some(audio::AudioModel::load(&mg).unwrap_or_else(|e| die(&e)));
        }
    }
    let cap = a.num("--ctx", 16384).max(1024);
    let (major, minor) = cuda::compute_capability(dev);
    let (_, total_mem) = cuda::mem_info();
    let tune_cache = std::env::var("HOME").ok().map(|h| {
        format!("{h}/.cache/d1rs/gemm-sm{major}{minor}-{}sm-{}m{}.plans", cuda::sm_count(dev), total_mem >> 20, if a.flag("--fast") { "-fast" } else { "" })
    });
    let cfg = engine::Config {
        tune: !a.flag("--no-tune"),
        tune_cache: a.get("--tune-cache").map(|s| s.to_string()).or(tune_cache),
        max_batch_tokens: a.num("--max-batch-tokens", 8192).max(64),
        cap_tokens: cap,
        media_cache_bytes: a.num("--media-cache-mb", 1536) << 20,
        fp16_acc: a.flag("--fast"),
    };
    let (has_vision, has_audio) = (vision.is_some(), audio.is_some());
    let conv = media::Converter::new(a.get("--convert").unwrap_or("auto"));
    eprintln!(
        "media: JPEG/PNG/BMP/PPM/WAV native; other formats {}",
        match &conv.ffmpeg { Some(f) => format!("converted with {f}"), None => "rejected (no converter; see --convert)".into() }
    );
    let eng = engine::start(engine::Models { text, vision, audio }, cfg);
    let (free, total) = cuda::mem_info();
    eprintln!(
        "loaded {name} in {:.2}s (vision: {has_vision}, audio: {has_audio}); GPU memory {} / {} MiB free",
        t0.elapsed().as_secs_f64(),
        free >> 20,
        total >> 20
    );
    Arc::new(service::Service {
        tok,
        dl,
        engine: eng,
        temps,
        limits: service::Limits { max_length: 16384, image_text_length: 896, audio_text_length: 15360, cap_tokens: cap },
        has_vision,
        has_audio,
        model_name: name,
        conv,
    })
}

const EXAMPLE: &str = r#"{"state": "I was charged twice this month, please refund one of them.",
 "questions": {
  "refund":  {"type": "noul", "instructions": "Is the customer asking for a refund?"},
  "team":    {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "fraud": "Suspected unauthorised use"}},
  "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}"#;

fn read_request_file(a: &Args) -> String {
    match a.get("--request") {
        Some(p) => std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}"))),
        None => EXAMPLE.to_string(),
    }
}

/// Replace local file paths in images/files/audio with data URIs (for `run` and `eval`).
fn inline_media(mut j: Json) -> Json {
    if let Json::Obj(o) = &mut j {
        for (k, v) in o.iter_mut() {
            if k != "images" && k != "files" && k != "audio" {
                continue;
            }
            let conv = |s: &str| -> Json {
                if s.starts_with("data:") || s.len() > 4096 {
                    return Json::str(s);
                }
                match std::fs::read(s) {
                    Ok(b) => {
                        let mime = if media::sniff(&b).is_image() { "image/octet-stream" } else { "audio/octet-stream" };
                        Json::str(format!("data:{mime};base64,{}", b64(&b)))
                    }
                    Err(_) => Json::str(s),
                }
            };
            let nv = match &*v {
                Json::Arr(a) => Json::Arr(a.iter().map(|x| x.as_str().map(conv).unwrap_or(x.clone())).collect()),
                Json::Str(s) => Json::Arr(vec![conv(s)]),
                other => other.clone(),
            };
            *v = nv;
        }
    }
    j
}

fn b64(d: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(d.len().div_ceil(3) * 4);
    for c in d.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

struct Lat {
    v: Mutex<Vec<f64>>,
}
impl Lat {
    fn report(&self, label: &str, secs: f64, questions: usize, tokens: usize) {
        let mut v = self.v.lock().unwrap().clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            println!("{label}: no completed requests");
            return;
        }
        let p = |q: f64| v[((v.len() as f64 - 1.0) * q).round() as usize];
        let tok = if tokens > 0 { format!(", {:.0} tokens/s", tokens as f64 / secs) } else { String::new() };
        println!(
            "{label}: {} requests in {:.2}s -> {:.1} req/s, {:.1} questions/s{tok} | latency ms p50 {:.2} p90 {:.2} p99 {:.2} max {:.2}",
            v.len(),
            secs,
            v.len() as f64 / secs,
            questions as f64 / secs,
            p(0.5),
            p(0.9),
            p(0.99),
            v[v.len() - 1]
        );
    }
}

fn http_post(stream: &mut std::net::TcpStream, reader: &mut std::io::BufReader<std::net::TcpStream>, host: &str, path: &str, body: &[u8]) -> Result<(u16, Vec<u8>), String> {
    let head = format!("POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len());
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let code: u16 = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or("bad status line")?;
    let mut len = 0;
    loop {
        line.clear();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let l = line.trim();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut b = vec![0u8; len];
    reader.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok((code, b))
}

fn main() {
    let a = parse_args();
    let cmd = a.pos.first().map(|s| s.as_str()).unwrap_or("");
    match cmd {
        "serve" => {
            let svc = build_service(&a);
            let addr = format!("{}:{}", a.get("--host").unwrap_or("127.0.0.1"), a.num("--port", 8080));
            if let Err(e) = server::serve(svc, &addr, a.num("--threads", 256), a.flag("-v")) {
                die(&format!("{addr}: {e}"));
            }
        }
        "run" => {
            let svc = build_service(&a);
            let body = match a.pos.get(1) {
                Some(p) => std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}"))),
                None => EXAMPLE.to_string(),
            };
            let req = inline_media(json::parse(&body).unwrap_or_else(|e| die(&e)));
            for i in 0..2 {
                let t0 = Instant::now();
                match svc.systemone(&req) {
                    Ok(r) => {
                        if i == 1 {
                            let mut s = String::new();
                            json::dumps(&r, &mut s, false);
                            println!("{s}");
                        }
                    }
                    Err(e) => die(&e.msg),
                }
                eprintln!("request {i}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        "eval" => {
            let svc = build_service(&a);
            let path = a.pos.get(1).unwrap_or_else(|| die("eval needs CASES.jsonl"));
            let f = std::fs::File::open(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
            let lines: Vec<String> = std::io::BufReader::new(f).lines().map_while(Result::ok).filter(|l| !l.trim().is_empty()).collect();
            // submit concurrently (exercises batching), print in order
            let out: Vec<String> = std::thread::scope(|s| {
                let hs: Vec<_> = lines
                    .iter()
                    .map(|l| {
                        let svc = &svc;
                        s.spawn(move || {
                            let req = inline_media(json::parse(l).unwrap_or_else(|e| die(&e)));
                            match svc.prepare(&req).and_then(|p| svc.run(p)) {
                                Ok((p, probs)) => {
                                    let o = Json::Obj(vec![
                                        ("probs".into(), Json::Obj(p.names.iter().zip(&probs).map(|(n, pr)| (n.clone(), Json::Arr(pr.iter().map(|&x| Json::num(x as f64)).collect()))).collect())),
                                        ("input_tokens".into(), Json::num(p.input_tokens as f64)),
                                    ]);
                                    json::to_string_compact(&o)
                                }
                                Err(e) => format!("{{\"error\":{}}}", json::to_string_compact(&Json::str(e.msg))),
                            }
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for l in out {
                println!("{l}");
            }
        }
        "bench" => {
            let svc = build_service(&a);
            let req = inline_media(json::parse(&read_request_file(&a)).unwrap_or_else(|e| die(&e)));
            let clients = a.num("--clients", 64);
            let secs = a.num("--seconds", 10) as f64;
            // warm up
            for _ in 0..3 {
                svc.systemone(&req).unwrap_or_else(|e| die(&e.msg));
            }
            let p0 = svc.prepare(&req).unwrap_or_else(|e| die(&e.msg));
            let (nq, ntok) = (p0.names.len(), p0.input_tokens);
            // single-request latency
            let lat1 = Lat { v: Mutex::new(vec![]) };
            let t0 = Instant::now();
            for _ in 0..50 {
                let t = Instant::now();
                svc.systemone(&req).unwrap();
                lat1.v.lock().unwrap().push(t.elapsed().as_secs_f64() * 1e3);
            }
            lat1.report("sequential (1 client)", t0.elapsed().as_secs_f64(), 50 * nq, 50 * ntok);
            let lat = Lat { v: Mutex::new(vec![]) };
            let done = AtomicUsize::new(0);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..clients {
                    s.spawn(|| {
                        while t0.elapsed().as_secs_f64() < secs {
                            let t = Instant::now();
                            svc.systemone(&req).unwrap();
                            lat.v.lock().unwrap().push(t.elapsed().as_secs_f64() * 1e3);
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
            let n = done.load(Ordering::Relaxed);
            lat.report(&format!("concurrent ({clients} clients)"), t0.elapsed().as_secs_f64(), n * nq, n * ntok);
            let s = &svc.engine.stats;
            let b = s.batches.load(Ordering::Relaxed).max(1);
            println!("engine: {} batches, avg {:.0} tokens / {:.1} questions per batch", b, s.tokens.load(Ordering::Relaxed) as f64 / b as f64, s.rows.load(Ordering::Relaxed) as f64 / b as f64);
        }
        "download" => {
            let (m, mm) = model_paths(&a);
            println!("{m}");
            if let Some(mm) = mm {
                println!("{mm}");
            }
        }
        "tokenize" => {
            // debug: one JSON string per stdin line -> token ids (no special tokens), as JSON arrays
            let (model, _) = model_paths(&a);
            let g = gguf::Gguf::open(&model).unwrap_or_else(|e| die(&e));
            let tok = tokenizer::Tokenizer::from_gguf(&g).unwrap_or_else(|e| die(&e));
            let stdin = std::io::stdin();
            let mut out = std::io::BufWriter::new(std::io::stdout());
            for line in stdin.lock().lines().map_while(Result::ok) {
                let Ok(Json::Str(s)) = json::parse(&line) else { continue };
                let mut ids = Vec::new();
                tok.encode(&s, &mut ids);
                writeln!(out, "{:?}", ids).ok();
            }
        }
        "gemm-bench" => {
            cuda::set_device(0);
            let st = cuda::new_stream();
            let mut blas = cuda::Blas::new(st);
            for &(m, n, k) in &[(4096usize, 4096usize, 4096usize), (154, 9216, 1024), (154, 1024, 4608), (2048, 9216, 1024), (2048, 1024, 4608), (8192, 9216, 1024), (8192, 1024, 4608)] {
                let x = cuda::DevBuf::new(m * k * 2);
                let w = cuda::DevBuf::new(n * k * 2);
                let y = cuda::DevBuf::new(m * n * 4);
                for (label, acc, out_f) in [("f16 out, f32 acc", false, false), ("f16 out, f16 acc", true, false), ("f32 out, f32 acc", false, true)] {
                    blas.fp16_acc = acc;
                    let out = if out_f { cuda::Out::F(y.f32()) } else { cuda::Out::H(y.f16()) };
                    let e0 = cuda::new_event(true);
                    let e1 = cuda::new_event(true);
                    for _ in 0..3 { blas.gemm(m, n, k, x.f16(), k, w.f16(), k, false, out, n, 1.0, 0.0); }
                    cuda::event_record(e0, st);
                    let iters = 20;
                    for _ in 0..iters { blas.gemm(m, n, k, x.f16(), k, w.f16(), k, false, out, n, 1.0, 0.0); }
                    cuda::event_record(e1, st);
                    cuda::event_sync(e1);
                    let ms = cuda::event_elapsed(e0, e1) / iters as f32;
                    println!("M={m:5} N={n:5} K={k:5} {label}: {ms:.3} ms  {:.1} TFLOPS", 2.0 * (m * n * k) as f64 / ms as f64 / 1e9);
                }
            }
        }
        "dump-patches" => {
            // debug: decoded + preprocessed patches as raw fp16, plus crop shapes on stderr
            let p = a.pos.get(1).unwrap_or_else(|| die("dump-patches IMG OUT"));
            let bytes = std::fs::read(p).unwrap();
            let t0 = Instant::now();
            let img = image::decode(&bytes).unwrap_or_else(|e| die(&e));
            let t1 = Instant::now();
            let _ = vision::preprocess(&img);
            eprintln!("decode {:.2} ms, preprocess {:.2} ms", (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
            std::fs::write(format!("{}.rgb", a.pos[2]), &img.rgb).unwrap();
            eprintln!("{} {}", img.w, img.h);
            let pt = vision::preprocess(&img);
            let bytes: Vec<u8> = pt.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&a.pos[2], bytes).unwrap();
            println!("{:?}", pt.shapes);
        }
        "loadtest" => {
            // --request a.json,b.json,...: every client cycles through the files (offset by client index), so the
            // server sees a mix of request types at all times; latency is reported overall and per file.
            let url = a.get("--url").unwrap_or("http://127.0.0.1:8080");
            let hostport = url.trim_start_matches("http://").trim_end_matches('/').to_string();
            let files: Vec<String> = a.get("--request").map(|r| r.split(',').map(|x| x.to_string()).collect()).unwrap_or_default();
            let bodies: Vec<(String, String, usize)> = if files.is_empty() {
                vec![("example".into(), EXAMPLE.into())]
            } else {
                files.iter().map(|f| (f.clone(), std::fs::read_to_string(f).unwrap_or_else(|e| die(&format!("{f}: {e}"))))).collect()
            }
            .into_iter()
            .map(|(n, t)| {
                let j = inline_media(json::parse(&t).unwrap_or_else(|e| die(&e)));
                let nq = match j.get("questions") {
                    Some(Json::Obj(o)) => o.len(),
                    _ => 1,
                };
                (n, json::to_string_compact(&j), nq)
            })
            .collect();
            let clients = a.num("--clients", 64);
            let secs = a.num("--seconds", 10) as f64;
            let lat = Lat { v: Mutex::new(vec![]) };
            let per: Vec<Lat> = bodies.iter().map(|_| Lat { v: Mutex::new(vec![]) }).collect();
            let errors = AtomicUsize::new(0);
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for c in 0..clients {
                    let (bodies, per, lat, errors, hostport) = (&bodies, &per, &lat, &errors, &hostport);
                    s.spawn(move || {
                        let mut conn = std::net::TcpStream::connect(hostport).unwrap_or_else(|e| die(&format!("{hostport}: {e}")));
                        conn.set_nodelay(true).ok();
                        let mut rd = std::io::BufReader::new(conn.try_clone().unwrap());
                        let mut i = c;
                        while t0.elapsed().as_secs_f64() < secs {
                            let k = i % bodies.len();
                            i += 1;
                            let t = Instant::now();
                            match http_post(&mut conn, &mut rd, hostport, "/v1/systemone", bodies[k].1.as_bytes()) {
                                Ok((200, _)) => {
                                    let ms = t.elapsed().as_secs_f64() * 1e3;
                                    lat.v.lock().unwrap().push(ms);
                                    per[k].v.lock().unwrap().push(ms);
                                }
                                Ok((c, b)) => {
                                    if errors.fetch_add(1, Ordering::Relaxed) < 3 {
                                        eprintln!("HTTP {c}: {}", String::from_utf8_lossy(&b));
                                    }
                                }
                                Err(e) => {
                                    errors.fetch_add(1, Ordering::Relaxed);
                                    eprintln!("{e}");
                                    return;
                                }
                            }
                        }
                    });
                }
            });
            let el = t0.elapsed().as_secs_f64();
            let nq: usize = bodies.iter().zip(&per).map(|(b, l)| b.2 * l.v.lock().unwrap().len()).sum();
            lat.report(&format!("HTTP ({clients} clients, all)"), el, nq, 0);
            if bodies.len() > 1 {
                for (b, l) in bodies.iter().zip(&per) {
                    let n = l.v.lock().unwrap().len();
                    l.report(&format!("  {}", b.0), el, n * b.2, 0);
                }
            }
            println!("errors: {}", errors.load(Ordering::Relaxed));
        }
        _ => {
            eprint!("{USAGE}");
            std::process::exit(if cmd.is_empty() || cmd == "help" { 0 } else { 1 });
        }
    }
}
