//! Minimal HTTP/1.1 server (keep-alive, Content-Length and chunked bodies) on std::net with a worker pool.

use crate::json::{self, Json};
use crate::service::{ApiError, Service};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_BODY: usize = 256 << 20;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    pub keep_alive: bool,
}

fn read_request(r: &mut BufReader<TcpStream>) -> Result<Option<Request>, String> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(None);
        }
        if !line.trim().is_empty() {
            break;
        }
    }
    let mut it = line.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("/").to_string();
    let version = it.next().unwrap_or("HTTP/1.1").to_string();
    let mut content_length = 0usize;
    let mut chunked = false;
    let mut keep_alive = version != "HTTP/1.0";
    let mut header_bytes = 0;
    loop {
        line.clear();
        let n = r.read_line(&mut line).map_err(|e| e.to_string())?;
        header_bytes += n;
        if n == 0 || header_bytes > 1 << 16 {
            return Err("bad headers".into());
        }
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            match k.as_str() {
                "content-length" => content_length = v.parse().map_err(|_| "bad content-length")?,
                "transfer-encoding" => chunked = v.to_ascii_lowercase().contains("chunked"),
                "connection" => {
                    let v = v.to_ascii_lowercase();
                    if v.contains("close") {
                        keep_alive = false;
                    } else if v.contains("keep-alive") {
                        keep_alive = true;
                    }
                }
                _ => {}
            }
        }
    }
    let mut body = Vec::new();
    if chunked {
        loop {
            line.clear();
            r.read_line(&mut line).map_err(|e| e.to_string())?;
            let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or("0"), 16).map_err(|_| "bad chunk")?;
            if size == 0 {
                // trailers
                loop {
                    line.clear();
                    if r.read_line(&mut line).map_err(|e| e.to_string())? <= 2 {
                        break;
                    }
                }
                break;
            }
            if body.len() + size > MAX_BODY {
                return Err("body too large".into());
            }
            let s = body.len();
            body.resize(s + size, 0);
            r.read_exact(&mut body[s..]).map_err(|e| e.to_string())?;
            line.clear();
            r.read_line(&mut line).map_err(|e| e.to_string())?;
        }
    } else if content_length > 0 {
        if content_length > MAX_BODY {
            return Err("body too large".into());
        }
        body.resize(content_length, 0);
        r.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    Ok(Some(Request { method, path, body, keep_alive }))
}

fn reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn respond(w: &mut TcpStream, code: u16, body: &str, keep_alive: bool) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nConnection: {}\r\n\r\n",
        reason(code),
        body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
    let mut buf = Vec::with_capacity(head.len() + body.len());
    buf.extend_from_slice(head.as_bytes());
    buf.extend_from_slice(body.as_bytes());
    w.write_all(&buf)
}

fn error_body(e: &ApiError) -> String {
    let t = if e.code < 500 { "invalid_request_error" } else { "server_error" };
    json::to_string_compact(&Json::Obj(vec![(
        "error".into(),
        Json::Obj(vec![
            ("code".into(), Json::Num(e.code as f64, Some(e.code.to_string()))),
            ("message".into(), Json::str(e.msg.clone())),
            ("type".into(), Json::str(t)),
        ]),
    )]))
}

fn handle(svc: &Service, req: &Request) -> (u16, String) {
    let path = req.path.split('?').next().unwrap_or("");
    match (req.method.as_str(), path) {
        ("OPTIONS", _) => (204, String::new()),
        ("GET", "/health") | ("GET", "/v1/health") => (200, r#"{"status":"ok"}"#.into()),
        ("GET", "/v1/models") | ("GET", "/models") => (
            200,
            json::to_string_compact(&Json::Obj(vec![
                ("object".into(), Json::str("list")),
                ("data".into(), Json::Arr(vec![Json::Obj(vec![("id".into(), Json::str(svc.model_name.clone())), ("object".into(), Json::str("model"))])])),
            ])),
        ),
        ("GET", "/metrics") | ("GET", "/stats") => {
            let s = &svc.engine.stats;
            let n = |v: u64| Json::Num(v as f64, Some(v.to_string()));
            (
                200,
                json::to_string_compact(&Json::Obj(vec![
                    ("requests".into(), n(s.requests.load(Ordering::Relaxed))),
                    ("questions".into(), n(s.rows.load(Ordering::Relaxed))),
                    ("tokens".into(), n(s.tokens.load(Ordering::Relaxed))),
                    ("batches".into(), n(s.batches.load(Ordering::Relaxed))),
                    ("media".into(), n(s.media.load(Ordering::Relaxed))),
                    ("queued".into(), n(s.queue.load(Ordering::Relaxed) as u64)),
                ])),
            )
        }
        ("POST", "/v1/systemone") | ("POST", "/systemone") => {
            let parsed = std::str::from_utf8(&req.body).map_err(|_| "body is not UTF-8".to_string()).and_then(json::parse);
            match parsed {
                Err(e) => (400, error_body(&ApiError { code: 400, msg: e })),
                Ok(j) => match svc.systemone(&j) {
                    Ok(r) => (200, json::to_string_compact(&r)),
                    Err(e) => (e.code, error_body(&e)),
                },
            }
        }
        ("POST", "/v1/systemone/batch") => {
            let parsed = std::str::from_utf8(&req.body).map_err(|_| "body is not UTF-8".to_string()).and_then(json::parse);
            let j = match parsed {
                Err(e) => return (400, error_body(&ApiError { code: 400, msg: e })),
                Ok(j) => j,
            };
            let Some(Json::Arr(items)) = j.get("requests") else {
                return (400, error_body(&ApiError { code: 400, msg: "`requests` must be a list".into() }));
            };
            // submit everything first so the engine batches across items
            let results: Vec<Json> = std::thread::scope(|s| {
                let hs: Vec<_> = items
                    .iter()
                    .map(|it| {
                        s.spawn(move || match svc.systemone(it) {
                            Ok(r) => r,
                            Err(e) => json::parse(&error_body(&e)).unwrap(),
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or(Json::Null)).collect()
            });
            (200, json::to_string_compact(&Json::Obj(vec![("results".into(), Json::Arr(results))])))
        }
        (_, "/v1/systemone") | (_, "/v1/systemone/batch") => (405, error_body(&ApiError { code: 405, msg: "use POST".into() })),
        _ => (404, error_body(&ApiError { code: 404, msg: format!("no route {path}") })),
    }
}

fn connection(svc: &Service, stream: TcpStream, verbose: bool) {
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
    let Ok(mut w) = stream.try_clone() else { return };
    let mut r = BufReader::with_capacity(64 << 10, stream);
    loop {
        let req = match read_request(&mut r) {
            Ok(Some(q)) => q,
            Ok(None) => return,
            Err(e) => {
                let _ = respond(&mut w, 400, &error_body(&ApiError { code: 400, msg: e }), false);
                return;
            }
        };
        let t0 = Instant::now();
        let (code, body) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(svc, &req)))
            .unwrap_or_else(|_| (500, error_body(&ApiError { code: 500, msg: "internal error".into() })));
        if verbose {
            eprintln!("{} {} -> {code} in {:.2} ms", req.method, req.path, t0.elapsed().as_secs_f64() * 1e3);
        }
        if respond(&mut w, code, &body, req.keep_alive).is_err() || !req.keep_alive {
            return;
        }
    }
}

pub fn serve(svc: Arc<Service>, addr: &str, threads: usize, verbose: bool) -> std::io::Result<()> {
    let listener = Arc::new(TcpListener::bind(addr)?);
    eprintln!("listening on http://{addr}  (POST /v1/systemone, /v1/systemone/batch; GET /health, /metrics)");
    let mut handles = Vec::with_capacity(threads);
    for i in 0..threads {
        let l = listener.clone();
        let s = svc.clone();
        handles.push(std::thread::Builder::new().name(format!("http-{i}")).stack_size(1 << 20).spawn(move || loop {
            match l.accept() {
                Ok((c, _)) => connection(&s, c, verbose),
                Err(e) => {
                    eprintln!("accept: {e}");
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        })?);
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}
