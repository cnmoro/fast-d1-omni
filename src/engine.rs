//! Continuous batching engine. One thread owns the GPU. Requests (jobs) arrive on a channel at any time;
//! every iteration the engine
//!   1. admits new jobs,
//!   2. encodes pending media (vision / audio) and runs their prefix through the trunk once (cached K/V),
//!   3. packs question rows from the queue (FIFO, across requests) into one variable-length batch under a token
//!      budget and launches it, keeping up to two batches in flight so the CPU work of batch N+1 overlaps the
//!      GPU work of batch N,
//!   4. hands finished rows' logits back to the waiting request threads.

use crate::audio::AudioModel;
use crate::cuda::{self, Blas, DevBuf, Stream};
use crate::model::{Pending, PrefixCache, PrefixSeq, Runner, TextModel, TextSeq, Workspace, D, SLOTS};
use crate::vision::{Patches, VisionModel};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Row {
    pub ids: Vec<i32>,
    pub markers: Vec<i32>,
    pub qtype: i32,
}

pub enum Media {
    None,
    Images(Vec<Patches>),
    Audio { mel: Vec<f32>, valid: usize },
}

impl Media {
    pub fn prefix_len(&self) -> usize {
        match self {
            Media::None => 0,
            Media::Images(p) => p.iter().map(|x| x.tokens()).sum(),
            Media::Audio { valid, .. } => {
                let mut l = *valid;
                for _ in 0..3 {
                    l = (l - 1) / 2 + 1;
                }
                l
            }
        }
    }
}

pub struct Job {
    pub rows: Vec<Row>,
    pub media: Media,
    pub reply: Sender<Result<Vec<Vec<f32>>, String>>,
}

#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub rows: AtomicU64,
    pub tokens: AtomicU64,
    pub batches: AtomicU64,
    pub media: AtomicU64,
    pub queue: AtomicUsize,
    pub busy_us: AtomicU64,
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: Sender<Job>,
    pub stats: Arc<Stats>,
}

impl EngineHandle {
    pub fn submit(&self, job: Job) -> Result<(), String> {
        self.stats.queue.fetch_add(1, Ordering::Relaxed);
        self.tx.send(job).map_err(|_| "engine stopped".to_string())
    }
}

pub struct Config {
    pub tune: bool,
    pub tune_cache: Option<String>,
    pub max_batch_tokens: usize,
    pub cap_tokens: usize,
    pub media_cache_bytes: usize,
    pub fp16_acc: bool,
}

struct JobState {
    job: Job,
    prefix: Option<PrefixCache>,
    media_done: bool,
    next_row: usize,
    done_rows: usize,
    results: Vec<Option<Vec<f32>>>,
    failed: Option<String>,
}

struct InFlight {
    pending: Option<Pending>,
    rows: Vec<(u64, usize, usize)>, // (job, row, n_markers)
}

pub struct Models {
    pub text: TextModel,
    pub vision: Option<VisionModel>,
    pub audio: Option<AudioModel>,
}

pub fn start(models: Models, cfg: Config) -> EngineHandle {
    let (tx, rx) = channel::<Job>();
    let stats = Arc::new(Stats::default());
    let st2 = stats.clone();
    let (ready_tx, ready_rx) = channel();
    std::thread::Builder::new()
        .name("d1-engine".into())
        .spawn(move || {
            let st = cuda::new_stream();
            let mut blas = Blas::new(st);
            blas.fp16_acc = cfg.fp16_acc;
            let mut ws = Workspace::new(&models.text, cfg.cap_tokens);
            if cfg.tune {
                tune(&models, &blas, &mut ws, st, &cfg);
            }
            let mut e = Engine {
                m: models,
                blas,
                st,
                ws,
                cfg,
                rx,
                jobs: HashMap::new(),
                order: VecDeque::new(),
                next_id: 0,
                inflight: VecDeque::new(),
                slot: 0,
                cache_bytes: 0,
                long_cost: 0.0,
                short_cost: 0.0,
                stats: st2,
            };
            ready_tx.send(()).ok();
            e.run();
        })
        .expect("spawn engine");
    ready_rx.recv().expect("engine failed to start");
    EngineHandle { tx, stats }
}

struct Engine {
    m: Models,
    blas: Blas,
    st: Stream,
    ws: Workspace,
    cfg: Config,
    rx: Receiver<Job>,
    jobs: HashMap<u64, JobState>,
    order: VecDeque<u64>,
    next_id: u64,
    inflight: VecDeque<InFlight>,
    slot: usize,
    cache_bytes: usize,
    long_cost: f64,
    short_cost: f64,
    stats: Arc<Stats>,
}

impl Engine {
    fn admit(&mut self, job: Job) {
        self.stats.queue.fetch_sub(1, Ordering::Relaxed);
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let id = self.next_id;
        self.next_id += 1;
        let n = job.rows.len();
        let media_done = matches!(job.media, Media::None);
        self.jobs.insert(
            id,
            JobState { job, prefix: None, media_done, next_row: 0, done_rows: 0, results: vec![None; n], failed: None },
        );
        self.order.push_back(id);
    }

    fn run(&mut self) {
        loop {
            // 1. admit
            if self.order.is_empty() && self.inflight.is_empty() {
                match self.rx.recv() {
                    Ok(j) => self.admit(j),
                    Err(_) => return,
                }
            }
            loop {
                match self.rx.try_recv() {
                    Ok(j) => self.admit(j),
                    Err(_) => break,
                }
            }
            let t0 = Instant::now();
            // 2. text first (latency), 3. then one bounded media group; the GPU executes them in this order
            let launched = self.launch_batch();
            let media = self.process_media();
            if !launched && !media {
                if let Some(f) = self.inflight.pop_front() {
                    self.complete(f);
                } else if !self.order.is_empty() {
                    // waiting for media budget; nothing in flight means nothing will free it -> force one
                    if !self.process_media_forced() {
                        // nothing runnable at all; avoid spinning
                        match self.rx.recv_timeout(Duration::from_millis(1)) {
                            Ok(j) => self.admit(j),
                            Err(RecvTimeoutError::Disconnected) => return,
                            Err(RecvTimeoutError::Timeout) => {}
                        }
                    }
                }
            } else if self.inflight.len() >= SLOTS {
                let f = self.inflight.pop_front().unwrap();
                self.complete(f);
            }
            self.stats.busy_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    fn next_slot(&mut self) -> usize {
        // a slot may only be reused once its previous batch has been collected
        if self.inflight.len() >= SLOTS {
            let f = self.inflight.pop_front().unwrap();
            self.complete(f);
        }
        let s = self.slot;
        self.slot = (self.slot + 1) % SLOTS;
        s
    }

    fn process_media_forced(&mut self) -> bool {
        let saved = self.cfg.media_cache_bytes;
        self.cfg.media_cache_bytes = usize::MAX;
        let before = self.jobs.values().filter(|j| j.media_done).count();
        self.process_media_limited(1);
        self.cfg.media_cache_bytes = saved;
        self.jobs.values().filter(|j| j.media_done).count() > before
    }

    /// One media group per engine iteration, so text batches keep flowing between media work.
    fn process_media(&mut self) -> bool {
        let before = self.stats.media.load(Ordering::Relaxed);
        self.process_media_limited(1);
        self.stats.media.load(Ordering::Relaxed) != before
    }

    /// Encode the media of queued jobs (FIFO, within the cache budget) in groups: images of several requests share
    /// one vision-tower pass, and all prefixes of a group share one trunk prefix pass.
    fn process_media_limited(&mut self, max_groups: usize) {
        for _ in 0..max_groups {
            let max_patches = self.m.vision.as_ref().map_or(usize::MAX, |v| v.max_patches);
            let mut group: Vec<u64> = Vec::new();
            let (mut tokens, mut patches, mut bytes) = (0usize, 0usize, 0usize);
            for &id in &self.order {
                let js = &self.jobs[&id];
                if js.media_done {
                    continue;
                }
                let p = js.job.media.prefix_len();
                let np: usize = match &js.job.media {
                    Media::Images(im) => im.iter().map(|x| x.shapes.iter().map(|(h, w)| h * w).sum::<usize>()).sum(),
                    _ => 0,
                };
                let need = PrefixCache::bytes_for(&self.m.text, p) + p * D * 4;
                if self.cache_bytes + bytes + need > self.cfg.media_cache_bytes && (self.cache_bytes + bytes) > 0 {
                    break; // FIFO: later jobs wait for memory too
                }
                // small groups keep the GPU time between text batches short
                if !group.is_empty() && (tokens + p > 2048 || patches + np > max_patches || group.len() >= 4) {
                    break;
                }
                group.push(id);
                tokens += p;
                patches += np;
                bytes += need;
            }
            if group.is_empty() {
                return;
            }
            self.encode_group(&group);
        }
    }

    fn encode_group(&mut self, group: &[u64]) {
        let st = self.st;
        // validate, compute offsets
        let mut ok: Vec<(u64, usize, usize)> = Vec::new(); // (job, offset, len)
        let mut total = 0;
        for &id in group {
            let p = self.jobs[&id].job.media.prefix_len();
            let err = if p == 0 || p > self.ws.cap_tokens {
                Some(format!("media prefix of {p} positions exceeds the context"))
            } else {
                match &self.jobs[&id].job.media {
                    Media::Images(_) if self.m.vision.is_none() => Some("this server was started without a vision encoder (--mmproj)".to_string()),
                    Media::Audio { .. } if self.m.audio.is_none() => Some("this server was started without an audio encoder (--mmproj)".to_string()),
                    _ => None,
                }
            };
            if let Some(e) = err {
                let js = self.jobs.get_mut(&id).unwrap();
                js.failed = Some(e);
                js.media_done = true;
                continue;
            }
            ok.push((id, total, p));
            total += p;
        }
        if ok.is_empty() {
            return;
        }
        let emb = unsafe { cuda::malloc_async(total * D * 4, st) } as *mut f32;
        {
            // vision: batch crops across jobs up to the tower capacity (an oversized image is chunked alone)
            let mut batch: Vec<Patches> = Vec::new();
            let mut batch_off = 0usize; // output row of the batch's first crop
            let mut batch_patches = 0usize;
            let flush = |batch: &mut Vec<Patches>, batch_off: &mut usize, batch_patches: &mut usize, next_off: usize| {
                if let (false, Some(v)) = (batch.is_empty(), self.m.vision.as_ref()) {
                    let refs: Vec<&Patches> = batch.iter().collect();
                    v.encode(&self.blas, st, &refs, unsafe { emb.add(*batch_off * D) });
                }
                batch.clear();
                *batch_off = next_off;
                *batch_patches = 0;
            };
            for &(id, off, _) in &ok {
                match &self.jobs[&id].job.media {
                    Media::Images(imgs) => {
                        let v = self.m.vision.as_ref().unwrap();
                        let mut out = off;
                        for img in imgs {
                            let mut i = 0;
                            while i < img.shapes.len() {
                                // largest run of crops starting at i that fits the capacity
                                let mut j = i;
                                let mut np = 0;
                                while j < img.shapes.len() && np + img.shapes[j].0 * img.shapes[j].1 <= v.max_patches {
                                    np += img.shapes[j].0 * img.shapes[j].1;
                                    j += 1;
                                }
                                if batch_patches + np > v.max_patches || (batch.is_empty() && batch_off != out) {
                                    flush(&mut batch, &mut batch_off, &mut batch_patches, out);
                                }
                                let start: usize = img.shapes[..i].iter().map(|(h, w)| h * w).sum::<usize>() * 768;
                                let sub = Patches { data: img.data[start..start + np * 768].to_vec(), shapes: img.shapes[i..j].to_vec() };
                                out += sub.tokens();
                                batch_patches += np;
                                batch.push(sub);
                                i = j;
                            }
                        }
                    }
                    Media::Audio { mel, valid } => {
                        flush(&mut batch, &mut batch_off, &mut batch_patches, off);
                        self.m.audio.as_ref().unwrap().encode(&self.blas, st, mel, *valid, unsafe { emb.add(off * D) });
                    }
                    Media::None => {}
                }
            }
            flush(&mut batch, &mut batch_off, &mut batch_patches, 0);
        }
        if let Ok(path) = std::env::var("D1_LOAD_PREFIX") {
            let b = std::fs::read(path).unwrap();
            let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            unsafe { cuda::h2d_async(emb as *mut c_void, v.as_ptr() as *const c_void, v.len().min(total * D) * 4, st) };
            cuda::stream_sync(st);
        }
        if let Ok(path) = std::env::var("D1_DUMP_PREFIX") {
            let mut v = vec![0f32; total * D];
            unsafe { cuda::d2h_async(v.as_mut_ptr() as *mut c_void, emb as *const c_void, total * D * 4, st) };
            cuda::stream_sync(st);
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(path, b).ok();
        }
        let caches: Vec<PrefixCache> = ok.iter().map(|&(_, _, p)| PrefixCache::alloc(&self.m.text, p, st)).collect();
        let slot = self.next_slot();
        {
            let items: Vec<PrefixSeq> = ok.iter().zip(&caches).map(|(&(_, off, _), c)| PrefixSeq { emb: unsafe { emb.add(off * D) }, cache: c }).collect();
            let r = Runner { m: &self.m.text, blas: &self.blas, st };
            r.launch_prefix(&mut self.ws, slot, &items);
        }
        self.inflight.push_back(InFlight { pending: None, rows: Vec::new() });
        unsafe { cuda::free_async(emb as *mut c_void, st) };
        self.short_cost += 3.0 * total as f64;
        for (&(id, _, _), cache) in ok.iter().zip(caches) {
            self.cache_bytes += cache.bytes;
            self.stats.media.fetch_add(1, Ordering::Relaxed);
            let js = self.jobs.get_mut(&id).unwrap();
            js.prefix = Some(cache);
            js.media_done = true;
        }
    }

    /// Pack rows FIFO across jobs into one batch and launch it.
    fn launch_batch(&mut self) -> bool {
        // A long question (more than half the budget) fills a whole batch on its own. After a batch with long rows,
        // the next few iterations prefer short rows and media, so short requests never queue behind a run of long
        // ones; long rows still run whenever nothing short is waiting.
        // While short work is waiting, long rows get at most about a quarter of the recent GPU cost (in tokens).
        if self.long_cost * 3.0 > self.short_cost && self.launch_batch_filtered(true) {
            return true;
        }
        self.launch_batch_filtered(false)
    }

    fn launch_batch_filtered(&mut self, short_only: bool) -> bool {
        let budget = self.cfg.max_batch_tokens;
        let long = (budget / 2).max(1);
        let mut picked: Vec<(u64, usize)> = Vec::new();
        let mut tokens = 0;
        let mut markers = 0;
        let mut failed = Vec::new();
        'outer: for &id in &self.order {
            let js = self.jobs.get(&id).unwrap();
            if js.failed.is_some() {
                failed.push(id);
                continue;
            }
            if !js.media_done {
                continue; // waiting for media memory; later jobs may proceed
            }
            for r in js.next_row..js.job.rows.len() {
                let n = js.job.rows[r].ids.len();
                let nm = js.job.rows[r].markers.len();
                if short_only && n > long {
                    continue 'outer;
                }
                if !picked.is_empty() && (tokens + n > budget || markers + nm > self.ws.cap_markers) {
                    break 'outer;
                }
                if n > self.ws.cap_tokens {
                    failed.push(id);
                    continue 'outer;
                }
                picked.push((id, r));
                tokens += n;
                markers += nm;
            }
        }
        for id in failed {
            let js = self.jobs.get_mut(&id).unwrap();
            if js.failed.is_none() {
                js.failed = Some("a question exceeds the batch capacity".into());
            }
            self.finish_job(id);
        }
        if picked.is_empty() {
            return false;
        }
        let _ = short_only;
        if picked.iter().any(|&(id, r)| self.jobs[&id].job.rows[r].ids.len() > long) {
            self.long_cost += tokens as f64;
        } else {
            self.short_cost += tokens as f64;
        }
        if self.long_cost + self.short_cost > 4e6 {
            self.long_cost *= 0.5;
            self.short_cost *= 0.5;
        }
        let slot = self.next_slot();
        let mut rows_meta = Vec::with_capacity(picked.len());
        let pending = {
            let mut seqs = Vec::with_capacity(picked.len());
            for &(id, r) in &picked {
                let js = &self.jobs[&id];
                let row = &js.job.rows[r];
                seqs.push(TextSeq { ids: &row.ids, markers: &row.markers, qtype: row.qtype, prefix: js.prefix.as_ref() });
                rows_meta.push((id, r, row.markers.len()));
            }
            let runner = Runner { m: &self.m.text, blas: &self.blas, st: self.st };
            runner.launch_text(&mut self.ws, slot, &seqs)
        };
        for &(id, r) in &picked {
            self.jobs.get_mut(&id).unwrap().next_row = r + 1;
        }
        self.stats.batches.fetch_add(1, Ordering::Relaxed);
        self.stats.rows.fetch_add(picked.len() as u64, Ordering::Relaxed);
        self.stats.tokens.fetch_add(tokens as u64, Ordering::Relaxed);
        self.inflight.push_back(InFlight { pending: Some(pending), rows: rows_meta });
        true
    }

    fn complete(&mut self, f: InFlight) {
        let Some(p) = f.pending else {
            return; // prefix pass: nothing to collect
        };
        let runner = Runner { m: &self.m.text, blas: &self.blas, st: self.st };
        let logits = runner.finish(&mut self.ws, &p);
        let mut off = 0;
        let mut finished = Vec::new();
        for (id, r, nm) in f.rows {
            let Some(js) = self.jobs.get_mut(&id) else {
                off += nm;
                continue;
            };
            js.results[r] = Some(logits[off..off + nm].to_vec());
            off += nm;
            js.done_rows += 1;
            if js.done_rows == js.job.rows.len() {
                finished.push(id);
            }
        }
        for id in finished {
            self.finish_job(id);
        }
    }

    fn finish_job(&mut self, id: u64) {
        let Some(js) = self.jobs.remove(&id) else { return };
        self.order.retain(|&x| x != id);
        if let Some(c) = js.prefix {
            self.cache_bytes -= c.bytes;
            c.free(self.st);
        }
        let res = match js.failed {
            Some(e) => Err(e),
            None => Ok(js.results.into_iter().map(|r| r.unwrap_or_default()).collect()),
        };
        js.job.reply.send(res).ok();
    }
}

/// Autotune the trunk/head GEMMs for every M bucket up to the batch budget (cached on disk).
fn tune(models: &Models, blas: &Blas, ws: &mut Workspace, st: Stream, cfg: &Config) {
    let m = &models.text;
    use crate::cuda::Shape;
    let loaded = cfg.tune_cache.as_ref().map(|p| blas.load_plans(p)).unwrap_or(-1);
    // record the shapes of one synthetic batch
    let ids: Vec<i32> = (0..128).map(|i| 1000 + i).collect();
    let markers = vec![20, 40, 60];
    let seqs: Vec<TextSeq> = (0..4).map(|_| TextSeq { ids: &ids, markers: &markers, qtype: 0, prefix: None }).collect();
    *blas.record.borrow_mut() = Some(Vec::new());
    let r = Runner { m, blas, st };
    let p = r.launch_text(ws, 0, &seqs);
    r.finish(ws, &p);
    let rec = blas.record.borrow_mut().take().unwrap_or_default();
    // (shape, largest M to tune for)
    let mut shapes: Vec<(Shape, usize)> = Vec::new();
    let mut add = |rec: Vec<Shape>, cap: Option<usize>| {
        for s in rec {
            let k = Shape { m: 0, ..s };
            let lim = cap.unwrap_or(s.m);
            match shapes.iter_mut().find(|(x, _)| *x == k) {
                Some(e) => e.1 = e.1.max(lim),
                None => shapes.push((k, lim)),
            }
        }
    };
    add(rec, Some(cfg.max_batch_tokens));
    // media encoders at their largest sizes (eleven 512 px crops; a 30 s clip)
    let emb = DevBuf::new(11 * 256 * D * 4);
    if let Some(v) = &models.vision {
        let n = v.max_patches / 1024;
        if n > 0 {
            let p = Patches { data: vec![0u16; n * 1024 * 768], shapes: vec![(32, 32); n] };
            *blas.record.borrow_mut() = Some(Vec::new());
            v.encode(blas, st, &[&p], emb.f32());
            add(blas.record.borrow_mut().take().unwrap_or_default(), None);
        }
    }
    if let Some(a) = &models.audio {
        let mel = vec![0f32; 3000 * 128];
        *blas.record.borrow_mut() = Some(Vec::new());
        a.encode(blas, st, &mel, 3000, emb.f32());
        add(blas.record.borrow_mut().take().unwrap_or_default(), None);
    }
    cuda::stream_sync(st);
    let buckets_upto = |lim: usize| {
        let mut buckets = Vec::new();
        let mut b = 1;
        loop {
            let nb = unsafe { cuda::d1_lt_bucket(b as i32) } as usize;
            if !buckets.contains(&nb) {
                buckets.push(nb);
            }
            if nb >= lim || nb >= 16384 {
                break;
            }
            b = nb + 1;
        }
        buckets
    };
    let work: Vec<(Shape, usize)> = shapes.iter().flat_map(|&(s, lim)| buckets_upto(lim.max(16)).into_iter().map(move |b| (s, b))).collect();
    let total = work.len();
    let t0 = Instant::now();
    let mut done = 0;
    let mut gain = (0f64, 0f64);
    for (s, bm) in work {
        let us = blas.tune(&Shape { m: bm, ..s }, 1, st);
        done += 1;
        if us > 0.0 {
            gain.0 += us as f64;
        }
        if done % 16 == 0 {
            eprint!("\rtuning GEMMs: {done}/{total} ({:.0}s)   ", t0.elapsed().as_secs_f64());
        }
    }
    let _ = gain.1;
    cuda::stream_sync(st);
    blas.free_tuning_scratch();
    if loaded <= 0 || t0.elapsed().as_secs_f64() > 1.0 {
        eprintln!("\rtuned {total} GEMM shapes in {:.1}s", t0.elapsed().as_secs_f64());
    }
    if let Some(p) = &cfg.tune_cache {
        if let Some(dir) = std::path::Path::new(p).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if !blas.save_plans(p) {
            eprintln!("warning: could not write GEMM plan cache {p}");
        }
    }
}
