# d1rs

A CUDA inference server for [LiquidAI d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M-GGUF), written in
Rust with no crates. It reads the GGUF files directly, runs text, image and audio questions, and serves the
`/v1/systemone` API with continuous batching.

The only external dependencies are the CUDA toolkit (`nvcc`, `libcudart`, `libcublas`, `libcublasLt`). The Rust code
implements the GGUF reader, the BPE tokenizer, JSON, the HTTP/1.1 server, JPEG/PNG/BMP decoding, WAV decoding,
resampling and the mel front end itself. The CUDA kernels (flash attention, fused short convolution, norms, the
vision and audio pieces) are in `cuda/kernels.cu`, and the GEMM autotuner is in `cuda/gemm.cu`.

## Model format: F16

d1-omni is a decision model. Each question is one bidirectional forward pass and there is no decoding, so every
token is prefill and the work is limited by tensor-core throughput, not memory bandwidth. Quantized weights don't
make that faster: they still have to be expanded to fp16 for the tensor cores. They only cost accuracy. Measured
against the fp32 PyTorch reference on 364 test questions:

| weights | mean \|Δlogit\| | max \|Δlogit\| | top answer flipped |
|---|---:|---:|---:|
| **F16 (this engine)** | **0.0052** | 0.044 | 0 |
| F16, PyTorch fp16 (reference code) | 0.0092 | 0.052 | 0 |
| Q8_0 (this engine) | 0.077 | 0.92 | 2 |

BF16 isn't an option either: Turing has no bf16 tensor cores, and the model card says bf16 changes answers. Use
`d1-omni-600M-F16.gguf` and `mmproj-d1-omni-600M-F16.gguf`. The other files load, but they're slower or less exact.

## Build

```bash
cargo build --release                 # nvcc -arch=native; override with D1_CUDA_ARCH=sm_75,sm_86
```

You need an NVIDIA GPU with compute capability 7.5 or newer (Turing, Ampere, Ada, ...) and the CUDA 12 toolkit.

## Run

```bash
./target/release/d1 serve --port 8080
```

That's all you need: with no `-m`, the server downloads `LiquidAI/d1-omni-600M-GGUF` (F16 model plus its F16
mmproj, about 1.2 GB) on first use. Other ways to pick the model:

```bash
d1 serve --hf LiquidAI/d1-omni-600M-GGUF:F16          # repo[:quant][@revision]; quant picks <name>-<QUANT>.gguf
d1 serve --hf LiquidAI/d1-omni-600M-GGUF --no-mmproj  # text only
d1 serve -m d1-omni-600M-F16.gguf --mmproj mmproj-d1-omni-600M-F16.gguf   # local files, no network
d1 download                                           # fetch into the cache and print the paths
```

How downloads work:

- There's no HTTP/TLS client in the code, so downloads use the system `curl` (or `wget` if curl is missing).
- Files are cached in `$D1_CACHE`, which defaults to `~/.cache/d1rs/models`.
- An interrupted download resumes where it stopped.
- Every file is checked against the SHA-256 that the Hub publishes for it. A file with the wrong size or hash is
  fetched again.
- Later starts reuse the cached files without the network. If the Hub can't be reached, the cached files are used.
- `HF_TOKEN` is sent for gated or private repos, and `HF_ENDPOINT` selects a mirror.

The first start autotunes the GEMMs (text, vision and audio shapes) for about three minutes. The result is cached in `~/.cache/d1rs/` and later starts
take about a second. The request and response format matches the model card:

```bash
curl http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund":  {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                             "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]}
  }
}'
```

- Images go in `"images": ["data:image/jpeg;base64,..."]` and audio in `"files": ["data:audio/mpeg;base64,..."]`.
  Clips are cut at 30 s. See [Media formats](#media-formats) for what is accepted.
- `POST /v1/systemone/batch` takes `{"requests": [...]}` and returns `{"results": [...]}`.
- `GET /health` reports status and `GET /metrics` reports engine counters.

Other commands:

```
d1 run      -m M.gguf [--mmproj P.gguf] REQUEST.json   run one request (local file paths are allowed for media)
d1 eval     -m M.gguf [--mmproj P.gguf] CASES.jsonl    per-question probabilities, one JSON line per case
d1 bench    -m M.gguf [--clients 64] [--seconds 10]    in-process latency and throughput
d1 loadtest --url http://127.0.0.1:8080 [--clients 64] [--request REQ.json]
```

Options: `--max-batch-tokens` (default 8192), `--ctx` (longest question, default 16384), `--media-cache-mb`,
`--convert`, `--fast`, `--no-tune`, `--threads`, `--device`, `-v`.

`loadtest --request a.json,b.json,c.json` cycles each client through the files, so the server sees a mix of request
types, and reports latency per file.

## Media formats

The server identifies each media blob by its content (magic bytes), not by the field it arrived in or its MIME type.
An MP3 sent in `images`, or a JPEG sent in `files`, still goes to the right encoder.

| | decoded natively | converted with ffmpeg (`--convert auto`) |
|---|---|---|
| images | JPEG (baseline, progressive), PNG (all color types, 1–16 bit, interlaced), BMP, PPM | WebP, GIF (first frame), TIFF, HEIC/HEIF, AVIF, JPEG 2000, ICO, QOI, CMYK or 12-bit JPEG, anything else ffmpeg reads |
| audio | WAV: PCM 8/16/24/32-bit or float, any sample rate or channel count (resampled to 16 kHz mono) | MP3, AAC/M4A, OGG Vorbis, Opus, FLAC, WebM/MKV, AIFF, AMR, WMA, CAF, the audio track of MP4/AVI video, ADPCM or μ-law WAV |

Conversion is optional:

- `--convert auto` (the default) uses `ffmpeg` from `PATH` if it is there.
- `--convert /path/to/ffmpeg` uses a specific binary.
- `--convert off` rejects non-native formats with a 400 error that says which format was found.

Images are converted to PPM and audio to 16 kHz mono 16-bit PCM, then they go through the same path as native files.
ffmpeg runs on untrusted input, so it may only read its own temporary input file (`-protocol_whitelist file`). It
also gets no stdin, a 30 s timeout and a cap on output size. Conversion adds about 40–80 ms per request (process
start plus decode). It runs on the HTTP thread, so it never stalls the GPU.

## How it works

**Packed variable-length batches.** Every question becomes one sequence:
`<bos><state>… <q>… <opt><mask>… </opt> … <decide>`. The sequences of all in-flight requests are concatenated
into one token stream with no padding. GEMMs run once over the whole stream. The attention kernel and the 3-tap
convolution use per-sequence metadata so sequences never see each other.

**Continuous batching.** One engine thread owns the GPU. HTTP threads do the CPU work (JSON, tokenizing, image and
audio decoding) in parallel and submit jobs. On each iteration the engine:

1. takes in new jobs;
2. encodes one group of pending media;
3. packs question rows FIFO across requests until the token budget is reached;
4. launches the batch.

Up to two batches are in flight, so building batch N+1 on the CPU overlaps with computing batch N on the GPU. Rows
of one request can be split across batches. A request returns as soon as its last row finishes.

**Media prefix caching.** In this model, image and audio positions attend only to each other and the convolution
never reads text into them, so the media prefix doesn't depend on the question. The engine runs it through the
trunk once per request and keeps every attention layer's K/V and the last conv state on the GPU. Each question then
computes only its own text tokens and attends to the cached prefix keys. For a 3-question image request this skips
recomputing 2×234 prefix tokens through the whole trunk. Images from several requests share one SigLIP2 pass, and
their prefixes share one trunk pass.

**Kernels.**
- Flash attention for head size 64 uses `mma.m16n8k8` and `ldmatrix` (works on sm_75 and up). It handles
  variable-length sequences, GQA and the cached-prefix keys, with fp32 softmax.
- The short convolution reads the in-projection output, computes `B·x`, the 3-tap filter and the `C·` gate in one
  kernel, and saves or loads the conv state at the media boundary.
- QK RMSNorm, RoPE and the K/V save to the prefix cache are one kernel.
- The residual stream stays in fp32. Output and down projections accumulate into it directly through the GEMM
  (beta=1). Biases are folded into the next LayerNorm.
- In the last decision-head layer, queries, the output projection and the FFN run only on the option-marker rows.

**GEMM autotuning.** cuBLAS heuristics are weak at the small row counts typical of latency-bound batches. At 154
rows the gate/up projection reached 13.8 TFLOPS with the default choice. For every (shape, M-bucket), `gemm.cu` times
the cuBLASLt algorithm × tile × split-K × swizzle space and caches the fastest. On the same GEMM that gives 1.3–1.5×
at small M.

**Numerics.** Kernels are built without `--use_fast_math`. The approximate `tanh`/`exp` it enables put a systematic
bias into GELU in the vision tower that showed up in image answers. Weights and activations are fp16, all
accumulation is fp32, and norms, softmax and the residual stream are fp32. `--fast` switches the fp16-output GEMMs to
fp16 accumulation. That runs about 1.3× faster and roughly triples the error (mean |Δlogit| 0.016 on text).

## Performance

Measured on an RTX 2060 SUPER 8 GB (Turing, 34 SMs, about 30 TFLOPS fp16 with fp32 accumulation). Another user's
jobs were using most of the CPU during these runs, which inflates the CPU-bound image and audio numbers.

| workload | result |
|---|---|
| text, 3 questions (154 tokens), 1 client over HTTP | **6.2 ms** p50 |
| text, 3 questions, 16 clients | 237 req/s, 67 ms p50, 36.5K tokens/s |
| text, 3 questions, 64 clients | ~250 req/s, ~39K tokens/s |
| text, `--fast`, 256 clients | 317 req/s, 49K tokens/s; single request 5.4 ms |
| image (640×480 JPEG) + 3 questions, 1 client | 42 ms p50 |
| image + 3 questions, 32 clients | 55 req/s |
| audio (11 s WAV) + 2 questions, 1 client | 39 ms p50 |
| 2 questions over a 16K-token state | 1.6 s |
| mixed: text, JSON, JPEG, WAV, WebP→ffmpeg, MP3→ffmpeg; 6 clients | 74 req/s; text p50 34 ms, image/voice 75 ms, converted 120–140 ms |
| mixed, 24 clients | 81 req/s (GPU at 100%); text p50 148 ms, media 340–380 ms |

Text batches are launched before media work on every engine iteration, and media are processed in small groups, so
text requests don't queue behind a burst of images. A media request costs about 15–20 ms of GPU time (SigLIP2 over
~900 patches plus one trunk pass for the prefix), so mixed traffic is GPU-bound at around 80 req/s on this card.

At steady state about 88% of GPU time is GEMMs, running at 80–90% of the fp32-accumulation tensor peak. Larger
batches can't add much, so `--fast` is the remaining throughput option.

## GPU memory

Measured with `nvidia-smi` as this process's usage (the CUDA context is included):

| configuration | idle | peak under load |
|---|---:|---:|
| text only, `--ctx 2048` | 0.99 GB | same |
| text only, `--ctx 4096` | 1.08 GB | same |
| text only, `--ctx 8192` | 1.26 GB | same |
| text only, default `--ctx 16384` | 1.61 GB | same (64 clients, or 16K-token states) |
| vision + audio, `--ctx 4096 --vision-patches 4096 --media-cache-mb 512` | 1.75 GB | 1.92 GB (mixed load) |
| vision + audio, defaults | 2.39 GB | 2.42 GB (640×480 images), 2.45 GB (30 s audio), 2.49 GB (13-crop tiled images) |
| vision + audio, defaults, everything mixed incl. 16K states and tiled images, 48 clients | 2.39 GB | 2.59–2.81 GB |

Where it goes:

- **Weights:** about 0.73 GB for the text model and 0.6 GB for the vision and audio encoders.
- **Text workspace:** about 44 KB per token of `--ctx`, so 0.7 GB at 16384. It is allocated once at startup, so text
  traffic never grows memory. `--max-batch-tokens` doesn't change memory; it is clamped to `--ctx`.
- **Vision scratch:** about 150 MB at the default `--vision-patches 11264`, which fits one fully tiled image (eleven
  512 px crops) in a single tower pass. Lower values still accept any image; the crops are processed in several
  passes.
- **Audio scratch:** about 170 MB, sized for a 30 s clip.
- **Media in flight:** each request with an image or audio keeps its cached prefix (K/V of the 6 attention layers)
  until its questions finish, at 12 KB per prefix token:
  - 2.9 MB for a 640×480 photo (234 tokens);
  - 4.6 MB for 30 s of audio (375 tokens);
  - 35 MB for a fully tiled image (2,816 tokens).

  These caches come from a stream-ordered pool capped by `--media-cache-mb` (default 1536). When the cap is reached,
  later media requests wait their turn.

For small GPUs, `--ctx` is the main lever. Use `--no-mmproj` (with `--hf`) or omit `--mmproj` (with `-m`) for text
only. Note that `--ctx` also caps the longest question; longer states are truncated, as in the reference
implementation.

## Scheduling

Rows are packed FIFO across requests. Two rules keep latency fair under mixed load:

- Text batches go to the GPU before media work on every iteration.
- A long question (more than half of `--max-batch-tokens`, such as a 16K-token state) fills a whole batch by itself.
  While short work is waiting, long rows get at most about a quarter of the recent GPU cost.

With clients sending 2 × 16K-token questions continuously alongside normal traffic, short text requests stay under
about 1 s; their worst case is waiting for one long batch already on the GPU. Without this rule they waited 3–12 s.

## Accuracy

Compared against the original PyTorch model (`trust_remote_code`, fp32) on 69 text cases, 14 image cases and 5 audio
cases, 364 questions in total. The cases cover multilingual text, JSON states, prompt-injection-like special tokens,
11+ options, truncation and 16K-token states. The image cases cover baseline, progressive, 4:2:2 and grayscale JPEG,
EXIF rotation, tiled large images, and RGB, RGBA, palette and grayscale PNG. The audio cases cover short (padded)
and long (cut) clips.

- 0 changed top answers and identical `input_tokens` everywhere; mean |Δp| 0.0007, max 0.013.
- The tokenizer matches Hugging Face `tokenizers` on 3,500 fuzzed strings: added tokens, contractions, Unicode
  classes and whitespace rules.
- JPEG and PNG decoding is bit-exact with PIL, and the resize matches torchvision to within fp16 rounding.

## Tests

```bash
cargo test --release                                # unit tests (JSON, prompt escaping, media sniffing, SHA-256)
python3 tests/e2e.py --url http://127.0.0.1:8080    # end to end against a running server
```

`tests/e2e.py` needs only the Python standard library and ffmpeg. It downloads the model card's sample photo and voice
clip, uses ffmpeg to make WebP/GIF/TIFF/BMP/AVIF/JPEG 2000 and MP3/OGG/Opus/FLAC/M4A/WebM/AIFF/44.1 kHz-stereo
variants, and runs 25 cases:

- text, JSON and multilingual states;
- null states;
- two images in one request;
- mislabeled fields and MIME types;
- every format variant;
- five error cases.

It then fires every case 8 times concurrently, 200 mixed requests in total, and checks that the answers match the
sequential ones (max |Δp| 0.0065).

Lossless conversions (PNG, TIFF, BMP, FLAC, AIFF) give the same answers as the originals. Lossy re-encodes (WebP,
GIF, MP3) shift the probabilities a little, because the test files themselves lost information.

`tools/` holds the scripts used for the accuracy numbers above. They read `D1_HF_DIR` (a download of
[LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) with `model.safetensors`), `D1_BIN` and
`D1_MODEL`: `ref_eval.py` runs the original PyTorch model,
`compare.py` and `logit_err.py` diff the results, `tok_fuzz.py` fuzzes the tokenizer, and `cmp_patches.py` checks
image preprocessing.

## Limitations

- Media must be sent inline. Remote URLs aren't fetched, because that would need an HTTP/TLS client.
- One GPU per process. Run one process per GPU behind a load balancer.
