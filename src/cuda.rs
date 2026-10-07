//! Minimal CUDA runtime + cuBLAS bindings and the kernel launchers from cuda/kernels.cu.
#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int};
use std::ptr;

pub type Stream = *mut c_void;
pub type Event = *mut c_void;
pub type Cublas = *mut c_void;

#[link(name = "cudart")]
extern "C" {
    fn cudaSetDevice(d: c_int) -> c_int;
    fn cudaMalloc(p: *mut *mut c_void, n: usize) -> c_int;
    fn cudaFree(p: *mut c_void) -> c_int;
    fn cudaMallocHost(p: *mut *mut c_void, n: usize) -> c_int;
    fn cudaFreeHost(p: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int, s: Stream) -> c_int;
    fn cudaMemsetAsync(dst: *mut c_void, v: c_int, n: usize, s: Stream) -> c_int;
    fn cudaStreamCreateWithFlags(s: *mut Stream, flags: u32) -> c_int;
    fn cudaStreamSynchronize(s: Stream) -> c_int;
    fn cudaEventCreateWithFlags(e: *mut Event, flags: u32) -> c_int;
    fn cudaEventRecord(e: Event, s: Stream) -> c_int;
    fn cudaEventSynchronize(e: Event) -> c_int;
    fn cudaEventQuery(e: Event) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, a: Event, b: Event) -> c_int;
    fn cudaEventDestroy(e: Event) -> c_int;
    fn cudaGetErrorString(e: c_int) -> *const c_char;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMallocAsync(p: *mut *mut c_void, n: usize, s: Stream) -> c_int;
    fn cudaFreeAsync(p: *mut c_void, s: Stream) -> c_int;
    fn cudaGetDeviceCount(n: *mut c_int) -> c_int;
}

#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(h: *mut Cublas) -> c_int;
    fn cublasSetStream_v2(h: Cublas, s: Stream) -> c_int;
    fn cublasSetMathMode(h: Cublas, mode: c_int) -> c_int;
    fn cublasSetWorkspace_v2(h: Cublas, ws: *mut c_void, bytes: usize) -> c_int;
    fn cublasGemmEx(
        h: Cublas, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const c_void, a: *const c_void,
        at: c_int, lda: c_int, b: *const c_void, bt: c_int, ldb: c_int, beta: *const c_void, c: *mut c_void, ct: c_int,
        ldc: c_int, compute: c_int, algo: c_int,
    ) -> c_int;
    fn cublasGemmStridedBatchedEx(
        h: Cublas, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const c_void, a: *const c_void,
        at: c_int, lda: c_int, sa: i64, b: *const c_void, bt: c_int, ldb: c_int, sb: i64, beta: *const c_void,
        c: *mut c_void, ct: c_int, ldc: c_int, sc: i64, batch: c_int, compute: c_int, algo: c_int,
    ) -> c_int;
}

pub type F16 = u16;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SeqInfo {
    pub start: i32,
    pub len: i32,
    pub pos_off: i32,
    pub qtype: i32,
    pub ext_bx: *const F16,
    pub save_bx: *mut F16,
    pub save_k: *mut F16,
    pub save_v: *mut F16,
}
unsafe impl Send for SeqInfo {}

impl Default for SeqInfo {
    fn default() -> Self {
        SeqInfo {
            start: 0,
            len: 0,
            pos_off: 0,
            qtype: 0,
            ext_bx: ptr::null(),
            save_bx: ptr::null_mut(),
            save_k: ptr::null_mut(),
            save_v: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnSeq {
    pub q_start: i32,
    pub q_len: i32,
    pub kv_start: i32,
    pub kv_len: i32,
    pub ext_len: i32,
    pub _pad: i32,
    pub ext_k: *const F16,
    pub ext_v: *const F16,
}
unsafe impl Send for AttnSeq {}

pub mod k {
    use super::*;
    extern "C" {
        pub fn d1_rmsnorm(x: *const f32, w: *const f32, y: *mut F16, rows: c_int, d: c_int, eps: f32, s: Stream) -> c_int;
        pub fn d1_layernorm(
            x: *mut f32, pend_bias: *const f32, pend_scale: f32, w: *const f32, b: *const f32, y: *mut F16, yf: *mut f32,
            rows: c_int, d: c_int, eps: f32, s: Stream,
        ) -> c_int;
        pub fn d1_head_init(
            h: *const f32, w: *const f32, type_emb: *const f32, tok_seq: *const i32, seqs: *const SeqInfo, out: *mut f32,
            rows: c_int, eps: f32, s: Stream,
        ) -> c_int;
        pub fn d1_embed(table: *const F16, ids: *const i32, h: *mut f32, rows: c_int, d: c_int, s: Stream) -> c_int;
        pub fn d1_swiglu(gu: *const F16, out: *mut F16, rows: c_int, f: c_int, s: Stream) -> c_int;
        pub fn d1_bias_act_h(x: *mut F16, bias: *const f32, rows: c_int, n: c_int, act: c_int, s: Stream) -> c_int;
        pub fn d1_bias_act_f(x: *mut f32, bias: *const f32, rows: c_int, n: c_int, act: c_int, s: Stream) -> c_int;
        pub fn d1_fill_rows(out: *mut f32, bias: *const f32, add: *const f32, rows: c_int, n: c_int, s: Stream) -> c_int;
        pub fn d1_f32_to_f16(x: *const f32, y: *mut F16, n: usize, s: Stream) -> c_int;
        pub fn d1_gather_rows(
            src: *const c_void, dst: *mut c_void, idx: *const i32, n: c_int, row_bytes: c_int, s: Stream,
        ) -> c_int;
        pub fn d1_shortconv(
            bcu: *const F16, convw: *const f32, tok_seq: *const i32, seqs: *const SeqInfo, out: *mut F16, rows: c_int,
            d: c_int, conv_idx: c_int, s: Stream,
        ) -> c_int;
        pub fn d1_qk_norm_rope(
            qkv: *mut F16, ld: c_int, h: c_int, kvh: c_int, qn: *const f32, kn: *const f32, rope: *const f32,
            tok_seq: *const i32, seqs: *const SeqInfo, attn_idx: c_int, rows: c_int, eps: f32, s: Stream,
        ) -> c_int;
        pub fn d1_flash_attn(
            q: *const F16, ldq: c_int, k: *const F16, v: *const F16, ldkv: c_int, ext_ld: c_int, bq: *const f32,
            bk: *const f32, bv: *const f32, o: *mut F16, ldo: c_int, seqs: *const AttnSeq, work: *const i32,
            n_work: c_int, heads: c_int, group: c_int, scale: f32, s: Stream,
        ) -> c_int;
        pub fn d1_scorer_out(
            x: *const F16, b1: *const f32, w2: *const F16, b2: f32, logits: *mut f32, rows: c_int, d: c_int, s: Stream,
        ) -> c_int;
        pub fn d1_unshuffle(x: *const F16, out: *mut F16, ph: c_int, pw: c_int, c: c_int, s: Stream) -> c_int;
        pub fn d1_sub_conv0(
            mel: *const f32, tin_valid: c_int, f: c_int, w: *const f32, b: *const f32, out: *mut f32, tout: c_int,
            fout: c_int, s: Stream,
        ) -> c_int;
        pub fn d1_sub_dwconv(
            x: *const f32, tin_valid: c_int, fin: c_int, w: *const f32, b: *const f32, out: *mut f32, tout: c_int,
            fout: c_int, s: Stream,
        ) -> c_int;
        pub fn d1_relpos_prep(
            qkv: *const F16, bq: *const f32, bk: *const f32, bv: *const f32, pu: *const f32, pv: *const f32,
            qu: *mut F16, qv: *mut F16, k: *mut F16, v: *mut F16, t: c_int, dm: c_int, s: Stream,
        ) -> c_int;
        pub fn d1_relpos_softmax(ac: *const f32, bd: *const f32, p: *mut F16, t: c_int, h: c_int, scale: f32, s: Stream)
            -> c_int;
        pub fn d1_conformer_dw(
            a: *const F16, b1: *const f32, dw: *const f32, dwb: *const f32, bn_s: *const f32, bn_b: *const f32,
            out: *mut F16, t: c_int, c: c_int, ksize: c_int, s: Stream,
        ) -> c_int;
    }
}

extern "C" {
    fn d1_lt_init(ws_bytes: usize) -> *mut c_void;
    fn d1_lt_gemm(
        ctx: *mut c_void, m: c_int, n: c_int, k: c_int, x: *const c_void, ldx: c_int, w: *const c_void, ldw: c_int,
        w_kn: c_int, y: *mut c_void, ldy: c_int, y_f32: c_int, f16acc: c_int, alpha: f32, beta: f32, s: Stream,
    ) -> c_int;
    fn d1_lt_tune(
        ctx: *mut c_void, m: c_int, n: c_int, k: c_int, ldx: c_int, ldw: c_int, w_kn: c_int, ldy: c_int, y_f32: c_int,
        f16acc: c_int, effort: c_int, s: Stream,
    ) -> f32;
    fn d1_lt_save(ctx: *mut c_void, path: *const c_char) -> c_int;
    fn d1_lt_load(ctx: *mut c_void, path: *const c_char) -> c_int;
    pub fn d1_lt_bucket(m: c_int) -> c_int;
}

pub fn err_str(e: c_int) -> String {
    unsafe { CStr::from_ptr(cudaGetErrorString(e)).to_string_lossy().into_owned() }
}

#[track_caller]
pub fn check(e: c_int) {
    if e != 0 {
        panic!("CUDA error {e}: {}", err_str(e));
    }
}

#[track_caller]
pub fn check_blas(e: c_int) {
    if e != 0 {
        panic!("cuBLAS error {e}");
    }
}

/// Owned device allocation.
pub struct DevBuf {
    pub ptr: *mut c_void,
    pub bytes: usize,
}
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl DevBuf {
    pub fn new(bytes: usize) -> DevBuf {
        let mut p = ptr::null_mut();
        if bytes > 0 {
            let e = unsafe { cudaMalloc(&mut p, bytes) };
            if e != 0 {
                panic!("cudaMalloc({} MiB) failed: {}", bytes >> 20, err_str(e));
            }
        }
        DevBuf { ptr: p, bytes }
    }
    pub fn from_slice<T: Copy>(data: &[T]) -> DevBuf {
        let bytes = std::mem::size_of_val(data);
        let b = DevBuf::new(bytes);
        if bytes > 0 {
            check(unsafe { cudaMemcpy(b.ptr, data.as_ptr() as *const c_void, bytes, 1) });
            // a pageable->device cudaMemcpy may return before the DMA lands, and the engine's non-blocking
            // stream does not order against the legacy stream: wait for it.
            check(unsafe { cudaStreamSynchronize(ptr::null_mut()) });
        }
        b
    }
    #[inline]
    pub fn f16(&self) -> *mut F16 {
        self.ptr as *mut F16
    }
    #[inline]
    pub fn f32(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
    #[inline]
    pub fn i32(&self) -> *mut i32 {
        self.ptr as *mut i32
    }
    pub fn to_host<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        check(unsafe { cudaMemcpy(v.as_mut_ptr() as *mut c_void, self.ptr, n * std::mem::size_of::<T>(), 2) });
        v
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr) };
        }
    }
}

/// Pinned host buffer (for async copies).
pub struct HostBuf {
    pub ptr: *mut u8,
    pub bytes: usize,
}
unsafe impl Send for HostBuf {}

impl HostBuf {
    pub fn new(bytes: usize) -> HostBuf {
        let mut p = ptr::null_mut();
        check(unsafe { cudaMallocHost(&mut p, bytes.max(16)) });
        HostBuf { ptr: p as *mut u8, bytes }
    }
}
impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe { cudaFreeHost(self.ptr as *mut c_void) };
    }
}

pub fn set_device(d: i32) {
    check(unsafe { cudaSetDevice(d) });
}
pub fn device_count() -> i32 {
    let mut n = 0;
    unsafe { cudaGetDeviceCount(&mut n) };
    n
}
pub fn mem_info() -> (usize, usize) {
    let (mut f, mut t) = (0, 0);
    check(unsafe { cudaMemGetInfo(&mut f, &mut t) });
    (f, t)
}
pub fn sm_count(dev: i32) -> i32 {
    let mut v = 0;
    unsafe { cudaDeviceGetAttribute(&mut v, 16, dev) };
    v
}
pub fn compute_capability(dev: i32) -> (i32, i32) {
    let (mut a, mut b) = (0, 0);
    unsafe {
        cudaDeviceGetAttribute(&mut a, 75, dev);
        cudaDeviceGetAttribute(&mut b, 76, dev);
    }
    (a, b)
}
pub fn device_sync() {
    check(unsafe { cudaDeviceSynchronize() });
}

pub fn new_stream() -> Stream {
    let mut s = ptr::null_mut();
    check(unsafe { cudaStreamCreateWithFlags(&mut s, 1) });
    s
}
pub fn stream_sync(s: Stream) {
    check(unsafe { cudaStreamSynchronize(s) });
}
pub fn new_event(timing: bool) -> Event {
    let mut e = ptr::null_mut();
    // 0x02 = cudaEventDisableTiming
    check(unsafe { cudaEventCreateWithFlags(&mut e, if timing { 0 } else { 2 }) });
    e
}
pub fn event_record(e: Event, s: Stream) {
    check(unsafe { cudaEventRecord(e, s) });
}
pub fn event_sync(e: Event) {
    check(unsafe { cudaEventSynchronize(e) });
}
pub fn event_elapsed(a: Event, b: Event) -> f32 {
    let mut ms = 0.0;
    check(unsafe { cudaEventElapsedTime(&mut ms, a, b) });
    ms
}

pub unsafe fn h2d_async(dst: *mut c_void, src: *const c_void, n: usize, s: Stream) {
    if n > 0 {
        check(cudaMemcpyAsync(dst, src, n, 1, s));
    }
}
pub unsafe fn d2h_async(dst: *mut c_void, src: *const c_void, n: usize, s: Stream) {
    if n > 0 {
        check(cudaMemcpyAsync(dst, src, n, 2, s));
    }
}
pub unsafe fn d2d_async(dst: *mut c_void, src: *const c_void, n: usize, s: Stream) {
    if n > 0 {
        check(cudaMemcpyAsync(dst, src, n, 3, s));
    }
}
pub unsafe fn memset_async(dst: *mut c_void, n: usize, s: Stream) {
    if n > 0 {
        check(cudaMemsetAsync(dst, 0, n, s));
    }
}
pub unsafe fn malloc_async(n: usize, s: Stream) -> *mut c_void {
    let mut p = ptr::null_mut();
    let e = cudaMallocAsync(&mut p, n.max(256), s);
    if e != 0 {
        panic!("cudaMallocAsync({} KiB) failed: {}", n >> 10, err_str(e));
    }
    p
}
pub unsafe fn free_async(p: *mut c_void, s: Stream) {
    if !p.is_null() {
        check(cudaFreeAsync(p, s));
    }
}

const CUDA_R_16F: c_int = 2;
const CUDA_R_32F: c_int = 0;
const COMPUTE_16F: c_int = 64;
const COMPUTE_32F: c_int = 68;
const ALGO_DEFAULT_TENSOR_OP: c_int = 99;

/// A GEMM shape as the tuner keys it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub ldx: usize,
    pub ldw: usize,
    pub w_kn: bool,
    pub ldy: usize,
    pub y_f32: bool,
    pub f16acc: bool,
}

pub struct Blas {
    pub h: Cublas,
    /// Accumulate fp16-output GEMMs in fp16 (2x tensor throughput on consumer Turing/Ampere cards).
    pub fp16_acc: bool,
    lt: *mut c_void,
    stream: Stream,
    /// When set, GEMM shapes are recorded (for tuning) in addition to being executed.
    pub record: std::cell::RefCell<Option<Vec<Shape>>>,
}
unsafe impl Send for Blas {}

/// Output of a GEMM: fp16 or fp32 matrix.
#[derive(Clone, Copy)]
pub enum Out {
    H(*mut F16),
    F(*mut f32),
}

impl Blas {
    pub fn new(stream: Stream) -> Blas {
        let mut h = ptr::null_mut();
        check_blas(unsafe { cublasCreate_v2(&mut h) });
        check_blas(unsafe { cublasSetStream_v2(h, stream) });
        // CUBLAS_TF32_TENSOR_OP_MATH(3) is irrelevant for fp16; default math allows tensor ops for fp16.
        let lt = unsafe { d1_lt_init(32 << 20) };
        Blas { h, fp16_acc: false, lt, stream, record: std::cell::RefCell::new(None) }
    }

    pub fn tune(&self, s: &Shape, effort: i32, st: Stream) -> f32 {
        unsafe {
            d1_lt_tune(
                self.lt, s.m as c_int, s.n as c_int, s.k as c_int, s.ldx as c_int, s.ldw as c_int, s.w_kn as c_int,
                s.ldy as c_int, s.y_f32 as c_int, s.f16acc as c_int, effort, st,
            )
        }
    }
    pub fn save_plans(&self, path: &str) -> bool {
        let c = std::ffi::CString::new(path).unwrap();
        unsafe { d1_lt_save(self.lt, c.as_ptr()) == 0 }
    }
    pub fn load_plans(&self, path: &str) -> i32 {
        let c = std::ffi::CString::new(path).unwrap();
        unsafe { d1_lt_load(self.lt, c.as_ptr()) }
    }

    /// Row-major `Y[M,N] = alpha * X[M,K] · W[N,K]^T + beta * Y` (W in nn.Linear layout).
    /// With `w_kn`, W is row-major [K,N] instead: `Y = X · W`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self, m: usize, n: usize, k: usize, x: *const F16, ldx: usize, w: *const F16, ldw: usize, w_kn: bool, y: Out,
        ldy: usize, alpha: f32, beta: f32,
    ) {
        if m == 0 || n == 0 {
            return;
        }
        let (yp, y_f32) = match y {
            Out::H(p) => (p as *mut c_void, false),
            Out::F(p) => (p as *mut c_void, true),
        };
        let f16acc = self.fp16_acc && !y_f32;
        if let Some(r) = self.record.borrow_mut().as_mut() {
            r.push(Shape { m, n, k, ldx, ldw, w_kn, ldy, y_f32, f16acc });
        }
        if !self.lt.is_null() {
            let e = unsafe {
                d1_lt_gemm(
                    self.lt, m as c_int, n as c_int, k as c_int, x as *const c_void, ldx as c_int, w as *const c_void,
                    ldw as c_int, w_kn as c_int, yp, ldy as c_int, y_f32 as c_int, f16acc as c_int, alpha, beta, self_st(self),
                )
            };
            if e == 0 {
                return;
            }
            if std::env::var("D1_DEBUG_GEMM").is_ok() {
                eprintln!("lt fallback {e}: m={m} n={n} k={k} ldx={ldx} ldw={ldw} ldy={ldy} f32={y_f32}");
            }
        }
        let ta = if w_kn { 0 } else { 1 };
        unsafe {
            let e = match y {
                Out::H(p) if self.fp16_acc => {
                    let a = f32_to_f16(alpha);
                    let b = f32_to_f16(beta);
                    cublasGemmEx(
                        self.h, ta, 0, n as c_int, m as c_int, k as c_int, &a as *const u16 as *const c_void,
                        w as *const c_void, CUDA_R_16F, ldw as c_int, x as *const c_void, CUDA_R_16F, ldx as c_int,
                        &b as *const u16 as *const c_void, p as *mut c_void, CUDA_R_16F, ldy as c_int, COMPUTE_16F,
                        ALGO_DEFAULT_TENSOR_OP,
                    )
                }
                Out::H(p) => cublasGemmEx(
                    self.h, ta, 0, n as c_int, m as c_int, k as c_int, &alpha as *const f32 as *const c_void,
                    w as *const c_void, CUDA_R_16F, ldw as c_int, x as *const c_void, CUDA_R_16F, ldx as c_int,
                    &beta as *const f32 as *const c_void, p as *mut c_void, CUDA_R_16F, ldy as c_int, COMPUTE_32F,
                    ALGO_DEFAULT_TENSOR_OP,
                ),
                Out::F(p) => cublasGemmEx(
                    self.h, ta, 0, n as c_int, m as c_int, k as c_int, &alpha as *const f32 as *const c_void,
                    w as *const c_void, CUDA_R_16F, ldw as c_int, x as *const c_void, CUDA_R_16F, ldx as c_int,
                    &beta as *const f32 as *const c_void, p as *mut c_void, CUDA_R_32F, ldy as c_int, COMPUTE_32F,
                    ALGO_DEFAULT_TENSOR_OP,
                ),
            };
            check_blas(e);
        }
    }

    /// fp32 GEMM: row-major `Y[M,N] = X[M,K] · W[N,K]^T + beta*Y`.
    pub fn sgemm(&self, m: usize, n: usize, k: usize, x: *const f32, ldx: usize, w: *const f32, ldw: usize, y: *mut f32, ldy: usize, beta: f32) {
        if m == 0 || n == 0 {
            return;
        }
        let alpha = 1.0f32;
        unsafe {
            check_blas(cublasGemmEx(
                self.h, 1, 0, n as c_int, m as c_int, k as c_int, &alpha as *const f32 as *const c_void,
                w as *const c_void, CUDA_R_32F, ldw as c_int, x as *const c_void, CUDA_R_32F, ldx as c_int,
                &beta as *const f32 as *const c_void, y as *mut c_void, CUDA_R_32F, ldy as c_int, COMPUTE_32F, -1,
            ));
        }
    }

    /// Batched row-major `Y_b[M,N] = X_b[M,K] · W_b^T` (or `X_b · W_b` with `w_kn`), fp32 accumulation.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched(
        &self, batch: usize, m: usize, n: usize, k: usize, x: *const F16, ldx: usize, sx: usize, w: *const F16,
        ldw: usize, sw: usize, w_kn: bool, y: Out, ldy: usize, sy: usize, alpha: f32,
    ) {
        let ta = if w_kn { 0 } else { 1 };
        let beta = 0.0f32;
        let (yp, yt) = match y {
            Out::H(p) => (p as *mut c_void, CUDA_R_16F),
            Out::F(p) => (p as *mut c_void, CUDA_R_32F),
        };
        unsafe {
            check_blas(cublasGemmStridedBatchedEx(
                self.h, ta, 0, n as c_int, m as c_int, k as c_int, &alpha as *const f32 as *const c_void,
                w as *const c_void, CUDA_R_16F, ldw as c_int, sw as i64, x as *const c_void, CUDA_R_16F, ldx as c_int,
                sx as i64, &beta as *const f32 as *const c_void, yp, yt, ldy as c_int, sy as i64, batch as c_int,
                COMPUTE_32F, ALGO_DEFAULT_TENSOR_OP,
            ));
        }
    }
}

// ------------------------------------------------------------------------------------------------ fp16 helpers

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rest > half || (rest == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 113u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn self_st(b: &Blas) -> Stream {
    b.stream
}

pub fn to_f16_vec(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_f16(x)).collect()
}
