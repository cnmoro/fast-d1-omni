// d1-omni CUDA kernels. Everything is launched through the extern "C" wrappers at the bottom, called from Rust.
//
// Conventions
//   * Residual streams are fp32, row-major [rows, D]. GEMM inputs/outputs are fp16 unless noted.
//   * A packed batch is a concatenation of variable-length sequences with no padding. Per-token metadata
//     (`tok_seq`) maps a row to its sequence; `SeqInfo` describes the sequence.
//   * Head dimension is 64 for every attention in the model (trunk, decision head, SigLIP2, conformer).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu

struct SeqInfo {
    int start;            // first row in the packed buffer
    int len;              // number of rows
    int pos_off;          // RoPE position of the first row (media prefix length for text after a prefix)
    int qtype;            // decision head question type
    const __half* ext_bx; // [n_conv][D] conv state of the prefix's last position (left neighbour), or null
    __half* save_bx;      // prefix pass: where to store this sequence's last-position conv state [n_conv][D]
    __half* save_k;       // prefix pass: [n_attn][len][kv_dim]
    __half* save_v;
};

struct AttnSeq {
    int q_start, q_len;       // query rows
    int kv_start, kv_len;     // local key/value rows
    int ext_len, _pad;        // external (cached prefix) keys, placed before the local ones
    const __half* ext_k;      // [ext_len][ext_ld] already offset to this layer and kv head 0
    const __half* ext_v;
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float apply_act(float x, int act) {
    switch (act) {
        case 1: return fmaxf(x, 0.0f);
        case 2: return gelu_erf(x);
        case 3: return gelu_tanh(x);
        case 4: return silu(x);
        default: return x;
    }
}

// ------------------------------------------------------------------------------------------------ norms
// One warp per row; the row lives in registers (NPL = D / 32 values per lane, D <= 1024).

template <int NPL>
__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, __half* __restrict__ y,
                          int rows, float eps) {
    const int D = NPL * 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * D;
    float v[NPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NPL; i++) {
        v[i] = xr[lane + 32 * i];
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    float r = rsqrtf(ss / D + eps);
    __half* yr = y + (size_t)row * D;
#pragma unroll
    for (int i = 0; i < NPL; i++) yr[lane + 32 * i] = __float2half(v[i] * r * w[lane + 32 * i]);
}

// x += pend_scale * pend_bias (written back), then y = LayerNorm(x) (fp16 and/or fp32 output, fp32 may alias x).
template <int NPL>
__global__ void k_layernorm(float* __restrict__ x, const float* __restrict__ pend_bias, float pend_scale,
                            const float* __restrict__ w, const float* __restrict__ b, __half* __restrict__ y,
                            float* yf, int rows, float eps) {
    const int D = NPL * 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    float* xr = x + (size_t)row * D;
    float v[NPL];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NPL; i++) {
        v[i] = xr[lane + 32 * i];
        if (pend_bias) {
            v[i] += pend_scale * pend_bias[lane + 32 * i];
            xr[lane + 32 * i] = v[i];
        }
        s += v[i];
    }
    float mean = warp_sum(s) / D;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NPL; i++) {
        float d = v[i] - mean;
        ss += d * d;
    }
    float r = rsqrtf(warp_sum(ss) / D + eps);
#pragma unroll
    for (int i = 0; i < NPL; i++) {
        int c = lane + 32 * i;
        float o = (v[i] - mean) * r * w[c] + b[c];
        if (y) y[(size_t)row * D + c] = __float2half(o);
        if (yf) yf[(size_t)row * D + c] = o;
    }
}

// Decision head input: hh = RMSNorm(h) * w + type_emb[qtype]   (fp32 -> fp32)
__global__ void k_head_init(const float* __restrict__ h, const float* __restrict__ w, const float* __restrict__ type_emb,
                            const int* __restrict__ tok_seq, const SeqInfo* __restrict__ seqs, float* __restrict__ out,
                            int rows, float eps) {
    const int NPL = 32, D = 1024;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* xr = h + (size_t)row * D;
    const float* te = type_emb + (size_t)seqs[tok_seq[row]].qtype * D;
    float v[NPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NPL; i++) {
        v[i] = xr[lane + 32 * i];
        ss += v[i] * v[i];
    }
    float r = rsqrtf(warp_sum(ss) / D + eps);
#pragma unroll
    for (int i = 0; i < NPL; i++) {
        int c = lane + 32 * i;
        out[(size_t)row * D + c] = v[i] * r * w[c] + te[c];
    }
}

// ------------------------------------------------------------------------------------------------ elementwise

__global__ void k_embed(const __half* __restrict__ table, const int* __restrict__ ids, float* __restrict__ h, int rows,
                        int D) {
    int row = blockIdx.x;
    const __half* src = table + (size_t)ids[row] * D;
    float* dst = h + (size_t)row * D;
    for (int c = threadIdx.x * 2; c < D; c += blockDim.x * 2) {
        float2 f = __half22float2(*(const __half2*)(src + c));
        dst[c] = f.x;
        dst[c + 1] = f.y;
    }
}

// out[t][i] = silu(gu[t][i]) * gu[t][F + i]
__global__ void k_swiglu(const __half* __restrict__ gu, __half* __restrict__ out, int rows, int F) {
    size_t n = (size_t)rows * (F / 2);
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        size_t r = idx / (F / 2);
        int c = (int)(idx % (F / 2)) * 2;
        float2 g = __half22float2(*(const __half2*)(gu + r * 2 * F + c));
        float2 u = __half22float2(*(const __half2*)(gu + r * 2 * F + F + c));
        *(__half2*)(out + r * F + c) = __floats2half2_rn(silu(g.x) * u.x, silu(g.y) * u.y);
    }
}

// x[t][i] = act(x[t][i] + bias[i])  (fp16 in place)
__global__ void k_bias_act_h(__half* __restrict__ x, const float* __restrict__ bias, int rows, int N, int act) {
    size_t n = (size_t)rows * (N / 2);
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        size_t r = idx / (N / 2);
        int c = (int)(idx % (N / 2)) * 2;
        __half2* p = (__half2*)(x + r * N + c);
        float2 v = __half22float2(*p);
        if (bias) {
            v.x += bias[c];
            v.y += bias[c + 1];
        }
        *p = __floats2half2_rn(apply_act(v.x, act), apply_act(v.y, act));
    }
}

// x[t][i] = act(x[t][i] * scale + bias[i])  (fp32 in place)
__global__ void k_bias_act_f(float* __restrict__ x, const float* __restrict__ bias, int rows, int N, int act) {
    size_t n = (size_t)rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % N);
        float v = x[idx] + (bias ? bias[c] : 0.f);
        x[idx] = apply_act(v, act);
    }
}

// out[t][i] = bias[i] (+ add[t % add_rows][i])  fp32 prefill for beta=1 GEMMs
__global__ void k_fill_rows(float* __restrict__ out, const float* __restrict__ bias, const float* __restrict__ add,
                            int rows, int N) {
    size_t n = (size_t)rows * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % N);
        float v = bias ? bias[c] : 0.f;
        if (add) v += add[idx];
        out[idx] = v;
    }
}

__global__ void k_f32_to_f16(const float* __restrict__ x, __half* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = __float2half(x[i]);
}

// dst[i] = src[idx[i]]  rows of `width` elements of `esize` bytes (width*esize multiple of 16)
__global__ void k_gather_rows(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ idx,
                              int n, int width16) {
    int r = blockIdx.x;
    if (r >= n) return;
    const uint4* s = src + (size_t)idx[r] * width16;
    uint4* d = dst + (size_t)r * width16;
    for (int c = threadIdx.x; c < width16; c += blockDim.x) d[c] = s[c];
}

// ------------------------------------------------------------------------------------------------ trunk ShortConv
// bcu: [T, 3D] = [b | c | u]; out[t] = c[t] * (w0*bx[t-1] + w1*bx[t] + w2*bx[t+1]), bx = b*u, zero outside the
// sequence; the left neighbour of a text sequence after a media prefix is the prefix's cached last bx.
__global__ void k_shortconv(const __half* __restrict__ bcu, const float* __restrict__ convw, const int* __restrict__ tok_seq,
                            const SeqInfo* __restrict__ seqs, __half* __restrict__ out, int D, int conv_idx) {
    int t = blockIdx.x;
    const SeqInfo s = seqs[tok_seq[t]];
    int local = t - s.start;
    bool has_left = local > 0, has_right = local + 1 < s.len;
    const __half* row = bcu + (size_t)t * 3 * D;
    for (int c = threadIdx.x * 2; c < D; c += blockDim.x * 2) {
        float2 b = __half22float2(*(const __half2*)(row + c));
        float2 cc = __half22float2(*(const __half2*)(row + D + c));
        float2 u = __half22float2(*(const __half2*)(row + 2 * D + c));
        float2 bx = make_float2(b.x * u.x, b.y * u.y);
        float2 l = make_float2(0.f, 0.f), r = make_float2(0.f, 0.f);
        if (has_left) {
            const __half* p = row - 3 * D;
            float2 lb = __half22float2(*(const __half2*)(p + c)), lu = __half22float2(*(const __half2*)(p + 2 * D + c));
            l = make_float2(lb.x * lu.x, lb.y * lu.y);
        } else if (s.ext_bx) {
            l = __half22float2(*(const __half2*)(s.ext_bx + (size_t)conv_idx * D + c));
        }
        if (has_right) {
            const __half* p = row + 3 * D;
            float2 rb = __half22float2(*(const __half2*)(p + c)), ru = __half22float2(*(const __half2*)(p + 2 * D + c));
            r = make_float2(rb.x * ru.x, rb.y * ru.y);
        } else if (s.save_bx) {
            *(__half2*)(s.save_bx + (size_t)conv_idx * D + c) = __floats2half2_rn(bx.x, bx.y);
        }
        const float* w0 = convw + c * 3;
        const float* w1 = convw + (c + 1) * 3;
        float y0 = w0[0] * l.x + w0[1] * bx.x + w0[2] * r.x;
        float y1 = w1[0] * l.y + w1[1] * bx.y + w1[2] * r.y;
        *(__half2*)(out + (size_t)t * D + c) = __floats2half2_rn(cc.x * y0, cc.y * y1);
    }
}

// ------------------------------------------------------------------------------------------------ trunk QK norm + RoPE
// qkv row: [q (H*64) | k (KVH*64) | v (KVH*64)], in place. One warp per head-vector, lane handles dims l and l+32.
__global__ void k_qk_norm_rope(__half* __restrict__ qkv, int ld, int H, int KVH, const float* __restrict__ qn,
                               const float* __restrict__ kn, const float2* __restrict__ rope /*[pos][32] (cos,sin)*/,
                               const int* __restrict__ tok_seq, const SeqInfo* __restrict__ seqs, int attn_idx,
                               float eps) {
    int t = blockIdx.x;
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const SeqInfo s = seqs[tok_seq[t]];
    int local = t - s.start;
    int pos = local + s.pos_off;
    float2 cs = rope[(size_t)pos * 32 + lane];
    __half* row = qkv + (size_t)t * ld;
    int units = H + KVH + (s.save_v ? KVH : 0);
    for (int u = warp; u < units; u += blockDim.x / 32) {
        if (u >= H + KVH) {  // v save
            int g = u - H - KVH;
            const __half* src = row + (H + KVH) * 64 + g * 64;
            __half* dst = s.save_v + ((size_t)attn_idx * s.len + local) * (KVH * 64) + g * 64;
            dst[lane] = src[lane];
            dst[lane + 32] = src[lane + 32];
            continue;
        }
        bool isq = u < H;
        __half* p = row + u * 64;  // k heads follow q heads contiguously
        const float* w = isq ? qn : kn;
        float a = __half2float(p[lane]), b = __half2float(p[lane + 32]);
        float r = rsqrtf(warp_sum(a * a + b * b) / 64.f + eps);
        a = a * r * w[lane];
        b = b * r * w[lane + 32];
        float oa = a * cs.x - b * cs.y;
        float ob = b * cs.x + a * cs.y;
        __half ha = __float2half(oa), hb = __float2half(ob);
        p[lane] = ha;
        p[lane + 32] = hb;
        if (!isq && s.save_k) {
            int g = u - H;
            __half* dst = s.save_k + ((size_t)attn_idx * s.len + local) * (KVH * 64) + g * 64;
            dst[lane] = ha;
            dst[lane + 32] = hb;
        }
    }
}

// ------------------------------------------------------------------------------------------------ flash attention
// Bidirectional varlen attention, head_dim 64, fp16 tensor cores (mma.m16n8k8, sm_75+), fp32 softmax.
// grid = (work items, heads), block = 128 (4 warps x 16 query rows). Keys = [ext prefix keys | local keys].

#define FA_BQ 64
#define FA_BK 64
#define FA_LD 72  // padded smem row (halfs): conflict-free fragment loads and ldmatrix

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *(uint32_t*)&h;
}

// Load 8 halfs (+ optional fp32 bias) as a uint4.
__device__ __forceinline__ uint4 load8(const __half* p, const float* bias) {
    uint4 v = *(const uint4*)p;
    if (bias) {
        __half2* h = (__half2*)&v;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f = __half22float2(h[i]);
            h[i] = __floats2half2_rn(f.x + bias[2 * i], f.y + bias[2 * i + 1]);
        }
    }
    return v;
}

__global__ void __launch_bounds__(128) k_flash_attn(const __half* __restrict__ Q, int ldq, const __half* __restrict__ K,
                                                    const __half* __restrict__ V, int ldkv, int ext_ld,
                                                    const float* __restrict__ bq, const float* __restrict__ bk,
                                                    const float* __restrict__ bv, __half* __restrict__ O, int ldo,
                                                    const AttnSeq* __restrict__ seqs, const int2* __restrict__ work,
                                                    int group, float scale_log2) {
    __shared__ __align__(16) __half sQ[FA_BQ * FA_LD];
    __shared__ __align__(16) __half sK[FA_BK * FA_LD];
    __shared__ __align__(16) __half sV[FA_BK * FA_LD];

    const int2 wi = work[blockIdx.x];
    const AttnSeq s = seqs[wi.x];
    const int q0 = wi.y;
    const int h = blockIdx.y, g = h / group;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;

    const float* bqh = bq ? bq + h * 64 : nullptr;
    const float* bkh = bk ? bk + g * 64 : nullptr;
    const float* bvh = bv ? bv + g * 64 : nullptr;

    // Q tile -> smem
    for (int i = tid; i < FA_BQ * 8; i += 128) {
        int r = i / 8, c = (i % 8) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q0 + r < s.q_len) v = load8(Q + (size_t)(s.q_start + q0 + r) * ldq + h * 64 + c, bqh ? bqh + c : nullptr);
        *(uint4*)&sQ[r * FA_LD + c] = v;
    }
    __syncthreads();

    uint32_t qa[8][2];
    {
        const int r0 = warp * 16 + gid;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            qa[kk][0] = *(const uint32_t*)&sQ[r0 * FA_LD + kk * 8 + 2 * tig];
            qa[kk][1] = *(const uint32_t*)&sQ[(r0 + 8) * FA_LD + kk * 8 + 2 * tig];
        }
    }

    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    const int total = s.ext_len + s.kv_len;
    for (int j0 = 0; j0 < total; j0 += FA_BK) {
        __syncthreads();
        for (int i = tid; i < FA_BK * 8; i += 128) {
            int r = i / 8, c = (i % 8) * 8;
            int j = j0 + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (j < s.ext_len) {
                kv = *(const uint4*)(s.ext_k + (size_t)j * ext_ld + g * 64 + c);
                vv = *(const uint4*)(s.ext_v + (size_t)j * ext_ld + g * 64 + c);
            } else if (j < total) {
                size_t row = (size_t)(s.kv_start + j - s.ext_len) * ldkv + g * 64 + c;
                kv = load8(K + row, bkh ? bkh + c : nullptr);
                vv = load8(V + row, bvh ? bvh + c : nullptr);
            }
            *(uint4*)&sK[r * FA_LD + c] = kv;
            *(uint4*)&sV[r * FA_LD + c] = vv;
        }
        __syncthreads();

        float sc[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; kk += 2) {
                // two k-steps of B fragments for key row nt*8+gid via one 32-bit load each
                uint32_t b0 = *(const uint32_t*)&sK[(nt * 8 + gid) * FA_LD + kk * 8 + 2 * tig];
                uint32_t b1 = *(const uint32_t*)&sK[(nt * 8 + gid) * FA_LD + (kk + 1) * 8 + 2 * tig];
                mma16816(sc[nt], qa[kk][0], qa[kk][1], b0);
                mma16816(sc[nt], qa[kk + 1][0], qa[kk + 1][1], b1);
            }
        }

        // mask + online softmax (rows gid and gid+8 of this warp)
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int key = j0 + nt * 8 + 2 * tig;
#pragma unroll
            for (int e = 0; e < 2; e++) {
                bool ok = key + e < total;
                sc[nt][e] = ok ? sc[nt][e] * scale_log2 : -INFINITY;
                sc[nt][2 + e] = ok ? sc[nt][2 + e] * scale_log2 : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][e]);
                mx1 = fmaxf(mx1, sc[nt][2 + e]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL_MASK, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL_MASK, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL_MASK, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL_MASK, mx1, 2));
        float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
        float a0 = exp2f(m0 - nm0), a1 = exp2f(m1 - nm1);
        m0 = nm0;
        m1 = nm1;
        float rs0 = 0.f, rs1 = 0.f;
        uint32_t pa[8][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            float p0 = exp2f(sc[nt][0] - nm0), p1 = exp2f(sc[nt][1] - nm0);
            float p2 = exp2f(sc[nt][2] - nm1), p3 = exp2f(sc[nt][3] - nm1);
            rs0 += p0 + p1;
            rs1 += p2 + p3;
            pa[nt][0] = pack_half2(p0, p1);
            pa[nt][1] = pack_half2(p2, p3);
        }
        l0 = l0 * a0 + rs0;
        l1 = l1 * a1 + rs1;
#pragma unroll
        for (int dt = 0; dt < 8; dt++) {
            o[dt][0] *= a0;
            o[dt][1] *= a0;
            o[dt][2] *= a1;
            o[dt][3] *= a1;
        }
        // O += P V
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
#pragma unroll
            for (int dq = 0; dq < 2; dq++) {
                uint32_t b[4];
                const __half* p = &sV[(kk * 8 + (lane & 7)) * FA_LD + (dq * 4 + (lane >> 3)) * 8];
                ldmatrix_x4_trans(b[0], b[1], b[2], b[3], p);
#pragma unroll
                for (int i = 0; i < 4; i++) mma16816(o[dq * 4 + i], pa[kk][0], pa[kk][1], b[i]);
            }
        }
    }

    l0 += __shfl_xor_sync(FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(FULL_MASK, l1, 2);
    float il0 = 1.f / l0, il1 = 1.f / l1;
    int r0 = q0 + warp * 16 + gid, r1 = r0 + 8;
#pragma unroll
    for (int dt = 0; dt < 8; dt++) {
        int c = h * 64 + dt * 8 + 2 * tig;
        if (r0 < s.q_len)
            *(__half2*)(O + (size_t)(s.q_start + r0) * ldo + c) = __floats2half2_rn(o[dt][0] * il0, o[dt][1] * il0);
        if (r1 < s.q_len)
            *(__half2*)(O + (size_t)(s.q_start + r1) * ldo + c) = __floats2half2_rn(o[dt][2] * il1, o[dt][3] * il1);
    }
}

// ------------------------------------------------------------------------------------------------ decision scorer
// logit[r] = sum_i gelu(x[r][i] + b1[i]) * w2[i] + b2
__global__ void k_scorer_out(const __half* __restrict__ x, const float* __restrict__ b1, const __half* __restrict__ w2,
                             float b2, float* __restrict__ logits, int rows, int D) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    float s = 0.f;
    for (int c = lane; c < D; c += 32) s += gelu_erf(__half2float(x[(size_t)row * D + c]) + b1[c]) * __half2float(w2[c]);
    s = warp_sum(s);
    if (lane == 0) logits[row] = s + b2;
}

// ------------------------------------------------------------------------------------------------ vision
// LFM2-VL projector pixel unshuffle: x [ph*pw, C] -> out [(ph/2)*(pw/2), 4C]
__global__ void k_unshuffle(const __half* __restrict__ x, __half* __restrict__ out, int ph, int pw, int C) {
    int o = blockIdx.x;  // output row
    int i = o / (pw / 2), j = o % (pw / 2);
    for (int k = threadIdx.x * 8; k < 4 * C; k += blockDim.x * 8) {
        int q = k / C, c = k % C;
        int y = 2 * i + (q >> 1), xx = 2 * j + (q & 1);
        *(uint4*)(out + (size_t)o * 4 * C + k) = *(const uint4*)(x + ((size_t)y * pw + xx) * C + c);
    }
}

// ------------------------------------------------------------------------------------------------ audio
// First subsampling conv: mel [Tin][F] (time-major, rows >= tin_valid are zero) -> out [Tout][Fout][256] (ReLU)
__global__ void k_sub_conv0(const float* __restrict__ mel, int tin_valid, int F, const float* __restrict__ w /*[256][9]*/,
                            const float* __restrict__ b, float* __restrict__ out, int Tout, int Fout) {
    int t = blockIdx.x, f = blockIdx.y, c = threadIdx.x;  // 256 threads
    float acc = b[c];
#pragma unroll
    for (int dt = 0; dt < 3; dt++) {
        int ti = 2 * t - 1 + dt;
        if (ti < 0 || ti >= tin_valid) continue;
#pragma unroll
        for (int df = 0; df < 3; df++) {
            int fi = 2 * f - 1 + df;
            if (fi < 0 || fi >= F) continue;
            acc += w[c * 9 + dt * 3 + df] * mel[(size_t)ti * F + fi];
        }
    }
    out[((size_t)t * Fout + f) * 256 + c] = fmaxf(acc, 0.f);
}

// Depthwise 3x3 stride-2 conv on [Tin][Fin][256] -> [Tout][Fout][256] (no activation)
__global__ void k_sub_dwconv(const float* __restrict__ x, int tin_valid, int Fin, const float* __restrict__ w,
                             const float* __restrict__ b, float* __restrict__ out, int Fout) {
    int t = blockIdx.x, f = blockIdx.y, c = threadIdx.x;
    float acc = b[c];
#pragma unroll
    for (int dt = 0; dt < 3; dt++) {
        int ti = 2 * t - 1 + dt;
        if (ti < 0 || ti >= tin_valid) continue;
#pragma unroll
        for (int df = 0; df < 3; df++) {
            int fi = 2 * f - 1 + df;
            if (fi < 0 || fi >= Fin) continue;
            acc += w[c * 9 + dt * 3 + df] * x[((size_t)ti * Fin + fi) * 256 + c];
        }
    }
    out[((size_t)t * Fout + f) * 256 + c] = acc;
}

// Rel-pos attention prep: qkv [T][3*Dm] (no bias) -> qu = q+bq+u, qv = q+bq+v, k+bk, v+bv, each [T][Dm] fp16
__global__ void k_relpos_prep(const __half* __restrict__ qkv, const float* __restrict__ bq, const float* __restrict__ bk,
                              const float* __restrict__ bv, const float* __restrict__ pu, const float* __restrict__ pv,
                              __half* __restrict__ qu, __half* __restrict__ qvv, __half* __restrict__ k,
                              __half* __restrict__ v, int T, int Dm) {
    int t = blockIdx.x;
    for (int c = threadIdx.x; c < Dm; c += blockDim.x) {
        const __half* r = qkv + (size_t)t * 3 * Dm;
        float q = __half2float(r[c]) + bq[c];
        qu[(size_t)t * Dm + c] = __float2half(q + pu[c]);
        qvv[(size_t)t * Dm + c] = __float2half(q + pv[c]);
        k[(size_t)t * Dm + c] = __float2half(__half2float(r[Dm + c]) + bk[c]);
        v[(size_t)t * Dm + c] = __float2half(__half2float(r[2 * Dm + c]) + bv[c]);
    }
}

// scores = (ac[h][i][j] + bd[h][i][j + T-1-i]) * scale -> softmax over j -> p fp16 [h][T][T]
__global__ void k_relpos_softmax(const float* __restrict__ ac, const float* __restrict__ bd, __half* __restrict__ p,
                                 int T, float scale) {
    int h = blockIdx.y, i = blockIdx.x;
    const float* a = ac + ((size_t)h * T + i) * T;
    const float* b = bd + ((size_t)h * T + i) * (2 * T - 1) + (T - 1 - i);
    __shared__ float red[32];
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < T; j += blockDim.x) mx = fmaxf(mx, (a[j] + b[j]) * scale);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL_MASK, mx, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) sum += expf((a[j] + b[j]) * scale - mx);
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    float inv = 1.f / red[0];
    __half* pr = p + ((size_t)h * T + i) * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) pr[j] = __float2half(expf((a[j] + b[j]) * scale - mx) * inv);
}

// Conformer conv module middle: a [T][2C] (pw1 output, no bias) -> GLU -> depthwise k (zero pad) + bias -> BN affine
// -> SiLU -> out [T][C] fp16
__global__ void k_conformer_dw(const __half* __restrict__ a, const float* __restrict__ b1, const float* __restrict__ dw,
                               const float* __restrict__ dwb, const float* __restrict__ bn_s, const float* __restrict__ bn_b,
                               __half* __restrict__ out, int T, int C, int ksize) {
    int t = blockIdx.x;
    int pad = (ksize - 1) / 2;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = dwb[c];
        for (int k = 0; k < ksize; k++) {
            int ti = t + k - pad;
            if (ti < 0 || ti >= T) continue;
            float x = __half2float(a[(size_t)ti * 2 * C + c]) + b1[c];
            float gt = __half2float(a[(size_t)ti * 2 * C + C + c]) + b1[C + c];
            acc += dw[c * ksize + k] * (x / (1.f + expf(-gt)));
        }
        acc = acc * bn_s[c] + bn_b[c];
        out[(size_t)t * C + c] = __float2half(silu(acc));
    }
}

// ================================================================================================ launchers
extern "C" {

static inline int nblk(size_t n, int t) { return (int)((n + t - 1) / t); }
static inline int egrid(size_t n) {
    size_t b = (n + 255) / 256;
    return (int)(b > 65535 * 8 ? 65535 * 8 : (b < 1 ? 1 : b));
}

int d1_rmsnorm(const float* x, const float* w, __half* y, int rows, int D, float eps, cudaStream_t st) {
    dim3 grid(nblk(rows, 4)), block(128);
    switch (D) {
        case 1024: k_rmsnorm<32><<<grid, block, 0, st>>>(x, w, y, rows, eps); break;
        case 768: k_rmsnorm<24><<<grid, block, 0, st>>>(x, w, y, rows, eps); break;
        case 512: k_rmsnorm<16><<<grid, block, 0, st>>>(x, w, y, rows, eps); break;
        default: return -1;
    }
    return (int)cudaGetLastError();
}

int d1_layernorm(float* x, const float* pend_bias, float pend_scale, const float* w, const float* b, __half* y, float* yf,
                 int rows, int D, float eps, cudaStream_t st) {
    dim3 grid(nblk(rows, 4)), block(128);
    switch (D) {
        case 1024: k_layernorm<32><<<grid, block, 0, st>>>(x, pend_bias, pend_scale, w, b, y, yf, rows, eps); break;
        case 768: k_layernorm<24><<<grid, block, 0, st>>>(x, pend_bias, pend_scale, w, b, y, yf, rows, eps); break;
        case 512: k_layernorm<16><<<grid, block, 0, st>>>(x, pend_bias, pend_scale, w, b, y, yf, rows, eps); break;
        default: return -1;
    }
    return (int)cudaGetLastError();
}

int d1_head_init(const float* h, const float* w, const float* type_emb, const int* tok_seq, const SeqInfo* seqs,
                 float* out, int rows, float eps, cudaStream_t st) {
    k_head_init<<<nblk(rows, 4), 128, 0, st>>>(h, w, type_emb, tok_seq, seqs, out, rows, eps);
    return (int)cudaGetLastError();
}

int d1_embed(const __half* table, const int* ids, float* h, int rows, int D, cudaStream_t st) {
    if (rows == 0) return 0;
    k_embed<<<rows, 256, 0, st>>>(table, ids, h, rows, D);
    return (int)cudaGetLastError();
}

int d1_swiglu(const __half* gu, __half* out, int rows, int F, cudaStream_t st) {
    k_swiglu<<<egrid((size_t)rows * F / 2), 256, 0, st>>>(gu, out, rows, F);
    return (int)cudaGetLastError();
}

int d1_bias_act_h(__half* x, const float* bias, int rows, int N, int act, cudaStream_t st) {
    k_bias_act_h<<<egrid((size_t)rows * N / 2), 256, 0, st>>>(x, bias, rows, N, act);
    return (int)cudaGetLastError();
}

int d1_bias_act_f(float* x, const float* bias, int rows, int N, int act, cudaStream_t st) {
    k_bias_act_f<<<egrid((size_t)rows * N), 256, 0, st>>>(x, bias, rows, N, act);
    return (int)cudaGetLastError();
}

int d1_fill_rows(float* out, const float* bias, const float* add, int rows, int N, cudaStream_t st) {
    k_fill_rows<<<egrid((size_t)rows * N), 256, 0, st>>>(out, bias, add, rows, N);
    return (int)cudaGetLastError();
}

int d1_f32_to_f16(const float* x, __half* y, size_t n, cudaStream_t st) {
    k_f32_to_f16<<<egrid(n), 256, 0, st>>>(x, y, n);
    return (int)cudaGetLastError();
}

int d1_gather_rows(const void* src, void* dst, const int* idx, int n, int row_bytes, cudaStream_t st) {
    if (n == 0) return 0;
    k_gather_rows<<<n, 128, 0, st>>>((const uint4*)src, (uint4*)dst, idx, n, row_bytes / 16);
    return (int)cudaGetLastError();
}

int d1_shortconv(const __half* bcu, const float* convw, const int* tok_seq, const SeqInfo* seqs, __half* out, int rows,
                 int D, int conv_idx, cudaStream_t st) {
    k_shortconv<<<rows, 256, 0, st>>>(bcu, convw, tok_seq, seqs, out, D, conv_idx);
    return (int)cudaGetLastError();
}

int d1_qk_norm_rope(__half* qkv, int ld, int H, int KVH, const float* qn, const float* kn, const float2* rope,
                    const int* tok_seq, const SeqInfo* seqs, int attn_idx, int rows, float eps, cudaStream_t st) {
    k_qk_norm_rope<<<rows, 256, 0, st>>>(qkv, ld, H, KVH, qn, kn, rope, tok_seq, seqs, attn_idx, eps);
    return (int)cudaGetLastError();
}

int d1_flash_attn(const __half* Q, int ldq, const __half* K, const __half* V, int ldkv, int ext_ld, const float* bq,
                  const float* bk, const float* bv, __half* O, int ldo, const AttnSeq* seqs, const int2* work, int n_work,
                  int heads, int group, float scale, cudaStream_t st) {
    if (n_work == 0) return 0;
    dim3 grid(n_work, heads);
    k_flash_attn<<<grid, 128, 0, st>>>(Q, ldq, K, V, ldkv, ext_ld, bq, bk, bv, O, ldo, seqs, work, group,
                                       scale * 1.4426950408889634f);
    return (int)cudaGetLastError();
}

int d1_scorer_out(const __half* x, const float* b1, const __half* w2, float b2, float* logits, int rows, int D,
                  cudaStream_t st) {
    if (rows == 0) return 0;
    k_scorer_out<<<nblk(rows, 4), 128, 0, st>>>(x, b1, w2, b2, logits, rows, D);
    return (int)cudaGetLastError();
}

int d1_unshuffle(const __half* x, __half* out, int ph, int pw, int C, cudaStream_t st) {
    k_unshuffle<<<(ph / 2) * (pw / 2), 128, 0, st>>>(x, out, ph, pw, C);
    return (int)cudaGetLastError();
}

int d1_sub_conv0(const float* mel, int tin_valid, int F, const float* w, const float* b, float* out, int Tout, int Fout,
                 cudaStream_t st) {
    dim3 grid(Tout, Fout);
    k_sub_conv0<<<grid, 256, 0, st>>>(mel, tin_valid, F, w, b, out, Tout, Fout);
    return (int)cudaGetLastError();
}

int d1_sub_dwconv(const float* x, int tin_valid, int Fin, const float* w, const float* b, float* out, int Tout, int Fout,
                  cudaStream_t st) {
    dim3 grid(Tout, Fout);
    k_sub_dwconv<<<grid, 256, 0, st>>>(x, tin_valid, Fin, w, b, out, Fout);
    return (int)cudaGetLastError();
}

int d1_relpos_prep(const __half* qkv, const float* bq, const float* bk, const float* bv, const float* pu, const float* pv,
                   __half* qu, __half* qv, __half* k, __half* v, int T, int Dm, cudaStream_t st) {
    k_relpos_prep<<<T, 256, 0, st>>>(qkv, bq, bk, bv, pu, pv, qu, qv, k, v, T, Dm);
    return (int)cudaGetLastError();
}

int d1_relpos_softmax(const float* ac, const float* bd, __half* p, int T, int H, float scale, cudaStream_t st) {
    dim3 grid(T, H);
    k_relpos_softmax<<<grid, 256, 0, st>>>(ac, bd, p, T, scale);
    return (int)cudaGetLastError();
}

int d1_conformer_dw(const __half* a, const float* b1, const float* dw, const float* dwb, const float* bn_s,
                    const float* bn_b, __half* out, int T, int C, int ksize, cudaStream_t st) {
    k_conformer_dw<<<T, 256, 0, st>>>(a, b1, dw, dwb, bn_s, bn_b, out, T, C, ksize);
    return (int)cudaGetLastError();
}

}  // extern "C"
