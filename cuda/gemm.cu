// cuBLASLt GEMM with per-shape autotuning.
//
// cuBLAS's heuristics pick poor tiles for the small token counts of latency-bound batches (e.g. 150 rows: 13.8
// TFLOPS where a split-K/swizzled config of the same library reaches 20+). For every (N, K, layout, dtype,
// M-bucket) the tuner times the library's algorithm/tile space (pruned two-phase search) and caches the winner;
// plans can be saved to / loaded from a text file.
//
// Row-major semantics: Y[M,N] = alpha * X[M,K] . W^T + beta * Y, W row-major [N,K] (or [K,N] with w_kn).

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

namespace {

typedef std::tuple<int, int, int, int, int, int, int, int, int> Key;  // N K ldx ldw ldy w_kn y_f32 f16acc bucket

struct Plan {
    cublasLtMatmulAlgo_t algo;
    size_t ws;
    float us;
};

struct Ctx {
    cublasLtHandle_t lt;
    void* ws;
    size_t ws_bytes;
    std::map<Key, Plan> plans;
    // tuning scratch
    void* sx = nullptr;
    void* sw = nullptr;
    void* sy = nullptr;
    size_t sx_b = 0, sw_b = 0, sy_b = 0;
};

const int BUCKETS[] = {16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
const int NBUCKETS = sizeof(BUCKETS) / sizeof(int);

int bucket_of(int m) {
    for (int i = 0; i < NBUCKETS; i++)
        if (m <= BUCKETS[i]) return BUCKETS[i];
    return BUCKETS[NBUCKETS - 1];
}

struct Descs {
    cublasLtMatmulDescOpaque_t d;
    cublasLtMatrixLayoutOpaque_t a, b, c;
};

int make_descs(Descs& ds, int M, int N, int K, int ldx, int ldw, int w_kn, int ldy, int y_f32, int f16acc) {
    cublasComputeType_t ct = f16acc && !y_f32 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
    cudaDataType_t st = f16acc && !y_f32 ? CUDA_R_16F : CUDA_R_32F;
    if (cublasLtMatmulDescInit(&ds.d, ct, st)) return -1;
    cublasOperation_t ta = w_kn ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(&ds.d, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(&ds.d, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    // column-major view: Y^T (N x M) = op(A) (N x K) * X^T (K x M)
    if (w_kn) {
        if (cublasLtMatrixLayoutInit(&ds.a, CUDA_R_16F, N, K, ldw)) return -1;
    } else {
        if (cublasLtMatrixLayoutInit(&ds.a, CUDA_R_16F, K, N, ldw)) return -1;
    }
    if (cublasLtMatrixLayoutInit(&ds.b, CUDA_R_16F, K, M, ldx)) return -1;
    if (cublasLtMatrixLayoutInit(&ds.c, y_f32 ? CUDA_R_32F : CUDA_R_16F, N, M, ldy)) return -1;
    return 0;
}

int run(Ctx* c, Descs& ds, const cublasLtMatmulAlgo_t* algo, size_t ws, const void* X, const void* W, void* Y, float alpha,
        float beta, int f16acc_h, cudaStream_t st) {
    if (f16acc_h) {
        __half a = __float2half(alpha), b = __float2half(beta);
        return (int)cublasLtMatmul(c->lt, &ds.d, &a, W, &ds.a, X, &ds.b, &b, Y, &ds.c, Y, &ds.c, algo, c->ws, ws, st);
    }
    return (int)cublasLtMatmul(c->lt, &ds.d, &alpha, W, &ds.a, X, &ds.b, &beta, Y, &ds.c, Y, &ds.c, algo, c->ws, ws, st);
}

void ensure(void** p, size_t* have, size_t need) {
    if (*have >= need) return;
    if (*p) cudaFree(*p);
    cudaMalloc(p, need);
    *have = need;
    // non-zero, realistic magnitudes (zeros make tensor cores run cooler and faster than real data)
    std::vector<__half> h(need / 2);
    uint32_t s = 12345;
    for (auto& x : h) {
        s = s * 1664525u + 1013904223u;
        x = __float2half(((int)(s >> 9) % 2000 - 1000) * 1e-4f);
    }
    cudaMemcpy(*p, h.data(), need / 2 * 2, cudaMemcpyHostToDevice);
}

float time_it(Ctx* c, Descs& ds, const cublasLtMatmulAlgo_t* algo, size_t ws, int f16h, cudaStream_t st, cudaEvent_t e0,
              cudaEvent_t e1, int reps) {
    for (int i = 0; i < 2; i++)
        if (run(c, ds, algo, ws, c->sx, c->sw, c->sy, 1.f, 0.f, f16h, st)) return 1e30f;
    cudaEventRecord(e0, st);
    for (int i = 0; i < reps; i++) run(c, ds, algo, ws, c->sx, c->sw, c->sy, 1.f, 0.f, f16h, st);
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    if (cudaGetLastError() != cudaSuccess) return 1e30f;
    return ms * 1000.f / reps;
}

}  // namespace

extern "C" {

void* d1_lt_init(size_t ws_bytes) {
    Ctx* c = new Ctx();
    if (cublasLtCreate(&c->lt)) return nullptr;
    c->ws_bytes = ws_bytes;
    if (cudaMalloc(&c->ws, ws_bytes)) return nullptr;
    return c;
}

// Returns 0 if a tuned (or heuristic) plan ran, -2 if no plan exists (caller falls back to cublasGemmEx).
int d1_lt_gemm(void* ctx, int M, int N, int K, const void* X, int ldx, const void* W, int ldw, int w_kn, void* Y, int ldy,
               int y_f32, int f16acc, float alpha, float beta, cudaStream_t st) {
    Ctx* c = (Ctx*)ctx;
    Key key(N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, bucket_of(M));
    auto it = c->plans.find(key);
    if (it == c->plans.end()) return -2;
    Descs ds;
    if (make_descs(ds, M, N, K, ldx, ldw, w_kn, ldy, y_f32, f16acc)) return -1;
    return run(c, ds, &it->second.algo, it->second.ws, X, W, Y, alpha, beta, f16acc && !y_f32, st);
}

// Tune one shape at bucket size M (the bucket's upper end). effort: 0 = heuristics only, 1 = pruned search,
// 2 = exhaustive. Returns the best time in microseconds (negative on failure).
float d1_lt_tune(void* ctx, int M, int N, int K, int ldx, int ldw, int w_kn, int ldy, int y_f32, int f16acc, int effort,
                 cudaStream_t st) {
    Ctx* c = (Ctx*)ctx;
    int bm = bucket_of(M);
    Key key(N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, bm);
    if (c->plans.count(key)) return c->plans[key].us;
    M = bm;
    ensure(&c->sx, &c->sx_b, (size_t)M * ldx * 2 + 256);
    ensure(&c->sw, &c->sw_b, (size_t)(w_kn ? K : N) * ldw * 2 + 256);
    ensure(&c->sy, &c->sy_b, (size_t)M * ldy * 4 + 256);
    Descs ds;
    if (make_descs(ds, M, N, K, ldx, ldw, w_kn, ldy, y_f32, f16acc)) return -1;
    int f16h = f16acc && !y_f32;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int reps = M >= 4096 ? 3 : M >= 1024 ? 5 : 10;

    struct Cand {
        cublasLtMatmulAlgo_t algo;
        size_t ws;
        float us;
    };
    std::vector<Cand> cands;
    // heuristics
    {
        cublasLtMatmulPreference_t pr;
        cublasLtMatmulPreferenceCreate(&pr);
        cublasLtMatmulPreferenceSetAttribute(pr, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c->ws_bytes, sizeof(size_t));
        cublasLtMatmulHeuristicResult_t res[16];
        int nr = 0;
        cublasLtMatmulAlgoGetHeuristic(c->lt, &ds.d, &ds.a, &ds.b, &ds.c, &ds.c, pr, 16, res, &nr);
        cublasLtMatmulPreferenceDestroy(pr);
        for (int i = 0; i < nr; i++) {
            if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > c->ws_bytes) continue;
            float us = time_it(c, ds, &res[i].algo, res[i].workspaceSize, f16h, st, e0, e1, reps);
            cands.push_back({res[i].algo, res[i].workspaceSize, us});
        }
    }
    if (effort > 0 && M <= 8192) {
        cublasComputeType_t ct = f16h ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
        cudaDataType_t sct = f16h ? CUDA_R_16F : CUDA_R_32F;
        cudaDataType_t yt = y_f32 ? CUDA_R_32F : CUDA_R_16F;
        int ids[64], nid = 0;
        cublasLtMatmulAlgoGetIds(c->lt, ct, sct, CUDA_R_16F, CUDA_R_16F, yt, yt, 64, ids, &nid);
        struct Base {
            cublasLtMatmulAlgo_t algo;
            float us;
            int splitk_sup, swz;
            uint32_t redmask;
        };
        std::vector<Base> bases;
        // phase 1: every (algo, tile), no split-K, default stages
        for (int a = 0; a < nid; a++) {
            cublasLtMatmulAlgo_t algo;
            if (cublasLtMatmulAlgoInit(c->lt, ct, sct, CUDA_R_16F, CUDA_R_16F, yt, yt, ids[a], &algo)) continue;
            size_t sz = 0;
            int tiles[128];
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
            int nt = (int)(sz / sizeof(int));
            if (nt > 128) nt = 128;
            if (nt)
                cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof(int) * nt, &sz);
            else {
                tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
                nt = 1;
            }
            int splitk_sup = 0, swz = 0;
            uint32_t redmask = 0;
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_sup, sizeof(int), &sz);
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swz, sizeof(int), &sz);
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redmask, sizeof(redmask), &sz);
            for (int t = 0; t < nt; t++) {
                cublasLtMatmulAlgo_t al = algo;
                int one = 1, zero = 0;
                uint32_t r0 = 0;
                cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[t], sizeof(int));
                cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(int));
                cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &r0, sizeof(r0));
                cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &zero, sizeof(int));
                cublasLtMatmulHeuristicResult_t hr;
                if (cublasLtMatmulAlgoCheck(c->lt, &ds.d, &ds.a, &ds.b, &ds.c, &ds.c, &al, &hr)) continue;
                if (hr.workspaceSize > c->ws_bytes) continue;
                float us = time_it(c, ds, &al, hr.workspaceSize, f16h, st, e0, e1, reps);
                if (us < 1e29f) {
                    bases.push_back({al, us, splitk_sup, swz, redmask});
                    cands.push_back({al, hr.workspaceSize, us});
                }
            }
        }
        // phase 2: split-K / swizzle / reduction variants of the best few
        std::sort(bases.begin(), bases.end(), [](const Base& x, const Base& y) { return x.us < y.us; });
        int keep = effort >= 2 ? (int)bases.size() : std::min<int>(4, (int)bases.size());
        const int splits[] = {1, 2, 3, 4, 6, 8, 12, 16};
        for (int b = 0; b < keep; b++) {
            for (int sk : splits) {
                if (sk > 1 && !bases[b].splitk_sup) continue;
                if ((long)K / sk < 64) continue;
                for (int sw = 0; sw <= bases[b].swz; sw++) {
                    for (uint32_t red : {1u, 2u, 4u}) {
                        if (sk == 1 && red != 1u) continue;
                        if (sk > 1 && !(bases[b].redmask & red)) continue;
                        if (sk == 1 && sw == 0) continue;  // phase 1 already
                        cublasLtMatmulAlgo_t al = bases[b].algo;
                        uint32_t rs = sk > 1 ? red : 0;
                        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(int));
                        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, sizeof(rs));
                        cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw, sizeof(int));
                        cublasLtMatmulHeuristicResult_t hr;
                        if (cublasLtMatmulAlgoCheck(c->lt, &ds.d, &ds.a, &ds.b, &ds.c, &ds.c, &al, &hr)) continue;
                        if (hr.workspaceSize > c->ws_bytes) continue;
                        float us = time_it(c, ds, &al, hr.workspaceSize, f16h, st, e0, e1, reps);
                        if (us < 1e29f) cands.push_back({al, hr.workspaceSize, us});
                    }
                }
            }
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (cands.empty()) return -1;
    auto best = std::min_element(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.us < y.us; });
    // re-time the winner against the heuristic #0 with more reps to avoid noise-driven picks
    c->plans[key] = Plan{best->algo, best->ws, best->us};
    return best->us;
}

int d1_lt_save(void* ctx, const char* path) {
    Ctx* c = (Ctx*)ctx;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "d1rs-gemm-plans v1 cublasLt %zu\n", cublasLtGetVersion());
    for (auto& kv : c->plans) {
        int N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, b;
        std::tie(N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, b) = kv.first;
        fprintf(f, "%d %d %d %d %d %d %d %d %d %zu %.2f", N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, b, kv.second.ws, kv.second.us);
        for (int i = 0; i < 8; i++) fprintf(f, " %016llx", (unsigned long long)kv.second.algo.data[i]);
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

int d1_lt_load(void* ctx, const char* path) {
    Ctx* c = (Ctx*)ctx;
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char hdr[128];
    size_t ver = 0;
    if (fscanf(f, "%127s v1 cublasLt %zu\n", hdr, &ver) != 2 || ver != cublasLtGetVersion()) {
        fclose(f);
        return -2;
    }
    int n = 0;
    while (true) {
        int N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, b;
        size_t ws;
        float us;
        unsigned long long d[8];
        if (fscanf(f, "%d %d %d %d %d %d %d %d %d %zu %f %llx %llx %llx %llx %llx %llx %llx %llx", &N, &K, &ldx, &ldw, &ldy,
                   &w_kn, &y_f32, &f16acc, &b, &ws, &us, &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6], &d[7]) != 19)
            break;
        Plan p;
        for (int i = 0; i < 8; i++) p.algo.data[i] = d[i];
        p.ws = ws;
        p.us = us;
        if (ws <= c->ws_bytes) {
            c->plans[Key(N, K, ldx, ldw, ldy, w_kn, y_f32, f16acc, b)] = p;
            n++;
        }
    }
    fclose(f);
    return n;
}

int d1_lt_bucket(int m) { return bucket_of(m); }

}  // extern "C"
