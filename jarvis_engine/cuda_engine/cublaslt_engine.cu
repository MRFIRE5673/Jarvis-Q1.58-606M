// =============================================================================
// cublaslt_engine.cu — cuBLAS GEMM Manager (SM120 / RTX 5070)
// =============================================================================
// Uses cublasGemmEx from CUDA 13.3 cublas64_13.dll (has SM120 BF16 TensorCore).
//
// CORRECT ROW-MAJOR GEMM FORMULA (derived and verified):
// For C_row[M,N] = alpha * A_row[M,K] * B_row[N,K]^T + beta * C_row[M,N]:
//
//   cublasGemmEx(T, N, N, M, K, B_ptr, K, A_ptr, K, C_ptr, N)
//
// WHY THIS WORKS:
//   opA=T on B_ptr: B_row[N,K] viewed as col-major[K,N] ld=K, transposed → B[N,K]
//   opB=N on A_ptr: A_row[M,K] viewed as col-major[K,M] ld=K, no trans    → A^T[K,M]
//   result: B[N,K] × A^T[K,M] = C^T[N,M] stored col-major at C_ptr ld=N
//   C^T_col[N,M] at C_ptr ≡ C_row[M,N] at C_ptr (identical byte layout!) ✓
//
// LEADING DIMENSION CONSTRAINTS (always satisfied):
//   lda=K (for opT): lda >= k=K  → K >= K  ✓ always
//   ldb=K (for opN): ldb >= k=K  → K >= K  ✓ always
//   ldc=N:           ldc >= m=N  → N >= N  ✓ always
// =============================================================================

#include "cublaslt_engine.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <cstdio>

#define CUBLAS_CHECK(call)                                                         \
    do {                                                                           \
        cublasStatus_t _s = (call);                                                \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
            char _buf[256];                                                        \
            snprintf(_buf, sizeof(_buf),                                           \
                "[cuBLAS] %s:%d status=%d", __FILE__, __LINE__, (int)_s);         \
            throw std::runtime_error(_buf);                                        \
        }                                                                          \
    } while(0)

// ---------------------------------------------------------------------------
// FP8 E4M3 helpers: dynamic per-tensor amax -> scaled cast; inv scale feeds
// cublasLt A/B scale pointers (dequant on output).
// ---------------------------------------------------------------------------
#include <cuda_fp8.h>
#include <cublasLt.h>

__global__ void amax_bf16_kernel(const bf16* __restrict__ x, int n4, float* __restrict__ amax) {
    float m = 0.f;
    const uint2* x4 = reinterpret_cast<const uint2*>(x);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        uint2 u = x4[i];
        float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.x));
        float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.y));
        m = fmaxf(m, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(reinterpret_cast<int*>(amax), __float_as_int(m));
}

// e5: 0 = E4M3 (max 448), 1 = E5M2 (max 57344, for gradients)
__global__ void cast_fp8_kernel(const bf16* __restrict__ x, uint8_t* __restrict__ y, int n4,
                                const float* __restrict__ amax, float* __restrict__ inv, int e5) {
    const float fmax_ = e5 ? 57344.0f : 448.0f;
    float am    = fmaxf(*amax, 1e-12f);
    float scale = fmax_ / am;
    if (blockIdx.x == 0 && threadIdx.x == 0) *inv = am / fmax_;
    const __nv_fp8_interpretation_t fmt = e5 ? __NV_E5M2 : __NV_E4M3;
    const uint2* x4 = reinterpret_cast<const uint2*>(x);
    uint32_t* y4 = reinterpret_cast<uint32_t*>(y);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        uint2 u = x4[i];
        float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.x));
        float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.y));
        uint32_t r = __nv_cvt_float2_to_fp8x2(make_float2(a.x * scale, a.y * scale), __NV_SATFINITE, fmt)
                   | ((uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(b.x * scale, b.y * scale), __NV_SATFINITE, fmt) << 16);
        y4[i] = r;
    }
}

// in [R][Cc] bf16 -> out [Cc][R] fp8 (scaled by amax), 32x32 tiles
__global__ void cast_fp8_T_kernel(const bf16* __restrict__ x, uint8_t* __restrict__ y, int R, int Cc,
                                  const float* __restrict__ amax, float* __restrict__ inv, int e5) {
    __shared__ uint8_t t[32][33];
    const float fmax_ = e5 ? 57344.0f : 448.0f;
    float am    = fmaxf(*amax, 1e-12f);
    float scale = fmax_ / am;
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) *inv = am / fmax_;
    const __nv_fp8_interpretation_t fmt = e5 ? __NV_E5M2 : __NV_E4M3;
    int c = blockIdx.x * 32 + threadIdx.x;
    for (int j = threadIdx.y; j < 32; j += 8) {
        int r = blockIdx.y * 32 + j;
        float v = (r < R && c < Cc) ? __bfloat162float(x[(size_t)r * Cc + c]) * scale : 0.f;
        t[j][threadIdx.x] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, fmt);
    }
    __syncthreads();
    int r0 = blockIdx.y * 32 + threadIdx.x;
    for (int j = threadIdx.y; j < 32; j += 8) {
        int c2 = blockIdx.x * 32 + j;
        if (r0 < R && c2 < Cc) y[(size_t)c2 * R + r0] = t[threadIdx.x][j];
    }
}

// src is [rows][cols]; if tr, dst is [cols][rows]. n = rows*cols.
static void quant_fp8(const bf16* x, uint8_t* y, int rows, int cols, bool tr, int e5,
                      float* amax, float* inv, cudaStream_t s) {
    int n = rows * cols, n4 = n / 4, blk = 256;
    int grd = min((n4 + blk - 1) / blk, 1024);
    cudaMemsetAsync(amax, 0, sizeof(float), s);
    amax_bf16_kernel<<<grd, blk, 0, s>>>(x, n4, amax);
    if (!tr) cast_fp8_kernel<<<grd, blk, 0, s>>>(x, y, n4, amax, inv, e5);
    else {
        dim3 g((cols + 31) / 32, (rows + 31) / 32), b(32, 8);
        cast_fp8_T_kernel<<<g, b, 0, s>>>(x, y, rows, cols, amax, inv, e5);
    }
}

// Non-static wrapper so full_engine.cu can fill the per-layer FP8 weight caches.
void quant_fp8_device(const bf16* x, uint8_t* y, int rows, int cols, bool tr, int e5,
                      float* d_amax, float* d_inv, cudaStream_t s) {
    quant_fp8(x, y, rows, cols, tr, e5, d_amax, d_inv, s);
}

void CublasLtEngine::gemm_fp8(int M, int N, int K, const bf16* X, const bf16* W, bf16* Y,
                              cudaStream_t s) const {
    gemm_fp8_ex(M, N, K, X, W, Y, false, false, false, 0.f, s);
}

// Y[M,N] = beta*Y + X[M,K] * W[N,K]^T. xT: X stored [K][M]. wT: W stored [K][N].
// grad: X is a gradient -> E5M2.
void CublasLtEngine::gemm_fp8_ex(int M, int N, int K, const bf16* X, const bf16* W, bf16* Y,
                                 bool xT, bool wT, bool grad, float beta_, cudaStream_t s) const {
    int e5 = grad ? 1 : 0;
    if (xT) quant_fp8(X, x8, K, M, true, e5, f8_stat + 0, f8_stat + 2, s);
    else    quant_fp8(X, x8, M, K, false, e5, f8_stat + 0, f8_stat + 2, s);
    if (wT) quant_fp8(W, w8, K, N, true, 0, f8_stat + 1, f8_stat + 3, s);
    else    quant_fp8(W, w8, N, K, false, 0, f8_stat + 1, f8_stat + 3, s);

    cublasLtMatmulDesc_t op;
    CUBLAS_CHECK((cublasStatus_t)cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn));
    const void* a_sc = f8_stat + 3;   // W inverse scale
    const void* b_sc = f8_stat + 2;   // X inverse scale
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sc, sizeof(a_sc));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sc, sizeof(b_sc));

    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&lb, grad ? CUDA_R_8F_E5M2 : CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t wsz = 32u << 20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
    cublasLtMatmulHeuristicResult_t r; int n = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &r, &n);
    if (!n) throw std::runtime_error("[cuBLASLt] no FP8 algo");

    float alpha = 1.f, beta = beta_;
    CUBLAS_CHECK((cublasStatus_t)cublasLtMatmul(lt, op, &alpha, w8, la, x8, lb, &beta,
                                                Y, lc, Y, lc, &r.algo, lt_ws, wsz, s));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

// gemm_fp8_cached: W already quantized in W_q8 (E4M3, [N,K] row-major), d_inv_w device ptr to inv scale.
// Quantizes X (E4M3) on the fly. C[M,N] = X[M,K] * W[N,K]^T.
void CublasLtEngine::gemm_fp8_cached(int M, int N, int K,
                                      const bf16* X, const uint8_t* W_q8,
                                      const float* d_inv_w, bf16* Y, cudaStream_t s) const {
    quant_fp8(X, x8, M, K, false, 0, f8_stat + 0, f8_stat + 2, s);

    cublasLtMatmulDesc_t op;
    CUBLAS_CHECK((cublasStatus_t)cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof(Nn));
    const void* a_sc = d_inv_w;        // pre-stored device inverse scale
    const void* b_sc = f8_stat + 2;   // activation inverse scale (just computed)
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sc, sizeof(a_sc));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sc, sizeof(b_sc));

    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t wsz = 32u << 20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
    cublasLtMatmulHeuristicResult_t r; int n = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &r, &n);
    if (!n) throw std::runtime_error("[cuBLASLt] no FP8 algo (cached)");

    float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK((cublasStatus_t)cublasLtMatmul(lt, op, &alpha, W_q8, la, x8, lb, &beta,
                                                Y, lc, Y, lc, &r.algo, lt_ws, wsz, s));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}


// ---------------------------------------------------------------------------
static CublasLtEngine g_cublaslt;
CublasLtEngine& get_cublaslt() { return g_cublaslt; }

// ---------------------------------------------------------------------------
// init — create cuBLAS handle, enable tensor ops
// ---------------------------------------------------------------------------
void CublasLtEngine::init(const JarvisConfig& cfg) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK((cublasStatus_t)cublasLtCreate(&lt));
    cudaMalloc(&lt_ws, 32u << 20);
    cudaMalloc(&x8, 16u << 20);
    cudaMalloc(&w8, 16u << 20);
    cudaMalloc(&f8_stat, 4 * sizeof(float));
    // SM120 peak: enable TensorCore for all ops, allow implicit type conversion
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    printf("[cuBLAS] Init OK. Handle ready for SM120 BF16 TensorCore (TF32_TENSOR_OP_MATH).\n");
    (void)cfg;
}

void CublasLtEngine::destroy() {
    if (handle) { cublasDestroy(handle); handle = nullptr; }
}

void CublasLtEngine::autotune_algorithms(const JarvisConfig& cfg) {
    printf("[cuBLAS] Using cublasGemmEx CUBLAS_GEMM_DEFAULT_TENSOR_OP (SM120 BF16).\n");
    (void)cfg;
}

// ---------------------------------------------------------------------------
// gemm — core NT BF16 GEMM for row-major data
//   C[M,N] = alpha * A[M,K] * B[N,K]^T + beta * C[M,N]
//
// Verified working form (all constraints always satisfied):
//   cublasGemmEx(T, N, m=N, n=M, k=K, B, lda=K, A, ldb=K, C, ldc=N)
// ---------------------------------------------------------------------------
void CublasLtEngine::gemm(const GemmDesc& d,
                           const void* A, const void* B, void* C,
                           float alpha, float beta, cudaStream_t s) const {
    CUBLAS_CHECK(cublasSetStream(handle, s));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,   // opA on B_ptr: B_row[N,K] as col[K,N] → transposed → B[N,K]
        CUBLAS_OP_N,   // opB on A_ptr: A_row[M,K] as col[K,M] → no-op → A^T[K,M]
        d.N, d.M, d.K, // m=N, n=M, k=K
        &alpha,
        B, CUDA_R_16BF, d.K,   // B_ptr: K rows col-major, lda=K >= k=K ✓
        A, CUDA_R_16BF, d.K,   // A_ptr: K rows col-major, ldb=K >= k=K ✓
        &beta,
        C, CUDA_R_16BF, d.N,   // C_ptr: N rows col-major = C[M,N] row-major ✓
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

}

// ---------------------------------------------------------------------------
// gemm_nn — NN GEMM: C[M,N] = A[M,K] * B[K,N] (both non-transposed, row-major)
// Used in backward passes where weight is not transposed.
// Row-major NN: cublasGemmEx(T, T, m=N, n=M, k=K, B, K, A, M, C, N)... wait:
// For C=A@B (std NN): B_row[K,N] at B_ptr, A_row[M,K] at A_ptr.
// Same trick: B^T_col[N,K] = B_ptr ld=N: op(B_ptr) T → B[K,N] NO.
//
// For C[M,N] = A[M,K] @ B[K,N]:
//   C^T[N,M] = B^T[N,K] @ A^T[K,M]
//   B_row[K,N] as col[N,K] ld=N → op with no transpose = B^T[N,K]... 
//   Actually: B_row[K,N] stored as B_ptr: B[i,j]=B_ptr[i*N+j].
//   As col-major [N,K] with ld=N: B_col[i,j]=B_ptr[i+j*N]=B_row[j,i]=B^T[i,j] ✓
//   opT on col[N,K] ld=N → transposed → [K,N] col → original B[K,N] ✓
//   A_row[M,K] as col[K,M] ld=K → A^T[K,M] ✓, opN → A^T
//   Result: op_first × op_second = B^T^T[K,N] × A^T[K,M]... wait that's B×A^T=[K,M] wrong dims.
//
// CORRECT for NN: use (T, T, N, M, K, B_ptr, N, A_ptr, K, C_ptr, N):
//   opA=T on B_ptr col[N,K] ld=N: B^T^T = B[K,N]. m=N, k=K. constraint: lda=N >= k=K? N>=K.
//   opB=T on A_ptr col[K,M] ld=K: A^T^T = A[M,K]. n=M, k=K. constraint: ldb=K >= n=M? K>=M.
//   Both may fail. Use (N, N, N, M, K, B, N, A, K, C, N) instead:
//   opA=N on B col[N,K] ld=N → B^T[N,K], m=N, k=K: lda=N >= m=N ✓. op = B^T[N,K]... 
// 
// Simplest: just use the same (T,N) trick but swap roles for NN:
//   C[M,N] = A[M,K] @ B[K,N] = (B^T @ A^T)^T
//   = cublasGemmEx(T,N, N,M,K, B_ptr, N, A_ptr, K, C_ptr, N) where now B=B^T[K,N]=B_row
//   Wait B_row[K,N] at B_ptr: as col[N,K] ld=N → B^T[N,K]. opT → B^T^T = B[K,N]. m=N, k=K: lda=N >= k=K? N>=K. Not always.
//   Try: cublasGemmEx(N, N, N, M, K, B_ptr, N, A_ptr, K, C_ptr, N):
//   opA=N on B col[N,K] ld=N: B^T[N,K], m=N: lda=N >= m=N ✓
//   opB=N on A col[K,M] ld=K: A^T[K,M], n=M, k=K: ldb=K >= k=K ✓ ALWAYS
//   Result: B^T × A^T = (A @ B)^T stored as col[N,M] = row[M,N] = C_row[M,N] ✓
//
// FORMULA for NN: cublasGemmEx(N, N, N, M, K, B_ptr, N, A_ptr, K, C_ptr, N)
// Constraint: lda=N >= m=N ✓, ldb=K >= k=K ✓, ldc=N >= m=N ✓. ALWAYS VALID!
// ---------------------------------------------------------------------------
static bool fp8_ok(const GemmDesc& d, float alpha) {
    const long long L = 16LL << 20;
    return alpha == 1.0f && d.K >= 64 && d.N >= 64 && d.M >= 64 && d.K % 16 == 0 && d.N % 16 == 0 && d.M % 16 == 0 &&
           (long long)d.N * d.K <= L && (long long)d.M * d.K <= L;
}
void CublasLtEngine::gemm_nn(const GemmDesc& d,
                              const void* A, const void* B, void* C,
                              float alpha, float beta, cudaStream_t s) const {
    // C[M,N]_row = A[M,K]_row @ B[K,N]_row  (no transpose)
    if (fp8_bwd && fp8_ok(d, alpha)) { gemm_fp8_ex(d.M, d.N, d.K, (const bf16*)A, (const bf16*)B, (bf16*)C, false, true, true, beta, s); return; }
    CUBLAS_CHECK(cublasSetStream(handle, s));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,   // opA on B_ptr: B_row[K,N] as col[N,K] ld=N → B^T[N,K], no transpose
        CUBLAS_OP_N,   // opB on A_ptr: A_row[M,K] as col[K,M] ld=K → A^T[K,M], no transpose
        d.N, d.M, d.K, // m=N, n=M, k=K
        &alpha,
        B, CUDA_R_16BF, d.N,   // B_ptr col[N,K]: lda=N >= m=N ✓
        A, CUDA_R_16BF, d.K,   // A_ptr col[K,M]: ldb=K >= k=K ✓ always
        &beta,
        C, CUDA_R_16BF, d.N,   // C col[N,M] = C_row[M,N] ✓
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

}

// gemm_tn — C[M,N] = A[K,M]^T * B[K,N] (row-major; weight gradient dW = dY^T X)
// col-major view: C^T[N,M] = B_col[N,K](lda=N) * op_T(A_col[M,K], ldb=M)
void CublasLtEngine::gemm_tn(const GemmDesc& d,
                              const void* A, const void* B, void* C,
                              float alpha, float beta, cudaStream_t s) const {
    if (fp8_bwd && fp8_ok(d, alpha)) { gemm_fp8_ex(d.M, d.N, d.K, (const bf16*)A, (const bf16*)B, (bf16*)C, true, true, true, beta, s); return; }
    CUBLAS_CHECK(cublasSetStream(handle, s));
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        d.N, d.M, d.K,
        &alpha,
        B, CUDA_R_16BF, d.N,
        A, CUDA_R_16BF, d.M,
        &beta,
        C, CUDA_R_16BF, d.N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------------
// gemm_tn — TN GEMM: C[M,N] = A[K,M]^T * B[K,N] (A is transposed, row-major)
// For weight gradient: dW[M,N] += dY[K,M]^T * X[K,N]
// C[M,N] = A^T @ B where A_row[K,M], B_row[K,N].
// = (B^T @ A)^T = same trick:
//   cublasGemmEx(T, N, N, M, K, B_ptr, N, A_ptr, M, C_ptr, N)
//   wait: now K is the "K" dimension and M is the inner dim... let me re-derive.
// A^T[M,K] @ B[K,N] where A_row[K,M], B_row[K,N]:
// Substitute A' = A^T so this is A'[M,K] @ B[K,N] = NN form with A'=A_row[K,M] transposed:
// gemm_nn with A'=A_ptr but A_ptr stores A[K,M]_row → as col[M,K] ld=M → A^T[M,K]. 
// So use gemm_nn formula with: B=B_ptr (K,N)_row, A=A_ptr (K,M)_row, M=M, N=N, K=K:
// cublasGemmEx(N, N, N, M, K, B_ptr, N, A_ptr, M, C_ptr, N)
// lda=N >=m=N ✓, ldb=M >=k=K? M>=K. For dW of QKV: M=3072, K=NT=2048, M>K ✓. But not guaranteed.
//
// Better: use same (T,N) trick but with A's role:
// C[M,N] = A[K,M]^T @ B[K,N]:
// C^T[N,M] = B^T[N,K] @ A[K,M]
// cublasGemmEx(T, N, N, M, K, B_ptr, K, A_ptr, K, C_ptr, N)
// But this is the same as gemm(d, A, B, C) = A^T @ B^T^T = ... hmm no.
// Let me just check: for TN case C[M,N]=A[K,M]^T @ B[K,N]:
// Sub A→A^T in NT formula: C[M,N] = A[K,M]^T @ B[K,N] (B not transposed, A transposed)
// Actually rewrite as: C^T[N,M] = B^T[N,K] @ A[K,M]
// Now B^T[N,K] = B_row[K,N] viewed as col[N,K] ld=N, transposed ← no, opT would give B.
// B_ptr = B_row[K,N]: as col[N,K] ld=N → B^T[N,K]. opN → B^T[N,K] shape N×K.
// A_ptr = A_row[K,M]: as col[M,K] ld=M → A^T[M,K]. Wait: A_row[K,M] means K rows, M cols.
// A_ptr[i*M + j] = A[i,j]. As col[M,K] with ld=M: col[j,i] = A_ptr[j + i*M] = A_ptr[i*M+j]... hmm.
// A_row[K,M] stores element A[i,j]=A_ptr[i*M+j]. As col[M,K] with ld=M: col_el[a,b]=A_ptr[a+b*M].
// col_el[a,b] = A_ptr[a+b*M] = A_row[b,a]? Only if b*M+a = a+b*M → yes iff a≡a ✓.
// So A_col[M,K] ld=M el[a,b] = A_row[b,a] = A^T[a,b]? No: A^T[a,b] = A[b,a] = A_ptr[b*M+a].
// col[a,b] = A_ptr[a+b*M] = A_ptr[b*M+a] = A[b,a] = A^T[a,b] ✓. So A_row[K,M] as col[M,K] ld=M = A^T[M,K].
// opN → A^T[M,K], op second arg: shape k×n = K×M with ldb=M, constraint ldb>=k → M>=K.
// For dW of attn_out bwd: M=1024, K=NT=2048. M<K → fails.
//
// USE THE TN version of the TN trick:
// C[M,N] = A[K,M]^T @ B[K,N] rewrite as C^T[N,M] = B^T @ A.
// cublasGemmEx(T, N, N, M, K, B_ptr, K, A_ptr, M, C_ptr, N) where:
//   opA=T on B_ptr: B_row[K,N] as col[N,K] ld=N... wait I want to use lda=K not N.
// B_row[K,N] as col[K,N]? col[K,N] ld=K means el[i,j]=B_ptr[i+j*K]. B_row[i,j]=B_ptr[i*N+j].
// These match only if i+j*K = i*N+j for all i,j... only if K=N and N=1, not general.
// So B_row[K,N] cannot be naturally interpreted as col[K,N].
//
// OK this is too complex for the TN case. Just use gemm_nn with explicit transposition:
// dW[M,N] = dY^T @ X where dY_row[NT,M], X_row[NT,N], K=NT.
// = gemm_nn form: A_eff[M,NT] = dY^T, B_eff[NT,N] = X.
// For gemm_nn (A[M,K], B[K,N]): A=dY^T (M×NT), B=X (NT×N), K=NT:
// cublasGemmEx(N, N, N, M, K=NT, X_ptr, lda=N, dY_ptr, lda_b=M, C, N)
// dY_ptr stores dY_row[NT,M]: as col[M,NT] ld=M: el[a,b]=dY_ptr[a+b*M]=dY_row[b,a]=dY^T[a,b] ✓.
// X_ptr stores X_row[NT,N]: as col[N,NT] ld=N: el[a,b]=X_ptr[a+b*N]=X_row[b,a]=X^T[a,b].
// opN on X_ptr col[N,NT]: B^T_eff[N,NT]. m=N, lda=N >= m=N ✓.
// opN on dY_ptr col[M,NT]: A^T_eff[M,NT]. k=NT, ldb=M >= k=NT? M>=NT? Not if M<NT.
// Ugh. ldb constraint still fails sometimes.
//
// FINAL: just use gemm(d) for all cases with C = A @ B^T formula.
// For NN case: C[M,N] = A[M,K] @ B[K,N] = A[M,K] @ (B^T[N,K])^T → NT form ✓
// For TN case: C[M,N] = A[K,M]^T @ B[K,N] = AT @ B:
//   A^T_row[M,K] is not contiguous (A is K×M row-major, A^T would be M×K).
//   Use: C[M,N] = A[K,M]^T @ B[K,N] = (B^T[N,K] @ A[K,M])^T... no that's NM matrix.
// ---------------------------------------------------------------------------

// Forward wrappers use gemm() (NT form: C = A @ W^T where W is stored [out,in])
void CublasLtEngine::qkv_proj_fwd(const bf16* X, const bf16* W, bf16* Y,
                                    int NT, int C, cudaStream_t s) {
    GemmDesc d = {NT, 3*C, C};
    if (fp8_fwd) { gemm_fp8(NT, 3*C, C, X, W, Y, s); return; }
    gemm(d, X, W, Y, 1.0f, 0.0f, s);
}

void CublasLtEngine::out_proj_fwd(const bf16* X, const bf16* W, bf16* Y,
                                   int NT, int C, cudaStream_t s) {
    GemmDesc d = {NT, C, C};
    if (fp8_fwd) { gemm_fp8(NT, C, C, X, W, Y, s); return; }
    gemm(d, X, W, Y, 1.0f, 0.0f, s);
}

void CublasLtEngine::router_fwd(const bf16* X, const bf16* W, bf16* Y,
                                 int NT, int C, int E, cudaStream_t s) {
    GemmDesc d = {NT, E, C};
    gemm(d, X, W, Y, 1.0f, 0.0f, s);
}

void CublasLtEngine::expert_w1_fwd(const bf16* X, const bf16* W, bf16* Y,
                                     int M, int hi_, int C, cudaStream_t s) {
    GemmDesc d = {M, hi_, C};
    if (fp8_fwd) { gemm_fp8(M, hi_, C, X, W, Y, s); return; }
    gemm(d, X, W, Y, 1.0f, 0.0f, s);
}

void CublasLtEngine::expert_w2_fwd(const bf16* X, const bf16* W, bf16* Y,
                                     int M, int C, int hi_, cudaStream_t s) {
    GemmDesc d = {M, C, hi_};
    if (fp8_fwd) { gemm_fp8(M, C, hi_, X, W, Y, s); return; }
    gemm(d, X, W, Y, 1.0f, 0.0f, s);
}

void CublasLtEngine::lm_head(const bf16* X, const bf16* W, bf16* Y,
                               int NT, int C, int Vpad, cudaStream_t s) {
    GemmDesc d = {NT, Vpad, C};
    gemm(d, X, W, Y, 1.0f, 0.0f, s);
}

// Backward: dx = dY @ W  (NN, W stored [N,K] i.e. [out,in])
// dX[M,K] = dY[M,N] @ W[N,K]  → NT with W^T substituted: dX = dY @ W where W not transposed.
// Use gemm_nn: dX[M,K] = dY[M,N] @ W[N,K], M=NT, N=C, K=in_features
void CublasLtEngine::lm_head_bwd_dx(const bf16* dY, const bf16* W, bf16* dX,
                                      int NT, int C, int Vpad, cudaStream_t s) {
    // dX[NT,C] = dY[NT,Vp] @ W[Vp,C]  (NN, W not transposed)
    GemmDesc d = {NT, C, Vpad};
    gemm_nn(d, dY, W, dX, 1.0f, 0.0f, s);
}

void CublasLtEngine::lm_head_bwd_dw(const bf16* dY, const bf16* X, bf16* dW,
                                      int NT, int C, int Vpad, float beta, cudaStream_t s) {
    // dW[Vp,C] += dY^T[Vp,NT] @ X[NT,C]
    // = NT form: dW[Vp,C] = dY^T @ X = (X^T @ dY)^T
    // Treat as: dW = dY_row[NT,Vp]^T @ X_row[NT,C]
    // Rewrite as NT: dW = X_row^T @ dY_row ? No...
    // dW[Vp,C] = dY^T[Vp,NT] @ X[NT,C]
    // Use gemm_nn: A=dY^T[Vp,NT], B=X[NT,C] → but dY^T is not contiguous.
    // Rewrite: dW^T[C,Vp] = X^T[C,NT] @ dY[NT,Vp]
    // Use gemm NT: dW^T[C,Vp] = X[NT,C]^T_stored... no.
    // OK use: dW[Vp,C] = dY[NT,Vp]^T @ X[NT,C]
    //       = (gemm NT form) where M=Vp, N=C, K=NT, A=dY stored[NT,Vp], B=X stored[NT,C]... wait
    // We need A^T @ B. The NT gemm computes A @ B^T. So:
    // dW = dY^T @ X = (X^T @ dY)^T and using NT on X:
    // Let tmp = X @ dY^T : X[NT,C] @ dY^T[Vp,NT]^T = X @ dY (NT×Vp)? No dY^T^T = dY.
    // dW^T[C,Vp] = X^T[C,NT] @ dY[NT,Vp]. This is NT: M=C, N=Vp, K=NT, A=X, B=dY.
    // gemm(M=C, N=Vp, K=NT, A=X_ptr, B=dY_ptr, C=dW^T_ptr)
    // dW^T_ptr = dW_ptr since dW_col[C,Vp] at dW_ptr with ldc=C = dW_row[Vp,C] at dW_ptr with stride C.
    // ✓ same bytes! So compute dW^T via gemm NT and store at dW_ptr:
    GemmDesc d = {Vpad, C, NT};
    gemm_tn(d, dY, X, dW, 1.0f, beta, s);
}

void CublasLtEngine::qkv_bwd_dx(const bf16* dY, const bf16* W, bf16* dX,
                                  int NT, int C, cudaStream_t s) {
    // dX[NT,C] = dY[NT,3C] @ W[3C,C]  (NN)
    GemmDesc d = {NT, C, 3*C};
    gemm_nn(d, dY, W, dX, 1.0f, 0.0f, s);
}

void CublasLtEngine::out_bwd_dx(const bf16* dY, const bf16* W, bf16* dX,
                                  int NT, int C, cudaStream_t s) {
    // dX[NT,C] = dY[NT,C] @ W[C,C]  (NN)
    GemmDesc d = {NT, C, C};
    gemm_nn(d, dY, W, dX, 1.0f, 0.0f, s);
}

void CublasLtEngine::generic_bwd_dw(const bf16* dY, const bf16* X, bf16* dW,
                                     int M, int N, int K, float beta, cudaStream_t s) {
    // dW[M,N] += dY[K,M]^T @ X[K,N]
    // dW^T[N,M] = X^T[N,K] @ dY[K,M] — use gemm NT: dW^T = X @ dY^T → M_eff=N,N_eff=M,K_eff=K
    // gemm(M=N, N=M, K=K, A=X_ptr[K,N], B=dY_ptr[K,M], dW_ptr)
    GemmDesc d = {M, N, K};
    gemm_tn(d, dY, X, dW, 1.0f, beta, s);
}
