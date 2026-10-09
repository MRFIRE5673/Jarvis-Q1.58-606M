#pragma once
// =============================================================================
// cublaslt_engine.h — cuBLAS GEMM Manager (SM120 / RTX 5070)
// =============================================================================
// Uses cublasGemmEx (cublas_v2.h) with verified row-major formulas.
// Links against cublas64_13.dll (CUDA 13.3) — has SM120 BF16 TensorCore.
//
// Core NT formula (C = A @ B^T, all row-major):
//   cublasGemmEx(T, N, N, M, K, B_ptr, K, A_ptr, K, C_ptr, N)
//   Constraints always satisfied: lda=K>=K, ldb=K>=K, ldc=N>=N
//
// Core NN formula (C = A @ B, all row-major):
//   cublasGemmEx(N, N, N, M, K, B_ptr, N, A_ptr, K, C_ptr, N)
//   Constraints: lda=N>=N, ldb=K>=K, ldc=N>=N — always satisfied
// =============================================================================
#include <cublas_v2.h>
#include <cublasLt.h>
#include "cuda_engine.h"

// GemmDesc: shape for one matrix multiply
struct GemmDesc {
    int M = 0, N = 0, K = 0;
    bool algo_valid = false;  // unused, kept for compat
};

struct CublasLtEngine {
    cublasHandle_t handle    = nullptr;
    void*          workspace = nullptr;  // unused, kept for compat
    size_t         ws_bytes  = 0;

    // FP8 (E4M3) forward path: dynamic per-tensor scaling, BF16 output.
    // fp8_fwd is toggled by the engine: true during forward, false in backward.
    bool             fp8_fwd  = false;
    cublasLtHandle_t lt       = nullptr;
    void*            lt_ws    = nullptr;       // 32 MB
    uint8_t*         x8       = nullptr;       // activation scratch (16 MB)
    uint8_t*         w8       = nullptr;       // weight scratch (16 MB)
    float*           f8_stat  = nullptr;       // [amax_x, amax_w, inv_x, inv_w]
    // C[M,N] = X[M,K] * W[N,K]^T  via FP8, same layout as gemm()
    void gemm_fp8(int M, int N, int K, const bf16* X, const bf16* W, bf16* Y, cudaStream_t s) const;
    bool fp8_bwd = false;   // route eligible backward GEMMs (gemm_nn/gemm_tn) through FP8 (E5M2 grads)
    void gemm_fp8_ex(int M, int N, int K, const bf16* X, const bf16* W, bf16* Y,
                     bool xT, bool wT, bool grad, float beta, cudaStream_t s) const;

    // GemmDesc members (kept for API compatibility)
    GemmDesc d_qkv_fwd;
    GemmDesc d_out_fwd;
    GemmDesc d_router;
    GemmDesc d_exp_w1;
    GemmDesc d_exp_w2;
    GemmDesc d_lm_head_fwd;
    GemmDesc d_lm_head_bwd_dx;
    GemmDesc d_lm_head_bwd_dw;
    GemmDesc d_qkv_bwd_dx;
    GemmDesc d_out_bwd_dx;

    void init(const JarvisConfig& cfg);
    void destroy();
    void autotune_algorithms(const JarvisConfig& cfg);

    // NT GEMM: C[M,N] = alpha*A[M,K]*B[N,K]^T + beta*C[M,N]  (all row-major)
    void gemm(const GemmDesc& d,
              const void* A, const void* B, void* C,
              float alpha, float beta, cudaStream_t s) const;

    // NN GEMM: C[M,N] = alpha*A[M,K]*B[K,N] + beta*C[M,N]  (all row-major)
    void gemm_nn(const GemmDesc& d,
                 const void* A, const void* B, void* C,
                 float alpha, float beta, cudaStream_t s) const;

    // TN GEMM: C[M,N] = alpha*A[K,M]^T*B[K,N] + beta*C  (weight gradients)
    void gemm_tn(const GemmDesc& d,
                 const void* A, const void* B, void* C,
                 float alpha, float beta, cudaStream_t s) const;

    // Forward wrappers
    void qkv_proj_fwd(const bf16* X, const bf16* W, bf16* Y, int NT, int C, cudaStream_t s);
    void out_proj_fwd(const bf16* X, const bf16* W, bf16* Y, int NT, int C, cudaStream_t s);
    void router_fwd(const bf16* X, const bf16* W, bf16* Y, int NT, int C, int E, cudaStream_t s);
    void expert_w1_fwd(const bf16* X, const bf16* W, bf16* Y, int M, int hi, int C, cudaStream_t s);
    void expert_w2_fwd(const bf16* X, const bf16* W, bf16* Y, int M, int C, int hi, cudaStream_t s);
    void lm_head(const bf16* X, const bf16* W, bf16* Y, int NT, int C, int Vpad, cudaStream_t s);

    // Backward wrappers
    void lm_head_bwd_dx(const bf16* dY, const bf16* W, bf16* dX, int NT, int C, int Vpad, cudaStream_t s);
    void lm_head_bwd_dw(const bf16* dY, const bf16* X, bf16* dW, int NT, int C, int Vpad, float beta, cudaStream_t s);
    void qkv_bwd_dx(const bf16* dY, const bf16* W, bf16* dX, int NT, int C, cudaStream_t s);
    void out_bwd_dx(const bf16* dY, const bf16* W, bf16* dX, int NT, int C, cudaStream_t s);
    void generic_bwd_dw(const bf16* dY, const bf16* X, bf16* dW, int M, int N, int K, float beta, cudaStream_t s);
};

CublasLtEngine& get_cublaslt();
