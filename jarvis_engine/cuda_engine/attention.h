// =============================================================================
// attention.h + attention.cu — O(N) Associative Linear Attention
// =============================================================================
// Architecture (architecture.md §4.1):
//   S_t = γ · S_{t-1} + V_t ⊗ K_t^T       [state update]
//   Z_t = S_t · Q_t                         [query against state]
//   Feature map: ELU(x)+1 (Katharopoulos 2020) on both Q and K
//   RoPE on Q and K (Su et al. 2021)
//   Chunk size = 64 tokens
//   State S ∈ R^{B,H,D,D} where D=64 → 4×16×64×64×2B = 0.524 MB (L2-resident)
//
// Kernel design for RTX 5070 (SM120, 48 SMs):
//   - ELU+RoPE: fused into single kernel, each thread handles D/2=32 elements
//     for one (b,h,t). Scalar ELU+RoPE (not vectorized) because:
//     a) Compute-bound: RoPE requires sinf/cosf transcendentals
//     b) Serializing 8 transcendentals per thread degrades SM ILP
//   - Chunked state update: per-(b,h) block, serial over chunk tokens
//     FP32 accumulation for S to avoid BF16 precision loss in outer product
//   - Intra-chunk attention: cuBLASLt handles this via scored GEMM
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Launch: Fused ELU+1 and RoPE on Q and K
// q_in/k_in: [B,H,T,D] BF16 → q_out/k_out: [B,H,T,D] BF16
// cos_tab/sin_tab: precomputed [T, D/2] FP32 (only half needed for RoPE)
void launch_fused_elu_rope(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in,
    const float* cos_tab,       const float* sin_tab,
    __nv_bfloat16* q_out,       __nv_bfloat16* k_out,
    int B, int H, int T, int D,
    cudaStream_t stream
);

// Backward through ELU+RoPE
void launch_fused_elu_rope_bwd(
    const __nv_bfloat16* dq_out, const __nv_bfloat16* dk_out,
    const __nv_bfloat16* q_in,   const __nv_bfloat16* k_in,
    const float* cos_tab,         const float* sin_tab,
    __nv_bfloat16* dq_in,         __nv_bfloat16* dk_in,
    int B, int H, int T, int D,
    cudaStream_t stream
);

// Precompute RoPE tables [T, D] FP32 on device
// Only the first D/2 columns are unique; the second D/2 are mirrored
void launch_build_rope_tables(
    float* cos_tab, float* sin_tab,
    int T, int D, float rope_base,
    cudaStream_t stream
);

// Associative state update forward: processes one full sequence in chunks
// S_in [B,H,D,D] BF16 → S_out [B,H,D,D] BF16 (updated in-place)
// attn_out [B,H,T,D] BF16 = output for each token
void launch_assoc_attn_fwd(
    const __nv_bfloat16* q,         // [B,H,T,D] post ELU+RoPE
    const __nv_bfloat16* k,         // [B,H,T,D] post ELU+RoPE
    const __nv_bfloat16* v,         // [B,H,T,D]
    const float*          log_gamma, // [H] log(sigmoid(gamma_raw))
    __nv_bfloat16*        attn_out,  // [B,H,T,D]
    __nv_bfloat16*        S,         // [B,H,D,D] state (updated in-place)
    int B, int H, int T, int D, int chunk_size,
    cudaStream_t stream
);
