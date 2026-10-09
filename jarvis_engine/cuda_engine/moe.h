// =============================================================================
// moe.h / moe.cu — Sparse MoE Kernels
// =============================================================================
// Architecture (architecture.md §4.3):
//   E=4 experts, K=2 Top-2 routing (LOCKED)
//   Router: x·W_router + N(0,σ²) → softmax → top-2 → normalize gate weights
//   Load balance loss: α_bal × E × Σ f_i × P_i
//   Expert FFN: W2(GELU(W1·x))  where W1:[hi,C] W2:[C,hi]  hi=2048
//
// Kernel optimization notes (RTX 5070 / 672 GB/s GDDR7):
//   Top-2 gating (E=4): fully unrolled in registers — no SMEM needed
//   Dispatch gather: vec8 uint4 (8×BF16 = 128-bit) — memory-bound, max BW
//   In-place GELU fwd: vec8 — memory-bound at W1 output buffer
//   GELU backward: SCALAR — transcendental chain (tanh + sech²) kills ILP if vec8
//   Scatter-combine: sorted by expert to enable coalesced writes per expert
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Top-K gating: logits[NT,E] → probs[NT,E], topk_idx[NT,K], topk_gates[NT,K]
void launch_top2_gating(
    const __nv_bfloat16* logits, float* probs,
    int* topk_idx, float* topk_gates,
    int NT, int E, cudaStream_t s
);

// Dispatch gather: x[NT,C] + topk_idx[NT,K] → dispatch_x[NTK,C]
void launch_dispatch_gather(
    const __nv_bfloat16* x, const int* topk_idx,
    __nv_bfloat16* dispatch_x,
    int NT, int K, int C, cudaStream_t s
);

// In-place GELU forward: x[NTK,hi] → x (overwritten)
void launch_gelu_fwd_inplace(
    __nv_bfloat16* x, int N, cudaStream_t s
);

// Fused GELU forward + stash write: h1_in → (stash=pre-GELU copy, h1_out=GELU)
// Eliminates a D2D memcpy per layer (16 MB saved per layer)
void launch_gelu_fwd_stash(
    const __nv_bfloat16* h1_in,
    __nv_bfloat16*       stash,
    __nv_bfloat16*       h1_out,
    int N, cudaStream_t s
);

// GELU backward: x_pre[NTK,hi], grad_out[NTK,hi] → grad_x[NTK,hi]
// SCALAR — do not vectorize (transcendental ILP constraint)
void launch_gelu_bwd(
    const __nv_bfloat16* x_pre, const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_x, int N, cudaStream_t s
);

// Scatter-combine: expert_out[NTK,C] + topk_idx/gates → moe_out[NT,C]
// Adds residual x (moe_out = sum(gate * expert_out) + x_residual)
void launch_scatter_combine(
    const __nv_bfloat16* expert_out,
    const __nv_bfloat16* x_residual,
    const int*   topk_idx,
    const float* topk_gates,
    __nv_bfloat16* moe_out,
    int NT, int K, int C, cudaStream_t s
);

// Load balance loss: probs[NT,E], topk_idx[NT,K] → l_balance[1]
void launch_load_balance_loss(
    const float* probs, const int* topk_idx,
    float* l_balance,
    float alpha_bal, int NT, int E, cudaStream_t s
);
