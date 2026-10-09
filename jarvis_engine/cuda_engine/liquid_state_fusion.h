// =============================================================================
// liquid_state_fusion.h / liquid_state_fusion.cu — LSF SNN Membrane Dynamics
// =============================================================================
// Architecture (architecture.md §4.4):
//   H_t = α_dyn · H_{t-1} + (1 - α_dyn) · M_t   [EMA membrane update]
//   α_dyn = α_min + (α_max - α_min) · σ(-scale · Var(M))
//   Membrane state H persists across token positions (temporal memory).
//
// Implementation (RTX 5070):
//   alpha computation: parallel reduction for mean/var of M[NT, C]
//     → one block, stride loop, warp-reduce, single-thread final
//   EMA forward: serial T-step scan per (b, dim_block) pair
//     → cannot be parallelized across T (causal dependency)
//     → parallelized across batch B and C dimension blocks
//   EMA backward: BPTT reversed scan (reversed T, same structure)
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

void launch_compute_lsf_alpha(
    const __nv_bfloat16* M,  // [NT, C]
    float var_scale, float alpha_min, float alpha_max,
    float* alpha_out,         // [1] FP32 output
    int NT, int C, cudaStream_t s
);

// Forward: H_t = α·H_{t-1} + (1-α)·M_t  (serial scan over T)
// H_io [B, C]: initial membrane state (updated in-place to H_T)
// H_out [B, T, C]: all per-token membrane states (needed for backward)
void launch_lsf_fwd(
    const __nv_bfloat16* M,   // [B, T, C] = moe_out reshaped
    const float*          alpha,
    __nv_bfloat16*        H_io,  // [B, C] membrane (updated in-place)
    __nv_bfloat16*        H_out, // [B, T, C] all states for backward
    int B, int T, int C, cudaStream_t s
);

// Backward: BPTT through EMA scan (reversed T)
void launch_lsf_bwd(
    const __nv_bfloat16* grad_H_out, // [B, T, C] upstream grad
    const __nv_bfloat16* M,          // [B, T, C] saved input
    const __nv_bfloat16* H_out,      // [B, T, C] saved forward states
    const float*          alpha,
    __nv_bfloat16*        grad_M,    // [B, T, C] output
    float*                grad_alpha_acc, // [1] FP32 accumulate d_alpha
    int B, int T, int C, cudaStream_t s
);

// Reflective regularization penalty (architecture.md §4.5)
void launch_reflective_penalty(
    const __nv_bfloat16* M,     // [NT, C] moe_out for stats
    float*       mu_running,     // [1] EMA mean (updated in-place)
    float*       l_reflect,      // [1] output loss
    float lambda, float tau_max, float ema_decay,
    int NT, int C, cudaStream_t s
);
