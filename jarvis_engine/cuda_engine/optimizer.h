// =============================================================================
// optimizer.h — Fused AdamW (vec4, graph-capturable)
// =============================================================================
// All scalars (bc1, bc2, clip_coef) are device pointers — no CPU-GPU sync.
// prepare_optimizer_step launches a tiny kernel that computes these values
// on device from d_step_count and d_grad_norm_sq. Fully graph-capturable.
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Prepare scalars on device (single-thread kernel): bc1, bc2, clip_coef
// Must be called BEFORE launch_adamw_fused in each optimizer step.
// d_step_count is an int on device — atomically incremented each call.
// ---------------------------------------------------------------------------
void launch_prepare_optimizer_step(
    int*         d_step_count,
    float*       d_bc1,
    float*       d_bc2,
    float*       d_clip_coef,
    float*       d_lr,
    const float* d_grad_norm_sq,
    float beta1, float beta2, float grad_clip, float base_lr,
    int warmup_steps, int decay_steps, float min_lr_ratio,
    cudaStream_t s
);

// ---------------------------------------------------------------------------
// Single-tensor fused AdamW (vec4 FP32)
// bc1, bc2, clip_coef are device pointers (filled by prepare_optimizer_step).
// No host-side sync required — fully graph-capturable.
// ---------------------------------------------------------------------------
void launch_adamw_fused(
    float*         p_fp32,
    __nv_bfloat16* p_bf16,
    float*         m,
    float*         v,
    const __nv_bfloat16* grad,
    float lr, float beta1, float beta2,
    float eps, float weight_decay,
    const float*   d_bc1,
    const float*   d_bc2,
    const float*   d_clip_coef,
    const float*   d_lr,
    int N, cudaStream_t s
);

// ---------------------------------------------------------------------------
// Descriptor for grouped launch (legacy, not used in capture path)
// ---------------------------------------------------------------------------
struct AdamWDesc {
    float*         p_fp32;
    __nv_bfloat16* p_bf16;
    float*         m;
    float*         v;
    const __nv_bfloat16* grad;
    int            N;
};

void launch_adamw_group(
    const AdamWDesc* descs,
    int num_tensors,
    float lr, float beta1, float beta2,
    float eps, float weight_decay,
    float bias_corr1, float bias_corr2,
    float clip_coef,
    cudaStream_t s
);
