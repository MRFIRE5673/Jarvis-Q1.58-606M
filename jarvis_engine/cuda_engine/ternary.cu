// =============================================================================
// ternary.cu — Q1.58 Spiking Ternary Quantization (BitNet b1.58)
// =============================================================================
// Architecture (architecture.md §4.2):
//   α = mean(|W|)                            AbsMean scaling
//   W_t = clamp(round(W_fp32 / (α+1e-8)), -1, 1)   ternary projection
//   W_q = W_t × α                            scaled back to BF16
//   STE: ∂L/∂W ≈ ∂L/∂W_q · 1{|W| ≤ 1.0}   straight-through estimator
//
// Memory.md §4: Quantizing without AbsMean causes 100% zero-weight collapse
//               (initial weight σ ≈ 0.025 << 0.5 threshold).
//               AbsMean is ESSENTIAL, not optional.
//
// Implementation:
//   abs_mean kernel: two-phase reduction (no per-block division inside atomic)
//     Phase 1: per-thread local sum of |W|
//     Phase 2: warp reduce → one atomic add per warp
//     Phase 3: separate kernel divides by N (avoids non-associative floats)
//
//   quantize kernel: vec4 (4 FP32 per thread) — compute-bound, fine to vectorize
//     (Unlike transcendentals, rintf + clamp is cheap arithmetic)
//
//   STE backward: scalar (per element) — simple comparison, no ILP concern
// =============================================================================
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp reduce sum helper
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// 1. ABS MEAN — Phase 1+2: accumulate sum(|W|)
//    Phase 3 (separate kernel): divide by N
//    IMPORTANT: We do NOT divide by N inside the atomic add (non-associative).
// ---------------------------------------------------------------------------
__global__ void abs_sum_kernel(
    const float* __restrict__ W,
    float*       __restrict__ out,  // pre-zeroed
    int N
) {
    float local = 0.0f;
    // Stride loop covers all elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        local += fabsf(W[i]);
    local = warp_sum(local);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, local);
}

__global__ void div_by_N_kernel(float* val, int N) {
    if (threadIdx.x == 0) *val /= (float)N;
}

void launch_ternary_abs_mean(const float* W, float* alpha, int N, cudaStream_t s) {
    cudaMemsetAsync(alpha, 0, sizeof(float), s);
    int BLOCK = 256;
    int GRID  = min((N + BLOCK - 1) / BLOCK, 2048);
    abs_sum_kernel<<<GRID, BLOCK, 0, s>>>(W, alpha, N);
    div_by_N_kernel<<<1, 32, 0, s>>>(alpha, N);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 2. TERNARY QUANTIZE FORWARD — vec4 FP32 per thread
//    W_q[i] = clamp(round(W[i] / (α+1e-8)), -1, 1) × α  → BF16
// ---------------------------------------------------------------------------
__global__ void ternary_quantize_kernel(
    const float* __restrict__ W_fp32,
    const float* __restrict__ alpha,
    __nv_bfloat16* __restrict__ W_bf16,
    int N
) {
    float a = *alpha + 1e-8f;
    float a_inv = 1.0f / a;

    // vec4: process 4 FP32 per thread
    int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(W_fp32 + i4);
        float4 q;
        q.x = fmaxf(-1.0f, fminf(1.0f, rintf(v.x * a_inv))) * *alpha;
        q.y = fmaxf(-1.0f, fminf(1.0f, rintf(v.y * a_inv))) * *alpha;
        q.z = fmaxf(-1.0f, fminf(1.0f, rintf(v.z * a_inv))) * *alpha;
        q.w = fmaxf(-1.0f, fminf(1.0f, rintf(v.w * a_inv))) * *alpha;
        // Pack 4 BF16 into 2×bfloat162
        __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(q.x, q.y));
        __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(q.z, q.w));
        *reinterpret_cast<__nv_bfloat162*>(W_bf16 + i4)     = lo;
        *reinterpret_cast<__nv_bfloat162*>(W_bf16 + i4 + 2) = hi;
    } else {
        for (int i = i4; i < N && i < i4 + 4; ++i) {
            float q = fmaxf(-1.0f, fminf(1.0f, rintf(W_fp32[i] * a_inv))) * *alpha;
            W_bf16[i] = __float2bfloat16(q);
        }
    }
}

void launch_ternary_quantize(const float* W_fp32, const float* alpha,
                              __nv_bfloat16* W_bf16, int N, cudaStream_t s) {
    int BLOCK = 256;
    int GRID  = ((N / 4) + BLOCK - 1) / BLOCK;
    ternary_quantize_kernel<<<GRID, BLOCK, 0, s>>>(W_fp32, alpha, W_bf16, N);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 3. STE BACKWARD — SCALAR (simple comparison, not worth vectorizing)
//    grad_W_fp32[i] = grad_W_q[i] × 1{|W_fp32[i]| ≤ 1.0}
// ---------------------------------------------------------------------------
__global__ void ternary_ste_bwd_kernel(
    const float*           __restrict__ W_fp32,
    const float*           __restrict__ alpha,
    const __nv_bfloat16*   __restrict__ grad_W_q,
    float*                 __restrict__ grad_W_fp32,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __bfloat162float(grad_W_q[i]);
    grad_W_fp32[i] = (fabsf(W_fp32[i]) <= *alpha) ? g : 0.0f;
}

void launch_ternary_ste_bwd(const float* W_fp32, const float* alpha,
                             const __nv_bfloat16* grad_W_q,
                             float* grad_W_fp32, int N, cudaStream_t s) {
    ternary_ste_bwd_kernel<<<(N + 255) / 256, 256, 0, s>>>(W_fp32, alpha, grad_W_q, grad_W_fp32, N);
    CUDA_CHECK_LAST();
}

// In-place straight-through estimator mask on BF16 gradient buffer:
// grad_W_q[i] = (|W_fp32[i]| <= 1.0) ? grad_W_q[i] : 0.0
__global__ void ternary_ste_mask_kernel(
    const float*         __restrict__ W_fp32,
    const float*         __restrict__ alpha,
    __nv_bfloat16*       __restrict__ grad_W_q,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (fabsf(W_fp32[i]) > *alpha) {
        grad_W_q[i] = __float2bfloat16(0.0f);
    }
}

void launch_ternary_ste_mask(const float* W_fp32, const float* alpha,
                             __nv_bfloat16* grad_W_q, int N, cudaStream_t s) {
    ternary_ste_mask_kernel<<<(N + 255) / 256, 256, 0, s>>>(W_fp32, alpha, grad_W_q, N);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 4. QUANTIZE ALL LAYERS — runs before forward pass each step
//    Quantizes QKV, Out, ExpertW1, ExpertW2 weights for all 24 layers.
//    Called once per optimizer step (not per microstep).
// ---------------------------------------------------------------------------
void quantize_all_layer_weights(
    __nv_bfloat16** qkv_bf16, float** qkv_fp32, float* alpha_buf,
    __nv_bfloat16** out_bf16, float** out_fp32,
    __nv_bfloat16** w1_bf16,  float** w1_fp32,
    __nv_bfloat16** w2_bf16,  float** w2_fp32,
    int L, int C, int hi, int E, cudaStream_t s
) {
    for (int l = 0; l < L; ++l) {
        int N_qkv = 3 * C * C;
        int N_out = C * C;
        int N_w1  = E * hi * C;
        int N_w2  = E * C * hi;

        // Each weight matrix gets its own alpha (per-tensor AbsMean)
        // For simplicity, alpha_buf is a scratch float[1] reused per launch
        // Production: pre-allocate alpha_buf[L × 4] for full parallelism

        launch_ternary_abs_mean(qkv_fp32[l], alpha_buf, N_qkv, s);
        launch_ternary_quantize(qkv_fp32[l], alpha_buf, qkv_bf16[l], N_qkv, s);

        launch_ternary_abs_mean(out_fp32[l], alpha_buf, N_out, s);
        launch_ternary_quantize(out_fp32[l], alpha_buf, out_bf16[l], N_out, s);

        launch_ternary_abs_mean(w1_fp32[l], alpha_buf, N_w1, s);
        launch_ternary_quantize(w1_fp32[l], alpha_buf, w1_bf16[l], N_w1, s);

        launch_ternary_abs_mean(w2_fp32[l], alpha_buf, N_w2, s);
        launch_ternary_quantize(w2_fp32[l], alpha_buf, w2_bf16[l], N_w2, s);
    }
}
