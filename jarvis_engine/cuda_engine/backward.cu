// =============================================================================
// backward.cu — Backward Pass Gradient Utilities
// =============================================================================
// Implements gradient flow helpers that don't belong in specific sub-modules:
//   1. Accumulate gradient into accumulator buffer (BF16 += BF16, scaled)
//   2. Copy gradient with scale (for accumulation step normalization)
//   3. Multi-tensor grad norm accumulation
//   4. Router backward (gating logits gradient)
// =============================================================================
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 1. Scaled gradient accumulation: acc += scale × grad (vec4 BF16)
//    Used to accumulate gradients across accum_steps microsteps.
//    scale = 1.0 / accum_steps so grads are averaged before optimizer step.
// ---------------------------------------------------------------------------
__global__ void grad_accum_vec4_kernel(
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16*       __restrict__ acc,
    float scale,
    int N4
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;
    int base = i * 4;

    __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(grad + base);
    __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(grad + base + 2);
    __nv_bfloat162 a01 = *reinterpret_cast<const __nv_bfloat162*>(acc  + base);
    __nv_bfloat162 a23 = *reinterpret_cast<const __nv_bfloat162*>(acc  + base + 2);

    float2 gf01 = __bfloat1622float2(g01);
    float2 gf23 = __bfloat1622float2(g23);
    float2 af01 = __bfloat1622float2(a01);
    float2 af23 = __bfloat1622float2(a23);

    af01.x += scale * gf01.x;  af01.y += scale * gf01.y;
    af23.x += scale * gf23.x;  af23.y += scale * gf23.y;

    *reinterpret_cast<__nv_bfloat162*>(acc + base)     = __float22bfloat162_rn(af01);
    *reinterpret_cast<__nv_bfloat162*>(acc + base + 2) = __float22bfloat162_rn(af23);
}

void launch_grad_accum(const __nv_bfloat16* grad, __nv_bfloat16* acc,
                        float scale, int N, cudaStream_t s) {
    int N4 = N / 4;
    grad_accum_vec4_kernel<<<(N4 + 255) / 256, 256, 0, s>>>(grad, acc, scale, N4);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 2. Grad scale + copy: dst = scale × src (vec4 BF16)
//    Used for first microstep where acc should be overwritten (not accumulated).
// ---------------------------------------------------------------------------
__global__ void grad_scale_copy_vec4(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*       __restrict__ dst,
    float scale, int N4
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N4) return;
    int base = i * 4;
    __nv_bfloat162 s01 = *reinterpret_cast<const __nv_bfloat162*>(src + base);
    __nv_bfloat162 s23 = *reinterpret_cast<const __nv_bfloat162*>(src + base + 2);
    float2 f01 = __bfloat1622float2(s01);
    float2 f23 = __bfloat1622float2(s23);
    f01.x *= scale; f01.y *= scale;
    f23.x *= scale; f23.y *= scale;
    *reinterpret_cast<__nv_bfloat162*>(dst + base)     = __float22bfloat162_rn(f01);
    *reinterpret_cast<__nv_bfloat162*>(dst + base + 2) = __float22bfloat162_rn(f23);
}

void launch_grad_scale_copy(const __nv_bfloat16* src, __nv_bfloat16* dst,
                             float scale, int N, cudaStream_t s) {
    int N4 = N / 4;
    grad_scale_copy_vec4<<<(N4 + 255) / 256, 256, 0, s>>>(src, dst, scale, N4);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 3. Multi-tensor grad norm accumulation (into pre-zeroed output)
//    Identical to cuda_engine_kernels.cu launch_grad_norm_sq but
//    exposed here for use in full_engine.cu's backward pass loop.
// ---------------------------------------------------------------------------
// (Declared in cuda_engine_kernels.cu; extern declared below)
extern void launch_grad_norm_sq(const __nv_bfloat16* g, float* out, int N, cudaStream_t s);

// Accumulate grad norm of FP32 gradient (for gamma scalars)
__global__ void grad_norm_sq_fp32_kernel(const float* g, float* out, int N) {
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        local += g[i] * g[i];
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, local);
}

void launch_grad_norm_sq_fp32(const float* g, float* out, int N, cudaStream_t s) {
    int GRID = min((N + 255) / 256, 512);
    grad_norm_sq_fp32_kernel<<<GRID, 256, 0, s>>>(g, out, N);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 4. Router backward
//    Gate logits grad: d_logits = softmax_grad * d_gate
//    For sparse top-2: only two experts receive gradient per token.
//    d_logits[tok, e] = d_gate[tok, k] × d(softmax[k]) for selected k,
//                     = 0 for non-selected experts.
//    This is a scalar kernel (NT × E = 2048 × 4 = tiny).
// ---------------------------------------------------------------------------
__global__ void router_bwd_kernel(
    const float* __restrict__ probs,     // [NT, E] softmax probs
    const float* __restrict__ d_gates,   // [NT, K] gate gradient
    const int*   __restrict__ topk_idx,  // [NT, K]
    __nv_bfloat16* __restrict__ d_logits,// [NT, E]
    int NT, int E, int K
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= NT) return;

    float p[4] = {0};  // E=4
    float d_p[4] = {0};

    for (int e = 0; e < E; ++e) p[e] = probs[tok * E + e];

    // Distribute d_gate through softmax Jacobian (dense form)
    for (int k = 0; k < K; ++k) {
        int e_sel = topk_idx[tok * K + k];
        float dg   = d_gates[tok * K + k];
        // softmax backward: d_logit[e] = p[e] * (delta_{e,e_sel} - p[e_sel]) * dg
        for (int e = 0; e < E; ++e) {
            float delta = (e == e_sel) ? 1.0f : 0.0f;
            d_p[e] += p[e] * (delta - p[e_sel]) * dg;
        }
    }

    for (int e = 0; e < E; ++e)
        d_logits[tok * E + e] = __float2bfloat16(d_p[e]);
}

void launch_router_bwd(
    const float* probs, const float* d_gates, const int* topk_idx,
    __nv_bfloat16* d_logits,
    int NT, int E, int K, cudaStream_t s
) {
    router_bwd_kernel<<<(NT + 255) / 256, 256, 0, s>>>(
        probs, d_gates, topk_idx, d_logits, NT, E, K);
    CUDA_CHECK_LAST();
}
