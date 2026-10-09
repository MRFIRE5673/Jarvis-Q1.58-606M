// =============================================================================
// optimizer.cu — Fused AdamW (vec4, runnable first)
// =============================================================================
// vec4 = float4 for FP32 m/v/p (4 elements per transaction).
// BF16 grad/weight use bfloat162 pairs (2 BF16 per instruction).
// This is the runnable baseline. Optimize to vec8 after it trains correctly.
// =============================================================================
#include "optimizer.h"
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Core AdamW kernel — vec4, one thread handles 4 elements
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Device-side AdamW with all scalars passed via uniform device memory
// bc1, bc2, clip_coef are device pointers to single float scalars.
// This makes the kernel graph-capturable (no CPU readback needed).
// ---------------------------------------------------------------------------
__global__ void adamw_vec4_kernel(
    float*         __restrict__ p_fp32,
    __nv_bfloat16* __restrict__ p_bf16,
    float*         __restrict__ m,
    float*         __restrict__ v,
    const __nv_bfloat16* __restrict__ grad,
    float lr, float beta1, float beta2,
    float eps, float wd,
    const float* __restrict__ d_bc1,
    const float* __restrict__ d_bc2,
    const float* __restrict__ d_clip_coef,
    int N4
) {
    // Load scalars once per block via uniform read (compiler hoists to smem)
    float bc1      = *d_bc1;
    float bc2      = *d_bc2;
    float clip_coef = *d_clip_coef;

    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= N4) return;
    int base = i4 * 4;

    float4 mp = *reinterpret_cast<const float4*>(m      + base);
    float4 vp = *reinterpret_cast<const float4*>(v      + base);
    float4 pp = *reinterpret_cast<const float4*>(p_fp32 + base);

    __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(grad + base);
    __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(grad + base + 2);
    float2 gf01 = __bfloat1622float2(g01);
    float2 gf23 = __bfloat1622float2(g23);
    float g[4] = {
        gf01.x * clip_coef, gf01.y * clip_coef,
        gf23.x * clip_coef, gf23.y * clip_coef
    };

    float* mf = reinterpret_cast<float*>(&mp);
    float* vf = reinterpret_cast<float*>(&vp);
    float* pf = reinterpret_cast<float*>(&pp);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        mf[j] = beta1 * mf[j] + (1.0f - beta1) * g[j];
        vf[j] = beta2 * vf[j] + (1.0f - beta2) * g[j] * g[j];
        float m_hat = mf[j] * bc1;
        float v_hat = vf[j] * bc2;
        pf[j] -= lr * (m_hat * rsqrtf(v_hat + eps * eps) + wd * pf[j]);
    }

    *reinterpret_cast<float4*>(m      + base) = mp;
    *reinterpret_cast<float4*>(v      + base) = vp;
    *reinterpret_cast<float4*>(p_fp32 + base) = pp;

    *reinterpret_cast<__nv_bfloat162*>(p_bf16 + base)     =
        __float22bfloat162_rn(make_float2(pf[0], pf[1]));
    *reinterpret_cast<__nv_bfloat162*>(p_bf16 + base + 2) =
        __float22bfloat162_rn(make_float2(pf[2], pf[3]));
}

// ---------------------------------------------------------------------------
// Small single-thread kernel: computes bc1, bc2 from step_count on device.
// Avoids any CPU-GPU sync during graph capture.
// ---------------------------------------------------------------------------
__global__ void prepare_optimizer_step_kernel(
    int* d_step_count,
    float* d_bc1, float* d_bc2,
    float* d_clip_coef,
    const float* d_grad_norm_sq,
    float beta1, float beta2, float grad_clip
) {
    if (threadIdx.x != 0) return;
    int t = atomicAdd(d_step_count, 1) + 1;  // pre-increment (t = new step)
    float b1t = 1.0f;
    float b2t = 1.0f;
    // Use fast iterative approach: beta^t computed on device
    // For small t this is fine; use __powf for large t
    *d_bc1 = 1.0f / (1.0f - __powf(beta1, (float)t));
    *d_bc2 = 1.0f / (1.0f - __powf(beta2, (float)t));
    (void)b1t; (void)b2t;

    // Compute clip coef: min(grad_clip / (sqrt(grad_norm_sq) + 1e-6), 1.0)
    float gnorm = sqrtf(*d_grad_norm_sq);
    *d_clip_coef = fminf(grad_clip / (gnorm + 1e-6f), 1.0f);
}

void launch_prepare_optimizer_step(
    int* d_step_count,
    float* d_bc1, float* d_bc2,
    float* d_clip_coef,
    const float* d_grad_norm_sq,
    float beta1, float beta2, float grad_clip,
    cudaStream_t s
) {
    prepare_optimizer_step_kernel<<<1, 32, 0, s>>>(
        d_step_count, d_bc1, d_bc2, d_clip_coef,
        d_grad_norm_sq, beta1, beta2, grad_clip);
    CUDA_CHECK_LAST();
}

void launch_adamw_fused(
    float* p_fp32, __nv_bfloat16* p_bf16,
    float* m, float* v,
    const __nv_bfloat16* grad,
    float lr, float beta1, float beta2,
    float eps, float wd,
    const float* d_bc1, const float* d_bc2,
    const float* d_clip_coef,
    int N, cudaStream_t s
) {
    int N4    = N / 4;
    int BLOCK = 256;
    int GRID  = (N4 + BLOCK - 1) / BLOCK;
    adamw_vec4_kernel<<<GRID, BLOCK, 0, s>>>(
        p_fp32, p_bf16, m, v, grad,
        lr, beta1, beta2, eps, wd,
        d_bc1, d_bc2, d_clip_coef, N4);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// Grouped AdamW — one 2D kernel for multiple tensors.
// blockIdx.y = tensor index, blockIdx.x = element tile.
// The device descriptor array is copied once per optimizer step.
// ---------------------------------------------------------------------------
__global__ void adamw_grouped_vec4_kernel(
    AdamWDesc* __restrict__ descs, int num_tensors,
    float lr, float beta1, float beta2,
    float eps, float wd, float bc1, float bc2, float clip_coef
) {
    int tid = blockIdx.y;
    if (tid >= num_tensors) return;

    AdamWDesc& d = descs[tid];
    int N4 = d.N / 4;
    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= N4) return;
    int base = i4 * 4;

    float4 mp = *reinterpret_cast<const float4*>(d.m      + base);
    float4 vp = *reinterpret_cast<const float4*>(d.v      + base);
    float4 pp = *reinterpret_cast<const float4*>(d.p_fp32 + base);

    __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162*>(d.grad + base);
    __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162*>(d.grad + base + 2);
    float2 gf01 = __bfloat1622float2(g01);
    float2 gf23 = __bfloat1622float2(g23);
    float g[4] = {
        gf01.x * clip_coef, gf01.y * clip_coef,
        gf23.x * clip_coef, gf23.y * clip_coef
    };

    float* mf = reinterpret_cast<float*>(&mp);
    float* vf = reinterpret_cast<float*>(&vp);
    float* pf = reinterpret_cast<float*>(&pp);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        mf[j] = beta1 * mf[j] + (1.0f - beta1) * g[j];
        vf[j] = beta2 * vf[j] + (1.0f - beta2) * g[j] * g[j];
        float m_hat = mf[j] * bc1;
        float v_hat = vf[j] * bc2;
        pf[j] -= lr * (m_hat * rsqrtf(v_hat + eps * eps) + wd * pf[j]);
    }

    *reinterpret_cast<float4*>(d.m      + base) = mp;
    *reinterpret_cast<float4*>(d.v      + base) = vp;
    *reinterpret_cast<float4*>(d.p_fp32 + base) = pp;

    *reinterpret_cast<__nv_bfloat162*>(d.p_bf16 + base)     =
        __float22bfloat162_rn(make_float2(pf[0], pf[1]));
    *reinterpret_cast<__nv_bfloat162*>(d.p_bf16 + base + 2) =
        __float22bfloat162_rn(make_float2(pf[2], pf[3]));
}

void launch_adamw_group(
    const AdamWDesc* descs, int num_tensors,
    float lr, float beta1, float beta2,
    float eps, float wd, float bc1, float bc2, float clip_coef,
    cudaStream_t s
) {
    if (num_tensors == 0) return;

    // Find largest tensor to size X grid
    int max_N4 = 0;
    for (int i = 0; i < num_tensors; ++i)
        max_N4 = max(max_N4, descs[i].N / 4);

    // Upload descriptor array to device scratch
    AdamWDesc* d_descs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_descs, num_tensors * sizeof(AdamWDesc)));
    CUDA_CHECK(cudaMemcpyAsync(d_descs, descs,
        num_tensors * sizeof(AdamWDesc), cudaMemcpyHostToDevice, s));

    int BLOCK = 256;
    dim3 grid((max_N4 + BLOCK - 1) / BLOCK, num_tensors);
    adamw_grouped_vec4_kernel<<<grid, BLOCK, 0, s>>>(
        d_descs, num_tensors, lr, beta1, beta2, eps, wd, bc1, bc2, clip_coef);
    CUDA_CHECK_LAST();

    // NOTE: d_descs is freed asynchronously after kernel completion.
    // For graph capture: this function is not used in the graph path.
    // CUDA will free the memory when the stream reaches this point.
    // For now, we keep a small leak-free approach: just don't free during capture.
    // (launch_adamw_group is not called from capture_graph)
    cudaFree(d_descs);  // safe: no sync needed, cudaFree is not async
}
