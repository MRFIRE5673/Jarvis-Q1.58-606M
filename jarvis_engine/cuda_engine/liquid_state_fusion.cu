// =============================================================================
// liquid_state_fusion.cu — LSF SNN Membrane Dynamics Implementation
// =============================================================================
#include "liquid_state_fusion.h"
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ===========================================================================
// 1. COMPUTE DYNAMIC ALPHA FROM M VARIANCE
//    α_dyn = α_min + (α_max - α_min) · σ(−scale · Var(M))
//    Parallel mean + var reduction over all NT×C elements.
//    Grid: (num_blocks,), Block: (256,)
// ===========================================================================
__global__ void compute_lsf_alpha_partials_kernel(
    const __nv_bfloat16* __restrict__ M,
    float* __restrict__ alpha_stats,
    int N  // NT * C
) {
    __shared__ float sums[8], squares[8];
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float v = __bfloat162float(M[i]);
        local_sum += v;
        local_sq  += v * v;
    }
    local_sum = warp_sum(local_sum);
    local_sq  = warp_sum(local_sq);

    if ((threadIdx.x & 31) == 0) {
        sums[threadIdx.x >> 5] = local_sum;
        squares[threadIdx.x >> 5] = local_sq;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f, sq = 0.0f;
        for (int w = 0; w < 8; ++w) { sum += sums[w]; sq += squares[w]; }
        alpha_stats[kLsfAlphaStatsOffset + blockIdx.x] = sum;
        alpha_stats[kLsfAlphaSqOffset + blockIdx.x] = sq;
    }
}

__global__ void finish_lsf_alpha_kernel(
    const float* __restrict__ alpha_stats, float* __restrict__ alpha_out,
    float var_scale, float alpha_min, float alpha_max, int N, int blocks
) {
    if (threadIdx.x != 0) return;
    float sum = 0.0f, sq = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        sum += alpha_stats[kLsfAlphaStatsOffset + b];
        sq += alpha_stats[kLsfAlphaSqOffset + b];
    }
    float mean = sum / (float)N;
    float var = fmaxf(0.0f, sq / (float)N - mean * mean);
    float sig = 1.0f / (1.0f + expf(var_scale * var));
    alpha_out[0] = alpha_min + (alpha_max - alpha_min) * sig;
    alpha_out[1] = mean;
}

void launch_compute_lsf_alpha(
    const __nv_bfloat16* M, float var_scale, float alpha_min, float alpha_max,
    float* alpha_out, int NT, int C, cudaStream_t s
) {
    int N     = NT * C;
    int BLOCK = 256;
    int GRID = min((N + BLOCK - 1) / BLOCK, kLsfReduceBlocks);
    compute_lsf_alpha_partials_kernel<<<GRID, BLOCK, 0, s>>>(M, alpha_out, N);
    CUDA_CHECK_LAST();
    finish_lsf_alpha_kernel<<<1, 1, 0, s>>>(
        alpha_out, alpha_out, var_scale, alpha_min, alpha_max, N, GRID);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 2. LSF FORWARD — Serial T-step EMA scan
//    H_t = α·H_{t-1} + (1-α)·M_t
//    Grid: (num_blocks_C, B), Block: (BLOCK_C = 64,)
//    Each (b, c_block) pair scans T=512 time steps serially.
//    FP32 accumulation for H to avoid precision drift over T steps.
// ===========================================================================
__global__ void lsf_fwd_kernel(
    const __nv_bfloat16* __restrict__ M,   // [B, T, C]
    const float*         __restrict__ alpha,
    __nv_bfloat16*       __restrict__ H_io, // [B, C] in-out
    __nv_bfloat16*       __restrict__ H_out,// [B, T, C]
    int B, int T, int C
) {
    int pid_b  = blockIdx.y;
    int pid_c  = blockIdx.x;
    int c      = pid_c * blockDim.x + threadIdx.x;
    if (c >= C || pid_b >= B) return;

    float a       = *alpha;
    float one_m_a = 1.0f - a;

    // Load initial membrane state for this (b, c)
    float h = __bfloat162float(H_io[pid_b * C + c]);

    for (int t = 0; t < T; ++t) {
        float m = __bfloat162float(M[(pid_b * T + t) * C + c]);
        h = a * h + one_m_a * m;
        H_out[(pid_b * T + t) * C + c] = __float2bfloat16(h);
    }

    // Write back final membrane state
    H_io[pid_b * C + c] = __float2bfloat16(h);
}

void launch_lsf_fwd(
    const __nv_bfloat16* M, const float* alpha,
    __nv_bfloat16* H_io, __nv_bfloat16* H_out,
    int B, int T, int C, cudaStream_t s
) {
    int BLOCK_C    = 64;
    int num_c_blks = (C + BLOCK_C - 1) / BLOCK_C;
    dim3 grid(num_c_blks, B);
    lsf_fwd_kernel<<<grid, BLOCK_C, 0, s>>>(M, alpha, H_io, H_out, B, T, C);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 3. LSF BACKWARD — Reversed BPTT through EMA
//    λ_T = grad_H_out[T-1]
//    for t = T-1 down to 0:
//      grad_M[t] = (1-α) · λ_t
//      d_alpha  += λ_t · (H_{t-1} - M_t)
//      λ_{t-1}  = α · λ_t + grad_H_out[t-1]
// ===========================================================================
__global__ void lsf_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_H_out, // [B, T, C]
    const __nv_bfloat16* __restrict__ M,          // [B, T, C]
    const __nv_bfloat16* __restrict__ H_out,      // [B, T, C]
    const float*         __restrict__ alpha,
    __nv_bfloat16*       __restrict__ grad_M,     // [B, T, C]
    float*               __restrict__ grad_alpha_acc,// [1] atomic acc
    int B, int T, int C
) {
    int pid_b = blockIdx.y;
    int pid_c = blockIdx.x;
    int c     = pid_c * blockDim.x + threadIdx.x;
    if (c >= C || pid_b >= B) return;

    float a       = *alpha;
    float one_m_a = 1.0f - a;
    float lambda  = 0.0f;  // adjoint state
    float d_alpha = 0.0f;

    for (int t = T - 1; t >= 0; --t) {
        int off = (pid_b * T + t) * C + c;

        lambda += __bfloat162float(grad_H_out[off]);

        grad_M[off] = __float2bfloat16(one_m_a * lambda);

        // d_alpha contribution from this step
        float H_prev = (t > 0) ?
            __bfloat162float(H_out[(pid_b * T + t - 1) * C + c]) : 0.0f;
        float Mt = __bfloat162float(M[off]);
        d_alpha += lambda * (H_prev - Mt);

        // Propagate adjoint backward
        lambda = a * lambda;
    }

    // Reduce d_alpha across all (b, c) pairs
    d_alpha = warp_sum(d_alpha);
    if ((threadIdx.x & 31) == 0) atomicAdd(grad_alpha_acc, d_alpha);
}

void launch_lsf_bwd(
    const __nv_bfloat16* grad_H_out,
    const __nv_bfloat16* M, const __nv_bfloat16* H_out,
    const float* alpha,
    __nv_bfloat16* grad_M, float* grad_alpha_acc,
    int B, int T, int C, cudaStream_t s
) {
    int BLOCK_C    = 64;
    int num_c_blks = (C + BLOCK_C - 1) / BLOCK_C;
    dim3 grid(num_c_blks, B);
    lsf_bwd_kernel<<<grid, BLOCK_C, 0, s>>>(
        grad_H_out, M, H_out, alpha, grad_M, grad_alpha_acc, B, T, C);
    CUDA_CHECK_LAST();
}

__global__ void lsf_alpha_bwd_kernel(
    const __nv_bfloat16* __restrict__ M,
    const float* __restrict__ alpha_stats,
    const float* __restrict__ grad_alpha,
    __nv_bfloat16* __restrict__ grad_M,
    float var_scale, float alpha_min, float alpha_max, int N
) {
    float range = alpha_max - alpha_min;
    if (range <= 0.0f) return;
    float alpha = alpha_stats[0];
    float mean = alpha_stats[1];
    float sigmoid = fminf(1.0f, fmaxf(0.0f, (alpha - alpha_min) / range));
    float d_alpha_d_var = -var_scale * range * sigmoid * (1.0f - sigmoid);
    float d_var_scale = 2.0f * (*grad_alpha) * d_alpha_d_var / (float)N;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float x = __bfloat162float(M[i]);
        float g = __bfloat162float(grad_M[i]) + d_var_scale * (x - mean);
        grad_M[i] = __float2bfloat16(g);
    }
}

void launch_lsf_alpha_bwd(
    const __nv_bfloat16* M, const float* alpha_stats, const float* grad_alpha,
    __nv_bfloat16* grad_M, float var_scale, float alpha_min, float alpha_max,
    int NT, int C, cudaStream_t s
) {
    int N = NT * C;
    int blocks = min((N + 255) / 256, kLsfReduceBlocks);
    lsf_alpha_bwd_kernel<<<blocks, 256, 0, s>>>(
        M, alpha_stats, grad_alpha, grad_M, var_scale, alpha_min, alpha_max, N);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 4. REFLECTIVE REGULARIZATION PENALTY (architecture.md §4.5)
//    L_ref = λ · [(μ_batch - μ_target)² + max(0, σ²_batch - τ_max)]
//    Computes mean and variance of M and writes l_reflect.
// ===========================================================================
__global__ void reflective_penalty_kernel(
    const __nv_bfloat16* __restrict__ M,
    float* mu_running,
    float* l_reflect,
    float lambda, float tau_max, float ema_decay,
    int N
) {
    __shared__ float ssum, ssq;
    if (threadIdx.x == 0) { ssum = 0.0f; ssq = 0.0f; }
    __syncthreads();

    float ls = 0.0f, lq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __bfloat162float(M[i]);
        ls += v; lq += v * v;
    }
    ls = warp_sum(ls); lq = warp_sum(lq);
    if ((threadIdx.x & 31) == 0) { atomicAdd(&ssum, ls); atomicAdd(&ssq, lq); }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mean = ssum / (float)N;
        float var  = ssq / (float)N - mean * mean;
        float mu_t = *mu_running;
        float p1   = (mean - mu_t) * (mean - mu_t);
        float p2   = fmaxf(0.0f, var - tau_max);
        *l_reflect  = lambda * (p1 + p2);
    }
}

void launch_reflective_penalty(
    const __nv_bfloat16* M,
    float* mu_running, float* l_reflect,
    float lambda, float tau_max, float ema_decay,
    int NT, int C, cudaStream_t s
) {
    reflective_penalty_kernel<<<1, 256, 0, s>>>(
        M, mu_running, l_reflect, lambda, tau_max, ema_decay, NT * C);
    CUDA_CHECK_LAST();
}

__global__ void reflective_penalty_bwd_kernel(
    const __nv_bfloat16* __restrict__ M,
    float* __restrict__ mu_running,
    __nv_bfloat16* __restrict__ grad_M,
    float lambda, float tau_max, float ema_decay,
    int N
) {
    __shared__ float ssum, ssq;
    if (threadIdx.x == 0) { ssum = 0.0f; ssq = 0.0f; }
    __syncthreads();

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __bfloat162float(M[i]);
        local_sum += v;
        local_sq += v * v;
    }
    local_sum = warp_sum(local_sum);
    local_sq = warp_sum(local_sq);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&ssum, local_sum);
        atomicAdd(&ssq, local_sq);
    }
    __syncthreads();

    float mean = ssum / (float)N;
    float var = ssq / (float)N - mean * mean;
    float mu_target = *mu_running;
    float mean_grad = 2.0f * lambda * (mean - mu_target) / (float)N;
    float var_grad = (var > tau_max) ? 2.0f * lambda / (float)N : 0.0f;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = __bfloat162float(M[i]);
        float g = mean_grad + var_grad * (x - mean);
        grad_M[i] = __float2bfloat16(__bfloat162float(grad_M[i]) + g);
    }
    __syncthreads();

    if (threadIdx.x == 0)
        *mu_running = ema_decay * mu_target + (1.0f - ema_decay) * mean;
}

void launch_reflective_penalty_bwd(
    const __nv_bfloat16* M, float* mu_running, __nv_bfloat16* grad_M,
    float lambda, float tau_max, float ema_decay,
    int NT, int C, cudaStream_t s
) {
    reflective_penalty_bwd_kernel<<<1, 256, 0, s>>>(
        M, mu_running, grad_M, lambda, tau_max, ema_decay, NT * C);
    CUDA_CHECK_LAST();
}
