// =============================================================================
// moe.cu — Sparse MoE CUDA Kernel Implementations
// =============================================================================
#include "moe.h"
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ===========================================================================
// 1. TOP-2 GATING (E=4 — fully register-resident, no SMEM)
//    Grid: (NT/BLOCK,), Block: (BLOCK=256,)
//    Each thread handles one token independently.
//    E=4 logits fit in 4 registers — no shared memory needed.
// ===========================================================================
__global__ void top2_gating_kernel(
    const __nv_bfloat16* __restrict__ logits,  // [NT, 4]
    float*               __restrict__ probs,   // [NT, 4]
    int*                 __restrict__ topk_idx,// [NT, 2]
    float*               __restrict__ topk_gates,// [NT, 2]
    int NT
) {
    int tok = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= NT) return;

    int base = tok * 4;
    float lg[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e)
        lg[e] = __bfloat162float(logits[base + e]);

    // Stable softmax over E=4
    float mx = fmaxf(fmaxf(lg[0], lg[1]), fmaxf(lg[2], lg[3]));
    float s  = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) { lg[e] = expf(lg[e] - mx); s += lg[e]; }
    float inv_s = 1.0f / (s + 1e-8f);
    #pragma unroll
    for (int e = 0; e < 4; ++e) { lg[e] *= inv_s; probs[base + e] = lg[e]; }

    // Top-2 selection (unrolled for E=4, no branches on sorted order)
    int i0 = 0;
    #pragma unroll
    for (int e = 1; e < 4; ++e) if (lg[e] > lg[i0]) i0 = e;

    int i1 = (i0 == 0) ? 1 : 0;
    #pragma unroll
    for (int e = 0; e < 4; ++e)
        if (e != i0 && lg[e] > lg[i1]) i1 = e;

    // Normalize gate weights to sum=1 over selected top-2
    float g_sum = lg[i0] + lg[i1] + 1e-8f;
    topk_idx  [tok * 2 + 0] = i0;  topk_idx  [tok * 2 + 1] = i1;
    topk_gates[tok * 2 + 0] = lg[i0] / g_sum;
    topk_gates[tok * 2 + 1] = lg[i1] / g_sum;
}

void launch_top2_gating(
    const __nv_bfloat16* logits, float* probs,
    int* topk_idx, float* topk_gates,
    int NT, int E, cudaStream_t s
) {
    (void)E;  // locked to 4
    top2_gating_kernel<<<(NT + 255) / 256, 256, 0, s>>>(
        logits, probs, topk_idx, topk_gates, NT);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 2. DISPATCH GATHER (vec8 uint4 — 128-bit, 8×BF16 per load)
//    Grid: (NTK,), Block: (C/8 = 128,)   for C=1024
//    Each block copies one row of dispatch_x from x[topk_idx[tok,k], :]
// ===========================================================================
__global__ void dispatch_gather_kernel(
    const __nv_bfloat16* __restrict__ x,         // [NT, C]
    const int*           __restrict__ topk_idx,  // [NT, K]
    __nv_bfloat16*       __restrict__ dispatch_x,// [NTK, C]
    int NT, int K, int C
) {
    int flat = blockIdx.x;  // flat ∈ [0, NTK)
    int tok  = flat / K;
    // No expert routing needed here — we just gather token copies
    // Expert routing happens in the scatter-combine after W2
    int src_row = tok;  // all K copies from the same token
    int vec8    = threadIdx.x;  // ∈ [0, C/8)
    if (vec8 >= C / 8) return;

    const uint4* src = reinterpret_cast<const uint4*>(x + src_row * C + vec8 * 8);
    uint4*       dst = reinterpret_cast<uint4*>(dispatch_x + flat * C + vec8 * 8);
    *dst = *src;
}

void launch_dispatch_gather(
    const __nv_bfloat16* x, const int* topk_idx,
    __nv_bfloat16* dispatch_x,
    int NT, int K, int C, cudaStream_t s
) {
    int NTK = NT * K;
    dispatch_gather_kernel<<<NTK, C / 8, 0, s>>>(x, topk_idx, dispatch_x, NT, K, C);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 3a. IN-PLACE GELU FORWARD (vec8 uint4 — memory-bound)
// ===========================================================================
__global__ void gelu_fwd_vec8_kernel(__nv_bfloat16* x, int N8) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N8) return;

    uint4 v = *reinterpret_cast<uint4*>(x + i * 8);
    __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&v);

    constexpr float kSqrt2Pi = 0.7978845608f;
    constexpr float kCoef    = 0.044715f;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v_ = __bfloat162float(e[j]);
        float t  = tanhf(kSqrt2Pi * (v_ + kCoef * v_ * v_ * v_));
        e[j] = __float2bfloat16(0.5f * v_ * (1.0f + t));
    }
    *reinterpret_cast<uint4*>(x + i * 8) = v;
}

void launch_gelu_fwd_inplace(__nv_bfloat16* x, int N, cudaStream_t s) {
    int N8 = N / 8, BLOCK = 256;
    gelu_fwd_vec8_kernel<<<(N8 + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(x, N8);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 3b. FUSED GELU + STASH — eliminates D2D memcpy for pre-GELU checkpoint
//     Reads h1_in once, writes stash (pre-GELU copy) AND gelu output.
//     This removes the 16 MB cudaMemcpyAsync per layer (384 MB/step total).
// ===========================================================================
__global__ void gelu_fwd_stash_vec8_kernel(
    const __nv_bfloat16* __restrict__ h1_in,   // [NTK, hi] pre-GELU input
    __nv_bfloat16*       __restrict__ stash,   // [NTK, hi] pre-GELU checkpoint
    __nv_bfloat16*       __restrict__ h1_out,  // [NTK, hi] post-GELU output
    int N8
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N8) return;

    uint4 v = *reinterpret_cast<const uint4*>(h1_in + i * 8);
    // Write pre-GELU stash (raw copy)
    *reinterpret_cast<uint4*>(stash + i * 8) = v;

    // Apply GELU in-register
    __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&v);
    constexpr float kSqrt2Pi = 0.7978845608f;
    constexpr float kCoef    = 0.044715f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float vf = __bfloat162float(e[j]);
        float t  = tanhf(kSqrt2Pi * (vf + kCoef * vf * vf * vf));
        e[j] = __float2bfloat16(0.5f * vf * (1.0f + t));
    }
    *reinterpret_cast<uint4*>(h1_out + i * 8) = v;
}

void launch_gelu_fwd_stash(
    const __nv_bfloat16* h1_in,
    __nv_bfloat16*       stash,
    __nv_bfloat16*       h1_out,
    int N, cudaStream_t s
) {
    int N8 = N / 8, BLOCK = 256;
    gelu_fwd_stash_vec8_kernel<<<(N8 + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(
        h1_in, stash, h1_out, N8);
    CUDA_CHECK_LAST();
}


// ===========================================================================
// 4. GELU BACKWARD — SCALAR (MANDATORY)
//    Compute chain: tanh, sech², multiply. Transcendental units on SM120 are
//    shared per warp. Vectorizing 8 chains per thread serializes them onto
//    the same unit → kills SM ILP. Scalar = one chain per thread = max ILP.
//    This matches the observation in memory.md §4.3.
// ===========================================================================
__global__ void gelu_bwd_kernel(
    const __nv_bfloat16* __restrict__ x_pre,
    const __nv_bfloat16* __restrict__ grad_out,
    __nv_bfloat16*       __restrict__ grad_x,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    constexpr float kSqrt2Pi = 0.7978845608f;
    constexpr float kCoef    = 0.044715f;

    float v  = __bfloat162float(x_pre[i]);
    float go = __bfloat162float(grad_out[i]);

    float x3   = v * v * v;
    float arg  = kSqrt2Pi * (v + kCoef * x3);
    float th   = tanhf(arg);
    float s2   = 1.0f - th * th;  // sech²(arg)
    float dgelu = 0.5f * (1.0f + th) +
                  0.5f * v * s2 * kSqrt2Pi * (1.0f + 3.0f * kCoef * v * v);
    grad_x[i] = __float2bfloat16(dgelu * go);
}

void launch_gelu_bwd(
    const __nv_bfloat16* x_pre, const __nv_bfloat16* grad_out,
    __nv_bfloat16* grad_x, int N, cudaStream_t s
) {
    // SCALAR — 1 element per thread, no vectorization
    gelu_bwd_kernel<<<(N + 255) / 256, 256, 0, s>>>(x_pre, grad_out, grad_x, N);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 5. SCATTER-COMBINE + RESIDUAL ADD
//    For each token tok, accumulate K=2 gate-weighted expert outputs:
//      moe_out[tok] = Σ_{k=0}^{K-1} topk_gates[tok,k] × expert_out[tok*K+k]
//                   + x_residual[tok]
//
//    Layout: expert_out [NTK, C] — contiguous dispatch, not per-expert batches
//    Each (tok, k) slice corresponds to one expert's output for that token.
//    Grid: (NT,), Block: (C/8=128,)  — vec8 for memory-bound combine
// ===========================================================================
__global__ void scatter_combine_kernel(
    const __nv_bfloat16* __restrict__ expert_out,  // [NTK, C]
    const __nv_bfloat16* __restrict__ x_res,       // [NT, C]
    const float*         __restrict__ topk_gates,  // [NT, K]
    __nv_bfloat16*       __restrict__ moe_out,     // [NT, C]
    int NT, int K, int C
) {
    int tok  = blockIdx.x;
    int c8   = threadIdx.x;  // column group (handles 8 BF16)
    if (tok >= NT || c8 >= C / 8) return;

    int col = c8 * 8;

    // Load residual
    float acc[8];
    const __nv_bfloat16* xr = x_res + tok * C + col;
    uint4 vx = *reinterpret_cast<const uint4*>(xr);
    const __nv_bfloat16* ex = reinterpret_cast<const __nv_bfloat16*>(&vx);
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = __bfloat162float(ex[j]);

    // Accumulate K=2 gate-weighted expert outputs
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        float gate = topk_gates[tok * K + k];
        int   flat = tok * K + k;
        const __nv_bfloat16* eo = expert_out + flat * C + col;
        uint4 ve = *reinterpret_cast<const uint4*>(eo);
        const __nv_bfloat16* ee = reinterpret_cast<const __nv_bfloat16*>(&ve);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += gate * __bfloat162float(ee[j]);
    }

    // Write output (pack 8 BF16 as uint4)
    uint4 vout;
    __nv_bfloat16* eo = reinterpret_cast<__nv_bfloat16*>(&vout);
    #pragma unroll
    for (int j = 0; j < 8; ++j) eo[j] = __float2bfloat16(acc[j]);
    *reinterpret_cast<uint4*>(moe_out + tok * C + col) = vout;
}

void launch_scatter_combine(
    const __nv_bfloat16* expert_out, const __nv_bfloat16* x_residual,
    const int* topk_idx, const float* topk_gates,
    __nv_bfloat16* moe_out,
    int NT, int K, int C, cudaStream_t s
) {
    (void)topk_idx;  // indices used by cuBLASLt expert GEMMs, not needed here
    scatter_combine_kernel<<<NT, C / 8, 0, s>>>(
        expert_out, x_residual, topk_gates, moe_out, NT, K, C);
    CUDA_CHECK_LAST();
}

// ===========================================================================
// 6. LOAD BALANCE LOSS
//    L_bal = α_bal × E × Σ_i f_i × P_i
//    f_i  = fraction of tokens whose top-1 is expert i
//    P_i  = mean gating probability for expert i
//    Single block, 32 threads (one warp), E=4 — tiny kernel
// ===========================================================================
__global__ void load_balance_kernel(
    const float* __restrict__ probs,    // [NT, E]
    const int*   __restrict__ topk_idx, // [NT, K]
    float*       __restrict__ l_balance,
    float alpha_bal, int NT, int E
) {
    __shared__ float sf[4], sp[4];
    if (threadIdx.x < E) { sf[threadIdx.x] = 0.0f; sp[threadIdx.x] = 0.0f; }
    __syncthreads();

    for (int tok = threadIdx.x; tok < NT; tok += blockDim.x) {
        int top1 = topk_idx[tok * 2];  // top-1 expert
        atomicAdd(&sf[top1], 1.0f / (float)NT);
        for (int e = 0; e < E; ++e)
            atomicAdd(&sp[e], probs[tok * E + e] / (float)NT);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float loss = 0.0f;
        for (int e = 0; e < E; ++e) loss += sf[e] * sp[e];
        *l_balance = alpha_bal * (float)E * loss;
    }
}

void launch_load_balance_loss(
    const float* probs, const int* topk_idx,
    float* l_balance,
    float alpha_bal, int NT, int E, cudaStream_t s
) {
    load_balance_kernel<<<1, 32, 0, s>>>(probs, topk_idx, l_balance, alpha_bal, NT, E);
    CUDA_CHECK_LAST();
}
