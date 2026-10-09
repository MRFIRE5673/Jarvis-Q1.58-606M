// =============================================================================
// attention.cu — Associative Linear Attention Kernels
// =============================================================================
// See attention.h for design rationale.
//
// Key SM120 considerations:
//   • ELU+RoPE: SCALAR per half-dim. Avoids serializing transcendentals across
//     8 elements (which kills ILP on SM's transcendental execution units).
//     Each thread does 1 sinf + 1 cosf for its dimension pair.
//   • State update: FP32 accumulation. BF16 outer product accumulation
//     introduces ~0.5% numerical drift in S over 512 tokens.
//   • Thread block per (b,h): 64 threads for D=64. Each thread owns one
//     row of S[d, :] = [D floats]. Full S fits in SMEM: 64×64×4B = 16 KB
//     (well within SM's 48 KB shared memory allocation).
// =============================================================================
#include "attention.h"
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float elu1(float x) {
    return (x >= 0.0f) ? (x + 1.0f) : expf(x);
}
__device__ __forceinline__ float d_elu1(float x) {
    return (x >= 0.0f) ? 1.0f : expf(x);
}

// ---------------------------------------------------------------------------
// 1. RoPE TABLE CONSTRUCTION
//    cos_tab[t, d] = cos(t × θ_d)   where θ_d = 1/base^(2d/D)
//    Grid: (T,), Block: (D/2,)
// ---------------------------------------------------------------------------
__global__ void build_rope_tables_kernel(
    float* cos_tab, float* sin_tab,
    int T, int D, float rope_base
) {
    int t = blockIdx.x;
    int d = threadIdx.x;  // d in [0, D/2)
    if (t >= T || d >= D / 2) return;

    float theta = 1.0f / powf(rope_base, 2.0f * (float)d / (float)D);
    float angle = (float)t * theta;
    // Store at both d and d+D/2 (standard RoPE layout for efficient indexing)
    cos_tab[t * D + d]         = cosf(angle);
    cos_tab[t * D + d + D / 2] = cosf(angle);
    sin_tab[t * D + d]         = sinf(angle);
    sin_tab[t * D + d + D / 2] = sinf(angle);
}

void launch_build_rope_tables(float* cos_tab, float* sin_tab,
                               int T, int D, float rope_base, cudaStream_t s) {
    build_rope_tables_kernel<<<T, D / 2, 0, s>>>(cos_tab, sin_tab, T, D, rope_base);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 2. FUSED ELU+1 + ROPE FORWARD
//    Layout: q_in [B, H, T, D]
//    Each block = one (b,h,t) triple, each thread = one d in [0, D/2)
//    Total blocks = B×H×T = 4×16×512 = 32,768
//    Each block does D/2 = 32 threads — perfect warp (one warp per token)
// ---------------------------------------------------------------------------
__global__ void fused_elu_rope_fwd_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const float*         __restrict__ cos_tab,
    const float*         __restrict__ sin_tab,
    __nv_bfloat16*       __restrict__ q_out,
    __nv_bfloat16*       __restrict__ k_out,
    float inv_sq_D,  // 1/sqrt(D) applied to Q only (key scaling)
    int B, int H, int T, int D
) {
    // blockIdx.x = flat index over B×H×T
    int bht = blockIdx.x;
    if (bht >= B * H * T) return;
    int t  = bht % T;
    // d = threadIdx.x ∈ [0, D/2)
    int d  = threadIdx.x;
    if (d >= D / 2) return;

    int base = bht * D;
    int tab  = t * D;

    // Load pair (d, d+D/2)
    float q0 = __bfloat162float(q_in[base + d]);
    float q1 = __bfloat162float(q_in[base + d + D/2]);
    float k0 = __bfloat162float(k_in[base + d]);
    float k1 = __bfloat162float(k_in[base + d + D/2]);

    // ELU+1 feature map
    q0 = elu1(q0); q1 = elu1(q1);
    k0 = elu1(k0); k1 = elu1(k1);

    // RoPE rotation: [x0, x1] → [x0·c - x1·s, x1·c + x0·s]
    float c = cos_tab[tab + d];
    float s = sin_tab[tab + d];

    q_out[base + d]       = __float2bfloat16((q0*c - q1*s) * inv_sq_D);
    q_out[base + d + D/2] = __float2bfloat16((q1*c + q0*s) * inv_sq_D);
    k_out[base + d]       = __float2bfloat16(k0*c - k1*s);
    k_out[base + d + D/2] = __float2bfloat16(k1*c + k0*s);
}

void launch_fused_elu_rope(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in,
    const float* cos_tab, const float* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    int B, int H, int T, int D, cudaStream_t s
) {
    int total = B * H * T;
    float inv_sq_D = 1.0f / sqrtf((float)D);
    // D/2 threads per block = 32 for D=64 (exactly one warp)
    fused_elu_rope_fwd_kernel<<<total, D / 2, 0, s>>>(
        q_in, k_in, cos_tab, sin_tab, q_out, k_out, inv_sq_D, B, H, T, D);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 3. FUSED ELU+ROPE BACKWARD
//    Same layout. Grad flows back through rotation and ELU.
// ---------------------------------------------------------------------------
__global__ void fused_elu_rope_bwd_kernel(
    const __nv_bfloat16* __restrict__ dq_out,
    const __nv_bfloat16* __restrict__ dk_out,
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const float*         __restrict__ cos_tab,
    const float*         __restrict__ sin_tab,
    __nv_bfloat16*       __restrict__ dq_in,
    __nv_bfloat16*       __restrict__ dk_in,
    float inv_sq_D,
    int B, int H, int T, int D
) {
    int bht = blockIdx.x;
    if (bht >= B * H * T) return;
    int t  = bht % T;
    int d  = threadIdx.x;
    if (d >= D / 2) return;

    int base = bht * D;
    int tab  = t * D;

    float gq0 = __bfloat162float(dq_out[base + d])       * inv_sq_D;
    float gq1 = __bfloat162float(dq_out[base + d + D/2]) * inv_sq_D;
    float gk0 = __bfloat162float(dk_out[base + d]);
    float gk1 = __bfloat162float(dk_out[base + d + D/2]);

    float c = cos_tab[tab + d];
    float s = sin_tab[tab + d];

    // Backward through RoPE: [ga, gb] → [ga·c + gb·s, -ga·s + gb·c]
    float dq0_pre = gq0 * c + gq1 * s;
    float dq1_pre = -gq0 * s + gq1 * c;
    float dk0_pre = gk0 * c + gk1 * s;
    float dk1_pre = -gk0 * s + gk1 * c;

    // Backward through ELU+1
    float q0 = __bfloat162float(q_in[base + d]);
    float q1 = __bfloat162float(q_in[base + d + D/2]);
    float k0 = __bfloat162float(k_in[base + d]);
    float k1 = __bfloat162float(k_in[base + d + D/2]);

    dq_in[base + d]       = __float2bfloat16(dq0_pre * d_elu1(q0));
    dq_in[base + d + D/2] = __float2bfloat16(dq1_pre * d_elu1(q1));
    dk_in[base + d]       = __float2bfloat16(dk0_pre * d_elu1(k0));
    dk_in[base + d + D/2] = __float2bfloat16(dk1_pre * d_elu1(k1));
}

void launch_fused_elu_rope_bwd(
    const __nv_bfloat16* dq_out, const __nv_bfloat16* dk_out,
    const __nv_bfloat16* q_in,   const __nv_bfloat16* k_in,
    const float* cos_tab,         const float* sin_tab,
    __nv_bfloat16* dq_in,         __nv_bfloat16* dk_in,
    int B, int H, int T, int D, cudaStream_t s
) {
    int total = B * H * T;
    float inv_sq_D = 1.0f / sqrtf((float)D);
    fused_elu_rope_bwd_kernel<<<total, D / 2, 0, s>>>(
        dq_out, dk_out, q_in, k_in, cos_tab, sin_tab, dq_in, dk_in, inv_sq_D, B, H, T, D);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// 4. ASSOCIATIVE LINEAR ATTENTION FORWARD (CHUNKED)
//    For one (b, h) pair (= one block), process T tokens in chunks of 64.
//
//    Intra-chunk: causal masked dot products within chunk (cuBLASLt handles this)
//    Inter-chunk: cross-state contribution: Z_cross[i] = γ^{i+1} * S_prev @ q_i
//    State update: S = γ^{cs} * S_prev + Σ_{i=0}^{cs-1} γ^{cs-1-i} * v_i ⊗ k_i^T
//
//    SMEM layout: S[D×D] = 64×64 FP32 = 16 KB per block (fine for 48 KB smem)
//    Block: (D,) = 64 threads. Each thread owns row d of S.
//    Grid: (B×H,) = 4×16 = 64 blocks.
//
//    NOTE: The intra-chunk quadratic term requires calling cuBLASLt for tiled
//    causal masked GEMMs. This kernel handles ONLY the cross-state contribution
//    and state update. Full integration is in full_engine.cu.
// ---------------------------------------------------------------------------
__global__ void assoc_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,         // [B,H,T,D]
    const __nv_bfloat16* __restrict__ k,         // [B,H,T,D]
    const __nv_bfloat16* __restrict__ v,         // [B,H,T,D]
    const float*         __restrict__ log_gamma, // [H]
    __nv_bfloat16*       __restrict__ attn_out,  // [B,H,T,D]
    __nv_bfloat16*       __restrict__ S_global,  // [B,H,D,D] state
    int B, int H, int T, int D, int cs
) {
    int bh  = blockIdx.x;
    int b   = bh / H;
    int h   = bh % H;
    if (bh >= B * H) return;

    int d = threadIdx.x;  // ∈ [0, D)
    float log_g = log_gamma[h];

    // SMEM layout: S[D*D] FP32 + q_smem[D] FP32
    // Total: D*D*4 + D*4 = 64*64*4 + 64*4 = 16640 bytes < 48 KB
    extern __shared__ float smem[];
    float* S      = smem;           // [D * D] — associative state
    float* q_smem = smem + D * D;  // [D]     — cached q for current token

    int base_S   = bh * D * D;
    int base_seq = (b * H + h) * T * D;

    // Load S from global → SMEM (BF16 → FP32)
    int D2 = D * D;
    for (int i = d; i < D2; i += D) {
        S[i] = __bfloat162float(S_global[base_S + i]);
    }
    __syncthreads();

    int n_chunks = (T + cs - 1) / cs;

    for (int ci = 0; ci < n_chunks; ++ci) {
        int t_start = ci * cs;
        int t_end   = min(t_start + cs, T);
        int chunk_len = t_end - t_start;
        float g     = expf(log_g);           // γ = exp(log_g): per-step decay factor
        float inv_g = expf(-log_g);          // 1/γ: for gw decrement
        float gamma_cs = expf(log_g * (float)chunk_len);  // γ^chunk_len for state decay
        // gc and gw are per-chunk scalars (all threads use same values — no divergence)
        float gc = g;                          // γ^1 at ti=t_start
        float gw = expf(log_g * (float)(chunk_len - 1));  // γ^(chunk_len-1) at ti=t_start

        for (int ti = t_start; ti < t_end; ++ti) {
            int toff = base_seq + ti * D;

            // Load q, k, v into registers once
            float qi = (d < D) ? __bfloat162float(q[toff + d]) : 0.0f;
            float ki = (d < D) ? __bfloat162float(k[toff + d]) : 0.0f;
            float vi = (d < D) ? __bfloat162float(v[toff + d]) : 0.0f;

            if (d < D) q_smem[d] = qi;
            __syncthreads();

            if (d < D) {
                // Cross-state: z[d] = gc * Σ_e S[d,e] * q[e]
                // gc updated incrementally: no expf() per token
                float z = 0.0f;
                #pragma unroll 8
                for (int e = 0; e < D; ++e) {
                    z += S[d * D + e] * q_smem[e];
                }
                attn_out[(b * H + h) * T * D + ti * D + d] =
                    __float2bfloat16(gc * z);
            }
            __syncthreads();

            // Reuse q_smem buffer as k_smem for state update
            if (d < D) q_smem[d] = ki;
            __syncthreads();

            if (d < D) {
                #pragma unroll 8
                for (int e = 0; e < D; ++e) {
                    S[d * D + e] = gamma_cs * S[d * D + e] + gw * vi * q_smem[e];
                }
            }
            __syncthreads();

            // Advance geometric factors by one step (no expf — multiply only)
            gc *= g;   // gc *= exp(log_g) = exp(log_g * (ti_rel + 2))
            gw *= inv_g;  // gw *= 1/exp(log_g) = exp(log_g * (chunk_len - 2 - ti_rel))
        }
    }

    // Write S back to global (FP32 → BF16)
    for (int i = d; i < D2; i += D) {
        S_global[base_S + i] = __float2bfloat16(S[i]);
    }
}

void launch_assoc_attn_fwd(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const float* log_gamma,
    __nv_bfloat16* attn_out, __nv_bfloat16* S,
    int B, int H, int T, int D, int chunk_size, cudaStream_t s
) {
    int grid       = B * H;       // 64 blocks for B=4, H=16
    int block      = D;           // 64 threads, one per S row
    // SMEM: S[D*D] FP32 + q_smem[D] FP32 = (D*D + D)*4 = 16640 bytes for D=64
    size_t smem_sz = (size_t)(D * D + D) * sizeof(float);

    assoc_attn_fwd_kernel<<<grid, block, smem_sz, s>>>(
        q, k, v, log_gamma, attn_out, S, B, H, T, D, chunk_size);
    CUDA_CHECK_LAST();
}
