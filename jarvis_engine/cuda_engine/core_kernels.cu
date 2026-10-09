// =============================================================================
// core_kernels.cu — correct forward/backward kernels for the training engine
//   RMSNorm fwd/bwd, linear attention (true per-token decay) fwd/bwd,
//   dense-expert MoE combine fwd/bwd, router bwd, embedding bwd, misc.
// Activations are BF16 in [NT, ...] token-major layout; math is FP32.
// =============================================================================
#include "cuda_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16  f2bf(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float elu1c(float x)   { return x >= 0.f ? x + 1.f : expf(x); }
__device__ __forceinline__ float d_elu1c(float x) { return x >= 0.f ? 1.f : expf(x); }

// Block-wide sum; every thread gets the result. sh must hold 32 floats.
__device__ __forceinline__ float blk_sum(float v, float* sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        float r = (l < (int)(blockDim.x >> 5)) ? sh[l] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffff, r, o);
        if (l == 0) sh[31] = r;
    }
    __syncthreads();
    return sh[31];
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
__global__ void rmsnorm_fwd_k(const bf16* x, const bf16* w, bf16* y, bf16* stash, int C, float eps) {
    __shared__ float sh[32];
    size_t row = (size_t)blockIdx.x * C;
    float ss = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) { float v = bf2f(x[row + i]); ss += v * v; }
    ss = blk_sum(ss, sh);
    float r = rsqrtf(ss / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        y[row + i] = f2bf(bf2f(x[row + i]) * r * bf2f(w[i]));
        if (stash) stash[row + i] = x[row + i];
    }
}
void launch_rmsnorm_fwd(const bf16* x, const bf16* w, bf16* y, bf16* stash, int NT, int C, cudaStream_t s) {
    rmsnorm_fwd_k<<<NT, 256, 0, s>>>(x, w, y, stash, C, 1e-6f);
    CUDA_CHECK_LAST();
}

// dx = r*w*dy - r^3/C * x * sum(w*dy*x) + dres      (dx may alias dres)
__global__ void rmsnorm_bwd_k(const bf16* x, const bf16* w, const bf16* dy, const bf16* dres,
                              bf16* dx, float* dw_acc, int C, float eps) {
    __shared__ float sh[32];
    size_t row = (size_t)blockIdx.x * C;
    float ss = 0.f, sd = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xv = bf2f(x[row + i]);
        ss += xv * xv;
        sd += bf2f(w[i]) * bf2f(dy[row + i]) * xv;
    }
    ss = blk_sum(ss, sh);
    sd = blk_sum(sd, sh);
    float r = rsqrtf(ss / (float)C + eps);
    float coef = r * r * r / (float)C * sd;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xv = bf2f(x[row + i]), dyv = bf2f(dy[row + i]);
        float g = r * bf2f(w[i]) * dyv - coef * xv;
        if (dres) g += bf2f(dres[row + i]);
        dx[row + i] = f2bf(g);
        if (dw_acc) atomicAdd(&dw_acc[i], dyv * xv * r);
    }
}
void launch_rmsnorm_bwd(const bf16* x, const bf16* w, const bf16* dy, const bf16* dres,
                        bf16* dx, float* dw_acc, int NT, int C, cudaStream_t s) {
    rmsnorm_bwd_k<<<NT, 256, 0, s>>>(x, w, dy, dres, dx, dw_acc, C, 1e-6f);
    CUDA_CHECK_LAST();
}

#if 0  // legacy serial attention kernels (replaced by scan16_k below)
// ---------------------------------------------------------------------------
// Linear attention with per-token decay.
//   S_t = g*S_{t-1} + v_t k_t^T ;  o_t = S_t q_t     (per b,h; q,k = rope(elu1(.)))
//   qkv: [NT, 3C] token-major (q | k | v), head h occupies cols h*D..h*D+D-1.
//   One block per (b,h), D threads. S is padded (D+1) to avoid bank conflicts.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void feat_maps(const bf16* row, int C, int D, int d,
                                          const float* cs, const float* sn, int t,
                                          float inv, float* qf, float* kf) {
    if (d < D / 2) {
        float q0 = elu1c(bf2f(row[d])),     q1 = elu1c(bf2f(row[d + D / 2]));
        float k0 = elu1c(bf2f(row[C + d])), k1 = elu1c(bf2f(row[C + d + D / 2]));
        float c = cs[t * D + d], s = sn[t * D + d];
        qf[d]         = (q0 * c - q1 * s) * inv;
        qf[d + D / 2] = (q1 * c + q0 * s) * inv;
        kf[d]         = k0 * c - k1 * s;
        kf[d + D / 2] = k1 * c + k0 * s;
    }
}

__global__ void attn_fwd2_k(const bf16* qkv, const float* log_gamma, const float* cs, const float* sn,
                            bf16* out, int H, int T, int D, int C) {
    int bh = blockIdx.x, b = bh / H, h = bh % H, d = threadIdx.x;
    float g = expf(log_gamma[h]);
    float inv = rsqrtf((float)D);
    extern __shared__ float sm[];
    float* S = sm; float* qf = S + D * (D + 1); float* kf = qf + D;
    for (int i = d; i < D * (D + 1); i += D) S[i] = 0.f;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const bf16* row = qkv + (size_t)(b * T + t) * 3 * C + h * D;
        feat_maps(row, C, D, d, cs, sn, t, inv, qf, kf);
        float v = bf2f(row[2 * C + d]);
        __syncthreads();
        float acc = 0.f;
        for (int e = 0; e < D; ++e) {
            float sv = g * S[d * (D + 1) + e] + v * kf[e];
            S[d * (D + 1) + e] = sv;
            acc += sv * qf[e];
        }
        out[(size_t)(b * T + t) * C + h * D + d] = f2bf(acc);
        __syncthreads();
    }
}
void launch_attn_fwd2(const bf16* qkv, const float* lg, const float* cs, const float* sn, bf16* out,
                      int B, int H, int T, int D, int C, cudaStream_t s) {
    size_t sm = (size_t)(D * (D + 1) + 2 * D) * sizeof(float);
    attn_fwd2_k<<<B * H, D, sm, s>>>(qkv, lg, cs, sn, out, H, T, D, C);
    CUDA_CHECK_LAST();
}

__global__ void attn_bwd2_k(const bf16* qkv, const float* log_gamma, const float* cs, const float* sn,
                            const bf16* dO, bf16* dqkv, int H, int T, int D, int C) {
    int bh = blockIdx.x, b = bh / H, h = bh % H, d = threadIdx.x;
    float g = expf(log_gamma[h]);
    float inv = rsqrtf((float)D);
    extern __shared__ float sm[];
    float* S = sm; float* qf = S + D * (D + 1); float* kf = qf + D;
    float* vv = kf + D; float* dov = vv + D; float* dxf = dov + D;
    for (int i = d; i < D * (D + 1); i += D) S[i] = 0.f;
    __syncthreads();

    // ---- pass 1 (forward in time): dq_t[e] = sum_d dO_t[d] * S_t[d,e]
    for (int t = 0; t < T; ++t) {
        size_t tok = (size_t)(b * T + t);
        const bf16* row = qkv + tok * 3 * C + h * D;
        feat_maps(row, C, D, d, cs, sn, t, inv, qf, kf);
        float v = bf2f(row[2 * C + d]);
        dov[d] = bf2f(dO[tok * C + h * D + d]);
        __syncthreads();
        for (int e = 0; e < D; ++e) S[d * (D + 1) + e] = g * S[d * (D + 1) + e] + v * kf[e];
        __syncthreads();
        float a = 0.f;
        for (int dd = 0; dd < D; ++dd) a += dov[dd] * S[dd * (D + 1) + d];
        dxf[d] = a;
        __syncthreads();
        if (d < D / 2) {
            float ga = dxf[d] * inv, gb = dxf[d + D / 2] * inv;
            float c = cs[t * D + d], s = sn[t * D + d];
            float p0 = ga * c + gb * s, p1 = -ga * s + gb * c;
            dqkv[tok * 3 * C + h * D + d]         = f2bf(p0 * d_elu1c(bf2f(row[d])));
            dqkv[tok * 3 * C + h * D + d + D / 2] = f2bf(p1 * d_elu1c(bf2f(row[d + D / 2])));
        }
        __syncthreads();
    }

    // ---- pass 2 (reverse in time): G_t = g*G_{t+1} + dO_t q_t^T ; dv = G k ; dk = G^T v
    for (int i = d; i < D * (D + 1); i += D) S[i] = 0.f;
    __syncthreads();
    for (int t = T - 1; t >= 0; --t) {
        size_t tok = (size_t)(b * T + t);
        const bf16* row = qkv + tok * 3 * C + h * D;
        feat_maps(row, C, D, d, cs, sn, t, inv, qf, kf);
        vv[d]  = bf2f(row[2 * C + d]);
        dov[d] = bf2f(dO[tok * C + h * D + d]);
        __syncthreads();
        for (int e = 0; e < D; ++e) S[d * (D + 1) + e] = g * S[d * (D + 1) + e] + dov[d] * qf[e];
        __syncthreads();
        float dv = 0.f;
        for (int e = 0; e < D; ++e) dv += S[d * (D + 1) + e] * kf[e];
        dqkv[tok * 3 * C + 2 * C + h * D + d] = f2bf(dv);
        float a = 0.f;
        for (int dd = 0; dd < D; ++dd) a += S[dd * (D + 1) + d] * vv[dd];
        dxf[d] = a;
        __syncthreads();
        if (d < D / 2) {
            float ga = dxf[d], gb = dxf[d + D / 2];
            float c = cs[t * D + d], s = sn[t * D + d];
            float p0 = ga * c + gb * s, p1 = -ga * s + gb * c;
            dqkv[tok * 3 * C + C + h * D + d]         = f2bf(p0 * d_elu1c(bf2f(row[C + d])));
            dqkv[tok * 3 * C + C + h * D + d + D / 2] = f2bf(p1 * d_elu1c(bf2f(row[C + d + D / 2])));
        }
        __syncthreads();
    }
}
void launch_attn_bwd2(const bf16* qkv, const float* lg, const float* cs, const float* sn,
                      const bf16* dO, bf16* dqkv, int B, int H, int T, int D, int C, cudaStream_t s) {
    size_t sm = (size_t)(D * (D + 1) + 5 * D) * sizeof(float);
    attn_bwd2_k<<<B * H, D, sm, s>>>(qkv, lg, cs, sn, dO, dqkv, H, T, D, C);
    CUDA_CHECK_LAST();
}
#endif  // legacy attention

// ---------------------------------------------------------------------------
// Linear attention, parallel register-scan formulation.
//   Every pass is the same recurrence:  X[u][w] = g*X[u][w] + A[u]*B[w] ;  out[u] = sum_w X[u][w]*C[w]
//   8 'u' rows per 32-thread block; each row is split over 4 threads x 16 'w' values held in registers.
//   No shared memory, no block syncs; time loop is serial but per-step latency is a few FMAs + 2 shuffles.
//   fwd (rows):  A=v  B=kf C=qf  -> o                S_t = g S + v k^T,   o = S q
//   dv  (rows, reverse): A=dO B=qf C=kf -> dv        G_t = g G + dO q^T,  dv = G k
//   dq  (cols):  A=kf B=v  C=dO -> dq_f              u=e, w=d
//   dk  (cols, reverse): A=qf B=dO C=v -> dk_f       u=e, w=d
// ---------------------------------------------------------------------------
__device__ __forceinline__ void unpack16(const bf16* p, float* f) {
    uint4 v0 = reinterpret_cast<const uint4*>(p)[0];
    uint4 v1 = reinterpret_cast<const uint4*>(p)[1];
    const bf16* a = reinterpret_cast<const bf16*>(&v0);
    const bf16* b = reinterpret_cast<const bf16*>(&v1);
    #pragma unroll
    for (int j = 0; j < 8; ++j) { f[j] = bf2f(a[j]); f[8 + j] = bf2f(b[j]); }
}

__global__ void scan16_k(const bf16* A, int sA, const bf16* Bv, int sB, const bf16* Cv, int sC,
                         bf16* out, int sO, const float* log_gamma, int reverse,
                         int T, int H, int D) {
    int per_bh = D / 8;
    int bh = blockIdx.x / per_bh, ub = blockIdx.x % per_bh;
    int b = bh / H, h = bh % H;
    int t32 = threadIdx.x;
    int u = ub * 8 + (t32 >> 2);
    int ws = (t32 & 3) * 16;
    float g = expf(log_gamma[h]);
    float X[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) X[j] = 0.f;
    #pragma unroll 2
    for (int i = 0; i < T; ++i) {
        int t = reverse ? (T - 1 - i) : i;
        size_t tok = (size_t)b * T + t;
        float a = bf2f(A[tok * sA + h * D + u]);
        float bb[16], cc[16];
        unpack16(Bv + tok * sB + h * D + ws, bb);
        unpack16(Cv + tok * sC + h * D + ws, cc);
        float acc = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) { X[j] = g * X[j] + a * bb[j]; acc += X[j] * cc[j]; }
        acc += __shfl_xor_sync(0xffffffff, acc, 1);
        acc += __shfl_xor_sync(0xffffffff, acc, 2);
        if ((t32 & 3) == 0) out[tok * sO + h * D + u] = f2bf(acc);
    }
}
static void scan16(const bf16* A, int sA, const bf16* Bv, int sB, const bf16* Cv, int sC,
                   bf16* out, int sO, const float* lg, int reverse,
                   int B, int H, int T, int D, cudaStream_t s) {
    scan16_k<<<B * H * (D / 8), 32, 0, s>>>(A, sA, Bv, sB, Cv, sC, out, sO, lg, reverse, T, H, D);
    CUDA_CHECK_LAST();
}

// ELU+1 and RoPE feature maps for q,k of every token -> qf (scaled by 1/sqrt(D)), kf  (bf16 [NT,C])
__global__ void feat_k(const bf16* qkv, const float* cs, const float* sn, bf16* qf, bf16* kf,
                       int T, int D, int C, float inv) {
    int tok = blockIdx.x, h = blockIdx.y, d = threadIdx.x, t = tok % T;
    const bf16* row = qkv + (size_t)tok * 3 * C + h * D;
    float q0 = elu1c(bf2f(row[d])),     q1 = elu1c(bf2f(row[d + D / 2]));
    float k0 = elu1c(bf2f(row[C + d])), k1 = elu1c(bf2f(row[C + d + D / 2]));
    float c = cs[t * D + d], s = sn[t * D + d];
    size_t o = (size_t)tok * C + h * D + d;
    qf[o]         = f2bf((q0 * c - q1 * s) * inv);
    qf[o + D / 2] = f2bf((q1 * c + q0 * s) * inv);
    kf[o]         = f2bf(k0 * c - k1 * s);
    kf[o + D / 2] = f2bf(k1 * c + k0 * s);
}

void launch_attn_fwd2(const bf16* qkv, const float* lg, const float* cs, const float* sn, bf16* out,
                      bf16* qf, bf16* kf, int B, int H, int T, int D, int C, cudaStream_t s) {
    int NT = B * T;
    feat_k<<<dim3(NT, H), D / 2, 0, s>>>(qkv, cs, sn, qf, kf, T, D, C, rsqrtf((float)D));
    CUDA_CHECK_LAST();
    scan16(qkv + 2 * C, 3 * C, kf, C, qf, C, out, C, lg, 0, B, H, T, D, s);
}

// raw-gradient conversion in place: d_qkv q/k slots hold grads wrt features -> grads wrt raw q,k
__global__ void feat_bwd_k(const bf16* qkv, const float* cs, const float* sn, bf16* dqkv,
                           int T, int D, int C, float inv) {
    int tok = blockIdx.x, h = blockIdx.y, d = threadIdx.x, t = tok % T;
    const bf16* row = qkv + (size_t)tok * 3 * C + h * D;
    bf16* g = dqkv + (size_t)tok * 3 * C + h * D;
    float c = cs[t * D + d], s = sn[t * D + d];
    {
        float ga = bf2f(g[d]) * inv, gb = bf2f(g[d + D / 2]) * inv;
        float p0 = ga * c + gb * s, p1 = -ga * s + gb * c;
        g[d]         = f2bf(p0 * d_elu1c(bf2f(row[d])));
        g[d + D / 2] = f2bf(p1 * d_elu1c(bf2f(row[d + D / 2])));
    }
    {
        float ga = bf2f(g[C + d]), gb = bf2f(g[C + d + D / 2]);
        float p0 = ga * c + gb * s, p1 = -ga * s + gb * c;
        g[C + d]         = f2bf(p0 * d_elu1c(bf2f(row[C + d])));
        g[C + d + D / 2] = f2bf(p1 * d_elu1c(bf2f(row[C + d + D / 2])));
    }
}

// qf,kf must already hold this layer's features (left by launch_attn_fwd2)
void launch_attn_bwd2(const bf16* qkv, const float* lg, const float* cs, const float* sn,
                      const bf16* dO, bf16* dqkv, const bf16* qf, const bf16* kf,
                      int B, int H, int T, int D, int C, cudaStream_t s) {
    const bf16* v = qkv + 2 * C;
    scan16(dO, C, qf, C, kf, C, dqkv + 2 * C, 3 * C, lg, 1, B, H, T, D, s);  // dv
    scan16(kf, C, v, 3 * C, dO, C, dqkv,       3 * C, lg, 0, B, H, T, D, s);  // dq (features)
    scan16(qf, C, dO, C, v, 3 * C, dqkv + C,   3 * C, lg, 1, B, H, T, D, s);  // dk (features)
    feat_bwd_k<<<dim3(B * T, H), D / 2, 0, s>>>(qkv, cs, sn, dqkv, T, D, C, rsqrtf((float)D));
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// MoE (dense experts, top-2 gate mask).  y: [E][NT][C]
// ---------------------------------------------------------------------------
__global__ void moe_combine_fwd_k(const bf16* y, const bf16* x2, const int* idx, const float* gates,
                                  bf16* m, int NT, int C) {
    int t = blockIdx.x;
    int e0 = idx[2 * t], e1 = idx[2 * t + 1];
    float g0 = gates[2 * t], g1 = gates[2 * t + 1];
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = bf2f(x2[(size_t)t * C + i])
                + g0 * bf2f(y[((size_t)e0 * NT + t) * C + i])
                + g1 * bf2f(y[((size_t)e1 * NT + t) * C + i]);
        m[(size_t)t * C + i] = f2bf(v);
    }
}
void launch_moe_combine_fwd(const bf16* y, const bf16* x2, const int* idx, const float* gates,
                            bf16* m, int NT, int C, cudaStream_t s) {
    moe_combine_fwd_k<<<NT, 256, 0, s>>>(y, x2, idx, gates, m, NT, C);
    CUDA_CHECK_LAST();
}

__global__ void moe_combine_bwd_k(const bf16* dm, const bf16* y, const int* idx, const float* gates,
                                  bf16* dy, float* dg, int NT, int C, int E) {
    __shared__ float sh[32];
    int t = blockIdx.x;
    int e0 = idx[2 * t], e1 = idx[2 * t + 1];
    float g0 = gates[2 * t], g1 = gates[2 * t + 1];
    float d0 = 0.f, d1 = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float dmv = bf2f(dm[(size_t)t * C + i]);
        for (int e = 0; e < E; ++e) {
            float val = (e == e0) ? g0 * dmv : ((e == e1) ? g1 * dmv : 0.f);
            dy[((size_t)e * NT + t) * C + i] = f2bf(val);
        }
        d0 += dmv * bf2f(y[((size_t)e0 * NT + t) * C + i]);
        d1 += dmv * bf2f(y[((size_t)e1 * NT + t) * C + i]);
    }
    d0 = blk_sum(d0, sh);
    d1 = blk_sum(d1, sh);
    if (threadIdx.x == 0) { dg[2 * t] = d0; dg[2 * t + 1] = d1; }
}
void launch_moe_combine_bwd(const bf16* dm, const bf16* y, const int* idx, const float* gates,
                            bf16* dy, float* dg, int NT, int C, int E, cudaStream_t s) {
    moe_combine_bwd_k<<<NT, 256, 0, s>>>(dm, y, idx, gates, dy, dg, NT, C, E);
    CUDA_CHECK_LAST();
}

// softmax -> top2 -> g_k = p_ik / (p_i0 + p_i1 + 1e-8); dg -> dlogits (E = 4)
__global__ void router_bwd2_k(const float* probs, const int* idx, const float* dg, bf16* dlogits, int NT) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NT) return;
    float p[4], dp[4] = {0.f, 0.f, 0.f, 0.f};
    for (int e = 0; e < 4; ++e) p[e] = probs[t * 4 + e];
    int i0 = idx[2 * t], i1 = idx[2 * t + 1];
    float s = p[i0] + p[i1] + 1e-8f, s2 = s * s;
    float d0 = dg[2 * t], d1 = dg[2 * t + 1];
    dp[i0] = (d0 - d1) * p[i1] / s2;
    dp[i1] = (d1 - d0) * p[i0] / s2;
    float dot = 0.f;
    for (int e = 0; e < 4; ++e) dot += p[e] * dp[e];
    for (int e = 0; e < 4; ++e) dlogits[t * 4 + e] = f2bf(p[e] * (dp[e] - dot));
}
void launch_router_bwd2(const float* probs, const int* idx, const float* dg, bf16* dlogits,
                        int NT, cudaStream_t s) {
    router_bwd2_k<<<(NT + 255) / 256, 256, 0, s>>>(probs, idx, dg, dlogits, NT);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// Embedding backward (scatter-add) and misc
// ---------------------------------------------------------------------------
__global__ void emb_bwd_k(const int* ids, const bf16* dx, bf16* demb, int C) {
    int t = blockIdx.x;
    size_t dst = (size_t)ids[t] * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        atomicAdd(&demb[dst + i], dx[(size_t)t * C + i]);
}
void launch_emb_bwd2(const int* ids, const bf16* dx, bf16* demb, int NT, int C, cudaStream_t s) {
    emb_bwd_k<<<NT, 256, 0, s>>>(ids, dx, demb, C);
    CUDA_CHECK_LAST();
}

__global__ void f32_to_bf16_k(const float* a, bf16* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = f2bf(a[i]);
}
void launch_f32_to_bf16(const float* a, bf16* b, int n, cudaStream_t s) {
    f32_to_bf16_k<<<(n + 255) / 256, 256, 0, s>>>(a, b, n);
    CUDA_CHECK_LAST();
}
