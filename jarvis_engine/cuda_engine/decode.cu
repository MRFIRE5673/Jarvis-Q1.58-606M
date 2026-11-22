// =============================================================================
// decode.cu — High-Performance Native CUDA Single-Token Inference & Generation
// =============================================================================
#include "decode.h"
#include "cuda_engine.h"
#include "cublaslt_engine.h"
#include "runtime.h"
#include "full_engine.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

__device__ __forceinline__ float d_bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 d_f2bf(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float d_elu1(float x) { return (x >= 0.0f) ? (x + 1.0f) : expf(x); }

namespace {

struct DecodeWorkspace {
    __nv_bfloat16* x          = nullptr; // [1024]
    __nv_bfloat16* norm1_out  = nullptr; // [1024]
    __nv_bfloat16* qkv_out    = nullptr; // [3072]
    __nv_bfloat16* q_rot      = nullptr; // [1024]
    __nv_bfloat16* k_rot      = nullptr; // [1024]
    __nv_bfloat16* v_raw      = nullptr; // [1024]
    __nv_bfloat16* attn_out   = nullptr; // [1024]
    __nv_bfloat16* x2         = nullptr; // [1024]
    __nv_bfloat16* norm2_out  = nullptr; // [1024]
    __nv_bfloat16* r_logits   = nullptr; // [4]
    float*         exp_gates  = nullptr; // [4]
    __nv_bfloat16* h1         = nullptr; // [2048]
    __nv_bfloat16* exp_out    = nullptr; // [4096]
    __nv_bfloat16* moe_out    = nullptr; // [1024]
    __nv_bfloat16* final_norm = nullptr; // [1024]
    float*         logits     = nullptr; // [50304]
    int*           out_tok    = nullptr; // [1]

    // Recurrent states resident on GPU
    float* S_state = nullptr; // [24, 16, 64, 64] FP32
    float* H_state = nullptr; // [24, 1024] FP32

    // Recent tokens ring buffer on GPU for repetition penalty
    int* recent_tokens = nullptr; // [256]
    int  recent_count  = 0;

    std::vector<float> host_logits;
    std::vector<int>   recent_history;

    bool initialized = false;
};

static DecodeWorkspace dws;

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

__global__ void emb_lookup_k(const __nv_bfloat16* tok_emb, int tok_id, __nv_bfloat16* x, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C) {
        x[idx] = tok_emb[(size_t)tok_id * C + idx];
    }
}

__global__ void rmsnorm_1d_k(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, int C, float eps) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float val = d_bf2f(x[i]);
        sum_sq += val * val;
    }
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    int lane = tid & 31;
    int wid  = tid >> 5;
    if (lane == 0) sdata[wid] = sum_sq;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) sdata[0] = val;
    }
    __syncthreads();

    float rsqrt_val = rsqrtf(sdata[0] / (float)C + eps);
    for (int i = tid; i < C; i += blockDim.x) {
        y[i] = d_f2bf(d_bf2f(x[i]) * rsqrt_val * d_bf2f(w[i]));
    }
}

__global__ void split_qkv_rope_elu_k(
    const __nv_bfloat16* qkv,
    const float* cos_tab, const float* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    int pos, int H, int D, int C, int T
) {
    int h = blockIdx.x; // [0, 16)
    int d = threadIdx.x; // [0, 32) -> handles d and d + 32
    if (h >= H || d >= D / 2) return;

    int base_h = h * D;
    int tab = (pos % T) * D;

    float inv_sq_D = 1.0f / sqrtf((float)D);

    // Q
    float q0 = d_elu1(d_bf2f(qkv[base_h + d]));
    float q1 = d_elu1(d_bf2f(qkv[base_h + d + D / 2]));
    // K
    float k0 = d_elu1(d_bf2f(qkv[C + base_h + d]));
    float k1 = d_elu1(d_bf2f(qkv[C + base_h + d + D / 2]));
    // V
    v_out[base_h + d]         = qkv[2 * C + base_h + d];
    v_out[base_h + d + D / 2] = qkv[2 * C + base_h + d + D / 2];

    float c = cos_tab[tab + d];
    float s = sin_tab[tab + d];

    q_out[base_h + d]         = d_f2bf((q0 * c - q1 * s) * inv_sq_D);
    q_out[base_h + d + D / 2] = d_f2bf((q1 * c + q0 * s) * inv_sq_D);
    k_out[base_h + d]         = d_f2bf(k0 * c - k1 * s);
    k_out[base_h + d + D / 2] = d_f2bf(k1 * c + k0 * s);
}

// 16 blocks (one per head), 64 threads (one per row d of S[64, 64])
__global__ void assoc_attn_step_k(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* S_layer, const float* log_gamma,
    __nv_bfloat16* out, int H, int D
) {
    int h = blockIdx.x;
    int d = threadIdx.x; // row d in S
    if (h >= H || d >= D) return;

    __shared__ float sh_k[64];
    __shared__ float sh_q[64];

    int base_h = h * D;
    sh_k[d] = d_bf2f(k[base_h + d]);
    sh_q[d] = d_bf2f(q[base_h + d]);
    float v_d = d_bf2f(v[base_h + d]);
    float raw_g = log_gamma[h];
    float gamma = 1.0f / (1.0f + expf(-raw_g)); // exact sigmoid(gamma_raw) matching PyTorch
    __syncthreads();

    float* S_row = S_layer + (size_t)h * D * D + d * D;

    float z_acc = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < 64; ++j) {
        float S_val = gamma * S_row[j] + v_d * sh_k[j];
        S_row[j] = S_val;
        z_acc += S_val * sh_q[j];
    }

    out[base_h + d] = d_f2bf(z_acc);
}

__global__ void vec_add_k(__nv_bfloat16* a, const __nv_bfloat16* b, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C) {
        a[i] = d_f2bf(d_bf2f(a[i]) + d_bf2f(b[i]));
    }
}

__global__ void router_top2_k(
    const __nv_bfloat16* r_logits,
    float* e_gates
) {
    if (threadIdx.x == 0) {
        float lg[4];
        float max_lg = -1e9f;
        for (int i = 0; i < 4; ++i) {
            lg[i] = d_bf2f(r_logits[i]);
            if (lg[i] > max_lg) max_lg = lg[i];
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < 4; ++i) {
            lg[i] = expf(lg[i] - max_lg);
            sum_exp += lg[i];
        }
        for (int i = 0; i < 4; ++i) lg[i] /= (sum_exp + 1e-8f);

        int i0 = 0;
        for (int i = 1; i < 4; ++i) if (lg[i] > lg[i0]) i0 = i;
        int i1 = (i0 == 0) ? 1 : 0;
        for (int i = 0; i < 4; ++i) if (i != i0 && lg[i] > lg[i1]) i1 = i;

        float g_sum = lg[i0] + lg[i1] + 1e-8f;
        for (int i = 0; i < 4; ++i) e_gates[i] = 0.0f;
        e_gates[i0] = lg[i0] / g_sum;
        e_gates[i1] = lg[i1] / g_sum;
    }
}

__global__ void gelu_k(__nv_bfloat16* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = d_bf2f(x[i]);
        // Fast GELU: 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
        float v3 = v * v * v;
        float inner = 0.79788456f * (v + 0.044715f * v3);
        float g = 0.5f * v * (1.0f + tanhf(inner));
        x[i] = d_f2bf(g);
    }
}

__global__ void moe_combine_k(
    const __nv_bfloat16* exp_out,
    const float* gates, __nv_bfloat16* moe_out, int C
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C) {
        float sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            float g = gates[e];
            if (g > 0.0f) {
                sum += g * d_bf2f(exp_out[(size_t)e * C + i]);
            }
        }
        moe_out[i] = d_f2bf(sum);
    }
}

__global__ void lsf_step_k(
    const __nv_bfloat16* moe_out, float* H_layer, __nv_bfloat16* x,
    int C, float a_min, float a_max, float scale
) {
    __shared__ float s_mean[32];
    __shared__ float s_var[32];
    int tid = threadIdx.x;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float v = d_bf2f(moe_out[i]);
        sum += v;
        sum_sq += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    int lane = tid & 31;
    int wid = tid >> 5;
    if (lane == 0) {
        s_mean[wid] = sum;
        s_var[wid] = sum_sq;
    }
    __syncthreads();

    if (wid == 0) {
        float m = (lane < (blockDim.x / 32)) ? s_mean[lane] : 0.0f;
        float v = (lane < (blockDim.x / 32)) ? s_var[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            m += __shfl_down_sync(0xffffffff, m, offset);
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0) {
            s_mean[0] = m;
            s_var[0] = v;
        }
    }
    __syncthreads();

    float mean = s_mean[0] / (float)C;
    float variance = (s_var[0] / (float)C) - (mean * mean);
    if (variance < 0.0f) variance = 0.0f;

    float alpha_raw = 1.0f / (1.0f + expf(scale * variance));
    float alpha = a_min + (a_max - a_min) * alpha_raw;

    for (int i = tid; i < C; i += blockDim.x) {
        float h_prev = H_layer[i];
        float m_val = d_bf2f(moe_out[i]);
        float h_new = alpha * h_prev + (1.0f - alpha) * m_val;
        H_layer[i] = h_new;
        x[i] = d_f2bf(d_bf2f(x[i]) + h_new);
    }
}

// Stochastic sampling: temperature, top-k, top-p nucleus, and repetition penalty
static int sample_token_stochastic(
    const float* raw_logits,
    int V,
    float temperature,
    float top_p,
    int top_k,
    float rep_penalty,
    const std::vector<int>& history
) {
    if (temperature <= 0.001f) {
        int best_idx = 0;
        float best_val = -1e30f;
        std::unordered_set<int> rec_set(history.begin(), history.end());
        for (int i = 0; i < V; ++i) {
            float v = raw_logits[i];
            if (rep_penalty > 1.0f && rec_set.find(i) != rec_set.end()) {
                v = (v > 0.0f) ? (v / rep_penalty) : (v * rep_penalty);
            }
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx;
    }

    std::vector<float> l(raw_logits, raw_logits + V);

    // 1. Frequency-weighted repetition penalty & anti-stutter
    if (rep_penalty > 1.0f && !history.empty()) {
        std::unordered_map<int, int> counts;
        for (int t : history) counts[t]++;
        for (const auto& kv : counts) {
            int t = kv.first;
            int count = kv.second;
            if (t >= 0 && t < V) {
                float penalty = std::pow(rep_penalty, (float)std::min(count, 4));
                float v = l[t];
                l[t] = (v > 0.0f) ? (v / penalty) : (v * penalty);
            }
        }
        int last_tok = history.back();
        if (last_tok >= 0 && last_tok < V) {
            l[last_tok] -= 2.0f;
        }
    }

    // 2. Temperature scaling
    float inv_temp = 1.0f / (temperature > 1e-5f ? temperature : 1e-5f);
    for (int i = 0; i < V; ++i) {
        l[i] *= inv_temp;
    }

    // 3. Top-k candidates
    int k = (top_k > 0 && top_k < V) ? top_k : V;
    std::vector<int> cand(V);
    std::iota(cand.begin(), cand.end(), 0);
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [&l](int a, int b) { return l[a] > l[b]; });

    // 4. Softmax over top-k
    float max_logit = l[cand[0]];
    std::vector<float> probs(k);
    float sum_p = 0.0f;
    for (int i = 0; i < k; ++i) {
        float p = std::exp(l[cand[i]] - max_logit);
        probs[i] = p;
        sum_p += p;
    }
    for (int i = 0; i < k; ++i) {
        probs[i] /= sum_p;
    }

    // 5. Top-p nucleus filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        int p_cutoff = k;
        for (int i = 0; i < k; ++i) {
            cum += probs[i];
            if (cum > top_p) {
                p_cutoff = i + 1;
                break;
            }
        }
        k = p_cutoff;
        probs.resize(k);
        float re_sum = 0.0f;
        for (int i = 0; i < k; ++i) re_sum += probs[i];
        for (int i = 0; i < k; ++i) probs[i] /= (re_sum > 0.0f ? re_sum : 1.0f);
    }

    // 6. Stochastic sampling
    static std::mt19937 rng(1337);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int chosen = dist(rng);
    return cand[chosen];
}

} // namespace

void init_decode_engine() {
    if (dws.initialized) return;
    auto& E = get_engine();
    const int C = E.cfg.d_model ? E.cfg.d_model : 1024;
    const int L = E.cfg.n_layers ? E.cfg.n_layers : 24;
    const int H = E.cfg.n_heads ? E.cfg.n_heads : 16;
    const int D = E.cfg.head_dim ? E.cfg.head_dim : 64;
    const int hi = E.cfg.expert_hidden ? E.cfg.expert_hidden : 2048;
    const int Vp = E.cfg.vocab_padded ? E.cfg.vocab_padded : 50304;

    cudaMalloc(&dws.x, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.norm1_out, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.qkv_out, 3 * C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.q_rot, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.k_rot, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.v_raw, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.attn_out, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.x2, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.norm2_out, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.r_logits, 4 * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.exp_gates, 4 * sizeof(float));
    cudaMalloc(&dws.h1, hi * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.exp_out, 4 * C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.moe_out, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.final_norm, C * sizeof(__nv_bfloat16));
    cudaMalloc(&dws.logits, Vp * sizeof(float));
    cudaMalloc(&dws.out_tok, sizeof(int));

    cudaMalloc(&dws.S_state, (size_t)L * H * D * D * sizeof(float));
    cudaMalloc(&dws.H_state, (size_t)L * C * sizeof(float));
    cudaMalloc(&dws.recent_tokens, 256 * sizeof(int));
    dws.host_logits.resize(Vp, 0.0f);

    reset_decode_state();
    dws.initialized = true;
    printf("[decode] Native CUDA inference decoding engine initialized.\n");
}

void reset_decode_state() {
    auto& E = get_engine();
    const int C = E.cfg.d_model ? E.cfg.d_model : 1024;
    const int L = E.cfg.n_layers ? E.cfg.n_layers : 24;
    const int H = E.cfg.n_heads ? E.cfg.n_heads : 16;
    const int D = E.cfg.head_dim ? E.cfg.head_dim : 64;
    if (dws.S_state) cudaMemset(dws.S_state, 0, (size_t)L * H * D * D * sizeof(float));
    if (dws.H_state) cudaMemset(dws.H_state, 0, (size_t)L * C * sizeof(float));
    if (dws.recent_tokens) cudaMemset(dws.recent_tokens, 0xff, 256 * sizeof(int));
    dws.recent_count = 0;
    dws.recent_history.clear();
}

// ---------------------------------------------------------------------------
// Single Token Decode Step
// ---------------------------------------------------------------------------
int decode_step_native(int input_tok, int pos, float temperature, float top_p, int top_k, float rep_penalty, bool update_recent) {
    if (!dws.initialized) init_decode_engine();
    auto& E = get_engine();
    auto& lt = get_cublaslt();
    cudaStream_t s = get_runtime().get_stream();
    cublasHandle_t cb = lt.handle;
    cublasSetStream(cb, s);

    const int C = E.cfg.d_model;
    const int L = E.cfg.n_layers;
    const int H = E.cfg.n_heads;
    const int D = E.cfg.head_dim;
    const int hi = E.cfg.expert_hidden;
    const int Vp = E.cfg.vocab_padded;
    const int V = E.cfg.vocab_size;

    float one_f = 1.0f, zero_f = 0.0f;

    // 1. Embedding lookup
    emb_lookup_k<<<1024 / 256, 256, 0, s>>>(E.params.tok_emb, input_tok, dws.x, C);

    // 2. Transformer layers
    for (int l = 0; l < L; ++l) {
        // RMSNorm 1
        rmsnorm_1d_k<<<1, 256, 0, s>>>(dws.x, E.params.norm1_w[l], dws.norm1_out, C, 1e-6f);

        // QKV Projection: (1, 3C) = (1, C) @ (3C, C)^T
        cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                     3 * C, 1, C,
                     &one_f, E.params.qkv_w[l], CUDA_R_16BF, C,
                     dws.norm1_out, CUDA_R_16BF, C,
                     &zero_f, dws.qkv_out, CUDA_R_16BF, 3 * C,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Split QKV + RoPE + ELU+1
        const int T = E.cfg.seq_len ? E.cfg.seq_len : 512;
        split_qkv_rope_elu_k<<<H, D / 2, 0, s>>>(
            dws.qkv_out, E.rope_cos, E.rope_sin,
            dws.q_rot, dws.k_rot, dws.v_raw, pos, H, D, C, T);

        // Associative Attention step
        float* S_layer = dws.S_state + (size_t)l * H * D * D;
        assoc_attn_step_k<<<H, D, 0, s>>>(
            dws.q_rot, dws.k_rot, dws.v_raw,
            S_layer, E.params.gamma_raw[l],
            dws.attn_out, H, D);

        // Output Projection: (1, C) = (1, C) @ (C, C)^T
        cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                     C, 1, C,
                     &one_f, E.params.out_w[l], CUDA_R_16BF, C,
                     dws.attn_out, CUDA_R_16BF, C,
                     &zero_f, dws.x2, CUDA_R_16BF, C,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Residual 1
        vec_add_k<<<1024 / 256, 256, 0, s>>>(dws.x, dws.x2, C);

        // RMSNorm 2
        rmsnorm_1d_k<<<1, 256, 0, s>>>(dws.x, E.params.norm2_w[l], dws.norm2_out, C, 1e-6f);

        // Router: (1, 4) = (1, C) @ (4, C)^T
        cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                     4, 1, C,
                     &one_f, E.params.router_w[l], CUDA_R_16BF, C,
                     dws.norm2_out, CUDA_R_16BF, C,
                     &zero_f, dws.r_logits, CUDA_R_16BF, 4,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Top-2 Gating
        router_top2_k<<<1, 32, 0, s>>>(dws.r_logits, dws.exp_gates);

        // Experts 0..3: W1 -> GELU -> W2 (executed entirely on GPU stream, zero host sync)
        for (int e = 0; e < 4; ++e) {
            cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                         hi, 1, C,
                         &one_f, E.params.expert_w1[l] + (size_t)e * hi * C, CUDA_R_16BF, C,
                         dws.norm2_out, CUDA_R_16BF, C,
                         &zero_f, dws.h1, CUDA_R_16BF, hi,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            gelu_k<<<hi / 256, 256, 0, s>>>(dws.h1, hi);
            cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                         C, 1, hi,
                         &one_f, E.params.expert_w2[l] + (size_t)e * C * hi, CUDA_R_16BF, hi,
                         dws.h1, CUDA_R_16BF, hi,
                         &zero_f, dws.exp_out + (size_t)e * C, CUDA_R_16BF, C,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        // MoE Combine on GPU
        moe_combine_k<<<1024 / 256, 256, 0, s>>>(dws.exp_out, dws.exp_gates, dws.moe_out, C);

        // Liquid State Fusion
        float* H_layer = dws.H_state + (size_t)l * C;
        lsf_step_k<<<1, 256, 0, s>>>(dws.moe_out, H_layer, dws.x, C, E.cfg.lsf_alpha_min, E.cfg.lsf_alpha_max, 1.0f);
    }

    // 3. Final RMSNorm
    rmsnorm_1d_k<<<1, 256, 0, s>>>(dws.x, E.params.final_norm_w, dws.final_norm, C, 1e-6f);

    // 4. Head Projection: (1, Vp) = (1, C) @ (Vp, C)^T
    // Output directly in FP32
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                 Vp, 1, C,
                 &one_f, E.params.tok_emb, CUDA_R_16BF, C,
                 dws.final_norm, CUDA_R_16BF, C,
                 &zero_f, dws.logits, CUDA_R_32F, Vp,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 5. Copy logits to host and perform temperature + top_k + top_p stochastic sampling
    cudaMemcpyAsync(dws.host_logits.data(), dws.logits, V * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    int next_tok = sample_token_stochastic(dws.host_logits.data(), V, temperature, top_p, top_k, rep_penalty, dws.recent_history);

    // Update recent tokens history if requested
    if (update_recent) {
        dws.recent_history.push_back(next_tok);
        if (dws.recent_history.size() > 256) {
            dws.recent_history.erase(dws.recent_history.begin());
        }
    }

    return next_tok;
}

std::vector<int> generate_native(
    const std::vector<int>& prompt,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    std::function<void(int)> token_callback,
    float rep_penalty
) {
    if (!dws.initialized) init_decode_engine();
    reset_decode_state();
    std::vector<int> generated;
    if (prompt.empty()) return generated;

    // Seed recent tokens buffer with prompt tokens for repetition penalty
    size_t prompt_start = prompt.size() > 256 ? prompt.size() - 256 : 0;
    dws.recent_history.assign(prompt.begin() + prompt_start, prompt.end());

    // Prefill prompt tokens (updates recurrent attention state S and liquid state H)
    int cur_tok = prompt[0];
    for (size_t i = 0; i < prompt.size(); ++i) {
        bool is_last = (i == prompt.size() - 1);
        cur_tok = decode_step_native(prompt[i], (int)i, temperature, top_p, top_k, rep_penalty, is_last);
    }

    // Autoregressive generation
    int pos = (int)prompt.size();
    for (int step = 0; step < max_new_tokens; ++step) {
        if (cur_tok == 50256) break; // EOT
        generated.push_back(cur_tok);
        if (token_callback) token_callback(cur_tok);

        cur_tok = decode_step_native(cur_tok, pos++, temperature, top_p, top_k, rep_penalty, true);
    }

    return generated;
}
