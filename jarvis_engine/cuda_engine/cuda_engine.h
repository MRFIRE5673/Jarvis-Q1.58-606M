// =============================================================================
// cuda_engine.h — Jarvis-Q1.58-500M Native CUDA Engine Core Types
// =============================================================================
// Architecture (IMMUTABLE per architecture.md):
//   L=24 layers, C=1024, H=16 heads, D=64, E=4 experts, K=2, hi=2048
//   V=50257, Vpad=50304, T=512, B=4, accum=2
//
// Hardware: RTX 5070 GB205 / SM120 / 48 SMs / 672 GB/s / 48 MB L2
//
// Engineering principles:
//   - All buffers sized from first principles, not invented benchmarks
//   - L2 budget: 48 MB. Hot activation working set must fit.
//   - BF16 for weights & activations, FP32 for optimizer moments & norms
//   - cuBLASLt for all matrix ops (algorithm heuristics queried at runtime)
//   - Single-stream CUDA Graph: eliminates CPU dispatch overhead
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cstdio>

#include "runtime.h"  // includes CUDA_CHECK, CUDA_CHECK_LAST, JarvisRuntime

// Convenience alias
using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// Model configuration (values from architecture.md — do not change)
// ---------------------------------------------------------------------------
struct JarvisConfig {
    // ---- Architecture (locked) ----
    int n_layers      = 24;
    int d_model       = 1024;    // C
    int n_heads       = 16;      // H
    int head_dim      = 64;      // D = C / H
    int n_experts     = 4;       // E
    int top_k         = 2;       // K
    int expert_hidden = 2048;    // hi
    int vocab_size    = 50257;
    int vocab_padded  = 50304;   // padded to multiple of 64 for TC alignment

    // ---- Training setup ----
    int   batch_size   = 8;
    int   seq_len      = 512;
    int   accum_steps  = 2;

    // ---- Optimizer (AdamW) ----
    float lr           = 1e-4f;
    float beta1        = 0.90f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float weight_decay = 0.01f;
    float grad_clip    = 1.0f;
    int   warmup_steps = 100;
    int   decay_steps  = 10000;
    float min_lr_ratio = 0.1f;

    // ---- Aux losses ----
    float balance_alpha   = 0.01f;   // MoE load balance
    float reflect_lambda  = 0.01f;   // reflective penalty
    float reflect_tau_max = 1.0f;    // variance threshold

    // ---- LSF membrane ----
    float lsf_alpha_min = 0.1f;
    float lsf_alpha_max = 0.99f;
    float lsf_var_scale = 1.0f;     // initial var_scale per layer

    // ---- Attention chunking ----
    int chunk_size = 64;

    // ---- Device ----
    int device = 0;

    // Derived
    int NT()  const { return batch_size * seq_len; }
    int NTK() const { return NT() * top_k; }
};

// ---------------------------------------------------------------------------
// Memory budget analysis (first principles, RTX 5070 / 12 GB)
// ---------------------------------------------------------------------------
// Per-layer forward activations (BF16, B=4, T=512):
//   layer_x        : 4×512×1024 × 2B = 4.19 MB
//   layer_x2       : 4.19 MB  (staging buffer for residual + norm input)
//   norm1_out      : 4.19 MB
//   qkv_out        : 4×512×3072 × 2B = 12.58 MB
//   q_rope/k_rope  : 4×16×512×64 × 2B = 4.19 MB each (×2 = 8.39 MB)
//   v_buf          : 4.19 MB
//   attn_state S   : 4×16×64×64 × 2B = 0.524 MB  ← O(BH D^2), not O(T)!
//   attn_out       : 4.19 MB
//   norm2_out      : 4.19 MB
//   router_logits  : 4×512×4 × 4B(FP32) = 0.016 MB
//   Sparse mode compacts dispatch_x/layer_h1/expert_out to NT*K route rows;
//   dense mode retains E*NT rows for the CUDA Graph path.
//   moe_out        : 4.19 MB
//   lsf_out        : 4.19 MB
// ─────────────────────────────────────────────────────────────────────────
// Total hot workspace ~= 88 MB > 48 MB L2
// Strategy: mark layer_x persisting; layer_x2 and attn_state use normal L2 policy.
// The rest streams through GDDR7 at 672 GB/s

// ---------------------------------------------------------------------------
// Parameters — all 24 layers
// Array-of-pointers layout (each element is a separate cudaMalloc).
// We keep norm/gamma/lsf as FP32 (they're tiny); quantized weights as BF16.
// ---------------------------------------------------------------------------
struct JarvisParams {
    // Embedding + final norm + LM head (tied embedding)
    bf16*  tok_emb         = nullptr;  // [Vpad, C] BF16
    float* tok_emb_fp32    = nullptr;  // [Vpad, C] FP32 master
    float* tok_emb_m       = nullptr;  // [Vpad, C] Adam m
    float* tok_emb_v       = nullptr;  // [Vpad, C] Adam v
    bf16*  final_norm_w    = nullptr;  // [C]
    float* final_norm_w_fp32 = nullptr;
    float* final_norm_m    = nullptr;
    float* final_norm_v    = nullptr;
    // lm_head weight-tied to tok_emb (pointer alias — no separate alloc)

    // Per-layer arrays of device pointers
    // Quantized projection weights: BF16 (ternary × alpha, re-quantized each step)
    bf16** qkv_w        = nullptr;    // [L] → [3C, C]
    bf16** out_w        = nullptr;    // [L] → [C, C]
    bf16** expert_w1    = nullptr;    // [L] → [E × hi × C]
    bf16** expert_w2    = nullptr;    // [L] → [E × C × hi]
    bf16** router_w     = nullptr;    // [L] → [C, E]  (not quantized — too small)

    // FP32 master copies for STE optimizer updates
    float** qkv_fp32    = nullptr;
    float** out_fp32    = nullptr;
    float** exp_w1_fp32 = nullptr;
    float** exp_w2_fp32 = nullptr;
    float** router_fp32 = nullptr;   // [L] -> [E, C] FP32 master for router

    // Adam moments for quantized weights
    float** qkv_m       = nullptr;  float** qkv_v       = nullptr;
    float** out_m       = nullptr;  float** out_v       = nullptr;
    float** exp_w1_m    = nullptr;  float** exp_w1_v    = nullptr;
    float** exp_w2_m    = nullptr;  float** exp_w2_v    = nullptr;
    float** router_m    = nullptr;  float** router_v    = nullptr;

    // Non-quantized small params (FP32 — BF16 granularity insufficient)
    bf16**  norm1_w     = nullptr;    // [L] → [C]
    bf16**  norm2_w     = nullptr;    // [L] → [C]
    float** norm1_fp32  = nullptr;
    float** norm2_fp32  = nullptr;
    float** norm1_m     = nullptr;  float** norm1_v     = nullptr;
    float** norm2_m     = nullptr;  float** norm2_v     = nullptr;

    float** gamma_raw   = nullptr;    // [L] → [H] per-head attention decay
    float** gamma_m     = nullptr;  float** gamma_v     = nullptr;
    float** lsf_vscale  = nullptr;    // [L] → [1] var_scale scalar
    float** lsf_vm      = nullptr;  float** lsf_vv      = nullptr;
};

// ---------------------------------------------------------------------------
// Workspace — per-step intermediate activations
// Allocated once, reused every step (static CUDA Graph requirement)
// ---------------------------------------------------------------------------
struct JarvisWorkspace {
    // Inputs
    int*   token_ids    = nullptr;  // [NT] int32

    // Activation pipeline
    bf16*  tok_emb_out  = nullptr;  // [NT, C]  after embedding lookup

    // Layer-resident buffers (overwritten each layer)
    bf16*  layer_x      = nullptr;  // [NT, C]  flowing residual stream  ← L2-persisting window
    bf16*  layer_x2     = nullptr;  // [NT, C]  staging / post-add
    bf16*  norm1_out    = nullptr;  // [NT, C]
    bf16*  qkv_out      = nullptr;  // [NT, 3C]
    bf16*  q_buf        = nullptr;  // [B, H, T, D]  post ELU+RoPE
    bf16*  k_buf        = nullptr;  // [B, H, T, D]  post ELU+RoPE
    bf16*  v_buf        = nullptr;  // [B, H, T, D]
    bf16*  attn_state   = nullptr;  // [B, H, D, D]  recurrent state S
    bf16*  attn_out     = nullptr;  // [NT, C]
    bf16*  norm2_out    = nullptr;  // [NT, C]
    bf16*  router_logits= nullptr;  // [NT, E] as BF16 (router is not quantized)
    float* router_probs = nullptr;  // [NT, E] FP32 softmax probs
    int*   topk_idx     = nullptr;  // [NT, K] int32
    float* topk_gates   = nullptr;  // [NT, K] normalized gate weights
    int*   route_pos    = nullptr;  // [NT, K] position in expert-major dispatch
    int*   expert_counts = nullptr; // [E] active rows per expert
    int*   expert_offsets = nullptr; // [E+1] compact expert segment offsets
    int*   route_write  = nullptr;  // [E] temporary compact dispatch cursors
    float* route_probs_cache = nullptr; // [L,NT,E]
    int*   topk_idx_cache = nullptr;    // [L,NT,K]
    float* topk_gates_cache = nullptr;  // [L,NT,K]
    int*   route_pos_cache = nullptr;   // [L,NT,K]
    int*   expert_counts_cache = nullptr; // [L,E]
    int*   expert_offsets_cache = nullptr; // [L,E+1]
    bf16*  dispatch_x   = nullptr;  // [NTK, C] compact expert-major token reps
    bf16*  layer_h1     = nullptr;  // [NTK, hi]  expert W1 output (in-place GELU)
    bf16*  expert_out   = nullptr;  // [NTK, C]   expert W2 output
    bf16*  moe_out      = nullptr;  // [NT, C]   scatter-combined
    float* lsf_alpha    = nullptr;  // [1] FP32 dynamic alpha per layer step
    bf16*  lsf_h        = nullptr;  // [L, B, C]  persistent membrane states
    bf16*  lsf_out      = nullptr;  // [NT, C]   output of LSF

    // LM head outputs
    bf16*  logits       = nullptr;  // [NT, Vpad]
    float* loss         = nullptr;  // [1] scalar

    // Backward stash (saved for backward pass — 24 layers)
    bf16*  stash_x      = nullptr;  // [L, NT, C]  pre-norm1 residual
    bf16*  stash_h1     = nullptr;  // [L, NTK, hi]  pre-GELU (for GELU bwd)

    // Pre-quantized FP8 weight caches (quantized once per step, E4M3)
    uint8_t** qkv_q8    = nullptr;  // [L] -> [3C*C] FP8
    uint8_t** out_q8    = nullptr;  // [L] -> [C*C]  FP8
    uint8_t** w1_q8     = nullptr;  // [L] -> [E*hi*C] FP8
    uint8_t** w2_q8     = nullptr;  // [L] -> [E*C*hi] FP8
    float**   qkv_inv   = nullptr;  // [L] -> [1] inv scale for qkv_q8
    float**   out_inv   = nullptr;  // [L] -> [1] inv scale for out_q8
    float**   w1_inv    = nullptr;  // [L] -> [1] inv scale for w1_q8
    float**   w2_inv    = nullptr;  // [L] -> [1] inv scale for w2_q8

    // Gradient buffers — reused across layers
    bf16*  d_x          = nullptr;  // [NT, C]   gradient flowing backward
    bf16*  d_norm1      = nullptr;  // [NT, C]
    bf16*  d_qkv        = nullptr;  // [NT, 3C]
    bf16*  d_attn_out   = nullptr;  // [NT, C]
    bf16*  d_norm2      = nullptr;  // [NT, C]
    bf16*  d_moe        = nullptr;  // [NT, C]
    bf16*  d_dispatch   = nullptr;  // [NTK, C]
    bf16*  d_h1         = nullptr;  // [NTK, hi]

    // Weight gradient accumulators (per layer, accumulated over accum_steps)
    bf16** d_qkv_w      = nullptr;  // [L] → [3C, C]
    bf16** d_out_w      = nullptr;  // [L] → [C, C]
    bf16** d_exp_w1     = nullptr;  // [L] → [E*hi*C]
    bf16** d_exp_w2     = nullptr;  // [L] → [E*C*hi]
    bf16** d_norm1_w    = nullptr;  // [L] → [C]
    bf16** d_norm2_w    = nullptr;  // [L] → [C]
    float** d_norm1_f   = nullptr;  // [L] → [C] FP32 accumulator for norm1 grad
    float** d_norm2_f   = nullptr;  // [L] → [C] FP32 accumulator for norm2 grad
    bf16** d_router_w   = nullptr;  // [L] → [E] (router grad)
    float** d_gamma     = nullptr;  // [L] → [H] FP32
    bf16*  d_tok_emb    = nullptr;  // [Vpad, C]  (accumulated, BF16)
    bf16*  d_final_norm = nullptr;  // [C]
    float* d_final_norm_f = nullptr; // [C] FP32 accumulator for final-norm grad
    float* dg           = nullptr;  // [NT, 2] gate grads
    bf16*  d_rlogits    = nullptr;  // [NT, E] router logit grads

    // Grad norm reduction
    float* grad_norm_sq = nullptr;  // [1] scalar

    // Aux losses
    float* l_balance    = nullptr;  // [1] MoE load balance loss
    float* l_reflect    = nullptr;  // [1] reflective penalty
    float* mu_running   = nullptr;  // [1] EMA mean for reflective penalty

    // AdamW state
    int*   adam_step    = nullptr;  // [1] int — current optimizer step
    float* d_bc1        = nullptr;  // [1] bias correction 1 (device scalar)
    float* d_bc2        = nullptr;  // [1] bias correction 2 (device scalar)
    float* d_clip_coef  = nullptr;  // [1] gradient clipping coefficient
    float* d_lr         = nullptr;  // [1] scheduled learning rate
};

// ---------------------------------------------------------------------------
// Allocation / free prototypes (implementations in cuda_engine.cpp)
// ---------------------------------------------------------------------------
void alloc_params(JarvisParams& p, const JarvisConfig& cfg);
void free_params(JarvisParams& p, const JarvisConfig& cfg);
void alloc_workspace(JarvisWorkspace& ws, const JarvisConfig& cfg);
void free_workspace(JarvisWorkspace& ws, const JarvisConfig& cfg);

// Utility: zero all grad accumulator buffers before an accumulation window
void zero_grad_buffers(JarvisWorkspace& ws, const JarvisConfig& cfg, cudaStream_t s);
