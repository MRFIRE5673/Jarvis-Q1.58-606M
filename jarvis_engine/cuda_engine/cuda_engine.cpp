// =============================================================================
// cuda_engine.cpp — Parameter & Workspace Memory Allocation
// =============================================================================
// Allocates all GPU memory. Called ONCE before the CUDA Graph is captured.
// All pointers are stable throughout the training run (CUDA Graph requirement).
// =============================================================================
#include "cuda_engine.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
static void dev_alloc_bf16(bf16** p, size_t n, const char* tag = nullptr) {
    size_t bytes = n * sizeof(bf16);
    if (cudaMalloc(p, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc BF16 failed");
    cudaMemset(*p, 0, bytes);
    if (tag) printf("  [alloc] %-35s  %9zu elems  %7.2f MB\n",
                     tag, n, bytes / 1048576.0);
}

static bool sparse_moe_requested_alloc() {
    const char* v = std::getenv("JARVIS_MOE_SPARSE");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static void dev_alloc_fp32(float** p, size_t n, const char* tag = nullptr) {
    size_t bytes = n * sizeof(float);
    if (cudaMalloc(p, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc FP32 failed");
    cudaMemset(*p, 0, bytes);
    if (tag) printf("  [alloc] %-35s  %9zu elems  %7.2f MB\n",
                     tag, n, bytes / 1048576.0);
}

static void dev_alloc_i32(int** p, size_t n, const char* tag = nullptr) {
    size_t bytes = n * sizeof(int);
    if (cudaMalloc(p, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc I32 failed");
    cudaMemset(*p, 0, bytes);
}

// Allocate an array of L device pointers, each pointing to n BF16 elements
static void alloc_layer_bf16(bf16*** arr, int L, size_t n_each) {
    *arr = new bf16*[L]();
    for (int l = 0; l < L; ++l) dev_alloc_bf16(&(*arr)[l], n_each);
}

// Allocate an array of L device pointers, each pointing to n FP32 elements
static void alloc_layer_fp32(float*** arr, int L, size_t n_each) {
    *arr = new float*[L]();
    for (int l = 0; l < L; ++l) dev_alloc_fp32(&(*arr)[l], n_each);
}

// ---------------------------------------------------------------------------
// alloc_params
// ---------------------------------------------------------------------------
void alloc_params(JarvisParams& p, const JarvisConfig& cfg) {
    const int L   = cfg.n_layers;
    const int C   = cfg.d_model;
    const int H   = cfg.n_heads;
    const int E   = cfg.n_experts;
    const int hi  = cfg.expert_hidden;
    const int Vp  = cfg.vocab_padded;

    printf("[alloc_params] Allocating model parameters...\n");

    // ---- Embedding ----
    dev_alloc_bf16(&p.tok_emb,      (size_t)Vp * C, "tok_emb BF16");
    dev_alloc_fp32(&p.tok_emb_fp32, (size_t)Vp * C, "tok_emb FP32 master");
    dev_alloc_fp32(&p.tok_emb_m,    (size_t)Vp * C, "tok_emb Adam m");
    dev_alloc_fp32(&p.tok_emb_v,    (size_t)Vp * C, "tok_emb Adam v");

    // ---- Final norm ----
    dev_alloc_bf16(&p.final_norm_w,      (size_t)C, "final_norm_w BF16");
    dev_alloc_fp32(&p.final_norm_w_fp32, (size_t)C, "final_norm_w FP32");
    dev_alloc_fp32(&p.final_norm_m,      (size_t)C, "final_norm Adam m");
    dev_alloc_fp32(&p.final_norm_v,      (size_t)C, "final_norm Adam v");

    // ---- Per-layer: QKV [3C, C] ----
    alloc_layer_bf16(&p.qkv_w,     L, (size_t)3 * C * C);
    alloc_layer_fp32(&p.qkv_fp32,  L, (size_t)3 * C * C);
    alloc_layer_fp32(&p.qkv_m,     L, (size_t)3 * C * C);
    alloc_layer_fp32(&p.qkv_v,     L, (size_t)3 * C * C);

    // ---- Per-layer: Out proj [C, C] ----
    alloc_layer_bf16(&p.out_w,     L, (size_t)C * C);
    alloc_layer_fp32(&p.out_fp32,  L, (size_t)C * C);
    alloc_layer_fp32(&p.out_m,     L, (size_t)C * C);
    alloc_layer_fp32(&p.out_v,     L, (size_t)C * C);

    // ---- Per-layer: Expert W1 [E, hi, C] ----
    alloc_layer_bf16(&p.expert_w1,   L, (size_t)E * hi * C);
    alloc_layer_fp32(&p.exp_w1_fp32, L, (size_t)E * hi * C);
    alloc_layer_fp32(&p.exp_w1_m,    L, (size_t)E * hi * C);
    alloc_layer_fp32(&p.exp_w1_v,    L, (size_t)E * hi * C);

    // ---- Per-layer: Expert W2 [E, C, hi] ----
    alloc_layer_bf16(&p.expert_w2,   L, (size_t)E * C * hi);
    alloc_layer_fp32(&p.exp_w2_fp32, L, (size_t)E * C * hi);
    alloc_layer_fp32(&p.exp_w2_m,    L, (size_t)E * C * hi);
    alloc_layer_fp32(&p.exp_w2_v,    L, (size_t)E * C * hi);

    // ---- Per-layer: Router [C, E] (not ternary quantized — too small) ----
    alloc_layer_bf16(&p.router_w,  L, (size_t)C * E);
    alloc_layer_fp32(&p.router_fp32, L, (size_t)C * E);
    alloc_layer_fp32(&p.router_m,  L, (size_t)C * E);
    alloc_layer_fp32(&p.router_v,  L, (size_t)C * E);

    // ---- Per-layer: RMSNorm weights [C] ----
    alloc_layer_bf16(&p.norm1_w,   L, (size_t)C);
    alloc_layer_fp32(&p.norm1_fp32,L, (size_t)C);
    alloc_layer_fp32(&p.norm1_m,   L, (size_t)C);
    alloc_layer_fp32(&p.norm1_v,   L, (size_t)C);

    alloc_layer_bf16(&p.norm2_w,   L, (size_t)C);
    alloc_layer_fp32(&p.norm2_fp32,L, (size_t)C);
    alloc_layer_fp32(&p.norm2_m,   L, (size_t)C);
    alloc_layer_fp32(&p.norm2_v,   L, (size_t)C);

    // ---- Per-layer: Per-head gamma [H] ----
    alloc_layer_fp32(&p.gamma_raw, L, (size_t)H);
    alloc_layer_fp32(&p.gamma_m,   L, (size_t)H);
    alloc_layer_fp32(&p.gamma_v,   L, (size_t)H);

    // ---- Per-layer: LSF var_scale [1] ----
    alloc_layer_fp32(&p.lsf_vscale, L, 1);
    alloc_layer_fp32(&p.lsf_vm,     L, 1);
    alloc_layer_fp32(&p.lsf_vv,     L, 1);

    printf("[alloc_params] Done.\n");
}

// ---------------------------------------------------------------------------
// free_params
// ---------------------------------------------------------------------------
void free_params(JarvisParams& p, const JarvisConfig& cfg) {
    const int L = cfg.n_layers;

    auto free_bf16 = [](bf16* ptr) { if (ptr) cudaFree(ptr); };
    auto free_fp32 = [](float* ptr) { if (ptr) cudaFree(ptr); };
    auto free_layer_bf16 = [&](bf16** arr) {
        if (!arr) return;
        for (int l = 0; l < L; ++l) if (arr[l]) cudaFree(arr[l]);
        delete[] arr;
    };
    auto free_layer_fp32 = [&](float** arr) {
        if (!arr) return;
        for (int l = 0; l < L; ++l) if (arr[l]) cudaFree(arr[l]);
        delete[] arr;
    };

    free_bf16(p.tok_emb);      free_fp32(p.tok_emb_fp32);
    free_fp32(p.tok_emb_m);    free_fp32(p.tok_emb_v);
    free_bf16(p.final_norm_w); free_fp32(p.final_norm_w_fp32);
    free_fp32(p.final_norm_m); free_fp32(p.final_norm_v);

    free_layer_bf16(p.qkv_w);    free_layer_fp32(p.qkv_fp32);
    free_layer_fp32(p.qkv_m);    free_layer_fp32(p.qkv_v);
    free_layer_bf16(p.out_w);    free_layer_fp32(p.out_fp32);
    free_layer_fp32(p.out_m);    free_layer_fp32(p.out_v);
    free_layer_bf16(p.expert_w1);free_layer_fp32(p.exp_w1_fp32);
    free_layer_fp32(p.exp_w1_m); free_layer_fp32(p.exp_w1_v);
    free_layer_bf16(p.expert_w2);free_layer_fp32(p.exp_w2_fp32);
    free_layer_fp32(p.exp_w2_m); free_layer_fp32(p.exp_w2_v);
    free_layer_bf16(p.router_w); free_layer_fp32(p.router_m);
    free_layer_fp32(p.router_v); free_layer_fp32(p.router_fp32);
    free_layer_bf16(p.norm1_w);  free_layer_fp32(p.norm1_fp32);
    free_layer_fp32(p.norm1_m);  free_layer_fp32(p.norm1_v);
    free_layer_bf16(p.norm2_w);  free_layer_fp32(p.norm2_fp32);
    free_layer_fp32(p.norm2_m);  free_layer_fp32(p.norm2_v);
    free_layer_fp32(p.gamma_raw);free_layer_fp32(p.gamma_m); free_layer_fp32(p.gamma_v);
    free_layer_fp32(p.lsf_vscale);free_layer_fp32(p.lsf_vm);free_layer_fp32(p.lsf_vv);

    memset(&p, 0, sizeof(p));
}

// ---------------------------------------------------------------------------
// alloc_workspace
// ---------------------------------------------------------------------------
void alloc_workspace(JarvisWorkspace& ws, const JarvisConfig& cfg) {
    const int L   = cfg.n_layers;
    const int C   = cfg.d_model;
    const int H   = cfg.n_heads;
    const int D   = cfg.head_dim;
    const int E   = cfg.n_experts;
    const int K   = cfg.top_k;
    const int hi  = cfg.expert_hidden;
    const int Vp  = cfg.vocab_padded;
    const int B   = cfg.batch_size;
    const int T   = cfg.seq_len;
    const int NT  = B * T;
    const int NTK = NT * K;
    const size_t route_rows = sparse_moe_requested_alloc() ? (size_t)NTK : (size_t)E * NT;

    printf("[alloc_workspace] Allocating training workspace...\n");

    dev_alloc_i32 (&ws.token_ids,    (size_t)NT * cfg.accum_steps + 1, "token_ids");
    // Capture and warmup execute before the first host batch is copied in.
    // Zero is a valid vocabulary ID and prevents undefined embedding reads.
    CUDA_CHECK(cudaMemset(ws.token_ids, 0, ((size_t)NT * cfg.accum_steps + 1) * sizeof(int)));
    dev_alloc_bf16(&ws.tok_emb_out,  (size_t)NT * C,     "tok_emb_out");

    // Hot path activations; layer_x receives the L2 persisting access window.
    dev_alloc_bf16(&ws.layer_x,      (size_t)NT * C,     "layer_x [L2-hot]");
    dev_alloc_bf16(&ws.layer_x2,     (size_t)NT * C,     "layer_x2 [L2-hot]");

    dev_alloc_bf16(&ws.norm1_out,    (size_t)NT * C,     "norm1_out");
    dev_alloc_bf16(&ws.qkv_out,      (size_t)NT * 3 * C, "qkv_out");
    dev_alloc_bf16(&ws.q_buf,        (size_t)B * H * T * D, "q_buf");
    dev_alloc_bf16(&ws.k_buf,        (size_t)B * H * T * D, "k_buf");
    dev_alloc_bf16(&ws.v_buf,        (size_t)B * H * T * D, "v_buf");

    // attn_state is O(BHD^2), not O(T).
    dev_alloc_bf16(&ws.attn_state,   (size_t)B * H * D * D, "attn_state [L2-hot]");
    dev_alloc_bf16(&ws.attn_out,     (size_t)NT * C,     "attn_out");

    dev_alloc_bf16(&ws.norm2_out,    (size_t)NT * C,     "norm2_out");
    dev_alloc_bf16(&ws.router_logits,(size_t)NT * E,     "router_logits");
    dev_alloc_fp32(&ws.router_probs, (size_t)NT * E,     "router_probs FP32");
    dev_alloc_i32 (&ws.topk_idx,     (size_t)NT * K,     "topk_idx");
    dev_alloc_fp32(&ws.topk_gates,   (size_t)NT * K,     "topk_gates FP32");
    dev_alloc_i32 (&ws.route_pos,    (size_t)NT * K,     "route_pos");
    dev_alloc_i32 (&ws.expert_counts, (size_t)E,          "expert_counts");
    dev_alloc_i32 (&ws.expert_offsets, (size_t)E + 1,     "expert_offsets");
    dev_alloc_i32 (&ws.route_write, (size_t)E,             "route_write");
    if (sparse_moe_requested_alloc()) {
        dev_alloc_fp32(&ws.route_probs_cache, (size_t)cfg.accum_steps * L * NT * E, "route_probs_cache [A,L,NT,E]");
        dev_alloc_i32(&ws.topk_idx_cache, (size_t)cfg.accum_steps * L * NT * K, "topk_idx_cache [A,L,NT,K]");
        dev_alloc_fp32(&ws.topk_gates_cache, (size_t)cfg.accum_steps * L * NT * K, "topk_gates_cache [A,L,NT,K]");
        dev_alloc_i32(&ws.route_pos_cache, (size_t)cfg.accum_steps * L * NT * K, "route_pos_cache [A,L,NT,K]");
        dev_alloc_i32(&ws.expert_counts_cache, (size_t)cfg.accum_steps * L * E, "expert_counts_cache [A,L,E]");
        dev_alloc_i32(&ws.expert_offsets_cache, (size_t)cfg.accum_steps * L * (E + 1), "expert_offsets_cache [A,L,E+1]");
    }
    dev_alloc_bf16(&ws.dispatch_x,   route_rows * C,     sparse_moe_requested_alloc() ? "dispatch_x [NTK,C] compact" : "dispatch_x [E,NT,C]");
    dev_alloc_bf16(&ws.layer_h1,     route_rows * hi,    sparse_moe_requested_alloc() ? "layer_h1 [NTK,hi] compact" : "layer_h1 [E,NT,hi]");
    dev_alloc_bf16(&ws.expert_out,   route_rows * C,     sparse_moe_requested_alloc() ? "expert_out [NTK,C] compact" : "expert_out [E,NT,C]");
    dev_alloc_bf16(&ws.moe_out,      (size_t)NT * C,     "moe_out");
    dev_alloc_fp32(&ws.lsf_alpha,    1,                  "lsf_alpha scalar");
    dev_alloc_bf16(&ws.lsf_h,        (size_t)L * B * C,  "lsf_h [L,B,C]");
    dev_alloc_bf16(&ws.lsf_out,      (size_t)NT * C,     "lsf_out");

    dev_alloc_bf16(&ws.logits,       (size_t)NT * Vp,    "logits [NT,Vpad]");
    dev_alloc_fp32(&ws.loss,         1,                  "loss scalar");

    // Backward stash
    dev_alloc_bf16(&ws.stash_x,      (size_t)L * NT * C, "stash_x [L,NT,C]");
    dev_alloc_bf16(&ws.stash_h1,     route_rows * hi, "stash_h1 route workspace");

    // FP8 weight caches (one uint8 byte per element, allocated per-layer)
    auto alloc_u8_arr = [&](uint8_t**& arr, size_t n_each) {
        arr = new uint8_t*[L]();
        for (int l = 0; l < L; ++l) { cudaMalloc(&arr[l], n_each); cudaMemset(arr[l], 0, n_each); }
    };
    auto alloc_f1_arr = [&](float**& arr) {
        arr = new float*[L]();
        for (int l = 0; l < L; ++l) { cudaMalloc(&arr[l], sizeof(float)); cudaMemset(arr[l], 0, sizeof(float)); }
    };
    alloc_u8_arr(ws.qkv_q8, (size_t)3*C*C);
    alloc_u8_arr(ws.out_q8, (size_t)C*C);
    alloc_u8_arr(ws.w1_q8,  (size_t)E*hi*C);
    alloc_u8_arr(ws.w2_q8,  (size_t)E*C*hi);
    alloc_f1_arr(ws.qkv_inv);
    alloc_f1_arr(ws.out_inv);
    alloc_f1_arr(ws.w1_inv);
    alloc_f1_arr(ws.w2_inv);
    printf("  [alloc] FP8 weight cache [L,per-layer]         %zu MB\n",
           ((size_t)3*C*C + C*C + E*hi*C + E*C*hi) * L / 1048576);

    // Grad buffers
    dev_alloc_bf16(&ws.d_x,          (size_t)NT * C,     "d_x");
    dev_alloc_bf16(&ws.d_norm1,      (size_t)NT * C,     "d_norm1");
    dev_alloc_bf16(&ws.d_qkv,        (size_t)NT * 3 * C, "d_qkv");
    dev_alloc_bf16(&ws.d_attn_out,   (size_t)NT * C,     "d_attn_out");
    dev_alloc_bf16(&ws.d_norm2,      (size_t)NT * C,     "d_norm2");
    dev_alloc_bf16(&ws.d_moe,        (size_t)NT * C,     "d_moe");
    dev_alloc_bf16(&ws.d_dispatch,   route_rows * C,     "d_dispatch route workspace");
    dev_alloc_bf16(&ws.d_h1,         route_rows * hi,    "d_h1 route workspace");
    dev_alloc_fp32(&ws.dg,           (size_t)NT * 2,      "dg");
    dev_alloc_bf16(&ws.d_rlogits,    (size_t)NT * E,      "d_rlogits");
    dev_alloc_fp32(&ws.d_final_norm_f, (size_t)C,         "d_final_norm_f");

    // Weight grad accumulators (per layer)
    ws.d_qkv_w   = new bf16*[L]();
    ws.d_out_w   = new bf16*[L]();
    ws.d_exp_w1  = new bf16*[L]();
    ws.d_exp_w2  = new bf16*[L]();
    ws.d_norm1_w = new bf16*[L]();
    ws.d_norm2_w = new bf16*[L]();
    ws.d_norm1_f = new float*[L]();
    ws.d_norm2_f = new float*[L]();
    ws.d_gamma   = new float*[L]();
    ws.d_router_w = new bf16*[L]();

    for (int l = 0; l < L; ++l) {
        dev_alloc_bf16(&ws.d_qkv_w[l],  (size_t)3 * C * C);
        dev_alloc_bf16(&ws.d_out_w[l],  (size_t)C * C);
        dev_alloc_bf16(&ws.d_exp_w1[l], (size_t)E * hi * C);
        dev_alloc_bf16(&ws.d_exp_w2[l], (size_t)E * C * hi);
        dev_alloc_bf16(&ws.d_norm1_w[l],(size_t)C);
        dev_alloc_bf16(&ws.d_norm2_w[l],(size_t)C);
        dev_alloc_fp32(&ws.d_norm1_f[l],(size_t)C);
        dev_alloc_fp32(&ws.d_norm2_f[l],(size_t)C);
        dev_alloc_fp32(&ws.d_gamma[l],  (size_t)H);
        dev_alloc_bf16(&ws.d_router_w[l], (size_t)C * E);
    }

    dev_alloc_bf16(&ws.d_tok_emb,    (size_t)Vp * C,   "d_tok_emb");
    dev_alloc_bf16(&ws.d_final_norm, (size_t)C,        "d_final_norm");
    dev_alloc_fp32(&ws.grad_norm_sq, 1,                "grad_norm_sq");
    dev_alloc_fp32(&ws.l_balance,    1,                "l_balance");
    dev_alloc_fp32(&ws.l_reflect,    1,                "l_reflect");
    dev_alloc_fp32(&ws.mu_running,   1,                "mu_running");
    dev_alloc_i32 (&ws.adam_step,    1,                "adam_step");
    dev_alloc_fp32(&ws.d_bc1,        1,                "d_bc1 (bias_corr1)");
    dev_alloc_fp32(&ws.d_bc2,        1,                "d_bc2 (bias_corr2)");
    dev_alloc_fp32(&ws.d_clip_coef,  1,                "d_clip_coef");
    dev_alloc_fp32(&ws.d_lr,         1,                "d_lr (scheduled lr)");

    printf("[alloc_workspace] Done.\n");
}

// ---------------------------------------------------------------------------
// free_workspace
// ---------------------------------------------------------------------------
void free_workspace(JarvisWorkspace& ws, const JarvisConfig& cfg) {
    const int L = cfg.n_layers;

    #define FD(ptr) do { if (ptr) { cudaFree(ptr); ptr = nullptr; } } while(0)

    FD(ws.token_ids); FD(ws.tok_emb_out);
    FD(ws.layer_x);   FD(ws.layer_x2);
    FD(ws.norm1_out); FD(ws.qkv_out);
    FD(ws.q_buf);     FD(ws.k_buf);   FD(ws.v_buf);
    FD(ws.attn_state);FD(ws.attn_out);
    FD(ws.norm2_out); FD(ws.router_logits);
    FD(ws.router_probs); FD(ws.topk_idx); FD(ws.topk_gates);
    FD(ws.route_pos); FD(ws.expert_counts); FD(ws.expert_offsets); FD(ws.route_write);
    FD(ws.route_probs_cache); FD(ws.topk_idx_cache); FD(ws.topk_gates_cache);
    FD(ws.route_pos_cache); FD(ws.expert_counts_cache); FD(ws.expert_offsets_cache);
    FD(ws.dispatch_x);FD(ws.layer_h1); FD(ws.expert_out); FD(ws.moe_out);
    FD(ws.lsf_alpha); FD(ws.lsf_h);   FD(ws.lsf_out);
    FD(ws.logits);    FD(ws.loss);
    FD(ws.stash_x);   FD(ws.stash_h1);

    auto fl_u8 = [&](uint8_t** arr) {
        if (!arr) return;
        for (int l = 0; l < L; ++l) if (arr[l]) cudaFree(arr[l]);
        delete[] arr;
    };
    auto fl_bf16 = [&](bf16** arr) {
        if (!arr) return;
        for (int l = 0; l < L; ++l) if (arr[l]) cudaFree(arr[l]);
        delete[] arr;
    };
    auto fl_fp32 = [&](float** arr) {
        if (!arr) return;
        for (int l = 0; l < L; ++l) if (arr[l]) cudaFree(arr[l]);
        delete[] arr;
    };

    // FP8 weight caches
    fl_u8(ws.qkv_q8); fl_u8(ws.out_q8); fl_u8(ws.w1_q8); fl_u8(ws.w2_q8);
    fl_fp32(ws.qkv_inv); fl_fp32(ws.out_inv); fl_fp32(ws.w1_inv); fl_fp32(ws.w2_inv);
    FD(ws.d_x);       FD(ws.d_norm1);
    FD(ws.d_qkv);     FD(ws.d_attn_out); FD(ws.d_norm2);
    FD(ws.d_moe);     FD(ws.d_dispatch); FD(ws.d_h1);
    FD(ws.d_tok_emb); FD(ws.d_final_norm); FD(ws.d_final_norm_f); FD(ws.dg); FD(ws.d_rlogits);
    FD(ws.grad_norm_sq); FD(ws.l_balance);
    FD(ws.l_reflect);    FD(ws.mu_running);
    FD(ws.adam_step);
    FD(ws.d_bc1);
    FD(ws.d_bc2);
    FD(ws.d_clip_coef);
    FD(ws.d_lr);

    #undef FD

    fl_bf16(ws.d_qkv_w);   fl_bf16(ws.d_out_w);
    fl_bf16(ws.d_exp_w1);  fl_bf16(ws.d_exp_w2);
    fl_bf16(ws.d_norm1_w); fl_bf16(ws.d_norm2_w); fl_bf16(ws.d_router_w);
    fl_fp32(ws.d_norm1_f); fl_fp32(ws.d_norm2_f);
    fl_fp32(ws.d_gamma);

    memset(&ws, 0, sizeof(ws));
}

// ---------------------------------------------------------------------------
// zero_grad_buffers — called at the start of each accumulation window
// Zeros grad accumulators so microstep-0 starts from clean state.
// Using cudaMemsetAsync to overlap with previous step's operations.
// ---------------------------------------------------------------------------
void zero_grad_buffers(JarvisWorkspace& ws, const JarvisConfig& cfg, cudaStream_t s) {
    const int L   = cfg.n_layers;
    const int C   = cfg.d_model;
    const int H   = cfg.n_heads;
    const int E   = cfg.n_experts;
    const int hi  = cfg.expert_hidden;
    const int Vp  = cfg.vocab_padded;
    const int NT  = cfg.NT();
    const int NTK = cfg.NTK();

    // Zero weight grad accumulators
    for (int l = 0; l < L; ++l) {
        cudaMemsetAsync(ws.d_qkv_w[l],  0, 3*C*C*sizeof(bf16),    s);
        cudaMemsetAsync(ws.d_out_w[l],  0, C*C*sizeof(bf16),       s);
        cudaMemsetAsync(ws.d_exp_w1[l], 0, E*hi*C*sizeof(bf16),    s);
        cudaMemsetAsync(ws.d_exp_w2[l], 0, E*C*hi*sizeof(bf16),    s);
        cudaMemsetAsync(ws.d_norm1_w[l],0, C*sizeof(bf16),          s);
        cudaMemsetAsync(ws.d_norm2_w[l],0, C*sizeof(bf16),          s);
        cudaMemsetAsync(ws.d_norm1_f[l],0, C*sizeof(float),         s);
        cudaMemsetAsync(ws.d_norm2_f[l],0, C*sizeof(float),         s);
        cudaMemsetAsync(ws.d_gamma[l],  0, H*sizeof(float),         s);
    }
    cudaMemsetAsync(ws.d_tok_emb,    0, Vp*C*sizeof(bf16),          s);
    cudaMemsetAsync(ws.d_final_norm, 0, C*sizeof(bf16),             s);
    cudaMemsetAsync(ws.grad_norm_sq, 0, sizeof(float),              s);
    cudaMemsetAsync(ws.l_balance,    0, sizeof(float),              s);
    cudaMemsetAsync(ws.l_reflect,    0, sizeof(float),              s);
    cudaMemsetAsync(ws.loss,         0, sizeof(float),              s);
}
