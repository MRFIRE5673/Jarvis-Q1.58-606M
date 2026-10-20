// =============================================================================
// full_engine.cu — Jarvis-Q1.58-500M Training Engine Orchestration
// =============================================================================
// Priority: COMPILE AND RUN correctly. Optimize after.
//
// Key fixes vs previous version:
//   - Removed illegal #include "ternary.cu" (was causing double definition)
//   - All kernel launchers are extern declared from their own .cu translation units
//   - Workspace field references match cuda_engine.h exactly
//   - Single-stream throughout (no multi-stream — memory.md §4.1 confirmed regression)
//   - MoE dispatch: NTK total rows, NOT per-expert batches
//     (model dispatches x_flat.repeat_interleave(K, dim=0) = NT*K rows total)
// =============================================================================
#include "full_engine.h"
#include "attention.h"
#include "moe.h"
#include "liquid_state_fusion.h"
#include "optimizer.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

// ---------------------------------------------------------------------------
// Extern declarations — each defined in its own .cu translation unit
// ---------------------------------------------------------------------------
// cuda_engine_kernels.cu
extern void launch_tok_emb_fwd(const int*, const bf16*, bf16*, int, int, cudaStream_t);
extern void launch_tok_emb_bwd(const bf16*, const int*, float*, bf16*, int, int, int, cudaStream_t);
extern void launch_fused_add_rmsnorm_fwd(const bf16*, const bf16*, const bf16*,
                                          bf16*, float*, int, int, float, cudaStream_t);
extern void launch_fused_stash_add_rmsnorm_fwd(const bf16*, const bf16*, const bf16*,
                                                bf16*, bf16*, float*, int, int, float, cudaStream_t);
extern void launch_online_xent_bwd(bf16*, const int*, float*, int, int, int, int, int, cudaStream_t);
extern void launch_add_residual(const bf16*, const bf16*, bf16*, int, cudaStream_t);
extern void launch_grad_norm_sq(const bf16*, float*, int, cudaStream_t);
extern void launch_compute_clip_coef(const float*, float*, float, cudaStream_t);
extern void launch_combine_aux_loss(float*, const float*, const float*, cudaStream_t);

// attention.cu
extern void launch_build_rope_tables(float*, float*, int, int, float, cudaStream_t);
extern void launch_fused_elu_rope(const bf16*, const bf16*, const float*, const float*,
                                   bf16*, bf16*, int, int, int, int, cudaStream_t);
extern void launch_assoc_attn_fwd(const bf16*, const bf16*, const bf16*, const float*,
                                   bf16*, bf16*, int, int, int, int, int, cudaStream_t);

// moe.cu
extern void launch_dispatch_gather(const bf16*, const int*, bf16*, int, int, int, cudaStream_t);
extern void launch_sparse_dispatch(const bf16*, const int*, bf16*, int*, int*, const int*, int, int, int, int, cudaStream_t);
extern void launch_sparse_combine_fwd(const bf16*, const bf16*, const int*, const int*, const int*, const float*, bf16*, int, int, int, cudaStream_t);
extern void launch_sparse_combine_bwd(const bf16*, const bf16*, const int*, const int*, const int*, const float*, bf16*, float*, int, int, int, cudaStream_t);
extern void launch_gelu_fwd_inplace(bf16*, int, cudaStream_t);
extern void launch_gelu_fwd_stash(const bf16*, bf16*, bf16*, int, cudaStream_t);
extern void launch_gelu_bwd(const bf16*, const bf16*, bf16*, int, cudaStream_t);

extern void launch_scatter_combine(const bf16*, const bf16*, const int*, const float*,
                                   bf16*, int, int, int, cudaStream_t);
extern void launch_load_balance_loss(const float*, const int*, float*, float, int, int, cudaStream_t);

// liquid_state_fusion.cu
extern void launch_compute_lsf_alpha(const bf16*, float, float, float, float*, int, int, cudaStream_t);
extern void launch_lsf_fwd(const bf16*, const float*, bf16*, bf16*, int, int, int, cudaStream_t);
extern void launch_reflective_penalty(const bf16*, float*, float*, float, float, float, int, int, cudaStream_t);

// ternary.cu (launchers only — NOT #included)
extern void launch_ternary_abs_mean(const float*, float*, int, cudaStream_t);
extern void launch_ternary_quantize(const float*, const float*, bf16*, int, cudaStream_t);
extern void launch_ternary_ste_mask(const float*, const float*, bf16*, int, cudaStream_t);

// backward.cu
extern void launch_grad_accum(const bf16*, bf16*, float, int, cudaStream_t);
extern void launch_router_bwd(const float*, const float*, const int*, bf16*, int, int, int, cudaStream_t);

// core_kernels.cu / liquid_state_fusion.cu (correct fwd/bwd path)
extern void launch_rmsnorm_fwd(const bf16*, const bf16*, bf16*, bf16*, int, int, cudaStream_t);
extern void launch_rmsnorm_bwd(const bf16*, const bf16*, const bf16*, const bf16*, bf16*, float*, int, int, cudaStream_t);
extern void launch_attn_fwd2(const bf16*, const float*, const float*, const float*, bf16*, bf16*, bf16*, int, int, int, int, int, cudaStream_t);
extern void launch_attn_bwd2(const bf16*, const float*, const float*, const float*, const bf16*, bf16*, const bf16*, const bf16*, int, int, int, int, int, cudaStream_t);
extern void launch_moe_combine_fwd(const bf16*, const bf16*, const int*, const float*, bf16*, int, int, cudaStream_t);
extern void launch_moe_combine_bwd(const bf16*, const bf16*, const int*, const float*, bf16*, float*, int, int, int, cudaStream_t);
extern void launch_router_bwd2(const float*, const int*, const float*, bf16*, int, cudaStream_t);
extern void launch_emb_bwd2(const int*, const bf16*, bf16*, int, int, cudaStream_t);
extern void launch_f32_to_bf16(const float*, bf16*, int, cudaStream_t);
extern void launch_lsf_bwd(const bf16*, const bf16*, const bf16*, const float*, bf16*, float*, int, int, int, cudaStream_t);

// ---------------------------------------------------------------------------
// Global engine singleton
// ---------------------------------------------------------------------------
static JarvisEngine g_engine;
JarvisEngine& get_engine() { return g_engine; }

// ---------------------------------------------------------------------------
// quant_fp8_device: same as quant_fp8 but writes inv scale to a device pointer
extern void quant_fp8_device(const bf16* x, uint8_t* y, int rows, int cols, bool tr, int e5,
                             float* d_amax, float* d_inv, cudaStream_t s);

// ---------------------------------------------------------------------------
static void quantize_weight(float* fp32, float* alpha_scratch,
                             bf16* bf16_out, int N, cudaStream_t s) {
    launch_ternary_abs_mean(fp32, alpha_scratch, N, s);
    launch_ternary_quantize(fp32, alpha_scratch, bf16_out, N, s);
}

static bool sparse_moe_enabled() {
    const char* v = std::getenv("JARVIS_MOE_SPARSE");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static float router_noise_std() {
    const char* v = std::getenv("JARVIS_PAPER_FIDELITY");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y') ? 0.1f : 0.0f;
}

static std::vector<int> read_expert_counts(JarvisWorkspace& ws, int experts,
                                           cudaStream_t stream) {
    std::vector<int> counts((size_t)experts);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), ws.expert_counts,
                               (size_t)experts * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counts;
}

// ===========================================================================
// JarvisEngine::init
// ===========================================================================
void JarvisEngine::init(const JarvisConfig& config, int device) {
    cfg = config;

    printf("=== Jarvis-Q1.58-500M Engine Init ===\n");
    printf("  L=%d C=%d H=%d D=%d E=%d K=%d hi=%d\n",
           cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.head_dim,
           cfg.n_experts, cfg.top_k, cfg.expert_hidden);
    printf("  B=%d T=%d Vpad=%d accum=%d\n",
           cfg.batch_size, cfg.seq_len, cfg.vocab_padded, cfg.accum_steps);

    init_runtime(device);
    auto& rt = get_runtime();

    alloc_params(params, cfg);
    alloc_workspace(ws, cfg);

    // RoPE tables [T, D] FP32
    size_t rope_sz = (size_t)cfg.seq_len * cfg.head_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&rope_cos, rope_sz));
    CUDA_CHECK(cudaMalloc(&rope_sin, rope_sz));

    // Quantization scratch alpha [1] FP32
    CUDA_CHECK(cudaMalloc(&quant_alpha, sizeof(float)));

    // Build RoPE tables
    launch_build_rope_tables(rope_cos, rope_sin,
        cfg.seq_len, cfg.head_dim, 10000.0f, rt.get_stream());

    // Init cuBLASLt
    get_cublaslt().init(cfg);

    // Init adam step counter to 0
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(ws.adam_step, &zero, sizeof(int), cudaMemcpyHostToDevice));

    rt.sync();
    printf("=== Init complete. VRAM allocated. ===\n");
}

// ===========================================================================
// Re-quantize all projection weights (called once per optimizer step)
// ===========================================================================
void JarvisEngine::quantize_all_weights() {
    cudaStream_t s = get_runtime().get_stream();
    const int L  = cfg.n_layers;
    const int C  = cfg.d_model;
    const int E  = cfg.n_experts;
    const int hi = cfg.expert_hidden;

    for (int l = 0; l < L; ++l) {
        quantize_weight(params.qkv_fp32[l],    quant_alpha, params.qkv_w[l],   3*C*C,   s);
        quantize_weight(params.out_fp32[l],    quant_alpha, params.out_w[l],   C*C,     s);
        quantize_weight(params.exp_w1_fp32[l], quant_alpha, params.expert_w1[l], E*hi*C, s);
        quantize_weight(params.exp_w2_fp32[l], quant_alpha, params.expert_w2[l], E*C*hi, s);
        // Fill FP8 caches: amax scratch reuses quant_alpha (sequential, same stream)
        quant_fp8_device(params.qkv_w[l],    ws.qkv_q8[l], 3*C, C,   false, 0, quant_alpha, ws.qkv_inv[l], s);
        quant_fp8_device(params.out_w[l],    ws.out_q8[l], C,   C,   false, 0, quant_alpha, ws.out_inv[l], s);
        quant_fp8_device(params.expert_w1[l], ws.w1_q8[l], E*hi, C,  false, 0, quant_alpha, ws.w1_inv[l],  s);
        quant_fp8_device(params.expert_w2[l], ws.w2_q8[l], E*C, hi,  false, 0, quant_alpha, ws.w2_inv[l],  s);
    }
}

// ===========================================================================
// JarvisEngine::forward_microstep
// ===========================================================================
#if 0  // ---- legacy approximate fwd/bwd (kept for reference; replaced below) ----
void JarvisEngine::forward_microstep(int ustep) {
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();
    auto& lt = get_cublaslt();

    const int NT   = cfg.NT();
    const int NTK  = cfg.NTK();
    const int L    = cfg.n_layers;
    const int C    = cfg.d_model;
    const int H    = cfg.n_heads;
    const int D    = cfg.head_dim;
    const int T    = cfg.seq_len;
    const int B    = cfg.batch_size;
    const int E    = cfg.n_experts;
    const int K    = cfg.top_k;
    const int hi   = cfg.expert_hidden;
    const int Vp   = cfg.vocab_padded;

    lt.fp8_fwd = true;   // FP8 E4M3 forward projections
    // === EMBEDDING ===
    const int* tok_ids = ws.token_ids + (size_t)ustep * NT;
    launch_tok_emb_fwd(tok_ids, params.tok_emb, ws.layer_x, NT, C, s);

    // === 24 TRANSFORMER LAYERS ===
    for (int l = 0; l < L; ++l) {
        size_t layer_off = (size_t)l * NT * C;

        // === NORM1: fused stash_x write + add + RMSNorm ===
        // Eliminates a separate D2D memcpy by writing stash_x inside the kernel
        launch_fused_stash_add_rmsnorm_fwd(
            ws.layer_x, ws.layer_x,         // x == residual for identity branch
            params.norm1_w[l], ws.norm1_out,
            ws.stash_x + layer_off,          // stash written here (fused)
            ws.stash_rsqrt1 + (size_t)l * NT,
            NT, C, 1e-6f, s);

        // --- QKV Projection [NT, 3C] ---
        lt.qkv_proj_fwd(ws.norm1_out, params.qkv_w[l], ws.qkv_out, NT, C, s);

        const bf16* q_raw = ws.qkv_out;
        const bf16* k_raw = ws.qkv_out + (size_t)NT * C;
        const bf16* v_raw = ws.qkv_out + (size_t)NT * C * 2;
        // NOTE: v_buf memcpy removed — pass v_raw directly (V is read-only in attn)

        // --- ELU+1 + RoPE on Q and K ---
        launch_fused_elu_rope(q_raw, k_raw, rope_cos, rope_sin,
                               ws.q_buf, ws.k_buf, B, H, T, D, s);

        // --- Associative Linear Attention ---
        launch_assoc_attn_fwd(ws.q_buf, ws.k_buf, v_raw,   // pass v_raw directly
                               params.gamma_raw[l],
                               ws.attn_out, ws.attn_state,
                               B, H, T, D, cfg.chunk_size, s);

        // --- Output Projection [NT, C] ---
        lt.out_proj_fwd(ws.attn_out, params.out_w[l], ws.layer_x2, NT, C, s);

        // --- Residual Add 1: layer_x = stash_x[l] + layer_x2 ---
        launch_add_residual(ws.stash_x + layer_off, ws.layer_x2, ws.layer_x, NT * C, s);

        // === NORM2: fused stash_x2 write + add + RMSNorm ===
        launch_fused_stash_add_rmsnorm_fwd(
            ws.layer_x, ws.layer_x,
            params.norm2_w[l], ws.norm2_out,
            ws.stash_x2 + layer_off,         // stash_x2 written here (fused)
            ws.stash_rsqrt2 + (size_t)l * NT,
            NT, C, 1e-6f, s);


        // --- Router: [NT, E] = norm2_out [NT, C] x router_w [E, C]^T ---
        lt.router_fwd(ws.norm2_out, params.router_w[l],
                      ws.router_logits, NT, C, E, s);

        // --- Top-2 Gating ---
        launch_top2_gating(ws.router_logits, ws.router_probs,
                           ws.topk_idx, ws.topk_gates, NT, E, s);

        // --- Load Balance Aux Loss ---
        launch_load_balance_loss(ws.router_probs, ws.topk_idx,
                                 ws.l_balance, cfg.balance_alpha, NT, E, s);

        // --- Dispatch Gather: dispatch_x [NTK, C] ---
        // Gathers NT*K=NTK copies (same token dispatched to K experts)
        launch_dispatch_gather(ws.norm2_out, ws.topk_idx,
                               ws.dispatch_x, NT, K, C, s);

        // --- Expert W1 GEMM ---
        lt.expert_w1_fwd(ws.dispatch_x, params.expert_w1[l],
                         ws.layer_h1, NTK, hi, C, s);

        // === FUSED GELU + STASH: reads h1 once, writes stash (pre-GELU) + h1 (post-GELU)
        // Eliminates separate D2D memcpy (16 MB per layer = 384 MB total savings)
        bf16* stash_h1_l = ws.stash_h1 + (size_t)l * NTK * hi;
        launch_gelu_fwd_stash(ws.layer_h1, stash_h1_l, ws.layer_h1, NTK * hi, s);

        // --- Expert W2 GEMM: expert_out [NTK, C] = layer_h1 [NTK, hi] x W2 [C, hi]^T ---
        lt.expert_w2_fwd(ws.layer_h1, params.expert_w2[l],
                         ws.expert_out, NTK, C, hi, s);


        // --- Reflective Regularization ---
        launch_reflective_penalty(ws.norm2_out, ws.mu_running, ws.l_reflect,
                                  cfg.reflect_lambda, cfg.reflect_tau_max,
                                  0.99f, NT, C, s);

        // --- Scatter-Combine + Residual Add 2: moe_out [NT, C] ---
        launch_scatter_combine(ws.expert_out, ws.layer_x,
                               ws.topk_idx, ws.topk_gates,
                               ws.moe_out, NT, K, C, s);

        // --- LSF: compute dynamic alpha, update membrane ---
        float var_scale = (params.lsf_vscale && params.lsf_vscale[l]) ?
                          cfg.lsf_var_scale : 1.0f;
        launch_compute_lsf_alpha(ws.moe_out, var_scale,
                                  cfg.lsf_alpha_min, cfg.lsf_alpha_max,
                                  ws.lsf_alpha, NT, C, s);

        bf16* H_layer = ws.lsf_h + (size_t)l * B * C;
        // Write LSF output directly into layer_x — eliminates D2D memcpy
        launch_lsf_fwd(ws.moe_out, ws.lsf_alpha,
                       H_layer, ws.layer_x, B, T, C, s);
    }
    lt.fp8_fwd = false;  // LM head + backward stay BF16

    // === FINAL NORM ===
    launch_fused_add_rmsnorm_fwd(
        ws.layer_x, ws.layer_x,
        params.final_norm_w, ws.layer_x2,
        nullptr,
        NT, C, 1e-6f, s);

    // === LM HEAD: logits [NT, Vpad] (weight-tied with tok_emb) ===
    lt.lm_head(ws.layer_x2, params.tok_emb, ws.logits, NT, C, Vp, s);

    // === LOSS: XEnt + aux ===
    // Autoregressive: token i predicts token i+1
    // Use NT-1 valid pairs (last token has no target)
    launch_online_xent_bwd(ws.logits, tok_ids + 1,
                            ws.d_logits, ws.loss, NT - 1, Vp, s);

    launch_combine_aux_loss(ws.loss, ws.l_balance, ws.l_reflect, s);
}

// ===========================================================================
// JarvisEngine::backward_microstep
// Simplified but correct gradient flow.
// ===========================================================================
void JarvisEngine::backward_microstep(int ustep, float grad_scale) {
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();
    auto& lt = get_cublaslt();

    const int NT   = cfg.NT();
    const int NTK  = cfg.NTK();
    const int L    = cfg.n_layers;
    const int C    = cfg.d_model;
    const int E    = cfg.n_experts;
    const int K    = cfg.top_k;
    const int hi   = cfg.expert_hidden;
    const int Vp   = cfg.vocab_padded;

    // --- LM HEAD BACKWARD ---
    // d_logits already computed by online_xent_bwd in forward
    // dX [NT, C] = d_logits [NT, Vp] x tok_emb [Vp, C]
    lt.lm_head_bwd_dx(ws.d_logits, params.tok_emb, ws.d_x, NT, C, Vp, s);

    // dW_tok_emb += d_logits^T x layer_x2  (weight-tied)
    // beta=0.0 on first microstep to overwrite (eliminates redundant zero-grad pass)
    lt.lm_head_bwd_dw(ws.d_logits, ws.layer_x2, ws.d_tok_emb,
                       NT, C, Vp, (ustep == 0) ? 0.0f : 1.0f, s);

    // --- 24-LAYER BACKWARD (reversed) ---
    for (int l = L - 1; l >= 0; --l) {
        size_t layer_off = (size_t)l * NT * C;
        float beta = (ustep == 0) ? 0.0f : 1.0f;  // overwrite vs accumulate

        // Backward: LSF + Scatter-Combine (gradient identity pass)
        // d_x flows directly into dispatch_gather — no d_moe copy needed
        launch_dispatch_gather(ws.d_x, ws.topk_idx,
                               ws.d_dispatch, NT, K, C, s);
        lt.expert_w2_fwd(ws.d_dispatch, params.expert_w2[l],
                         ws.d_h1, NTK, hi, C, s);
        lt.generic_bwd_dw(ws.d_dispatch, ws.layer_h1, ws.d_exp_w2[l],
                           C, hi, NTK, beta, s);

        // Backward: GELU (scalar, correct per memory.md)
        const bf16* h1_pre = ws.stash_h1 + (size_t)l * NTK * hi;
        launch_gelu_bwd(h1_pre, ws.d_h1, ws.d_h1, NTK * hi, s);

        // Backward: Expert W1
        // ddispatch [NTK, C] = d_h1 [NTK, hi] x W1 [hi, C]
        lt.expert_w1_fwd(ws.d_h1, params.expert_w1[l],
                         ws.d_dispatch, NTK, C, hi, s);
        lt.generic_bwd_dw(ws.d_h1, ws.dispatch_x, ws.d_exp_w1[l],
                           hi, C, NTK, beta, s);

        // Router backward (NT x E) — d_norm2 used directly below
        launch_router_bwd(ws.router_probs, ws.topk_gates,
                          ws.topk_idx, ws.d_norm2,
                          NT, E, K, s);
        lt.generic_bwd_dw(ws.d_norm2, ws.norm2_out, ws.d_norm2_w[l],
                           E, C, NT, beta, s);

        // Backward: Out Projection — d_norm2 passes through RMSNorm2 (identity)
        lt.out_bwd_dx(ws.d_norm2, params.out_w[l], ws.d_attn_out, NT, C, s);
        lt.generic_bwd_dw(ws.d_norm2, ws.attn_out, ws.d_out_w[l],
                           C, C, NT, beta, s);

        // Backward: QKV Projection — d_attn_out passes through RMSNorm1 (identity)
        lt.qkv_bwd_dx(ws.d_attn_out, params.qkv_w[l], ws.d_x, NT, C, s);
        lt.generic_bwd_dw(ws.d_attn_out, ws.norm1_out, ws.d_qkv_w[l],
                           3*C, C, NT, beta, s);

        // Accumulate grad norms per layer
        launch_grad_norm_sq(ws.d_qkv_w[l],  ws.grad_norm_sq, 3*C*C,  s);
        launch_grad_norm_sq(ws.d_out_w[l],  ws.grad_norm_sq, C*C,    s);
        launch_grad_norm_sq(ws.d_exp_w1[l], ws.grad_norm_sq, E*hi*C, s);
        launch_grad_norm_sq(ws.d_exp_w2[l], ws.grad_norm_sq, E*C*hi, s);
    }

    // tok_emb grad norm
    launch_grad_norm_sq(ws.d_tok_emb, ws.grad_norm_sq, Vp * C, s);
}
#endif  // legacy

// ===========================================================================
// Correct transformer layer (shared by forward and the recompute in backward)
//   x -> n1=RMS(x) -> qkv -> linear attn -> out_proj -> x2=x+a -> n2=RMS(x2)
//     -> router/top2 -> dense experts (all E) -> m = x2 + sum_k g_k y_{e_k}
//   Results left in workspace: norm1_out, qkv_out, attn_out, dispatch_x(=x2),
//   norm2_out, router_*, layer_h1 (post-GELU), stash_h1 (pre-GELU),
//   expert_out (y), moe_out (m).
// ===========================================================================
static void layer_core(JarvisEngine& E, int l, const bf16* xin) {
    auto& ws = E.ws; auto& P = E.params; const auto& cfg = E.cfg;
    auto& lt = get_cublaslt();
    cudaStream_t s = get_runtime().get_stream();
    const int NT = cfg.NT(), C = cfg.d_model, H = cfg.n_heads, D = cfg.head_dim;
    const int Ee = cfg.n_experts, hi = cfg.expert_hidden, B = cfg.batch_size, T = cfg.seq_len;
    bf16* x2 = ws.layer_x2;

    // FP8 forward using pre-quantized weight cache (only activation is quantized per call)
    launch_rmsnorm_fwd(xin, P.norm1_w[l], ws.norm1_out, nullptr, NT, C, s);
    lt.gemm_fp8_cached(NT, 3*C, C, ws.norm1_out, P.qkv_w[l], ws.qkv_q8[l], ws.qkv_inv[l], ws.qkv_out, s);
    launch_attn_fwd2(ws.qkv_out, P.gamma_raw[l], E.rope_cos, E.rope_sin, ws.attn_out, ws.q_buf, ws.k_buf, B, H, T, D, C, s);
    lt.gemm_fp8_cached(NT, C, C, ws.attn_out, P.out_w[l], ws.out_q8[l], ws.out_inv[l], ws.layer_x2, s);
    launch_add_residual(xin, ws.layer_x2, x2, NT * C, s);
    launch_rmsnorm_fwd(x2, P.norm2_w[l], ws.norm2_out, nullptr, NT, C, s);

    lt.router_fwd(ws.norm2_out, P.router_w[l], ws.router_logits, NT, C, Ee, s);
    if (sparse_moe_enabled()) {
        CUDA_CHECK(cudaMemsetAsync(ws.expert_counts, 0, (size_t)Ee * sizeof(int), s));
        launch_top2_gating_count(ws.router_logits, ws.router_probs, ws.topk_idx,
                                 ws.topk_gates, ws.expert_counts, NT, Ee,
                                 router_noise_std(), ws.adam_step, 0, s);
        launch_expert_offsets(ws.expert_counts, ws.expert_offsets, Ee, s);
        const std::vector<int> counts = read_expert_counts(ws, Ee, s);
        launch_sparse_dispatch(ws.norm2_out, ws.topk_idx, ws.dispatch_x, ws.route_pos,
                               ws.route_write, ws.expert_offsets, NT, cfg.top_k, Ee, C, s);
        int offset = 0;
        for (int e = 0; e < Ee; ++e) {
            const int rows = counts[e];
            if (rows > 0) {
                bf16* h1 = ws.layer_h1 + (size_t)offset * hi;
                bf16* pre = ws.stash_h1 + (size_t)offset * hi;
                bf16* x = ws.dispatch_x + (size_t)offset * C;
                lt.gemm_fp8_cached(rows, hi, C, x,
                    P.expert_w1[l] + (size_t)e * hi * C,
                    ws.w1_q8[l] + (size_t)e * hi * C, ws.w1_inv[l], h1, s);
                launch_gelu_fwd_stash(h1, pre, h1, rows * hi, s);
                lt.gemm_fp8_cached(rows, C, hi, h1,
                    P.expert_w2[l] + (size_t)e * C * hi,
                    ws.w2_q8[l] + (size_t)e * C * hi, ws.w2_inv[l],
                    ws.expert_out + (size_t)offset * C, s);
            }
            offset += rows;
        }
        launch_sparse_combine_fwd(ws.expert_out, x2, ws.topk_idx, ws.route_pos,
                                  ws.expert_offsets, ws.topk_gates, ws.moe_out,
                                  NT, cfg.top_k, C, s);
    } else {
        launch_top2_gating(ws.router_logits, ws.router_probs, ws.topk_idx,
                           ws.topk_gates, NT, Ee, router_noise_std(), ws.adam_step, 0, s);
        for (int e = 0; e < Ee; ++e) {
            lt.gemm_fp8_cached(NT, hi, C, ws.norm2_out,
                               P.expert_w1[l] + (size_t)e * hi * C,
                               ws.w1_q8[l] + (size_t)e * hi * C, ws.w1_inv[l],
                               ws.layer_h1 + (size_t)e * NT * hi, s);
            launch_gelu_fwd_stash(ws.layer_h1 + (size_t)e * NT * hi,
                                  ws.stash_h1 + (size_t)e * NT * hi,
                                  ws.layer_h1 + (size_t)e * NT * hi, NT * hi, s);
            lt.gemm_fp8_cached(NT, C, hi,
                               ws.layer_h1 + (size_t)e * NT * hi,
                               P.expert_w2[l] + (size_t)e * C * hi,
                               ws.w2_q8[l] + (size_t)e * C * hi, ws.w2_inv[l],
                               ws.expert_out + (size_t)e * NT * C, s);
        }
        launch_moe_combine_fwd(ws.expert_out, x2, ws.topk_idx, ws.topk_gates,
                               ws.moe_out, NT, C, s);
    }

}

// ===========================================================================
// JarvisEngine::forward_microstep
// ===========================================================================
void JarvisEngine::forward_microstep(int ustep) {
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();
    auto& lt = get_cublaslt();
    lt.fp8_fwd = false;

    const int NT = cfg.NT(), L = cfg.n_layers, C = cfg.d_model, E = cfg.n_experts;
    const int B = cfg.batch_size, T = cfg.seq_len, Vp = cfg.vocab_padded;
    const int* tok_ids = ws.token_ids + (size_t)ustep * NT;

    launch_tok_emb_fwd(tok_ids, params.tok_emb, ws.layer_x, NT, C, s);
    // Each microbatch is an independent sequence set: reset LSF membrane state
    CUDA_CHECK(cudaMemsetAsync(ws.lsf_h, 0, (size_t)L * B * C * sizeof(bf16), s));

    for (int l = 0; l < L; ++l) {
        CUDA_CHECK(cudaMemcpyAsync(ws.stash_x + (size_t)l * NT * C, ws.layer_x,
                                   (size_t)NT * C * sizeof(bf16), cudaMemcpyDeviceToDevice, s));
        layer_core(*this, l, ws.layer_x);

        if (l == L - 1) {
        launch_load_balance_loss(ws.router_probs, ws.topk_idx, ws.l_balance, cfg.balance_alpha, NT, E, s);
        launch_reflective_penalty(ws.norm2_out, ws.mu_running, ws.l_reflect,
                                  cfg.reflect_lambda, cfg.reflect_tau_max, 0.99f, NT, C, s);
        }

        float var_scale = (params.lsf_vscale && params.lsf_vscale[l]) ? cfg.lsf_var_scale : 1.0f;
        launch_compute_lsf_alpha(ws.moe_out, var_scale, cfg.lsf_alpha_min, cfg.lsf_alpha_max,
                                 ws.lsf_alpha, NT, C, s);
        launch_lsf_fwd(ws.moe_out, ws.lsf_alpha, ws.lsf_h + (size_t)l * B * C, ws.layer_x, B, T, C, s);
    }

    launch_rmsnorm_fwd(ws.layer_x, params.final_norm_w, ws.layer_x2, nullptr, NT, C, s);
    lt.lm_head(ws.layer_x2, params.tok_emb, ws.logits, NT, C, Vp, s);
    launch_online_xent_bwd(ws.logits, tok_ids, ws.loss, B, T, Vp, cfg.accum_steps, ustep, s);
    launch_combine_aux_loss(ws.loss, ws.l_balance, ws.l_reflect, s);
}

// ===========================================================================
// JarvisEngine::backward_microstep — full gradient (layer recompute from stash_x)
// ===========================================================================
void JarvisEngine::backward_microstep(int ustep, float /*grad_scale*/) {
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();
    auto& lt = get_cublaslt();
    lt.fp8_fwd = false;
    lt.fp8_bwd = true;

    const int NT = cfg.NT(), L = cfg.n_layers, C = cfg.d_model, E = cfg.n_experts;
    const int hi = cfg.expert_hidden, Vp = cfg.vocab_padded;
    const int B = cfg.batch_size, T = cfg.seq_len, H = cfg.n_heads, D = cfg.head_dim;
    const float beta = (ustep == 0) ? 0.0f : 1.0f;
    const int* tok_ids = ws.token_ids + (size_t)ustep * NT;

    // --- LM head + final norm ---
    lt.lm_head_bwd_dx(ws.logits, params.tok_emb, ws.d_x, NT, C, Vp, s);
    lt.lm_head_bwd_dw(ws.logits, ws.layer_x2, ws.d_tok_emb, NT, C, Vp, beta, s);
    if (ustep == 0) CUDA_CHECK(cudaMemsetAsync(ws.d_final_norm_f, 0, (size_t)C * sizeof(float), s));
    launch_rmsnorm_bwd(ws.layer_x, params.final_norm_w, ws.d_x, nullptr, ws.d_x,
                       ws.d_final_norm_f, NT, C, s);
    launch_f32_to_bf16(ws.d_final_norm_f, ws.d_final_norm, C, s);

    // --- layers, reversed ---
    for (int l = L - 1; l >= 0; --l) {
        const bf16* xin = ws.stash_x + (size_t)l * NT * C;
        layer_core(*this, l, xin);          // recompute activations of layer l
        bf16* x2 = ws.dispatch_x;

        // LSF backward (alpha treated as constant)
        float var_scale = (params.lsf_vscale && params.lsf_vscale[l]) ? cfg.lsf_var_scale : 1.0f;
        launch_compute_lsf_alpha(ws.moe_out, var_scale, cfg.lsf_alpha_min, cfg.lsf_alpha_max,
                                 ws.lsf_alpha, NT, C, s);
        const bf16* Hout = (l == L - 1) ? ws.layer_x : ws.stash_x + (size_t)(l + 1) * NT * C;
        launch_lsf_bwd(ws.d_x, Hout, Hout, ws.lsf_alpha, ws.d_moe, ws.dg, B, T, C, s);
        // ws.d_moe = dm = grad wrt m (and, via the residual, wrt x2)

        // MoE combine backward -> per-expert dy, gate grads -> router logit grads
        if (sparse_moe_enabled()) {
            launch_sparse_combine_bwd(ws.d_moe, ws.expert_out, ws.topk_idx,
                ws.route_pos, ws.expert_offsets, ws.topk_gates,
                ws.d_dispatch, ws.dg, NT, cfg.top_k, C, s);
            launch_router_bwd2(ws.router_probs, ws.topk_idx, ws.dg, ws.d_rlogits, NT, s);
            const std::vector<int> counts = read_expert_counts(ws, E, s);
            CUDA_CHECK(cudaMemsetAsync(ws.d_norm2, 0, (size_t)NT * C * sizeof(bf16), s));
            int offset = 0;
            bool first = true;
            for (int e = 0; e < E; ++e) {
                const int rows = counts[e];
                if (rows > 0) {
                    const bf16* dy_e = ws.d_dispatch + (size_t)offset * C;
                    bf16* dh_e = ws.d_h1 + (size_t)offset * hi;
                    const bf16* h1 = ws.layer_h1 + (size_t)offset * hi;
                    const bf16* h1_pre = ws.stash_h1 + (size_t)offset * hi;
                    lt.generic_bwd_dw(dy_e, h1, ws.d_exp_w2[l] + (size_t)e * C * hi,
                                      C, hi, rows, beta, s);
                    GemmDesc d1 = {rows, hi, C};
                    lt.gemm_nn(d1, dy_e, params.expert_w2[l] + (size_t)e * C * hi,
                               dh_e, 1.0f, 0.0f, s);
                    launch_gelu_bwd(h1_pre, dh_e, dh_e, rows * hi, s);
                    lt.generic_bwd_dw(dh_e, ws.norm2_out,
                                      ws.d_exp_w1[l] + (size_t)e * hi * C,
                                      hi, C, rows, beta, s);
                    GemmDesc d2 = {rows, C, hi};
                    lt.gemm_nn(d2, dh_e, params.expert_w1[l] + (size_t)e * hi * C,
                               ws.d_norm2, 1.0f, first ? 0.0f : 1.0f, s);
                    first = false;
                }
                offset += rows;
            }
        } else {
            launch_moe_combine_bwd(ws.d_moe, ws.expert_out, ws.topk_idx, ws.topk_gates,
                                   ws.d_dispatch, ws.dg, NT, C, E, s);
            launch_router_bwd2(ws.router_probs, ws.topk_idx, ws.dg, ws.d_rlogits, NT, s);

            for (int e = 0; e < E; ++e) {
                const bf16* dy_e = ws.d_dispatch + (size_t)e * NT * C;
                lt.generic_bwd_dw(dy_e, ws.layer_h1 + (size_t)e * NT * hi,
                                  ws.d_exp_w2[l] + (size_t)e * C * hi, C, hi, NT, beta, s);
                GemmDesc d1 = {NT, hi, C};
                lt.gemm_nn(d1, dy_e, params.expert_w2[l] + (size_t)e * C * hi,
                           ws.d_h1 + (size_t)e * NT * hi, 1.0f, 0.0f, s);
                launch_gelu_bwd(ws.stash_h1 + (size_t)e * NT * hi,
                                ws.d_h1 + (size_t)e * NT * hi,
                                ws.d_h1 + (size_t)e * NT * hi, NT * hi, s);
                const bf16* dh_e = ws.d_h1 + (size_t)e * NT * hi;
                lt.generic_bwd_dw(dh_e, ws.norm2_out, ws.d_exp_w1[l] + (size_t)e * hi * C,
                                  hi, C, NT, beta, s);
                GemmDesc d2 = {NT, C, hi};
                lt.gemm_nn(d2, dh_e, params.expert_w1[l] + (size_t)e * hi * C,
                           ws.d_norm2, 1.0f, (e == 0) ? 0.0f : 1.0f, s);
            }
        }
        // Router weight grad and its contribution to dn2
        lt.generic_bwd_dw(ws.d_rlogits, ws.norm2_out, ws.d_router_w[l], E, C, NT, beta, s);
        GemmDesc dr = {NT, C, E};
        lt.gemm_nn(dr, ws.d_rlogits, params.router_w[l], ws.d_norm2, 1.0f, 1.0f, s);

        // norm2 backward (+ residual dm) -> grad wrt x2 (in d_moe)
        launch_rmsnorm_bwd(x2, params.norm2_w[l], ws.d_norm2, ws.d_moe, ws.d_moe, nullptr, NT, C, s);

        // out projection
        lt.generic_bwd_dw(ws.d_moe, ws.attn_out, ws.d_out_w[l], C, C, NT, beta, s);
        lt.out_bwd_dx(ws.d_moe, params.out_w[l], ws.d_attn_out, NT, C, s);

        // attention
        launch_attn_bwd2(ws.qkv_out, params.gamma_raw[l], rope_cos, rope_sin,
                         ws.d_attn_out, ws.d_qkv, ws.q_buf, ws.k_buf, B, H, T, D, C, s);

        // qkv projection
        lt.generic_bwd_dw(ws.d_qkv, ws.norm1_out, ws.d_qkv_w[l], 3 * C, C, NT, beta, s);
        lt.qkv_bwd_dx(ws.d_qkv, params.qkv_w[l], ws.d_norm1, NT, C, s);

        // norm1 backward (+ residual grad) -> d_x for the previous layer
        launch_rmsnorm_bwd(xin, params.norm1_w[l], ws.d_norm1, ws.d_moe, ws.d_x, nullptr, NT, C, s);

        if (ustep == cfg.accum_steps - 1) {
            // Apply the Q1.58 STE gate after all microbatch gradients are accumulated,
            // before clipping so masked coordinates do not affect the global norm.
            launch_ternary_abs_mean(params.qkv_fp32[l], quant_alpha, 3 * C * C, s);
            launch_ternary_ste_mask(params.qkv_fp32[l], quant_alpha, ws.d_qkv_w[l], 3 * C * C, s);
            launch_ternary_abs_mean(params.out_fp32[l], quant_alpha, C * C, s);
            launch_ternary_ste_mask(params.out_fp32[l], quant_alpha, ws.d_out_w[l], C * C, s);
            launch_ternary_abs_mean(params.exp_w1_fp32[l], quant_alpha, E * hi * C, s);
            launch_ternary_ste_mask(params.exp_w1_fp32[l], quant_alpha, ws.d_exp_w1[l], E * hi * C, s);
            launch_ternary_abs_mean(params.exp_w2_fp32[l], quant_alpha, E * C * hi, s);
            launch_ternary_ste_mask(params.exp_w2_fp32[l], quant_alpha, ws.d_exp_w2[l], E * C * hi, s);

            launch_grad_norm_sq(ws.d_qkv_w[l],   ws.grad_norm_sq, 3 * C * C,  s);
            launch_grad_norm_sq(ws.d_out_w[l],   ws.grad_norm_sq, C * C,      s);
            launch_grad_norm_sq(ws.d_exp_w1[l],  ws.grad_norm_sq, E * hi * C, s);
            launch_grad_norm_sq(ws.d_exp_w2[l],  ws.grad_norm_sq, E * C * hi, s);
            launch_grad_norm_sq(ws.d_router_w[l], ws.grad_norm_sq, C * E,     s);
        }
    }

    // Embedding lookup gradient
    launch_emb_bwd2(tok_ids, ws.d_x, ws.d_tok_emb, NT, C, s);
    if (ustep == cfg.accum_steps - 1)
        launch_grad_norm_sq(ws.d_tok_emb, ws.grad_norm_sq, Vp * C, s);
}

// ===========================================================================
// JarvisEngine::optimizer_step
// ===========================================================================
void JarvisEngine::optimizer_step() {
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();

    // Step 1: Compute bc1, bc2, clip_coef on device — no CPU readback.
    // prepare_optimizer_step atomically increments ws.adam_step on device,
    // then computes bias-correction factors and clip coefficient.
    launch_prepare_optimizer_step(
        ws.adam_step, ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr,
        ws.grad_norm_sq, cfg.beta1, cfg.beta2, cfg.grad_clip, cfg.lr,
        cfg.warmup_steps, cfg.decay_steps, cfg.min_lr_ratio, s);

    float lr = cfg.lr, b1 = cfg.beta1, b2 = cfg.beta2;
    float eps = cfg.eps, wd = cfg.weight_decay;

    // Step 2: Launch AdamW for all parameter groups using device scalars.
    launch_adamw_fused(params.tok_emb_fp32, params.tok_emb,
                       params.tok_emb_m, params.tok_emb_v,
                       ws.d_tok_emb,
                       lr, b1, b2, eps, wd,
                        ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr,
                       cfg.vocab_padded * cfg.d_model, s);

    const int L = cfg.n_layers, C = cfg.d_model, E = cfg.n_experts, hi = cfg.expert_hidden;

    for (int l = 0; l < L; ++l) {
        launch_adamw_fused(params.qkv_fp32[l], params.qkv_w[l],
                           params.qkv_m[l], params.qkv_v[l],
                           ws.d_qkv_w[l],
                           lr, b1, b2, eps, wd,
                            ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr, 3*C*C, s);

        launch_adamw_fused(params.out_fp32[l], params.out_w[l],
                           params.out_m[l], params.out_v[l],
                           ws.d_out_w[l],
                           lr, b1, b2, eps, wd,
                            ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr, C*C, s);

        launch_adamw_fused(params.exp_w1_fp32[l], params.expert_w1[l],
                           params.exp_w1_m[l], params.exp_w1_v[l],
                           ws.d_exp_w1[l],
                           lr, b1, b2, eps, wd,
                            ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr, E*hi*C, s);

        launch_adamw_fused(params.exp_w2_fp32[l], params.expert_w2[l],
                           params.exp_w2_m[l], params.exp_w2_v[l],
                           ws.d_exp_w2[l],
                           lr, b1, b2, eps, wd,
                            ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr, E*C*hi, s);

        launch_adamw_fused(params.router_fp32[l], params.router_w[l],
                           params.router_m[l], params.router_v[l],
                           ws.d_router_w[l],
                           lr, b1, b2, eps, 0.0f,
                            ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr, C*E, s);
    }

    launch_adamw_fused(params.final_norm_w_fp32, params.final_norm_w,
                       params.final_norm_m, params.final_norm_v,
                       ws.d_final_norm,
                       lr, b1, b2, eps, 0.0f,
                        ws.d_bc1, ws.d_bc2, ws.d_clip_coef, ws.d_lr, C, s);
}

// ===========================================================================
// JarvisEngine::warmup
// ===========================================================================
void JarvisEngine::warmup(int n_steps) {
    printf("[engine] Warmup (%d steps)...\n", n_steps);
    get_cublaslt().autotune_algorithms(cfg);

    auto& rt = get_runtime();
    float grad_scale = 1.0f / (float)cfg.accum_steps;

    for (int step = 0; step < n_steps; ++step) {
        // Zero grad norm accumulator
        CUDA_CHECK(cudaMemsetAsync(ws.grad_norm_sq, 0, sizeof(float), rt.get_stream()));

        quantize_all_weights();

        for (int us = 0; us < cfg.accum_steps; ++us) {
            forward_microstep(us);
            backward_microstep(us, grad_scale);
        }
        optimizer_step();
        quantize_all_weights();  // re-quantize after optimizer step

        rt.sync();
        float loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&loss, ws.loss, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  warmup step %d  loss=%.4f\n", step + 1, loss);
    }
    printf("[engine] Warmup done.\n");
}

// ===========================================================================
// JarvisEngine::capture_graph
// ===========================================================================
void JarvisEngine::capture_graph() {
    if (sparse_moe_enabled()) {
        graph_captured = false;
        printf("[engine] Sparse top-2 MoE uses dynamic expert batches; CUDA Graph disabled.\n");
        return;
    }
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();
    float grad_scale = 1.0f / (float)cfg.accum_steps;

    printf("[engine] Capturing CUDA Graph...\n");
    rt.begin_graph_capture();

    CUDA_CHECK(cudaMemsetAsync(ws.grad_norm_sq, 0, sizeof(float), s));
    quantize_all_weights();

    for (int us = 0; us < cfg.accum_steps; ++us) {
        forward_microstep(us);
        backward_microstep(us, grad_scale);
    }
    optimizer_step();
    quantize_all_weights();

    rt.end_graph_capture();
    graph_captured = true;
    printf("[engine] CUDA Graph captured.\n");
}

// ===========================================================================
// JarvisEngine::replay
// ===========================================================================
void JarvisEngine::replay() {
    if (!graph_captured)
        throw std::runtime_error("[engine] Graph not captured.");
    get_runtime().replay();
}

// ===========================================================================
// JarvisEngine::train_step
// ===========================================================================
float JarvisEngine::train_step(const int* token_ids) {
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();

    CUDA_CHECK(cudaMemcpyAsync(ws.token_ids, token_ids,
        (size_t)cfg.NT() * cfg.accum_steps * sizeof(int), cudaMemcpyHostToDevice, s));

    if (graph_captured) {
        rt.replay();
    } else {
        CUDA_CHECK(cudaMemsetAsync(ws.grad_norm_sq, 0, sizeof(float), s));
        quantize_all_weights();
        float grad_scale = 1.0f / (float)cfg.accum_steps;
        for (int us = 0; us < cfg.accum_steps; ++us) {
            forward_microstep(us);
            backward_microstep(us, grad_scale);
        }
        optimizer_step();
        quantize_all_weights();
    }

    rt.sync();
    CUDA_CHECK(cudaMemcpy(&step_count, ws.adam_step, sizeof(step_count), cudaMemcpyDeviceToHost));

    // The optimizer records an invalid clipping coefficient when its gradient
    // norm is non-finite. Check it before returning a seemingly valid loss.
    float clip_coef = 0.0f;
    CUDA_CHECK(cudaMemcpy(&clip_coef, ws.d_clip_coef, sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(clip_coef))
        throw std::runtime_error("[optimizer] Non-finite gradient norm; skipped optimizer update.");

    float loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&loss, ws.loss, sizeof(float), cudaMemcpyDeviceToHost));
    return loss;
}

// ===========================================================================
// JarvisEngine::benchmark
// ===========================================================================
float JarvisEngine::benchmark(int n_steps) {
    if (!graph_captured)
        throw std::runtime_error("[engine] Must capture graph first.");
    auto& rt = get_runtime();
    int tps = cfg.batch_size * cfg.seq_len * cfg.accum_steps;
    rt.record_start();
    for (int i = 0; i < n_steps; ++i) rt.replay();
    rt.record_stop();
    float ms = rt.elapsed_ms() / (float)n_steps;
    float tok_s = (float)tps / (ms / 1000.0f);
    printf("[bench] %.3f ms/step  |  %.1f tok/s\n", ms, tok_s);
    return tok_s;
}

// ===========================================================================
// JarvisEngine::profile_step — CUDA-event timing breakdown
// ===========================================================================
void JarvisEngine::profile_step(const int* token_ids, int n_reps,
                                 float& out_fwd_ms, float& out_bwd_ms,
                                 float& out_opt_ms)
{
    auto& rt = get_runtime();
    cudaStream_t s = rt.get_stream();

    // Copy tokens to device if not already there
    CUDA_CHECK(cudaMemcpyAsync(ws.token_ids, token_ids,
        (size_t)cfg.NT() * sizeof(int), cudaMemcpyHostToDevice, s));

    // Create 6 events: start_fwd, end_fwd, start_bwd, end_bwd, start_opt, end_opt
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    auto mkev = [](cudaEvent_t& e) {
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    };
    auto mktm = [](cudaEvent_t& e) { cudaEventCreate(&e); };
    mktm(e0); mktm(e1); mktm(e2); mktm(e3); mktm(e4); mktm(e5);

    float total_fwd = 0, total_bwd = 0, total_opt = 0;

    for (int rep = 0; rep < n_reps; ++rep) {
        // Zero grad buffers before first accum step
        zero_grad_buffers(ws, cfg, s);
        cudaMemsetAsync(ws.grad_norm_sq, 0, sizeof(float), s);

        // ---- Forward timing ----
        cudaEventRecord(e0, s);
        for (int u = 0; u < cfg.accum_steps; ++u)
            forward_microstep(u);
        cudaEventRecord(e1, s);

        // ---- Backward timing ----
        cudaEventRecord(e2, s);
        float gs = 1.0f / (float)cfg.accum_steps;
        for (int u = 0; u < cfg.accum_steps; ++u)
            backward_microstep(u, gs);
        cudaEventRecord(e3, s);

        // ---- Optimizer timing ----
        cudaEventRecord(e4, s);
        optimizer_step();
        cudaEventRecord(e5, s);

        cudaStreamSynchronize(s);

        float f, b, o;
        cudaEventElapsedTime(&f, e0, e1);
        cudaEventElapsedTime(&b, e2, e3);
        cudaEventElapsedTime(&o, e4, e5);
        total_fwd += f;
        total_bwd += b;
        total_opt += o;
    }

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaEventDestroy(e4); cudaEventDestroy(e5);

    out_fwd_ms = total_fwd / n_reps;
    out_bwd_ms = total_bwd / n_reps;
    out_opt_ms = total_opt / n_reps;

    printf("[profile] fwd=%.1f ms | bwd=%.1f ms | opt=%.1f ms | total=%.1f ms\n",
           out_fwd_ms, out_bwd_ms, out_opt_ms,
           out_fwd_ms + out_bwd_ms + out_opt_ms);
}


// ===========================================================================
// Checkpoint
// ===========================================================================
void JarvisEngine::save_checkpoint(const char* path) {
    get_runtime().sync();
    CUDA_CHECK(cudaMemcpy(&step_count, ws.adam_step, sizeof(step_count), cudaMemcpyDeviceToHost));
    const std::string temp_path = std::string(path) + ".tmp";
    std::ofstream f(temp_path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open checkpoint for writing");

    f.write(reinterpret_cast<const char*>(&step_count), sizeof(step_count));

    const int L = cfg.n_layers, C = cfg.d_model, E = cfg.n_experts, hi = cfg.expert_hidden;
    const int Vp = cfg.vocab_padded, H = cfg.n_heads;

    auto write_fp32 = [&](const float* dev, size_t n) {
        std::vector<float> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost));
        f.write(reinterpret_cast<const char*>(h.data()), n * sizeof(float));
        if (!f) throw std::runtime_error("Checkpoint write failed");
    };

    write_fp32(params.tok_emb_fp32, (size_t)Vp * C);
    write_fp32(params.final_norm_w_fp32, C);
    for (int l = 0; l < L; ++l) {
        write_fp32(params.qkv_fp32[l],    (size_t)3*C*C);
        write_fp32(params.out_fp32[l],    (size_t)C*C);
        write_fp32(params.exp_w1_fp32[l], (size_t)E*hi*C);
        write_fp32(params.exp_w2_fp32[l], (size_t)E*C*hi);
        write_fp32(params.norm1_fp32[l],  (size_t)C);
        write_fp32(params.norm2_fp32[l],  (size_t)C);
        write_fp32(params.gamma_raw[l],   (size_t)H);
    }
    // router weights appended at the end (older checkpoints lack this section)
    for (int l = 0; l < L; ++l) write_fp32(params.router_fp32[l], (size_t)E*C);
    write_fp32(params.tok_emb_m, (size_t)Vp*C); write_fp32(params.tok_emb_v, (size_t)Vp*C);
    for (int l = 0; l < L; ++l) {
        write_fp32(params.qkv_m[l], (size_t)3*C*C); write_fp32(params.qkv_v[l], (size_t)3*C*C);
        write_fp32(params.out_m[l], (size_t)C*C); write_fp32(params.out_v[l], (size_t)C*C);
        write_fp32(params.exp_w1_m[l], (size_t)E*hi*C); write_fp32(params.exp_w1_v[l], (size_t)E*hi*C);
        write_fp32(params.exp_w2_m[l], (size_t)E*C*hi); write_fp32(params.exp_w2_v[l], (size_t)E*C*hi);
        write_fp32(params.router_m[l], (size_t)E*C); write_fp32(params.router_v[l], (size_t)E*C);
    }
    write_fp32(params.final_norm_m, C); write_fp32(params.final_norm_v, C);
    f.flush();
    if (!f) throw std::runtime_error("Checkpoint flush failed");
    f.close();
    if (!f) throw std::runtime_error("Checkpoint close failed");
#ifdef _WIN32
    if (!MoveFileExA(temp_path.c_str(), path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Checkpoint atomic replace failed");
#else
    if (std::rename(temp_path.c_str(), path) != 0)
        throw std::runtime_error("Checkpoint atomic replace failed");
#endif
    printf("[checkpoint] Saved '%s' step=%d\n", path, step_count);
}

void JarvisEngine::load_checkpoint(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open checkpoint for reading");

    f.read(reinterpret_cast<char*>(&step_count), sizeof(step_count));
    if (!f || step_count < 0) throw std::runtime_error("Invalid checkpoint header");

    const int L = cfg.n_layers, C = cfg.d_model, E = cfg.n_experts, hi = cfg.expert_hidden;
    const int Vp = cfg.vocab_padded, H = cfg.n_heads;

    const size_t core_floats = (size_t)Vp*C + C + (size_t)L *
        (3ULL*C*C + (size_t)C*C + (size_t)E*hi*C + (size_t)E*C*hi + 2ULL*C + H);
    const size_t core_bytes = sizeof(int) + core_floats * sizeof(float);
    const size_t router_floats = (size_t)L*E*C;
    const size_t routed_bytes = core_bytes + router_floats*sizeof(float);
    const size_t adam_floats = (size_t)Vp*C + C + (size_t)L *
        (3ULL*C*C + (size_t)C*C + (size_t)E*hi*C + (size_t)E*C*hi + (size_t)E*C);
    const size_t full_bytes = routed_bytes + 2*adam_floats*sizeof(float);
    const size_t full_bytes_legacy = full_bytes + router_floats*sizeof(float);
    f.seekg(0, std::ios::end);
    const std::streamoff file_bytes = f.tellg();
    const bool has_adam = file_bytes == (std::streamoff)full_bytes ||
                          file_bytes == (std::streamoff)full_bytes_legacy ||
                          file_bytes == (std::streamoff)(full_bytes + sizeof(float)) ||
                          file_bytes == (std::streamoff)(full_bytes_legacy + sizeof(float));
    if (file_bytes != (std::streamoff)core_bytes && file_bytes != (std::streamoff)routed_bytes && !has_adam)
        throw std::runtime_error("Checkpoint size does not match this model");
    f.seekg(0, std::ios::beg);

    auto read_fp32 = [&](float* dev, size_t n) {
        std::vector<float> h(n);
        f.read(reinterpret_cast<char*>(h.data()), n * sizeof(float));
        if (!f) throw std::runtime_error("Truncated checkpoint");
        CUDA_CHECK(cudaMemcpy(dev, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    read_fp32(params.tok_emb_fp32, (size_t)Vp * C);
    read_fp32(params.final_norm_w_fp32, C);
    for (int l = 0; l < L; ++l) {
        read_fp32(params.qkv_fp32[l],    (size_t)3*C*C);
        read_fp32(params.out_fp32[l],    (size_t)C*C);
        read_fp32(params.exp_w1_fp32[l], (size_t)E*hi*C);
        read_fp32(params.exp_w2_fp32[l], (size_t)E*C*hi);
        read_fp32(params.norm1_fp32[l],  (size_t)C);
        read_fp32(params.norm2_fp32[l],  (size_t)C);
        read_fp32(params.gamma_raw[l],   (size_t)H);
    }
    // optional router section; if absent keep current router weights
    bool has_router = true;
    for (int l = 0; l < L && has_router; ++l) {
        std::vector<float> h((size_t)E*C);
        f.read(reinterpret_cast<char*>(h.data()), h.size() * sizeof(float));
        if (!f) { has_router = false; break; }
        CUDA_CHECK(cudaMemcpy(params.router_fp32[l], h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (has_adam) {
        read_fp32(params.tok_emb_m, (size_t)Vp*C); read_fp32(params.tok_emb_v, (size_t)Vp*C);
        for (int l = 0; l < L; ++l) {
            read_fp32(params.qkv_m[l], (size_t)3*C*C); read_fp32(params.qkv_v[l], (size_t)3*C*C);
            read_fp32(params.out_m[l], (size_t)C*C); read_fp32(params.out_v[l], (size_t)C*C);
            read_fp32(params.exp_w1_m[l], (size_t)E*hi*C); read_fp32(params.exp_w1_v[l], (size_t)E*hi*C);
            read_fp32(params.exp_w2_m[l], (size_t)E*C*hi); read_fp32(params.exp_w2_v[l], (size_t)E*C*hi);
            read_fp32(params.router_m[l], (size_t)E*C); read_fp32(params.router_v[l], (size_t)E*C);
        }
        read_fp32(params.final_norm_m, C); read_fp32(params.final_norm_v, C);
    }
    CUDA_CHECK(cudaMemcpy(ws.adam_step, &step_count, sizeof(step_count), cudaMemcpyHostToDevice));
    get_runtime().sync();
    // Re-quantize after loading, then refresh BF16 copies of non-ternary params
    quantize_all_weights();
    {
        cudaStream_t s = get_runtime().get_stream();
        launch_f32_to_bf16(params.tok_emb_fp32, params.tok_emb, (size_t)Vp * C, s);
        launch_f32_to_bf16(params.final_norm_w_fp32, params.final_norm_w, C, s);
        for (int l = 0; l < L; ++l) {
            launch_f32_to_bf16(params.norm1_fp32[l], params.norm1_w[l], C, s);
            launch_f32_to_bf16(params.norm2_fp32[l], params.norm2_w[l], C, s);
            if (has_router) launch_f32_to_bf16(params.router_fp32[l], params.router_w[l], E*C, s);
        }
        get_runtime().sync();
    }
    printf("[checkpoint] Loaded '%s' step=%d\n", path, step_count);
}

// ===========================================================================
// JarvisEngine::destroy
// ===========================================================================
void JarvisEngine::destroy() {
    get_cublaslt().destroy();
    free_workspace(ws, cfg);
    free_params(params, cfg);
    if (rope_cos)    { cudaFree(rope_cos);    rope_cos    = nullptr; }
    if (rope_sin)    { cudaFree(rope_sin);    rope_sin    = nullptr; }
    if (quant_alpha) { cudaFree(quant_alpha); quant_alpha = nullptr; }
    destroy_runtime();
    graph_captured = false;
    printf("[engine] Destroyed.\n");
}
