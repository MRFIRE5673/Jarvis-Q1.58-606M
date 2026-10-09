#pragma once
// =============================================================================
// full_engine.h — Jarvis-Q1.58-500M Full Training Engine
// =============================================================================
#include "cuda_engine.h"
#include "cublaslt_engine.h"
#include "runtime.h"

struct JarvisEngine {
    JarvisConfig    cfg;
    JarvisParams    params;
    JarvisWorkspace ws;
    int             step_count    = 0;
    bool            graph_captured = false;

    // RoPE sinusoidal tables
    float* rope_cos    = nullptr;
    float* rope_sin    = nullptr;

    // Ternary quantization scratch (alpha scalar)
    float* quant_alpha = nullptr;

    // Lifecycle
    void init(const JarvisConfig& config, int device = 0);
    void warmup(int n_steps = 3);
    void capture_graph();
    void replay();
    float train_step(const int* token_ids);
    float benchmark(int n_steps = 100);
    // Returns {fwd_ms, bwd_ms, opt_ms} (averaged over n_reps GPU-timed steps)
    void  profile_step(const int* token_ids, int n_reps,
                       float& fwd_ms, float& bwd_ms, float& opt_ms);
    void save_checkpoint(const char* path);
    void load_checkpoint(const char* path);
    void destroy();

    // Internal steps
    void forward_microstep(int ustep);
    void backward_microstep(int ustep, float grad_scale);
    void optimizer_step();
    void quantize_all_weights();
};

JarvisEngine& get_engine();
