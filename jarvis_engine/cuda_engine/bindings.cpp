// =============================================================================
// bindings.cpp — PyBind11 Python Interface for native_cuda_engine
// =============================================================================
// Exposes the JarvisEngine to Python with a clean, minimal API.
// The training loop itself runs entirely in C++/CUDA under a static CUDA Graph.
// Python only needs to: init, load_weights, set token_ids, step/replay, get_loss.
// =============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

#include "full_engine.h"
#include "runtime.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static const int* int_ptr(const py::array_t<int>& a) {
    return static_cast<const int*>(a.request().ptr);
}

// ---------------------------------------------------------------------------
// Engine API
// ---------------------------------------------------------------------------

static void py_init(
    int batch_size, int seq_len,
    int d_model, int n_heads, int n_layers,
    int n_experts, int top_k, int expert_hidden,
    int vocab_size, int vocab_padded,
    int chunk_size, int accum_steps,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float grad_clip,
    float balance_alpha, float reflect_lambda,
    float lsf_alpha_min, float lsf_alpha_max,
    int device
) {
    JarvisConfig cfg;
    cfg.batch_size    = batch_size;
    cfg.seq_len       = seq_len;
    cfg.d_model       = d_model;
    cfg.n_heads       = n_heads;
    cfg.head_dim      = d_model / n_heads;
    cfg.n_layers      = n_layers;
    cfg.n_experts     = n_experts;
    cfg.top_k         = top_k;
    cfg.expert_hidden = expert_hidden;
    cfg.vocab_size    = vocab_size;
    cfg.vocab_padded  = vocab_padded;
    cfg.chunk_size    = chunk_size;
    cfg.accum_steps   = accum_steps;
    cfg.lr            = lr;
    cfg.beta1         = beta1;
    cfg.beta2         = beta2;
    cfg.eps           = eps;
    cfg.weight_decay  = weight_decay;
    cfg.grad_clip     = grad_clip;
    cfg.balance_alpha = balance_alpha;
    cfg.reflect_lambda= reflect_lambda;
    cfg.lsf_alpha_min = lsf_alpha_min;
    cfg.lsf_alpha_max = lsf_alpha_max;
    cfg.device        = device;
    get_engine().init(cfg, device);
}

static void py_init_default(int device = 0) {
    JarvisConfig cfg;
    cfg.device = device;
    get_engine().init(cfg, device);
}

static void py_warmup(int n_steps = 3) {
    get_engine().warmup(n_steps);
}

static void py_capture_graph() {
    get_engine().capture_graph();
}

static void py_replay() {
    get_engine().replay();
}

static float py_train_step(const py::array_t<int>& token_ids) {
    return get_engine().train_step(int_ptr(token_ids));
}

static float py_benchmark(int n_steps = 100) {
    return get_engine().benchmark(n_steps);
}

static py::dict py_profile_step(const py::array_t<int>& token_ids, int n_reps = 5) {
    float fwd_ms, bwd_ms, opt_ms;
    get_engine().profile_step(int_ptr(token_ids), n_reps, fwd_ms, bwd_ms, opt_ms);
    py::dict d;
    d["fwd_ms"] = fwd_ms;
    d["bwd_ms"] = bwd_ms;
    d["opt_ms"] = opt_ms;
    d["total_ms"] = fwd_ms + bwd_ms + opt_ms;
    return d;
}

static float py_get_loss() {
    float v = 0.0f;
    cudaMemcpy(&v, get_engine().ws.loss, sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

static void py_save(const std::string& path) {
    get_engine().save_checkpoint(path.c_str());
}

static void py_load(const std::string& path) {
    get_engine().load_checkpoint(path.c_str());
}

static void py_sync() {
    get_runtime().sync();
}

static void py_destroy() {
    get_engine().destroy();
}

static py::dict py_memory_info() {
    size_t fr, tot;
    cudaMemGetInfo(&fr, &tot);
    py::dict d;
    d["free_mb"]  = (int)(fr  >> 20);
    d["total_mb"] = (int)(tot >> 20);
    d["used_mb"]  = (int)((tot - fr) >> 20);
    return d;
}

static bool py_graph_ready() {
    return get_runtime().is_graph_ready();
}

// Load a single weight tensor from a numpy array (FP32) onto device
// key: "qkv.0", "out.5", "emb", "norm1.3", etc.
// Convert FP32 host data to BF16 on device (BF16 copies are otherwise zero until the first optimizer step)
static void put_bf16(bf16* dst, const float* src, size_t n) {
    std::vector<bf16> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = __float2bfloat16(src[i]);
    cudaMemcpy(dst, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice);
}

static void py_load_weight_np(const std::string& key, const py::array_t<float>& arr) {
    auto buf = arr.request();
    const float* src = static_cast<const float*>(buf.ptr);
    size_t n = buf.size;

    auto& e = get_engine();
    auto& p = e.params;
    const auto& cfg = e.cfg;

    // Dispatch by key prefix
    if (key == "emb") {
        cudaMemcpy(p.tok_emb_fp32, src, n * sizeof(float), cudaMemcpyHostToDevice);
        put_bf16(p.tok_emb, src, n);
    } else if (key.rfind("qkv.", 0) == 0) {
        int l = std::stoi(key.substr(4));
        cudaMemcpy(p.qkv_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
    } else if (key.rfind("out.", 0) == 0) {
        int l = std::stoi(key.substr(4));
        cudaMemcpy(p.out_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
    } else if (key.rfind("w1.", 0) == 0) {
        int l = std::stoi(key.substr(3));
        cudaMemcpy(p.exp_w1_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
    } else if (key.rfind("w2.", 0) == 0) {
        int l = std::stoi(key.substr(3));
        cudaMemcpy(p.exp_w2_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
    } else if (key.rfind("router.", 0) == 0) {
        int l = std::stoi(key.substr(7));
        cudaMemcpy(p.router_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
        put_bf16(p.router_w[l], src, n);
    } else if (key.rfind("norm1.", 0) == 0) {
        int l = std::stoi(key.substr(6));
        cudaMemcpy(p.norm1_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
        put_bf16(p.norm1_w[l], src, n);
    } else if (key.rfind("norm2.", 0) == 0) {
        int l = std::stoi(key.substr(6));
        cudaMemcpy(p.norm2_fp32[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
        put_bf16(p.norm2_w[l], src, n);
    } else if (key.rfind("gamma.", 0) == 0) {
        int l = std::stoi(key.substr(6));
        cudaMemcpy(p.gamma_raw[l], src, n * sizeof(float), cudaMemcpyHostToDevice);
    } else if (key == "final_norm") {
        cudaMemcpy(p.final_norm_w_fp32, src, n * sizeof(float), cudaMemcpyHostToDevice);
        put_bf16(p.final_norm_w, src, n);
    } else {
        fprintf(stderr, "[bindings] Unknown weight key: '%s'\n", key.c_str());
    }
}

// ---------------------------------------------------------------------------
// MODULE DEFINITION
// ---------------------------------------------------------------------------
PYBIND11_MODULE(native_cuda_engine, m) {
    m.doc() = R"(
Jarvis-Q1.58-500M Native CUDA Training Engine
==============================================
Hardware target: NVIDIA GeForce RTX 5070 (GB205, Blackwell SM120)
  48 SMs | 6144 CUDA cores | 12 GB GDDR7 | 672 GB/s | 48 MB L2 | 2.51 GHz

Architecture (LOCKED per architecture.md):
  24 layers | d_model=1024 | 16 heads | D=64
  O(N) Associative Linear Attention with ELU+1+RoPE feature maps
  Q1.58 Spiking Ternary Quantization (BitNet b1.58) with AbsMean + STE
  4-expert Top-2 MoE with load-balance aux loss
  Liquid State Fusion (LSF) SNN membrane dynamics
  Reflective regularization penalty

Usage:
    import native_cuda_engine as ce
    ce.init_default(device=0)
    ce.warmup(3)
    ce.capture_graph()
    for batch in loader:
        loss = ce.train_step(batch)  # or: ce.replay(); loss = ce.get_loss()
)";

    m.def("init", &py_init,
          "Initialize engine with explicit config",
          py::arg("batch_size")=4, py::arg("seq_len")=512,
          py::arg("d_model")=1024, py::arg("n_heads")=16, py::arg("n_layers")=24,
          py::arg("n_experts")=4, py::arg("top_k")=2, py::arg("expert_hidden")=2048,
          py::arg("vocab_size")=50257, py::arg("vocab_padded")=50304,
          py::arg("chunk_size")=64, py::arg("accum_steps")=2,
          py::arg("lr")=1e-4f, py::arg("beta1")=0.9f, py::arg("beta2")=0.999f,
          py::arg("eps")=1e-8f, py::arg("weight_decay")=0.01f, py::arg("grad_clip")=1.0f,
          py::arg("balance_alpha")=0.01f, py::arg("reflect_lambda")=0.01f,
          py::arg("lsf_alpha_min")=0.1f, py::arg("lsf_alpha_max")=0.99f,
          py::arg("device")=0);

    m.def("init_default",   &py_init_default,  "Init with default Jarvis config", py::arg("device")=0);
    m.def("warmup",         &py_warmup,        "Warmup steps before graph capture", py::arg("n_steps")=3);
    m.def("capture_graph",  &py_capture_graph, "Capture CUDA Graph for zero-overhead replay");
    m.def("replay",         &py_replay,        "Replay captured CUDA Graph");
    m.def("train_step",     &py_train_step,    "Run one training step, return loss", py::arg("token_ids"));
    m.def("benchmark",      &py_benchmark,     "Benchmark N replays, return tok/s", py::arg("n_steps")=100);
    m.def("profile_step",   &py_profile_step,  "Profile fwd/bwd/opt timing breakdown",
          py::arg("token_ids"), py::arg("n_reps")=5);
    m.def("get_loss",       &py_get_loss,      "Read loss scalar from device");
    m.def("save_checkpoint",&py_save,          "Save checkpoint", py::arg("path"));
    m.def("load_checkpoint",&py_load,          "Load checkpoint", py::arg("path"));
    m.def("load_weight",    &py_load_weight_np,"Load single weight from numpy FP32 array",
          py::arg("key"), py::arg("arr"));
    m.def("synchronize",    &py_sync,          "Synchronize CUDA stream");
    m.def("destroy",        &py_destroy,       "Destroy engine and free GPU memory");
    m.def("memory_info",    &py_memory_info,   "Return {free_mb, total_mb, used_mb}");
    m.def("is_graph_ready", &py_graph_ready,   "True if CUDA Graph is captured and ready");
}
