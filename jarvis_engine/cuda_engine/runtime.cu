// =============================================================================
// runtime.cu — JarvisRuntime Implementation
// =============================================================================
// RTX 5070 / SM120 / CUDA 12.8+
// =============================================================================
#include "runtime.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------
static JarvisRuntime g_runtime;

JarvisRuntime& get_runtime()    { return g_runtime; }
void init_runtime(int device)   { g_runtime.init(device); }
void destroy_runtime()          { g_runtime.destroy(); }

// ---------------------------------------------------------------------------
// JarvisRuntime::init
// ---------------------------------------------------------------------------
void JarvisRuntime::init(int device) {
    device_id = device;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    // Validate: we expect SM120 (compute 12.0) on the RTX 5070
    if (props.major < 10) {
        fprintf(stderr,
            "[runtime] WARNING: Expected compute 12.x (Blackwell SM120), "
            "got compute %d.%d — some optimizations may not apply.\n",
            props.major, props.minor);
    }

    // memoryClockRate removed in CUDA 13.3 — report bus width only.
    // RTX 5070 GDDR7: 192-bit bus @ ~672-768 GB/s (known from golden_gpu_state.json).
    float est_bw_gbs = props.memoryBusWidth * 2.0f / 8.0f;  // rough estimate in GB/s at 1 GHz
    printf("[runtime] Device %d: %s  |  SM %d.%d  |  "
           "%d SMs  |  %d-bit bus  |  %.0f MB L2\n",
           device,
           props.name,
           props.major, props.minor,
           props.multiProcessorCount,
           props.memoryBusWidth,
           (float)props.l2CacheSize / 1048576.0f);
    (void)est_bw_gbs;

    // Non-blocking stream — allows async kernel overlap in future expansion
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Timing events
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Keep the measured 24 MB default; override with JARVIS_L2_PIN_MB to tune.
    if (props.persistingL2CacheMaxSize > 0) {
        size_t max_persisting_cache_size = (size_t)props.persistingL2CacheMaxSize;
        persisting_cache_size = std::min(max_persisting_cache_size,
                                         (size_t)24 * 1024 * 1024);
        if (const char* env = std::getenv("JARVIS_L2_PIN_MB")) {
            char* end = nullptr;
            long mb = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && mb >= 0) {
                persisting_cache_size = (size_t)mb > max_persisting_cache_size / (1024 * 1024)
                    ? max_persisting_cache_size : (size_t)mb * 1024 * 1024;
            }
        }
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting_cache_size));
        printf("[runtime] L2 persisting set-aside: %.0f MB (of %.0f MB total L2)\n",
               persisting_cache_size / 1048576.0f,
               (float)props.l2CacheSize / 1048576.0f);
    }
}

void JarvisRuntime::set_persisting_window(void* base, size_t bytes) {
    size_t window_bytes = std::min(bytes, (size_t)props.accessPolicyMaxWindowSize);
    window_bytes = std::min(window_bytes, persisting_cache_size);
    if (!base || !window_bytes) return;

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = base;
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    printf("[runtime] L2 persisting window: %.2f MB\n", window_bytes / 1048576.0f);
}

// ---------------------------------------------------------------------------
// JarvisRuntime::destroy
// ---------------------------------------------------------------------------
void JarvisRuntime::destroy() {
    if (graph_exec)  { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
    if (graph)       { cudaGraphDestroy(graph);           graph      = nullptr; }
    if (ev_start)    { cudaEventDestroy(ev_start);        ev_start   = nullptr; }
    if (ev_stop)     { cudaEventDestroy(ev_stop);         ev_stop    = nullptr; }
    if (stream)      { cudaStreamDestroy(stream);         stream     = nullptr; }
    graph_ready = false;
    printf("[runtime] Destroyed.\n");
}

// ---------------------------------------------------------------------------
// Graph capture
// ---------------------------------------------------------------------------
void JarvisRuntime::begin_graph_capture() {
    // Invalidate any previous graph
    if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
    if (graph)      { cudaGraphDestroy(graph);           graph      = nullptr; }
    graph_ready = false;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    printf("[runtime] CUDA Graph capture started.\n");
}

void JarvisRuntime::end_graph_capture() {
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate with auto-free (replaces previous exec if re-captured)
    cudaGraphExecUpdateResult update_result;
    if (graph_exec) {
        cudaGraphNode_t error_node;
        cudaGraphExecUpdate(graph_exec, graph, &error_node, &update_result);
        if (update_result != cudaGraphExecUpdateSuccess) {
            cudaGraphExecDestroy(graph_exec);
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        }
    } else {
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    }

    graph_ready = true;
    printf("[runtime] CUDA Graph captured and instantiated.\n");
}

void JarvisRuntime::replay() {
    if (!graph_ready || !graph_exec)
        throw std::runtime_error("[runtime] Graph not ready — call end_graph_capture() first.");
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
}

void JarvisRuntime::invalidate_graph() {
    if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
    if (graph)      { cudaGraphDestroy(graph);           graph      = nullptr; }
    graph_ready = false;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
void JarvisRuntime::record_start() {
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
}
void JarvisRuntime::record_stop() {
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
}
float JarvisRuntime::elapsed_ms() const {
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return ms;
}

// ---------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------
void JarvisRuntime::sync() const {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
