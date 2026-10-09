// =============================================================================
// runtime.cu — JarvisRuntime Implementation
// =============================================================================
// RTX 5070 / SM120 / CUDA 12.8+
// =============================================================================
#include "runtime.h"
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

    // Enable L2 cache persistence for the hot working set
    // L2 = 48 MB on RTX 5070 — pin up to 24 MB for our hot activations
    // (other 24 MB left for texture/constant caches, driver overhead)
    if (props.persistingL2CacheMaxSize > 0) {
        size_t pin_size = min(
            (size_t)props.persistingL2CacheMaxSize,
            24ULL * 1024 * 1024  // 24 MB
        );
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pin_size));
        printf("[runtime] L2 persistence: %.0f MB pinned (of %.0f MB total L2)\n",
               pin_size / 1048576.0f,
               (float)props.l2CacheSize / 1048576.0f);
    }
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
