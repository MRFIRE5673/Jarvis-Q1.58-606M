// =============================================================================
// runtime.h — CUDA Runtime Management
// =============================================================================
// Target: NVIDIA GeForce RTX 5070 12GB Gaming Trio  |  GB205  |  Blackwell SM120
//
// VERIFIED HARDWARE — Source: golden_gpu_state.json (direct nvml query on this card)
//
//   SM Count:          46   (cudaGetDeviceProperties: sm_count=46)
//   VRAM:              12 GB GDDR7  |  192-bit bus
//   Memory clock OC:   16,001 MHz effective  |  32 Gbps per pin
//   Peak BW (OC):      768.048 GB/s  (192-bit × 32 Gbps / 8)
//   L2 Cache:          48.0 MB  (50,331,648 bytes)
//   Core clock OC:     3,330–3,375 MHz (sustained under training load)
//   Power OC:          237–240 W load draw  |  250 W limit
//   Compute Cap:       12.0  (sm_120)
//   CUDA toolkit:      13.3  (not 12.8)
//   Driver:            581.57
//   OS:                Windows 11 x64
//   PyTorch:           2.7.0.dev20250228+cu128  (or 2.12.0 depending on session)
//
// MEASURED PERFORMANCE FACTS (from phase24_optimization_ledger.md):
//   AdamW DRAM util:   673.8 GB/s actual (vs 768 OC peak — serial bandwidth limited)
//   AdamW BW factor:   673.8 / 672.0 rated = 100.27% of stock spec
//   GEMM util:         76.35 TFLOPS/s sustained across all 9 shapes
//
// ARCHITECTURE NOTES:
//   TMA support:       YES (PipelineTmaAsync, like Hopper sm_90)
//   TMEM / tcgen05:    NO  (datacenter SM100 only — H100/B100)
//   FP8 support:       YES (E4M3 / E5M2 via cuBLASLt on SM120)
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <cstdio>
#include <string>

// ---------------------------------------------------------------------------
// Verified hardware constants — ALL from golden_gpu_state.json
// ---------------------------------------------------------------------------
namespace hw {
    constexpr int   SM_COUNT          = 46;          // nvml verified
    constexpr long  L2_CACHE_BYTES    = 50331648LL;  // 48.0 MB exact
    constexpr int   L2_CACHE_MB       = 48;
    constexpr float MEM_BW_GBS_OC     = 768.048f;   // OC at 16001 MHz × 32 Gbps
    constexpr float MEM_BW_GBS_RATED  = 672.0f;     // stock 28 Gbps rated
    constexpr float MEM_BW_GBS_MEASURED = 673.8f;   // AdamW measured utilization
    constexpr int   MEM_CLOCK_MHZ     = 16001;       // OC memory clock
    constexpr int   CORE_CLOCK_MHZ_OC = 3337;        // typical OC boost
    constexpr int   TDP_WATTS         = 250;
    constexpr int   POWER_LOAD_WATTS  = 238;         // typical training load
    constexpr int   VRAM_MB           = 12288;
    constexpr int   COMPUTE_MAJOR     = 12;
    constexpr int   COMPUTE_MINOR     = 0;
    // L2 pinning budget: leave 24 MB free for cuBLASLt workspace staging
    // Pin: layer_x (2MB) + layer_x2 (2MB) + attn_state (0.5MB) = 4.5MB hot path
    constexpr long  L2_PIN_BUDGET     = 24LL * 1024 * 1024;
}

// ---------------------------------------------------------------------------
// Error macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            char _buf[512];                                                    \
            snprintf(_buf, sizeof(_buf),                                       \
                "[CUDA_CHECK] %s:%d  '%s'  → %s",                             \
                __FILE__, __LINE__, #call, cudaGetErrorString(_e));            \
            throw std::runtime_error(_buf);                                    \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t _e = cudaGetLastError();                                   \
        if (_e != cudaSuccess) {                                               \
            char _buf[512];                                                    \
            snprintf(_buf, sizeof(_buf),                                       \
                "[CUDA_CHECK_LAST] %s:%d → %s",                               \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
            throw std::runtime_error(_buf);                                    \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// JarvisRuntime — owns the single authoritative CUDA stream & graph
// ---------------------------------------------------------------------------
struct JarvisRuntime {
    cudaStream_t    stream       = nullptr;
    cudaGraph_t     graph        = nullptr;
    cudaGraphExec_t graph_exec   = nullptr;
    cudaEvent_t     ev_start     = nullptr;
    cudaEvent_t     ev_stop      = nullptr;
    bool            graph_ready  = false;

    cudaDeviceProp  props        = {};
    int             device_id    = 0;

    void init(int device = 0);
    void destroy();

    // Graph lifecycle
    void begin_graph_capture();     // Call after uncaptured warmup steps
    void end_graph_capture();       // Instantiates the executable graph
    void replay();                  // Zero-CPU-overhead graph replay
    void invalidate_graph();

    // Timing
    void  record_start();
    void  record_stop();
    float elapsed_ms() const;       // Blocking — waits for ev_stop

    // Sync
    void sync() const;

    bool         is_graph_ready() const { return graph_ready; }
    cudaStream_t get_stream()     const { return stream; }

    // Roofline helper: given bytes of DRAM traffic, return min time in ms
    float roofline_ms(size_t dram_bytes) const {
        // 672 GB/s = 672e9 bytes/s
        return (float)dram_bytes / (672e9f) * 1000.0f;
    }
};

// Global singleton
JarvisRuntime& get_runtime();
void init_runtime(int device = 0);
void destroy_runtime();
