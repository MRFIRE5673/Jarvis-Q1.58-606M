import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jarvis_engine.cuda_engine import eng, _ENGINE_AVAILABLE

if not _ENGINE_AVAILABLE or eng is None:
    print("[bench_engine] Native CUDA engine not built or unavailable on this platform.")
    print("               To build on an NVIDIA GPU system: cd jarvis_engine/cuda_engine && python build_engine.py")
    sys.exit(0)

try:
    eng.init(8, 512, 1024, 16, 24, 4, 2, 2048, 50257, 50304, 64, 2, 1e-4, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.01, 0.01, 0.1, 0.99, 0)
    eng.warmup(1)
    eng.capture_graph()
    toks = eng.benchmark(10)
    print("TOKS", toks)
except Exception as e:
    print(f"[bench_engine] Benchmark failed: {e}")
