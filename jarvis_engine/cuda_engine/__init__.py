"""
jarvis_engine/cuda_engine/__init__.py
======================================
Auto-resolving loader for the native CUDA engine extension.

Cross-platform support:
  - Windows: loads .pyd and dependent DLLs (cublasLt, cudart, c10, torch_cpu, torch_cuda)
  - Linux: loads .so and dependent shared objects from standard CUDA and PyTorch lib paths
  - Safe fallback: if native CUDA engine is absent, symbols default gracefully without crash
"""

import os
import sys
import ctypes
import glob

# ---------------------------------------------------------------------------
# 1. Pre-load CUDA DLLs / Shared Libraries
# ---------------------------------------------------------------------------
_CUDA_SEARCH_PATHS = [
    os.environ.get("CUDA_HOME", ""),
    os.environ.get("CUDA_PATH", ""),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
    r"C:\CUDA\v13.3\bin\x64",
    r"C:\CUDA\v13.3\bin",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib",
] + glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\x64") + glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin") + glob.glob("/usr/local/cuda-*/lib64")

_dll_handles = []
for _cuda_dir in _CUDA_SEARCH_PATHS:
    if _cuda_dir and os.path.isdir(_cuda_dir):
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                _dll_handles.append(os.add_dll_directory(_cuda_dir))
            except Exception:
                pass
        if _cuda_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cuda_dir + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------------------------
# 2. Pre-load PyTorch Libraries
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                _dll_handles.append(os.add_dll_directory(_torch_lib))
            except Exception:
                pass
        if _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 3. Import the compiled native extension (.pyd / .so)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import native_cuda_engine as _eng  # type: ignore
    eng = _eng
    init            = getattr(_eng, "init", None)
    init_default    = getattr(_eng, "init_default", None)
    warmup          = getattr(_eng, "warmup", None)
    capture_graph   = getattr(_eng, "capture_graph", None)
    replay          = getattr(_eng, "replay", None)
    train_step      = getattr(_eng, "train_step", None)
    benchmark       = getattr(_eng, "benchmark", None)
    save_checkpoint = getattr(_eng, "save_checkpoint", None)
    load_checkpoint = getattr(_eng, "load_checkpoint", None)
    get_loss        = getattr(_eng, "get_loss", None)
    memory_info     = getattr(_eng, "memory_info", None)
    synchronize     = getattr(_eng, "synchronize", None)
    is_graph_ready  = getattr(_eng, "is_graph_ready", None)
    _ENGINE_AVAILABLE = True

except ImportError as _e:
    _ENGINE_AVAILABLE = False
    eng = None
    init = None
    init_default = None
    warmup = None
    capture_graph = None
    replay = None
    train_step = None
    benchmark = None
    save_checkpoint = None
    load_checkpoint = None
    get_loss = None
    memory_info = None
    synchronize = None
    is_graph_ready = None

__all__ = [
    "eng",
    "init", "init_default", "warmup", "capture_graph", "replay",
    "train_step", "benchmark", "save_checkpoint", "load_checkpoint",
    "get_loss", "memory_info", "synchronize", "is_graph_ready",
    "_ENGINE_AVAILABLE",
]
