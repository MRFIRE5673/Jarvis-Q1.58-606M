"""
jarvis_engine/cuda_engine/__init__.py
======================================
Auto-resolving loader for the native CUDA engine .pyd extension.

The extension links against:
  - cublasLt64_13.dll / cudart64_13.dll  (CUDA 13.3 toolkit: bin/x64/)
  - c10.dll / torch_cpu.dll              (PyTorch lib/)

Windows requires these to be pre-loaded before importing the .pyd.
This module handles that transparently.

Usage:
    from jarvis_engine.cuda_engine import eng
    eng.init()
    eng.warmup(3)
    eng.capture_graph()
    loss = eng.train_step(token_ids_np)
"""

import os
import sys
import ctypes
import glob

# ---------------------------------------------------------------------------
# 1. Pre-load CUDA 13.x DLLs
# ---------------------------------------------------------------------------
_CUDA_BIN_DIRS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
    r"C:\CUDA\v13.3\bin\x64",
]

_CUDA_DLL_LOADED = False
for _cuda_dir in _CUDA_BIN_DIRS:
    if os.path.isdir(_cuda_dir):
        for _dll in os.listdir(_cuda_dir):
            if _dll.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(_cuda_dir, _dll))
                except Exception:
                    pass
        _CUDA_DLL_LOADED = True
        break

if not _CUDA_DLL_LOADED:
    import warnings
    warnings.warn(
        "[cuda_engine] CUDA 13.x DLLs not found in standard locations. "
        "Set CUDA_PATH or add CUDA bin/x64 to PATH.",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# 2. Pre-load PyTorch DLLs (c10.dll, torch_cpu.dll, torch_cuda.dll etc.)
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        for _dll in os.listdir(_torch_lib):
            if _dll.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(_torch_lib, _dll))
                except Exception:
                    pass
except ImportError:
    pass  # PyTorch not installed — bindings won't work but won't crash import

# ---------------------------------------------------------------------------
# 3. Import the compiled .pyd extension
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import native_cuda_engine as _eng  # type: ignore
    # Re-export the module as `eng` for convenient use
    eng = _eng
    # Also expose top-level symbols
    init           = _eng.init
    init_default   = _eng.init_default
    warmup         = _eng.warmup
    capture_graph  = _eng.capture_graph
    replay         = _eng.replay
    train_step     = _eng.train_step
    benchmark      = _eng.benchmark
    save_checkpoint = _eng.save_checkpoint
    load_checkpoint = _eng.load_checkpoint
    get_loss       = _eng.get_loss
    memory_info    = _eng.memory_info
    synchronize    = _eng.synchronize
    is_graph_ready = _eng.is_graph_ready
    _ENGINE_AVAILABLE = True

except ImportError as _e:
    _ENGINE_AVAILABLE = False
    eng = None
    import warnings
    warnings.warn(
        f"[cuda_engine] Native engine not available: {_e}\n"
        "Run: cd jarvis_engine/cuda_engine && python setup.py build_ext --inplace",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = [
    "eng",
    "init", "init_default", "warmup", "capture_graph", "replay",
    "train_step", "benchmark", "save_checkpoint", "load_checkpoint",
    "get_loss", "memory_info", "synchronize", "is_graph_ready",
    "_ENGINE_AVAILABLE",
]
