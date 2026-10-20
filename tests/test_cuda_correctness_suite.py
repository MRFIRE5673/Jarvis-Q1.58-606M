"""Small real-engine CUDA smoke test; requires the built extension and GPU."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE_DIR = os.path.join(ROOT, "jarvis_engine", "cuda_engine")
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

torch = pytest.importorskip("torch")
_DLL_HANDLES = []
if hasattr(os, "add_dll_directory"):
    for dll_dir in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
        os.path.join(os.path.dirname(torch.__file__), "lib"),
    ):
        if os.path.isdir(dll_dir):
            _DLL_HANDLES.append(os.add_dll_directory(dll_dir))
nce = pytest.importorskip("native_cuda_engine")


def test_native_cuda_training_step_is_finite():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")

    nce.init_default(device=0)
    rng = np.random.default_rng(42)
    vocab_padded, width, layers, experts, hidden, heads = 50304, 1024, 24, 4, 2048, 16
    n = lambda *shape: rng.standard_normal(shape, dtype=np.float32) * 0.02
    nce.load_weight("emb", n(vocab_padded, width))
    nce.load_weight("final_norm", np.ones(width, dtype=np.float32))
    for layer in range(layers):
        nce.load_weight(f"qkv.{layer}", n(3 * width, width))
        nce.load_weight(f"out.{layer}", n(width, width))
        nce.load_weight(f"w1.{layer}", n(experts * hidden, width))
        nce.load_weight(f"w2.{layer}", n(experts * width, hidden))
        nce.load_weight(f"norm1.{layer}", np.ones(width, dtype=np.float32))
        nce.load_weight(f"norm2.{layer}", np.ones(width, dtype=np.float32))
        nce.load_weight(f"gamma.{layer}", np.full(heads, np.log(0.95), dtype=np.float32))
        nce.load_weight(f"router.{layer}", n(experts, width))

    tokens = rng.integers(0, 50257, size=8 * 512 * 2, dtype=np.int32)
    loss = nce.train_step(tokens)
    assert np.isfinite(loss), f"native CUDA engine returned non-finite loss: {loss}"
