"""
build_engine.py — Minimal build patcher for cpu-torch + CUDA 13.3
"""
import os, sys

CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["CUDA_PATH"] = CUDA_HOME
os.environ["PATH"] = os.path.join(CUDA_HOME, "bin") + ";" + os.environ.get("PATH", "")

import torch
if torch.version.cuda is None:
    torch.version.cuda = "13.3"

import torch.utils.cpp_extension as cpp_ext

# Patch _join_cuda_home
def _fixed_join(*parts):
    return os.path.join(CUDA_HOME, *parts)
cpp_ext._join_cuda_home = _fixed_join

# Patch library_paths to add CUDA 13.3 lib/x64
CUDA_LIB = os.path.join(CUDA_HOME, "lib", "x64")
_orig_lp = cpp_ext.library_paths
def _fixed_lp(*a, **kw):
    paths = _orig_lp(*a, **kw)
    if CUDA_LIB not in paths:
        paths.append(CUDA_LIB)
    return paths
cpp_ext.library_paths = _fixed_lp

# Patch CUDAExtension.libraries to remove cuda-torch-only libs after init
_orig_ext_init = cpp_ext.CUDAExtension.__init__
def _patched_ext_init(self, *args, **kwargs):
    _orig_ext_init(self, *args, **kwargs)
    remove = {'c10_cuda', 'torch_cuda', 'cudnn'}
    self.libraries = [l for l in (self.libraries or []) if l not in remove]
cpp_ext.CUDAExtension.__init__ = _patched_ext_init

# Suppress version checks
for _fn in ('_check_cuda_version', '_check_compiler_abi_compatibility'):
    _orig = getattr(cpp_ext, _fn, None)
    if _orig:
        def _noop(*a, _f=_orig, **kw):
            try: _f(*a, **kw)
            except: pass
        setattr(cpp_ext, _fn, _noop)

print(f"[build_engine] CUDA: {CUDA_HOME}  torch: {torch.__version__}")
sys.argv = ["setup.py", "build_ext", "--inplace"]
import runpy
runpy.run_path("setup.py", run_name="__main__")
