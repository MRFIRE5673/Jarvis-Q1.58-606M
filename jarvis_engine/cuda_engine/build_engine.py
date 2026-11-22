"""Build the native engine against the installed CUDA toolkit."""
import os, sys, glob

CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if not CUDA_HOME:
    candidates = []
    if sys.platform == "win32":
        cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(cuda_base):
            candidates = sorted(
                (os.path.join(cuda_base, n) for n in os.listdir(cuda_base) if n.startswith("v")),
                reverse=True
            )
    else:
        candidates = sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
        if os.path.isdir("/usr/local/cuda"):
            candidates.insert(0, "/usr/local/cuda")
        if os.path.isdir("/opt/cuda"):
            candidates.append("/opt/cuda")

    CUDA_HOME = candidates[0] if candidates else None

if not CUDA_HOME or not os.path.isdir(CUDA_HOME):
    raise RuntimeError("CUDA toolkit not found; set CUDA_HOME or CUDA_PATH to your CUDA installation directory.")

os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["CUDA_PATH"] = CUDA_HOME
cuda_bin = os.path.join(CUDA_HOME, "bin")
if cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

os.environ["JARVIS_ALLOW_CUDA_MISMATCH"] = "1"
import torch
import torch.utils.cpp_extension as cpp_ext

# Patch _check_cuda_version to allow CUDA version mismatch
cpp_ext._check_cuda_version = lambda *args, **kwargs: None

# Patch _join_cuda_home
def _fixed_join(*parts):
    return os.path.join(CUDA_HOME, *parts)
cpp_ext._join_cuda_home = _fixed_join

# Patch library_paths to add appropriate lib directory
if sys.platform == "win32":
    CUDA_LIB = os.path.join(CUDA_HOME, "lib", "x64")
else:
    CUDA_LIB = os.path.join(CUDA_HOME, "lib64")
    if not os.path.isdir(CUDA_LIB):
        CUDA_LIB = os.path.join(CUDA_HOME, "lib")

_orig_lp = cpp_ext.library_paths
def _fixed_lp(*a, **kw):
    paths = _orig_lp(*a, **kw)
    if os.path.isdir(CUDA_LIB) and CUDA_LIB not in paths:
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

print(f"[build_engine] CUDA: {CUDA_HOME}  torch: {torch.__version__} (platform: {sys.platform})")
engine_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(engine_dir)
sys.argv = ["setup.py", "build_ext", "--inplace"]
import runpy
runpy.run_path(os.path.join(engine_dir, "setup.py"), run_name="__main__")
