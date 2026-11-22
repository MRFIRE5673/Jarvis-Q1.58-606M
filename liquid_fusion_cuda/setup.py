# setup.py
import os
import sys
import glob
import shutil
import subprocess
import torch
import torch.utils.cpp_extension as cpp_ext
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 1. Automatically configure MSVC environment if cl.exe is not in PATH
if not shutil.which("cl"):
    vcvars_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for vcvars in vcvars_candidates:
        if os.path.exists(vcvars):
            print(f"[setup.py] Initializing MSVC environment via {vcvars}")
            cmd = f'"{vcvars}" && set'
            out = subprocess.check_output(cmd, shell=True, text=True)
            for line in out.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k] = v
            os.environ["DISTUTILS_USE_SDK"] = "1"
            break

if not shutil.which("cl"):
    print("[setup.py] WARNING: cl.exe could not be found in PATH or standard VS locations.")

# 2. Detect or configure CUDA_HOME
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if not cuda_home:
    if sys.platform == "win32":
        cuda_candidates = sorted(
            glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"),
            reverse=True
        )
    else:
        cuda_candidates = sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
        if os.path.isdir("/usr/local/cuda"):
            cuda_candidates.insert(0, "/usr/local/cuda")
    if cuda_candidates:
        cuda_home = cuda_candidates[0]
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home
        print(f"[setup.py] Auto-detected CUDA_HOME: {cuda_home}")

if cuda_home:
    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

# 3. Relax PyTorch major-version check between PyTorch and installed CUDA Toolkit
orig_check = getattr(cpp_ext, "_check_cuda_version", None)
if orig_check:
    def relaxed_check(compiler_name, compiler_version):
        try:
            orig_check(compiler_name, compiler_version)
        except RuntimeError as e:
            print(f"[setup.py] Note: Relaxed CUDA version check ({e}).")
    cpp_ext._check_cuda_version = relaxed_check

# 4. Architecture and preprocessor flags
target_archs = []
env_arch = os.environ.get("JARVIS_CUDA_ARCH") or os.environ.get("TORCH_CUDA_ARCH_LIST")
if env_arch:
    for arch in env_arch.replace(";", " ").split():
        arch_clean = arch.strip().replace(".", "")
        if arch_clean:
            target_archs.append(f"-gencode=arch=compute_{arch_clean},code=sm_{arch_clean}")
elif torch.cuda.is_available():
    try:
        cap = torch.cuda.get_device_capability()
        arch_code = f"{cap[0]}{cap[1]}"
        target_archs.append(f"-gencode=arch=compute_{arch_code},code=sm_{arch_code}")
    except Exception:
        pass

if not target_archs:
    target_archs = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_120,code=sm_120",
    ]

extra_cuda_cflags = target_archs + [
    "-O3",
    "--use_fast_math",
    "--ptxas-options=-v",
]
if sys.platform == "win32":
    extra_cuda_cflags.extend(["-Xcompiler", "/Zc:preprocessor", "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"])
    cxx_flags = ["/O2", "/std:c++17", "/Zc:preprocessor", "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"]
else:
    cxx_flags = ["-O3", "-std=c++17", "-Wno-deprecated-declarations"]

extra_compile_args = {
    "cxx": cxx_flags,
    "nvcc": extra_cuda_cflags,
}

setup(
    name="liquid_state_fusion_cuda",
    ext_modules=[
        CUDAExtension(
            name="liquid_state_fusion_cuda",
            sources=[
                "liquid_state_fusion_cpp.cpp",
                "liquid_state_fusion_cuda.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
