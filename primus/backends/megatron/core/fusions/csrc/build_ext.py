"""Build script for hipBLASLt FP8 grouped GEMM extension."""
import os
import torch
from torch.utils.cpp_extension import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")

ext = load(
    name="hipblaslt_fp8_gg",
    sources=[os.path.join(SCRIPT_DIR, "hipblaslt_fp8_grouped_gemm.cpp")],
    extra_include_paths=[
        os.path.join(ROCM_PATH, "include"),
        os.path.join(ROCM_PATH, "include", "hipblaslt"),
    ],
    extra_ldflags=[
        f"-L{ROCM_PATH}/lib",
        "-lhipblaslt",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
    with_cuda=True,
)

print("Extension loaded successfully!")
print(f"  fp8_grouped_gemm_fwd: {ext.fp8_grouped_gemm_fwd}")
print(f"  fp8_grouped_gemm_dA: {ext.fp8_grouped_gemm_dA}")
