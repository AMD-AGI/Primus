"""
Setup script for ODC package using torch cpp_extension for CUDA/HIP extensions.

ROCm adaptation
---------------
This is the ROCm-compatible fork of ODC's setup.py. The two changes versus
the upstream sail-sg/odc are:

1. On ROCm platforms (``torch.version.hip is not None``):
   * Depend on ``amd_mori>=1.1.1`` instead of ``nvidia-nvshmem-cu12`` /
     ``nvshmem4py-cu12``. MORI provides an OpenSHMEM-style symmetric memory
     and RDMA library that is API-compatible with NVSHMEM at the Triton
     bitcode level, and is the ROCm-native equivalent used by ODC's
     replacement on AMD GPUs.
   * Drop ``libraries=["cuda"]`` (no ``libcuda.so`` on ROCm). The HIP runtime
     symbols that ``tensor_ipc.cu`` needs (hipMemGetAddressRange,
     hipIpcGetMemHandle, hipIpcOpenMemHandle) are already provided by the
     PyTorch ROCm build.

2. ``CUDAExtension`` is platform-agnostic: PyTorch's ``CUDAExtension`` runs
   the hipify pass on the CUDA sources transparently on ROCm. The
   PyTorch-maintained CUDA-to-HIP mapping covers every symbol used by
   ``csrc/tensor_ipc/tensor_ipc.cu`` (verified empirically on PyTorch
   2.10a + ROCm 7.2 / MI300X).
"""

from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Platform detection. On ROCm, torch.version.hip is a string like "7.2.0";
# on CUDA it is None.
IS_ROCM = torch.version.hip is not None

# Define the tensor_ipc CUDA/HIP extension. Sources are unchanged from
# upstream; hipify takes care of the symbol translation on ROCm.
tensor_ipc_extension = CUDAExtension(
    name="odc.primitives.tensor_ipc",
    sources=[
        "csrc/tensor_ipc/binding.cpp",
        "csrc/tensor_ipc/tensor_ipc.cu",
    ],
    libraries=[] if IS_ROCM else ["cuda"],
)


# Note: nvshmem bitcode files are built using CMake, not during pip install
# The BuildNVSHMEMBitcode class has been removed to avoid duplicate builds

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Communication backend dependency per platform.
if IS_ROCM:
    backend_requires = ["amd_mori>=1.1.1"]
else:
    backend_requires = ["nvidia-nvshmem-cu12", "nvshmem4py-cu12"]

setup(
    name="odc",
    description="On-Demand Communication (ROCm adaptation)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sail-sg/odc",
    packages=find_packages(),
    ext_modules=[tensor_ipc_extension],
    cmdclass={
        "build_ext": BuildExtension,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "triton>=3.4.0",
    ]
    + backend_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
