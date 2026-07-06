"""
Setup script for ODC (ROCm) using torch cpp_extension for the HIP extension.

Notes
-----
* The symmetric-memory / RDMA backend is ``amd_mori`` (an OpenSHMEM-style
  symmetric-memory and RDMA library for AMD GPUs), declared as an install
  dependency below.
* The ``tensor_ipc`` extension is built from the HIP source
  ``csrc/tensor_ipc/tensor_ipc.hip`` (uses hipMemGetAddressRange,
  hipIpcGetMemHandle, hipIpcOpenMemHandle — all provided by the PyTorch ROCm
  build, so no extra ``libraries`` are required).
"""

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the tensor_ipc HIP extension. On ROCm, ``CUDAExtension`` compiles
# ``.hip`` sources directly with hipcc.
tensor_ipc_extension = CUDAExtension(
    name="odc.primitives.tensor_ipc",
    sources=[
        "csrc/tensor_ipc/binding.cpp",
        "csrc/tensor_ipc/tensor_ipc.hip",
    ],
    libraries=[],
)


# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Communication backend dependency (ROCm).
backend_requires = ["amd_mori>=1.1.1"]

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
