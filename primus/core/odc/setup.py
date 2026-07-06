"""Setup script for ODC (ROCm).

Notes
-----
* The symmetric-memory / RDMA backend is ``amd_mori`` (an OpenSHMEM-style
  symmetric-memory and RDMA library for AMD GPUs), declared as an install
  dependency below.
* The rocSHMEM host bindings (single-node XGMI IPC and multi-node GPU-direct
  GDA) are built from source by ``build_rocshmem_backend.sh``; this package is
  pure Python and ships no compiled extension.
"""

from pathlib import Path

from setuptools import find_packages, setup

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
