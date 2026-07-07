"""Setup script for ODC (ROCm).

Notes
-----
* The default symmetric-memory / RDMA backend is ``amd_mori`` (an OpenSHMEM-style
  symmetric-memory and RDMA library for AMD GPUs), declared as an install
  dependency below.
* The rocSHMEM backend (single-node XGMI IPC host API and multi-node GPU-direct
  GDA) is no longer built in-tree. It is provided by Primus-Turbo as the
  ``primus_turbo.pytorch._C.odc_rocshmem_host`` / ``odc_rocshmem_gda`` pybind
  submodules and consumed by ``odc/primitives/_rocshmem_backend.py``. This
  package remains pure Python and ships no compiled extension.

  TODO(primus-turbo): once the Primus-Turbo PR that adds the ODC rocSHMEM ops is
  merged, pin the exact merge commit here (and wherever Primus-Turbo is fetched)
  so the rocSHMEM backend is reproducible:
      PRIMUS_TURBO_COMMIT = "<fill-with-merge-commit-hash-after-turbo-PR-merges>"
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Default communication backend dependency (ROCm). The optional rocSHMEM backend
# is provided by Primus-Turbo (see the module docstring / PRIMUS_TURBO_COMMIT
# TODO) rather than a declared version here, since it is pinned by commit.
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
