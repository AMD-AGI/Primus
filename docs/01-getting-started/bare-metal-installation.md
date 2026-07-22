# Bare-metal installation: build the Primus training stack from source (no Docker)

This guide explains how to build the **full Primus training software stack directly on a host machine**, without using the AMD published training Docker image. It is intended for users who, for policy or operational reasons, cannot run containers and need to reproduce the same environment on bare metal.

It is derived from the official training `Dockerfile` and installs the same components and versions. Wherever possible, everything is installed **inside a Python virtual environment and without `sudo`**. The only steps that require root are a small set of OS-level system libraries (installed with `apt`), and a couple of optional networking packages used for multi-node training.

> **Important**: This is a long, build-heavy process. A full from-source build (Flash Attention, TransformerEngine, aiter, Primus-Turbo, FBGEMM, rocSHMEM, etc.) can take **several hours** and needs a machine with many CPU cores, plenty of RAM, and tens of GB of free disk. The Docker image remains the recommended and best-supported path. Use this guide only when containers are not an option.

---

## Quick path: automated install scripts

If you just want the environment built for you, use the helper scripts in
[tools/installation/](https://github.com/AMD-AGI/Primus/tree/main/tools/installation). They automate everything in
Sections 3 (Python venv) of this guide — venv creation, the ROCm/PyTorch
wheels, every source-built kernel library, and Primus itself — and provide a
single `env.sh` to activate the environment before each job. Read the rest of
this document if you want to understand or customize what they do, or if you
need the multi-node networking stack (Section 4), which the scripts do not
build.

There are two files:

- **env.sh** — defines the install location and exports every environment
  variable the build and runtime need (ROCm paths, TransformerEngine/`NVTE_*`
  flags, cache locations, etc.). Source it both during the build and every time
  you use the environment.
- `**setup.sh**` — runs the install in re-runnable **stages**. It sources
`env.sh` automatically.

### Choose where it installs (important)

Everything lives under `PRIMUS_BASE` (venv, kept checkouts), with transient
build sources on `SRC_DIR` (defaults to local `/tmp` for fast compile I/O). The
**default `PRIMUS_BASE` is site-specific** and may not exist on your host, so
set it to a directory you can write to that has tens of GB free:

```bash
export PRIMUS_BASE=/path/to/big/disk/primus-env   # venv + checkouts (persistent)
export SRC_DIR=/tmp/primus-build                  # transient build sources (optional override)
```

The scripts **auto-detect your GPU architecture** (`env.sh` reads `rocminfo`, or
falls back to the kernel KFD sysfs `gfx_target_version` when no ROCm is
installed yet), and install the matching device wheels — `gfx942` (MI300X/MI325X),
`gfx950` (MI350X/MI355X), or both. To force a target (e.g. to build a portable
environment for both archs), export it before running:

```bash
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
```

### Build the environment

```bash
cd tools/installation

bash setup.sh            # run all default stages, in order
bash setup.sh --list     # list available stages
bash setup.sh te         # re-run a single stage (e.g. rebuild TransformerEngine)
bash setup.sh venv torch # run a subset of stages
```

Stages are idempotent and re-runnable, so if a build fails you can fix the
cause and re-run just that stage. On failure the script now stops immediately
and prints which stage failed.

Default stages (single-node training path):

```
venv → torch → flash_attn → te → torchtune → torchao → pydeps
     → grouped_gemm → causal_conv1d → mamba → primus → aiter → turbo
     → boto → manifest
```

Optional: `torchrec` (DLRM / recommendation stack).

### Use the environment for a training job

```bash
# Use the SAME PRIMUS_BASE you built with
export PRIMUS_BASE=/path/to/big/disk/primus-env
source tools/installation/env.sh    # activates the venv + sets all ROCm/NVTE vars

python -c "import torch; print('gpu:', torch.cuda.is_available())"

# Primus is checked out under $WORKSPACE_DIR
cd "$WORKSPACE_DIR/Primus"
./primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

### What the scripts do NOT do

- **System (`apt`) packages** (Section 2): skipped — they need root. A C++compiler (`g++`/`make`),` git`, and the build basics must already be present.
- **Multi-node networking** (Section 4: UCX, OpenMPI, rocSHMEM, AINIC): not
built. Single-node training works without them; follow Section 4 manually if
you need distributed-over-RDMA.
- **FBGEMM / Flux / DLRM**: not built (`torchrec` is an optional stage).

The scripts also bake in a few host-specific fixes beyond the Dockerfile (mamba
installed via `pip` rather than `setup.py`, `apache-tvm-ffi` and `z3-solver`
pinned for `mamba_ssm`, and the Megatron `helpers_cpp` extension compiled). See
`[tools/installation/README.md](https://github.com/AMD-AGI/Primus/blob/main/tools/installation/README.md)` for the full
rationale.

---

## 0. The key idea: ROCm comes from pip, not from a system install

The most important thing to understand is that this stack **does not require a system-wide ROCm installation**. ROCm is delivered as Python wheels (AMD "TheRock" multi-arch nightly wheels):

- `rocm-sdk-devel`, `rocm-sdk-device-gfx942`, `rocm-sdk-device-gfx950` provide the ROCm toolchain (HIP, hipBLASLt, compilers, headers, libraries) **inside the virtual environment**.
- `torch`, `torchvision`, `torchaudio` and the `amd-*-device-gfx`* packages provide a ROCm-enabled PyTorch built against those wheels.

This means almost the entire stack can be installed **without root** into a venv. The only host-level requirement from the system administrator is:

- The **AMD GPU kernel driver (amdgpu / ROCm KMD)** must already be installed and loaded on the host (`/dev/kfd` and `/dev/dri` must exist, and the user must have permission to access them — typically by being in the `video` and `render` groups).
- A small set of **build/runtime system libraries** (see Section 2).

You do **not** need to install the full ROCm user-space stack system-wide.

---

## 1. Required software stack for distributed LLM training

The complete environment is composed of the following layers:


| Layer                   | Component                                                                                   | Source                  | Needs root?                      |
| ----------------------- | ------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------- |
| Kernel / hardware       | AMD GPU driver (amdgpu KMD), GPU device access                                              | OS / admin              | Yes (one-time, by admin)         |
| OS libraries            | Build toolchain + runtime libs (`g++`, `git`, RDMA, hwloc, etc.)                            | `apt`                   | Yes (one-time)                   |
| ROCm user-space         | `rocm-sdk-devel` + device packages                                                          | pip (TheRock wheels)    | No (venv)                        |
| Deep learning framework | PyTorch ROCm (`torch`, `torchvision`, `torchaudio`, `apex`)                                 | pip (TheRock wheels)    | No (venv)                        |
| Accelerated kernels     | Flash Attention, TransformerEngine, aiter, grouped_gemm, Primus-Turbo, causal-conv1d, mamba | build from source       | No (venv)                        |
| Training frameworks     | torchtune, torchao, torchrec, FBGEMM                                                        | build from source / pip | No (venv)                        |
| Multi-node comms        | UCX, OpenMPI, rocSHMEM, AMD AINIC (libionic)                                                | build from source / apt | Mostly no (AINIC libs need root) |
| Primus                  | Primus + submodules (Megatron-LM, TorchTitan, etc.)                                         | git + pip               | No (venv)                        |
| Python deps             | datasets, transformers, accelerate, trl, wandb, etc.                                        | pip                     | No (venv)                        |


For a **distributed (multi-node) training job** specifically, beyond PyTorch and ROCm you additionally need:

- **RCCL** (AMD's collective library) — provided by the ROCm SDK wheels.
- **UCX + OpenMPI** — point-to-point transport and the MPI launcher.
- **rocSHMEM** — GPU-initiated communication (optional but used by Primus-Turbo for advanced overlap).
- **AMD AINIC / RDMA stack** (`libibverbs`, `rdma-core`, `libionic`) — for high-performance networking on AMD Pensando NICs.
- Correct GPU/NIC device permissions and (often) hugepages / `ulimit -l unlimited` for RDMA.

---

## 2. System packages (require `sudo` / administrator, one-time)

These are OS-level libraries needed to *build* the rest of the stack and to run RDMA networking. They must be installed by someone with root, but this is a **one-time** action; everything afterward is done unprivileged in a venv. On a shared/managed host, ask your administrator to install them once.

> If you genuinely cannot get root at all, these packages must already be present on the host. There is no supported way to install system `.deb` packages without root. The remainder of the guide (Sections 3+) then runs entirely without root.

### 2.1 Build toolchain and core libraries

```bash
sudo apt update
sudo apt install -y \
    gfortran git git-lfs ninja-build g++ pkg-config xxd patchelf \
    automake libtool autoconf flex ccache \
    python3-venv python3-dev python3-pip python-is-python3 \
    libegl1-mesa-dev liblzma-dev libdw1 libdrm-dev libz3-dev \
    wget xz-utils ffmpeg numactl pciutils
```

### 2.2 RDMA / networking libraries (needed for multi-node training)

```bash
sudo apt install -y \
    rdma-core libibverbs-dev ibverbs-utils infiniband-diags \
    ethtool kmod dpkg-dev jq \
    libevent-dev libhwloc-dev libmunge-dev \
    software-properties-common
```

### 2.3 AMD AINIC library (optional, for AMD Pensando NICs)

This pulls a vendor `.deb` from the AMD radeon repository. Skip it if you are not using AMD AINIC networking.

```bash
# Pin to the version used by the reference image
AINIC_BUNDLE_VERSION="1.117.5-a-77"

sudo add-apt-repository -y \
  "deb https://repo.radeon.com/amdainic/pensando/ubuntu/${AINIC_BUNDLE_VERSION} noble main"
sudo apt update --allow-insecure-repositories
sudo apt install -y --allow-unauthenticated libionic-dev
```

---

## 3. Build the Python environment (no `sudo` from here on)

Everything below runs as a regular user inside a virtual environment.

### 3.1 Create and activate the virtual environment

```bash
# Pick a stable location, e.g. ~/primus-env
python -m venv ~/primus-env
source ~/primus-env/bin/activate

# Build/runtime knobs (match the Dockerfile)
export MAX_JOBS=128                              # lower this if you have fewer CPU cores / less RAM
export PYTORCH_ROCM_ARCH="gfx942;gfx950"         # MI300/MI325 = gfx942, MI350/MI355 = gfx950
export ROCM_AMDGPU_TARGETS="gfx942,gfx950"
```

> Set `PYTORCH_ROCM_ARCH` to only the architecture(s) you actually have to speed up source builds (e.g. `"gfx942"` for MI300X-only).

### 3.2 Bootstrap build tooling

```bash
pip install --upgrade pip
pip install \
    pybind11 typeguard \
    wheel==0.45.1 \
    cmake==3.31.6 \
    ninja==1.11.1.3 \
    packaging==25.0 \
    setuptools==75.1.0
```

### 3.3 Workaround environment variables

```bash
# Avoids HSA_STATUS_ERROR_OUT_OF_RESOURCES on some configurations
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1
```

### 3.4 Install ROCm + PyTorch from the TheRock multi-arch wheels

This is the step that replaces a system ROCm install. The versions below match the reference image; check [https://rocm.nightlies.amd.com/whl-multi-arch/torch/](https://rocm.nightlies.amd.com/whl-multi-arch/torch/) for the current pin.

```bash
# Base scientific / build deps first
pip install \
    cxxfilt==0.3.0 tqdm==4.67.3 pyyaml==6.0.3 pytest==9.0.3 \
    matplotlib==3.10.9 pandas==2.3.3 py-cpuinfo==9.0.0 build==1.5.0

# ROCm SDK + ROCm-enabled PyTorch
PYTORCH_VERSION=2.12.0+rocm7.14.0a20260608

python -m pip uninstall -y torch
python -m pip install \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch \
    --pre \
    torch==${PYTORCH_VERSION} \
    amd-torch-device-gfx942==${PYTORCH_VERSION} \
    amd-torch-device-gfx950==${PYTORCH_VERSION} \
    rocm-sdk-devel \
    rocm-sdk-device-gfx942 \
    rocm-sdk-device-gfx950 \
    torchaudio \
    torchvision \
    amd-torchvision-device-gfx942 \
    amd-torchvision-device-gfx950 \
    apex
```

> Install only the `*-gfx942` **or** `*-gfx950` device packages matching your hardware if you want a smaller install.

### 3.5 Initialize the ROCm SDK and export ROCm paths

`rocm-sdk init` materializes the ROCm toolchain inside the venv. The environment variables below point the rest of the build at that in-venv ROCm. **These must be set every time you use the environment** — Section 5 shows how to make them persistent.

```bash
rocm-sdk init

# ROCm lives inside the venv, under site-packages
export ROCM_PATH=$(python -c 'import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))')
# Primus uses ROCM_HOME; set both to be safe
export ROCM_HOME=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
export PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"
export LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"
export CPATH=$HIP_INCLUDE_PATH
export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"
```

> The Dockerfile hardcodes `ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel`. The `python -c ...` form above derives it automatically so it works with any venv location or Python version.

Quick check before continuing:

```bash
hipcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.6 Build the accelerated kernel libraries from source

These are compiled against your in-venv ROCm and selected `PYTORCH_ROCM_ARCH`. Do these in a scratch build directory.

```bash
export GPU_ARCHS="${PYTORCH_ROCM_ARCH}"
mkdir -p ~/primus-build && cd ~/primus-build
```

**Flash Attention (ROCm fork):**

```bash
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout 6387433156558135a998d5568a9d74c1778666d8
python setup.py install
cd .. && rm -rf flash-attention
```

**TransformerEngine (ROCm fork):**

```bash
export NVTE_USE_HIPBLASLT=1
export NVTE_FRAMEWORK=pytorch
export NVTE_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_USE_ROCM=1

git clone --recursive https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine
git checkout e6ede467a49cfda1859b145e045109e2f330bccc
git submodule update --init --recursive
pip install psutil
MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation .
cd ..
```

**grouped_gemm:**

```bash
git clone https://github.com/caaatch22/grouped_gemm.git
cd grouped_gemm
git checkout rocm
git submodule update --init --recursive
pip install --no-build-isolation .
cd .. && rm -rf grouped_gemm
```

**causal-conv1d:**

```bash
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export HIP_ARCHITECTURES=gfx942,gfx950

git clone https://github.com/Dao-AILab/causal-conv1d causal-conv1d
cd causal-conv1d
git checkout e940ead2fd962c56854455017541384909ca669f
pip install --no-build-isolation .
cd .. && rm -rf causal-conv1d
```

**mamba:**

```bash
git clone --branch enable-primus-hybrid-models https://github.com/AndreasKaratzas/mamba.git
cd mamba
pip install "apache-tvm-ffi==0.1.6"
pip install --no-build-isolation .
cd ..
```

**aiter (AMD inference/training kernels):**

```bash
pip uninstall aiter amd-aiter -y
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
git checkout 0f3c58e6edb6754940bcf9fd5f09ccb6f389f52e
git submodule update --init --recursive
PREBUILD_KERNELS=3 pip install --no-cache-dir --use-pep517 .
cd ..
```

**Primus-Turbo:**

```bash
export HCC_AMDGPU_TARGET="gfx942,gfx950"

git clone https://github.com/AMD-AGI/Primus-Turbo.git --recursive
cd Primus-Turbo
git checkout 3974fc246be594d989156dd83e91da618274b0c8
git submodule update --init --recursive
pip install -r requirements.txt
pip install --no-build-isolation . -v
cd ..
```

### 3.7 Install training-framework Python dependencies

```bash
pip install \
    datasets==3.6.0 av==16.0.1 transformers==4.55.0 optree==0.18.0 sympy \
    accelerate==1.9.0 trl==0.21.0 tensorboard==2.20.0 peft scipy einops \
    flask-restful nltk pytest pytest-cov pytest_mock pytest-csv \
    pytest-random-order sentencepiece wrapt \
    zarr==2.18.7 numcodecs==0.12.1 xarray wandb tensorstore==0.1.45 \
    pybind11 tiktoken pynvml "huggingface_hub[cli]"

python3 -m nltk.downloader punkt_tab
```

**torchtune (with the Primus patch):**

```bash
cd ~/primus-build
git clone https://github.com/pytorch/torchtune.git
cd torchtune
git checkout b4c98ac2a37f0397d64c22579aed415ce7264db6
# Primus patch: disable grouped_mm in the MoE path
sed -i 's/use_grouped_mm = True/use_grouped_mm = False/g' torchtune/modules/moe/utils.py
pip install .
cd .. && rm -rf torchtune
```

**torchao (with the Primus patches):**

```bash
git clone https://github.com/pytorch/ao.git
cd ao
git checkout e9c7bead90b840b280f97374308255957108ce47
# Primus patches for fp8 + ROCm swizzle
sed -i 's/pad_inner_dim: bool = False/pad_inner_dim: bool = True/g' torchao/float8/config.py
sed -i 's/if defined(HIPBLASLT_VEC_EXT)/if false/g' torchao/csrc/rocm/swizzle/swizzle.cpp
pip install --no-build-isolation .
cd .. && rm -rf ao
```

**torchrec + FBGEMM (for DLRM / recommendation workloads, optional):**

```bash
# torchrec + helpers
pip install --no-deps torchrec
pip install tensordict iopath torchmetrics==1.0.3 \
    git+https://github.com/mlperf/logging.git \
    --extra-index-url https://rocm.nightlies.amd.com/whl-multi-arch

# FBGEMM (GPU)
export BUILD_ROCM_VERSION='7.2'
export FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1
export ROCM_VERSION=72000
export HIPBLAS_V2=1

git clone https://github.com/pytorch/FBGEMM.git
cd FBGEMM
git checkout fbda30f767186b7cc6f5663fd17c268a5d853c3e
cd fbgemm_gpu
git clean -dfx
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
pip install setuptools==75.1.0
python setup.py --build-variant rocm --build-target default --package_channel nightly \
    -DHIP_ROOT_DIR=$ROCM_PATH \
    -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
    -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA" \
    build develop 2>&1 | tee build.log
cd ../..

# AWS SDK (used by some data pipelines)
pip install boto3==1.35.42 botocore==1.35.99
```

> The Flux (`AMDiffusionBenchmark`) and DLRM (`DLRMBenchmark`) repos in the Dockerfile are benchmark workloads, not core training dependencies. Clone them only if you need those specific benchmarks.

### 3.8 Install Primus and its submodules

```bash
cd ~/primus-build
# Required to resolve a post-v26.2 attention backend issue
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout 9a1547cd5885c4de6ad0935a5d59c08303dd0674
git submodule update --init --recursive
pip install -r requirements.txt
```

If you already have a local Primus checkout (e.g. this repository), you can instead just run `pip install -r requirements.txt` from its root and skip the clone.

---

## 4. Multi-node communication stack (UCX, OpenMPI, rocSHMEM)

These are only needed for **multi-node distributed** training. They build from source and install into user-writable prefixes (no root needed, except the AINIC `.deb` already handled in Section 2.3).

### 4.1 UCX

```bash
cd ~/primus-build
UCX_VERSION="1.18.0"
wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz
mkdir -p ucx-${UCX_VERSION}
tar -zxf ucx-${UCX_VERSION}.tar.gz -C ucx-${UCX_VERSION} --strip-components=1
cd ucx-${UCX_VERSION}
mkdir build && cd build
../configure --prefix=$HOME/primus-build/ucx-${UCX_VERSION}/install --with-rocm=${ROCM_PATH}
make -j 16 && make install
cd ../..

export UCX_INSTALL_DIR=$HOME/primus-build/ucx-${UCX_VERSION}/install
```

### 4.2 OpenMPI

```bash
MPI_VERSION="4.1.6"
wget https://download.open-mpi.org/release/open-mpi/v$(echo "${MPI_VERSION}" | cut -d. -f1-2)/openmpi-${MPI_VERSION}.tar.gz
mkdir -p ompi-${MPI_VERSION}
tar -zxf openmpi-${MPI_VERSION}.tar.gz -C ompi-${MPI_VERSION} --strip-components=1
cd ompi-${MPI_VERSION}
mkdir build && cd build
# Install to a user-writable prefix instead of /opt to avoid sudo
../configure --prefix=$HOME/primus-build/openmpi --with-ucx=${UCX_INSTALL_DIR} \
    --disable-oshmem --disable-mpi-fortran
make -j 16 && make install
cd ../..

export PATH="$HOME/primus-build/openmpi/bin:${PATH}"
export LD_LIBRARY_PATH="$HOME/primus-build/openmpi/lib:${LD_LIBRARY_PATH}"
```

> The Dockerfile installs OpenMPI to `/opt/openmpi` (needs root). The `$HOME/primus-build/openmpi` prefix above keeps it unprivileged. Use `/opt/openmpi` only if you have root and want to match the image exactly.

### 4.3 rocSHMEM

rocSHMEM depends on UCX and OpenMPI, so build it last.

```bash
ROCSHMEM_HOME=$HOME/primus-build/rocshmem
export UCX_HOME=${UCX_INSTALL_DIR}
export MPI_HOME=$HOME/primus-build/openmpi

cd ~/primus-build
git clone https://github.com/ROCm/rocSHMEM.git
cd rocSHMEM
git checkout 17ff985c026f9f97f85068647e863ab541dd5645
mkdir build && cd build
MPI_ROOT=${MPI_HOME} \
UCX_ROOT=${UCX_HOME} \
INSTALL_PREFIX=${ROCSHMEM_HOME} \
../scripts/build_configs/gda \
    -DROCM_PATH=${ROCM_PATH} \
    -DGDA_IONIC=ON \
    -DGDA_MLX5=ON \
    -DGDA_BNXT=ON \
    -DUSE_IPC=ON
cd ~/primus-build
export ROCSHMEM_HOME
```

> `GDA_IONIC=ON` targets AMD AINIC, `GDA_MLX5=ON` targets Mellanox/NVIDIA NICs, `GDA_BNXT=ON` targets Broadcom NICs. You can turn off the ones that don't match your hardware.

---

## 5. Make the environment reproducible (activation script)

Many of the variables above (especially the ROCm paths) must be present in **every** shell that runs training. Append them to the venv's activation script so they're set whenever you `source ~/primus-env/bin/activate`:

```bash
cat >> ~/primus-env/bin/activate <<'EOF'

# ---- Primus host environment ----
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
export ROCM_AMDGPU_TARGETS="gfx942,gfx950"
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1

export ROCM_PATH=$(python -c 'import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))')
export ROCM_HOME=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
export PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$HOME/primus-build/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$HOME/primus-build/openmpi/lib"
export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"
export CPATH=$HIP_INCLUDE_PATH
export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"

# TransformerEngine / attention backend selection used by Primus
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

# Runtime: let aiter detect the local GPU architecture.
# NOTE: this is the runtime value. If you rebuild any source kernel later,
#       re-export GPU_ARCHS="$PYTORCH_ROCM_ARCH" first, then set it back to native.
export GPU_ARCHS=native

# Multi-node comms (only if built)
export UCX_HOME=$HOME/primus-build/ucx-1.18.0/install
export MPI_HOME=$HOME/primus-build/openmpi
export ROCSHMEM_HOME=$HOME/primus-build/rocshmem
# ---- end Primus host environment ----
EOF
```

---

## 6. Verify the installation

```bash
source ~/primus-env/bin/activate

# GPUs visible to ROCm?
rocm-smi || ls -l /dev/kfd /dev/dri

# PyTorch sees the GPUs?
python -c "import torch; print('torch', torch.__version__); \
print('gpu available:', torch.cuda.is_available()); \
print('device count:', torch.cuda.device_count()); \
print('device 0:', torch.cuda.get_device_name(0))"

# Key kernel libs import cleanly?
python -c "import transformer_engine; import flash_attn; import aiter; print('TE/FA/aiter OK')"

# Run a Primus benchmark / training directly (no container)
cd ~/primus-build/Primus   # or your Primus checkout
./primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096
./primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

Use `primus-cli direct` (not `container`) since you are running on bare metal with everything installed in your environment.

---

## 7. Other important considerations

- **GPU device access without root**: the user running training must be able to read/write `/dev/kfd` and `/dev/dri/*`. This usually means membership in the `video` and `render` groups (`sudo usermod -aG video,render $USER`, then re-login). This is a one-time admin action.
- **Hugging Face access**: if your config downloads weights or tokenizers from the Hub, export your token: `export HF_TOKEN=hf_xxx` (and/or `huggingface-cli login`). The token is needed for gated models like Llama.
- **RDMA / multi-node limits**: high-performance networking typically requires locked-memory limits raised (`ulimit -l unlimited`) and possibly hugepages. These are configured in `/etc/security/limits.conf` and need admin help. Verify NICs with `ibv_devinfo` and `ibstat`.
- **Disk and time**: source builds of TransformerEngine, aiter, FBGEMM and rocSHMEM are large and slow. Reserve plenty of disk in `~/primus-build` and expect a multi-hour first build. Lower `MAX_JOBS` if the build runs out of memory.
- `**ccache`**: already installed in Section 2.1; it dramatically speeds up rebuilds. No extra config needed for a basic speedup.
- **Architecture pinning**: building for only your actual GPU arch (e.g. `gfx942` for MI300X/MI325X, `gfx950` for MI350X/MI355X) significantly reduces build time and binary size versus building both.
- **Version drift**: the TheRock nightly wheels and the source-build commits above are pinned to a specific release. If you change one, you may need to update others to keep them compatible. The Docker image is the authoritative, tested combination — match its `Dockerfile` ARGs when in doubt.
- **Automated scripts**: the manual steps in Section 3 are automated by `[tools/installation/](https://github.com/AMD-AGI/Primus/tree/main/tools/installation)` (see the *Quick path* section above). The scripts cover the single-node venv build; the multi-node networking stack in Section 4 is still manual. Treat the reference `Dockerfile` as the source of truth for exact versions.

---

## 8. Quick reference: minimal vs. full install

If you only need **single-node Megatron/TorchTitan LLM pretraining**, you can skip several optional components:


| Component                                               | Needed for                           |
| ------------------------------------------------------- | ------------------------------------ |
| Flash Attention, TransformerEngine, aiter, Primus-Turbo | Core LLM training (install these)    |
| grouped_gemm                                            | MoE models                           |
| causal-conv1d, mamba                                    | Hybrid / Mamba-family models         |
| torchtune, torchao                                      | Post-training (SFT/LoRA), fp8        |
| torchrec, FBGEMM, DLRM                                  | Recommendation (DLRM) workloads only |
| Flux / AMDiffusionBenchmark                             | Diffusion benchmark only             |
| UCX, OpenMPI, rocSHMEM, AINIC                           | Multi-node distributed training      |


Install the core rows first, validate with Section 6, then add the optional components as your workload requires.
