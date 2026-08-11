# Bare-metal installation: build the Primus training stack from source (no Docker)

This guide explains how to build the **full Primus training software stack directly on a host machine**, without using the AMD published training Docker image. It is intended for users who, for policy or operational reasons, cannot run containers and need to reproduce the same environment on bare metal.

It is derived from the official training Dockerfile — currently
[`Dockerfile.primus-v26.5`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.5) — and installs the same components and versions. Wherever possible, everything is installed **inside a Python virtual environment and without `sudo`**. The only steps that require root are a small set of OS-level system libraries (installed with `apt`), and a couple of optional networking packages used for multi-node training.

> **Important**: This is a long, build-heavy process. A full from-source build (Flash Attention, aiter, Primus-Turbo, and on older hosts TransformerEngine) can take **several hours** and needs a machine with many CPU cores, plenty of RAM, and tens of GB of free disk. The Docker image remains the recommended and best-supported path. Use this guide only when containers are not an option.

> **Python 3.12 is required.** The pinned torch nightly publishes a **cp312 Linux wheel and nothing else**, so a Python 3.10 environment (the default on Ubuntu 22.04) cannot install it. The automated scripts provision a private CPython 3.12 for you without root; if you build manually you must supply one yourself. See [Section 4.1](#41-create-the-virtual-environment-python-312).

> **Using the JAX MaxText backend instead?** This guide builds the PyTorch stack (Megatron-LM / TorchTitan). For the JAX / MaxText backend, follow the leaner [JAX bare-metal installation guide](bare-metal-installation-jax.md) instead.

---

## 0. The key idea: ROCm comes from pip, not from a system install

The most important thing to understand is that this stack **does not require a system-wide ROCm installation**. ROCm is delivered as Python wheels (AMD "TheRock" multi-arch nightly wheels):

- `rocm-sdk-devel`, `rocm-sdk-device-gfx942`, `rocm-sdk-device-gfx950` provide the ROCm toolchain (HIP, hipBLASLt, compilers, headers, libraries) **inside the virtual environment**.
- `torch`, `torchvision`, `torchaudio` and the `amd-*-device-gfx*` packages provide a ROCm-enabled PyTorch built against those wheels.

This means almost the entire stack can be installed **without root** into a venv. The only host-level requirements from the system administrator are:

- The **AMD GPU kernel driver (amdgpu / ROCm KMD)** must already be installed and loaded on the host (`/dev/kfd` and `/dev/dri` must exist, and the user must have permission to access them — typically by being in the `video` and `render` groups).
- A small set of **build/runtime system libraries** (see [Section 3](#3-system-packages-need-root-one-time)).

You do **not** need to install the full ROCm user-space stack system-wide.

---

## 1. Recommended path: the automated scripts

The helper scripts in [tools/installation/](https://github.com/AMD-AGI/Primus/tree/main/tools/installation) build the entire single-node environment for you: the Python 3.12 interpreter, the venv, the ROCm/PyTorch wheels, every source-built kernel library, and Primus itself. They are the maintained path and are kept in sync with the reference Dockerfile — prefer them over the manual reference in [Section 4](#4-manual-build-reference).

There are two files:

- **`env.sh`** — defines the install location and exports every environment variable the build and runtime need (ROCm paths, `NVTE_*` flags, cache locations). Source it both during the build and every time you use the environment.
- **`setup.sh`** — runs the install in re-runnable **stages**. It sources `env.sh` automatically.

### 1.1 Before you start

- System packages from [Section 3](#3-system-packages-need-root-one-time) must already be present (a C++ compiler, `git`, `make` and the build basics). These need root, so the scripts do not install them.
- The GPU driver must be loaded and your user must be able to access `/dev/kfd` and `/dev/dri/*`.
- Python 3.12 is handled for you: if no suitable interpreter is found, `setup.sh` fetches a standalone CPython 3.12 with [`uv`](https://docs.astral.sh/uv/). No root, no `apt`.
- **`uv` itself is optional to pre-install.** If it is missing, `setup.sh` downloads it into `$PRIMUS_BASE/bin` from `astral.sh`. Install it yourself if that download is blocked, or if you prefer not to pipe a script into a shell — see [Installing `uv`](#installing-uv).

### 1.2 Choose where it installs (required)

`PRIMUS_BASE` **has no default and must be exported.** The right location is site-specific, so the scripts refuse to guess and stop with an error if it is unset. Point it at a directory you can write to with tens of GB free; it holds the venv, the provisioned interpreter, and the kept checkouts.

```bash
export PRIMUS_BASE="$HOME/envs/primus-env"   # venv + interpreter + checkouts (persistent)
export SRC_DIR=/tmp/primus-build             # transient build sources (optional override)
```

The scripts detect your GPU architecture automatically and build only for it — `gfx942` (MI300X/MI325X) or `gfx950` (MI350X/MI355X) — which keeps the build as short as possible. Override it with `export PYTORCH_ROCM_ARCH="gfx942;gfx950"` only if you need a different target, such as one environment shared across both.

### 1.3 Build the environment

```bash
cd tools/installation

bash setup.sh            # run all default stages, in order
bash setup.sh --list     # list available stages (works without PRIMUS_BASE)
bash setup.sh te         # re-run a single stage (e.g. reinstall TransformerEngine)
bash setup.sh venv torch # run a subset of stages
```

Expect hours rather than minutes. Running it detached avoids losing the build to a dropped connection:

```bash
nohup bash setup.sh > ~/primus-setup.log 2>&1 &
tail -f ~/primus-setup.log
```

Stages are idempotent, so if a build fails you can fix the cause and re-run just that stage — exporting the same `PRIMUS_BASE` again, since that is how it finds the venv. On failure the script stops immediately and prints which stage failed.

Default stages (single-node training path):

```
venv → torch → flash_attn → te → torchtune → torchao → pydeps
     → grouped_gemm → causal_conv1d → mamba → primus → aiter → turbo
     → boto → cleanup → manifest
```

Optional: `torchrec` (DLRM / recommendation stack).

Order matters if you cherry-pick: `te` needs `torch` and `flash_attn` to have run first, because the TransformerEngine package index serves only TE packages and cannot supply anything else.

### 1.4 Use the environment for a training job

```bash
# Use the SAME PRIMUS_BASE you built with
export PRIMUS_BASE="$HOME/envs/primus-env"
source tools/installation/env.sh    # activates the venv + sets all ROCm/NVTE vars

python -c "import torch; print('gpu:', torch.cuda.is_available())"

# Primus is checked out under $WORKSPACE_DIR
cd "$WORKSPACE_DIR/Primus"
./primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

Use `primus-cli direct` (not `container`), since you are running on bare metal with everything installed in your environment.

### 1.5 What the scripts do NOT do

- **System (`apt`) packages** ([Section 3](#3-system-packages-need-root-one-time)): skipped — they need root.
- **Multi-node networking** ([Section 5](#5-multi-node-communication-stack-ucx-and-openmpi)): UCX, OpenMPI and AMD AINIC are not built. Single-node training works without them.
- **FBGEMM / Flux / DLRM benchmarks**: not built (`torchrec` is an optional stage; FBGEMM additionally needs apt `libtbb-dev`).
- **MLPerf `primus_mllog`**: v26.5 installs it from `training_results_v6.0`; it is not part of the core training path and is not installed.

### 1.6 Where the scripts deliberately differ from the Dockerfile

A handful of places diverge on purpose, because copying the Dockerfile exactly produces an environment that does not work outside the container — several of its floating (unpinned) dependencies have since drifted to versions that break. The short list:

- **TransformerEngine** is installed from the wheel index only where glibc is new enough, otherwise built from the equivalent source commit ([Section 4.6](#46-transformerengine)).
- **Companion wheels are pinned** (`torchaudio`, `torchvision`, `apex`) to the same nightly as `torch`, because the Dockerfile's floating versions no longer resolve.
- **`nvidia-cutlass-dsl` is pinned to 4.5.3** and **`flydsl` to 0.2.4**, both to versions that the Dockerfile itself resolved to when it was built. Newer releases silently break `import mamba_ssm` and aiter's CK/HIP kernels respectively.
- **`NVTE_CK_IS_V3_ATOMIC_FP32=1` on gfx942** instead of the Dockerfile's `0`: without fp32 atomics the CK v3 backward attention kernel produces `Inf` gradients on MI300X/MI325X at the first training step. This matches the MI300X/MI325X block in [training recipes](../02-user-guide/training-recipes.md); gfx950 keeps the Dockerfile value.
- **pip is constrained** so no later install can replace the ROCm `torch`/`triton` with upstream CUDA builds.

Each one is documented with the exact failure it avoids in
[tools/installation/README.md](https://github.com/AMD-AGI/Primus/blob/main/tools/installation/README.md). Read that before "correcting" any of them back.

---

## 2. Required software stack for distributed LLM training

The complete environment is composed of the following layers:

| Layer                   | Component                                                                                   | Source                  | Needs root?                      |
| ----------------------- | ------------------------------------------------------------------------------------------- | ----------------------- | -------------------------------- |
| Kernel / hardware       | AMD GPU driver (amdgpu KMD), GPU device access                                              | OS / admin              | Yes (one-time, by admin)         |
| OS libraries            | Build toolchain + runtime libs (`g++`, `git`, RDMA, hwloc, etc.)                            | `apt`                   | Yes (one-time)                   |
| Python                  | CPython 3.12 (required by the pinned wheels)                                                | `uv` / standalone build | No                               |
| ROCm user-space         | `rocm-sdk-devel` + device packages                                                          | pip (TheRock wheels)    | No (venv)                        |
| Deep learning framework | PyTorch ROCm (`torch`, `torchvision`, `torchaudio`, `apex`)                                 | pip (TheRock wheels)    | No (venv)                        |
| Accelerated kernels     | Flash Attention, TransformerEngine, aiter, grouped_gemm, Primus-Turbo, causal-conv1d, mamba | pip / build from source | No (venv)                        |
| Training frameworks     | torchtune, torchao, torchrec, FBGEMM                                                        | build from source / pip | No (venv)                        |
| Multi-node comms        | UCX, OpenMPI, AMD AINIC (libionic)                                                          | build from source / apt | Mostly no (AINIC libs need root) |
| Primus                  | Primus + submodules (Megatron-LM, TorchTitan, etc.)                                         | git + pip               | No (venv)                        |
| Python deps             | datasets, transformers, accelerate, trl, wandb, etc.                                        | pip                     | No (venv)                        |

For a **distributed (multi-node) training job** specifically, beyond PyTorch and ROCm you additionally need:

- **RCCL** (AMD's collective library) — provided by the ROCm SDK wheels.
- **UCX + OpenMPI** — point-to-point transport and the MPI launcher.
- **AMD AINIC / RDMA stack** (`libibverbs`, `rdma-core`, `libionic`) — for high-performance networking on AMD Pensando NICs.
- Correct GPU/NIC device permissions and (often) hugepages / `ulimit -l unlimited` for RDMA.

---

## 3. System packages (need root, one-time)

These are OS-level libraries needed to *build* the rest of the stack and to run RDMA networking. They must be installed by someone with root, but this is a **one-time** action; everything afterward is done unprivileged in a venv. On a shared/managed host, ask your administrator to install them once.

> If you genuinely cannot get root at all, these packages must already be present on the host. There is no supported way to install system `.deb` packages without root. The remainder of the guide then runs entirely without root.

### 3.1 Build toolchain and core libraries

```bash
sudo apt update
sudo apt install -y \
    gfortran git git-lfs ninja-build g++ pkg-config xxd patchelf \
    automake libtool autoconf flex ccache \
    python3-venv python3-dev python3-pip python-is-python3 \
    libegl1-mesa-dev liblzma-dev libdw1 libdrm-dev libz3-dev \
    wget xz-utils ffmpeg numactl pciutils
```

`libtbb-dev` is additionally required if you intend to build FBGEMM.

### 3.2 RDMA / networking libraries (needed for multi-node training)

```bash
sudo apt install -y \
    rdma-core libibverbs-dev ibverbs-utils infiniband-diags \
    ethtool kmod dpkg-dev jq \
    libevent-dev libhwloc-dev libmunge-dev \
    software-properties-common
```

### 3.3 AMD AINIC library (optional, for AMD Pensando NICs)

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

## 4. Manual build reference

This section is **reference material**, not a maintained walkthrough: it records the exact versions and the non-obvious build steps so you can audit or adapt the process. For an actual install, use the scripts in [Section 1](#1-recommended-path-the-automated-scripts) — they encode everything below plus the workarounds listed in [Section 1.6](#16-where-the-scripts-deliberately-differ-from-the-dockerfile).

The authoritative source of versions is always the reference Dockerfile.

### 4.1 Create the virtual environment (Python 3.12)

The pinned torch nightly ships a cp312 Linux wheel only, so the interpreter must be 3.12. Ubuntu 22.04 has 3.10, and installing 3.12 with `apt` needs root — the no-root option is a standalone build, which is what `uv` provides.

#### Installing `uv`

Either method works and neither needs root. `setup.sh` performs the second one automatically when `uv` is missing, so you only need to do this by hand for a manual build, or if the `astral.sh` download is blocked on your network.

```bash
# A. via pip, using whatever Python you already have.
#    --user puts it in ~/.local/bin, one of the locations setup.sh searches.
python3 -m pip install --user uv

# B. via the standalone installer (what setup.sh uses)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Make sure the result is on your `PATH` (`uv --version` should work). If you cannot install `uv` at all, supply your own 3.12 interpreter instead and skip it entirely — the scripts accept `PRIMUS_PYTHON=/path/to/python3.12`, and for a manual build just point `PY312` at it below.

#### Create the venv

```bash
uv python install 3.12
PY312="$(uv python find 3.12)"

"$PY312" -m venv ~/primus-env
source ~/primus-env/bin/activate
python --version          # must report 3.12.x

# Build/runtime knobs (match the Dockerfile)
export MAX_JOBS=128                              # lower if you have fewer cores / less RAM
export PYTORCH_ROCM_ARCH="gfx942;gfx950"         # MI300/MI325 = gfx942, MI350/MI355 = gfx950
export ROCM_AMDGPU_TARGETS="gfx942,gfx950"
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0        # avoids HSA_STATUS_ERROR_OUT_OF_RESOURCES
export HSA_NO_SCRATCH_RECLAIM=1
```

> Set `PYTORCH_ROCM_ARCH` to only the architecture(s) you actually have to speed up source builds (e.g. `"gfx942"` for MI300X-only).

### 4.2 Bootstrap build tooling

```bash
pip install --upgrade pip
pip install \
    pybind11 typeguard \
    wheel==0.45.1 cmake==3.31.6 ninja==1.11.1.3 \
    packaging==25.0 setuptools==75.1.0
```

### 4.3 Install ROCm + PyTorch from the TheRock multi-arch wheels

This replaces a system ROCm install. Install the base deps first — `apex` declares `cxxfilt`, `pytest` and `ninja` requirements that the nightly index does not serve, so they must be present before the torch resolve.

```bash
pip install \
    cxxfilt==0.3.0 tqdm==4.67.3 pyyaml==6.0.3 pytest==9.0.3 \
    matplotlib==3.10.9 pandas==2.3.3 py-cpuinfo==9.0.0 build==1.5.0

# One coherent nightly. The Dockerfile pins only `torch` and lets the companions
# float, which no longer resolves; pin them all to the same date.
NIGHTLY="rocm7.15.0a20260720"

python -m pip uninstall -y torch
python -m pip install \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch --pre \
    torch==2.12.0+${NIGHTLY} \
    amd-torch-device-gfx942==2.12.0+${NIGHTLY} \
    amd-torch-device-gfx950==2.12.0+${NIGHTLY} \
    rocm-sdk-devel==7.15.0a20260720 \
    rocm-sdk-device-gfx942==7.15.0a20260720 \
    rocm-sdk-device-gfx950==7.15.0a20260720 \
    torchaudio==2.11.0+${NIGHTLY} \
    torchvision==0.27.0+${NIGHTLY} \
    amd-torchvision-device-gfx942==0.27.0+${NIGHTLY} \
    amd-torchvision-device-gfx950==0.27.0+${NIGHTLY} \
    apex==1.12.0+${NIGHTLY}
```

> Install only the `*-gfx942` **or** `*-gfx950` device packages matching your hardware for a smaller install.
>
> Nightly indexes are pruned. If this date has disappeared, pick a newer one that publishes a *complete* cp312 set — `torch`, `amd-torch-device-*`, `rocm-sdk-*`, `torchaudio`, `torchvision`, `amd-torchvision-device-*` and `apex` must all come from the same date.

### 4.4 Initialize the ROCm SDK and export ROCm paths

`rocm-sdk init` materializes the ROCm toolchain inside the venv. The variables below point the rest of the build at that in-venv ROCm and **must be set every time you use the environment** — see [Section 6](#6-make-the-environment-reproducible).

```bash
rocm-sdk init

export ROCM_PATH=$(python -c 'import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))')
export ROCM_HOME=$ROCM_PATH          # Primus uses ROCM_HOME; set both
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

> The Dockerfile hardcodes `ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel`. The `python -c ...` form derives it from the venv instead, so it works wherever you put the environment.

Quick check before continuing:

```bash
hipcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.5 Source-built components and their pins

Build these against your in-venv ROCm and selected `PYTORCH_ROCM_ARCH`, in a scratch directory:

```bash
export GPU_ARCHS="${PYTORCH_ROCM_ARCH}"   # cross-compile during builds; see Section 6 for runtime
mkdir -p ~/primus-build && cd ~/primus-build
```

| Component | Repository | Pin | Install command | Notes |
|---|---|---|---|---|
| Flash Attention | `ROCm/flash-attention` | `6387433156558135a998d5568a9d74c1778666d8` | `python setup.py install` | clone `--recursive` |
| grouped_gemm | `caaatch22/grouped_gemm` | branch `rocm` | `pip install --no-build-isolation .` | MoE models |
| causal-conv1d | `Dao-AILab/causal-conv1d` | `e940ead2fd962c56854455017541384909ca669f` | `pip install --no-build-isolation .` | needs `CAUSAL_CONV1D_FORCE_BUILD=TRUE`, `HIP_ARCHITECTURES=gfx942,gfx950` |
| mamba | `AndreasKaratzas/mamba` | branch `enable-primus-hybrid-models` | `pip install --no-build-isolation .` | see note below |
| aiter | `ROCm/aiter` | `0f3c58e6edb6754940bcf9fd5f09ccb6f389f52e` | `PREBUILD_KERNELS=3 pip install --no-cache-dir --use-pep517 .` | clone `--recursive`; `pip uninstall aiter amd-aiter` first |
| Primus-Turbo | `AMD-AGI/Primus-Turbo` | `edc8d2ccb0be4888e80ee7c6e765fd3956026a32` | `pip install -r requirements.txt` then `pip install --no-build-isolation . -v` | needs `HCC_AMDGPU_TARGET="gfx942,gfx950"`; see note below |
| torchtune | `pytorch/torchtune` | `b4c98ac2a37f0397d64c22579aed415ce7264db6` | `pip install .` | patch first: `sed -i 's/use_grouped_mm = True/use_grouped_mm = False/g' torchtune/modules/moe/utils.py` |
| torchao | `pytorch/ao` | `e9c7bead90b840b280f97374308255957108ce47` | `pip install --no-build-isolation .` | two patches: `pad_inner_dim` → `True` in `torchao/float8/config.py`, and `if defined(HIPBLASLT_VEC_EXT)` → `if false` in `torchao/csrc/rocm/swizzle/swizzle.cpp` |

**mamba.** Install with `pip`, **not** `python setup.py install`: the legacy `easy_install` path ignores pip-installed packages and re-fetches the latest of every unpinned dependency as `.egg`s, which clobbers the pins. Pin `apache-tvm-ffi==0.1.11` beforehand, and `nvidia-cutlass-dsl==4.5.3` — newer CUTLASS DSL releases removed a symbol that `mamba_ssm`'s `quack-kernels` dependency imports, breaking `import mamba_ssm`. `mamba_ssm` pulls in `tilelang`, which v26.5 uninstalls again once everything is built.

**Primus-Turbo.** Its `setup.py` hard-pins upstream `triton==3.7.0`, which conflicts with the ROCm `triton` that `torch` requires; installing it replaces the ROCm build and then segfaults on import, because ROCm's HIP runtime already loads its own LLVM. Install with `--no-deps` and supply `scipy` and `flydsl==0.2.4` yourself. Primus-Turbo also probes for rocSHMEM and mistakes the pip ROCm SDK for one; set `ROCSHMEM_HOME` to a non-existent path to make the probe fail cleanly and disable internode DeepEP.

### 4.6 TransformerEngine

v26.5 changed this: TE is no longer built from source but installed from the ROCm staging index.

```bash
# Runtime tuning flags (see Section 6 for the full set)
export NVTE_USE_CAST_TRANSPOSE_TRITON=1

# TE's own dependencies must be present first: the staging index serves only the
# transformer-engine packages, so pip cannot fetch anything else while it is selected.
pip install pybind11==3.0.4 importlib-metadata==8.7.1 onnxscript==0.7.0 \
            pydantic==2.13.4 nvdlfw_inspect==0.2.2 einops onnx

pip install \
    --index-url https://rocm.frameworks-nightlies.amd.com/whl-staging/device-all/ \
    --pre --no-build-isolation \
    transformer_engine_rocm_torch==2.15.0.dev0+rocm7.15.0a20260716.a07e607
```

> **Those wheels need glibc ≥ 2.38.** They are built on Ubuntu 24.04, and `libtransformer_engine.so` requires `GLIBC_2.38` plus `GLIBCXX_3.4.32`. Ubuntu 22.04 has glibc 2.35, and glibc cannot be side-loaded via `LD_LIBRARY_PATH`, so on 22.04 the wheels install fine but fail at import with `version 'GLIBC_2.38' not found`.
>
> On such hosts, build the equivalent source commit instead — the same commit the version label refers to:
>
> ```bash
> git clone --recursive https://github.com/ROCm/TransformerEngine.git
> cd TransformerEngine
> git checkout a07e607f14a5330807ffdafeeb6224f2d7dffacc
> git submodule update --init --recursive
> pip install psutil
> MAX_JOBS=${MAX_JOBS} NVTE_FRAMEWORK=pytorch NVTE_USE_ROCM=1 \
>   NVTE_USE_HIPBLASLT=1 NVTE_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
>   pip install --no-build-isolation .
> ```
>
> `setup.sh` picks between the two automatically from the host's glibc; override with `PRIMUS_TE_MODE=wheel|source`.

Either way, apply the same fix v26.5 applies inside the image: concurrent CK JIT compiles race to publish the same `.so`, and without this the loser aborts the run.

```bash
CK_JIT=$(python -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "transformer_engine/lib/ck_jit/ck_jit_compile.sh"))')
sed -i 's|    mv -n "$_TMP_SO" "$OUTPUT"$|    mv -n "$_TMP_SO" "$OUTPUT" 2>/dev/null \|\| true|' "$CK_JIT"
```

### 4.7 Training-framework Python dependencies

```bash
pip install \
    datasets==3.6.0 av==16.0.1 transformers==4.55.0 optree==0.18.0 sympy \
    accelerate==1.9.0 trl==0.21.0 tensorboard==2.20.0 peft scipy einops \
    flask-restful nltk pytest pytest-cov pytest_mock pytest-csv \
    pytest-random-order sentencepiece wrapt \
    zarr==2.18.7 numcodecs==0.12.1 xarray wandb tensorstore==0.1.45 \
    pybind11 tiktoken pynvml "huggingface_hub[cli]"

python3 -m nltk.downloader punkt_tab

# AWS SDK (used by some data pipelines)
pip install boto3==1.35.42 botocore==1.35.99
```

### 4.8 Install Primus and its submodules

```bash
cd ~/primus-build
# Required to resolve a post-v26.2 attention backend issue
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout b511d1b66b0068715308ea9bfe8ba147ea1a3860   # release/v26.5
git submodule update --init --recursive
pip install -r requirements.txt

# Megatron's dataset indexing needs a compiled pybind11 extension, or training
# fails with "MockGPTDataset failed to build as a mock data generator".
# Pass LIBEXT explicitly: the Makefile reads it from `python3-config`, which a
# venv does not provide, so it otherwise emits a filename 3.12 will never import.
EXT=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
make -C third_party/Megatron-LM/megatron/core/datasets LIBEXT="$EXT"
```

If you already have a local Primus checkout, run `pip install -r requirements.txt` from its root and skip the clone — but still build `helpers_cpp` in *that* checkout, since it is the one you will launch training from.

### 4.9 Optional: torchrec + FBGEMM (DLRM / recommendation workloads)

```bash
pip install --no-deps torchrec
pip install tensordict iopath torchmetrics==1.0.3 \
    git+https://github.com/mlperf/logging.git \
    --extra-index-url https://rocm.nightlies.amd.com/whl-multi-arch

# FBGEMM (GPU) — needs apt libtbb-dev
export BUILD_ROCM_VERSION='7.14'

git clone https://github.com/pytorch/FBGEMM.git
cd FBGEMM
git checkout 80bd3c077dc41b55cd16ed4dcad15cf7c1c1d76a
cd fbgemm_gpu
git clean -dfx && git submodule sync && git submodule update --init --recursive
pip install -r requirements.txt
pip install setuptools==75.1.0
python setup.py install \
    --build-variant=rocm \
    --build-target=default \
    -DAMDGPU_TARGETS=$PYTORCH_ROCM_ARCH \
    -DHIP_ROOT_DIR=$ROCM_PATH \
    -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
    -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"
cd ../..
```

> The Flux (`AMDiffusionBenchmark`) and DLRM (`DLRMBenchmark`) repos in the Dockerfile are benchmark workloads, not core training dependencies. Clone them only if you need those specific benchmarks.

---

## 5. Multi-node communication stack (UCX and OpenMPI)

These are only needed for **multi-node distributed** training. They build from source into user-writable prefixes (no root needed, except the AINIC `.deb` already handled in [Section 3.3](#33-amd-ainic-library-optional-for-amd-pensando-nics)).

### 5.1 UCX

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

### 5.2 OpenMPI

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

---

## 6. Make the environment reproducible

The ROCm paths and `NVTE_*` flags must be present in **every** shell that runs training.

If you used the scripts, this is already handled — `source tools/installation/env.sh` sets everything (and refuses to run if `PRIMUS_BASE` is unset). For a manual install, append the equivalent to the venv's activation script:

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

# TransformerEngine: attention backend selection used by Primus
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

# TransformerEngine: CK performance knobs
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_USES_BWD_V3=1
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_CK_HOW_V3_BF16_CVT=2
# gfx942 (MI300X/MI325X) needs fp32 atomics for the CK v3 backward kernel: with
# the Dockerfile's 0 it produces Inf gradients at the first step. Use 0 on gfx950.
export NVTE_CK_IS_V3_ATOMIC_FP32=1

# Runtime: let aiter detect the local GPU architecture.
# NOTE: this is the runtime value. If you rebuild any source kernel later,
#       re-export GPU_ARCHS="$PYTORCH_ROCM_ARCH" first, then set it back to native.
export GPU_ARCHS=native

# Multi-node comms (only if built)
export UCX_HOME=$HOME/primus-build/ucx-1.18.0/install
export MPI_HOME=$HOME/primus-build/openmpi
# ---- end Primus host environment ----
EOF
```

---

## 7. Verify the installation

```bash
# Scripts: export the same PRIMUS_BASE and source env.sh
# Manual:  source ~/primus-env/bin/activate

# GPUs visible to ROCm?
rocm-smi || ls -l /dev/kfd /dev/dri

# PyTorch sees the GPUs?
python -c "import torch; print('torch', torch.__version__); \
print('gpu available:', torch.cuda.is_available()); \
print('device count:', torch.cuda.device_count()); \
print('device 0:', torch.cuda.get_device_name(0))"

# Key kernel libs import cleanly?
python -c "import transformer_engine, flash_attn, aiter, primus_turbo, mamba_ssm; print('kernels OK')"

# Run a Primus benchmark / training directly (no container)
cd "$WORKSPACE_DIR/Primus"   # or your Primus checkout
./primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096
./primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
  --train_iters 10
```

A healthy run reports a decreasing `lm loss`, `number of nan iterations: 0`, and a steady `throughput per GPU (TFLOP/s/GPU)` after the first couple of iterations.

Use `primus-cli direct` (not `container`) since you are running on bare metal with everything installed in your environment.

---

## 8. Other important considerations

- **GPU device access without root**: the user running training must be able to read/write `/dev/kfd` and `/dev/dri/*`. This usually means membership in the `video` and `render` groups (`sudo usermod -aG video,render $USER`, then re-login). This is a one-time admin action.
- **Hugging Face access**: if your config downloads weights or tokenizers from the Hub, export your token: `export HF_TOKEN=hf_xxx` (and/or `huggingface-cli login`). The token is needed for gated models like Llama.
- **RDMA / multi-node limits**: high-performance networking typically requires locked-memory limits raised (`ulimit -l unlimited`) and possibly hugepages. These are configured in `/etc/security/limits.conf` and need admin help. Verify NICs with `ibv_devinfo` and `ibstat`.
- **Disk and time**: source builds of aiter, Primus-Turbo, FBGEMM and (on older glibc) TransformerEngine are large and slow. Reserve plenty of disk and expect a multi-hour first build. Lower `MAX_JOBS` if the build runs out of memory.
- **`ccache`**: installed in [Section 3.1](#31-build-toolchain-and-core-libraries); it dramatically speeds up rebuilds, with no extra configuration needed for a basic speedup.
- **Architecture pinning**: building for only your actual GPU arch (e.g. `gfx942` for MI300X/MI325X, `gfx950` for MI350X/MI355X) significantly reduces build time and binary size versus building both. The scripts already do this automatically; it only needs stating explicitly for a manual build.
- **Version drift is the main hazard.** The nightly wheels and source commits are a tested combination; several of the Dockerfile's *unpinned* transitive dependencies have since released versions that break the build or silently disable kernels. The scripts pin those explicitly — see [Section 1.6](#16-where-the-scripts-deliberately-differ-from-the-dockerfile). Treat the reference `Dockerfile` as the source of truth for versions, and the scripts as the source of truth for the workarounds.

---

## 9. Quick reference: minimal vs. full install

If you only need **single-node Megatron/TorchTitan LLM pretraining**, you can skip several optional components:

| Component                                               | Needed for                           |
| ------------------------------------------------------- | ------------------------------------ |
| Flash Attention, TransformerEngine, aiter, Primus-Turbo | Core LLM training (install these)    |
| grouped_gemm                                            | MoE models                           |
| causal-conv1d, mamba                                    | Hybrid / Mamba-family models         |
| torchtune, torchao                                      | Post-training (SFT/LoRA), fp8        |
| torchrec, FBGEMM, DLRM                                  | Recommendation (DLRM) workloads only |
| Flux / AMDiffusionBenchmark                             | Diffusion benchmark only             |
| UCX, OpenMPI, AINIC                                     | Multi-node distributed training      |

Install the core rows first, validate with [Section 7](#7-verify-the-installation), then add the optional components as your workload requires.
