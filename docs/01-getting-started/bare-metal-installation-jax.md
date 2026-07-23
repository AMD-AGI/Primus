# Bare-metal installation (JAX / MaxText): build the Primus JAX training stack from source (no Docker)

This guide explains how to build the **Primus JAX / MaxText training software
stack directly on a host machine**, without using the AMD published JAX training
Docker image. It is intended for users who, for policy or operational reasons,
cannot run containers and need to reproduce the same environment on bare metal.

It is derived from the official JAX training `Dockerfile`
(`unified-training-dockers/jax_training_the_rock/docker/Dockerfile`) and
installs the same components and versions. Wherever possible, everything is
installed **inside a Python virtual environment and without `sudo`**. The only
steps that require root are a small set of OS-level system libraries (installed
with `apt`), and a couple of optional networking packages used for multi-node
training.

> **Looking for the PyTorch/Megatron/TorchTitan stack instead?** See
> [bare-metal-installation.md](bare-metal-installation.md). This document is the
> JAX **MaxText** counterpart. It is leaner than the PyTorch stack — no Flash
> Attention / aiter / Primus-Turbo / FBGEMM / rocSHMEM builds — but v26.5 still
> compiles **TensorFlow (CPU) and RCCL from source** (and TransformerEngine from
> source on hosts with glibc < 2.38), so it is not build-free.

> **Python 3.12+ required.** MaxText requires Python ≥ 3.12 (the reference image
> is built on Ubuntu 24.04). Unlike the PyTorch recipe, Python 3.10 is **not**
> sufficient here.

> **⚠️ Host OS: Ubuntu 24.04 / glibc ≥ 2.38 strongly recommended.** The prebuilt
> `transformer_engine_rocm_jax` wheel (and other JAX training wheels) are built
> against the Dockerfile's `ubuntu:24.04` base — they need **`glibc ≥ 2.38`** and
> **`libstdc++` with `GLIBCXX_3.4.32`** (GCC 13/14). On an older host such as
> **Ubuntu 22.04 (glibc 2.35)** the TransformerEngine shared library fails to
> load with `version 'GLIBC_2.38' not found` (and this failure is *silently*
> swallowed by the launcher — training just exits right after JAX initializes
> the GPUs). `libstdc++` can be side-loaded via `LD_LIBRARY_PATH`, but **glibc
> cannot**. For a **manual** install on Ubuntu 22.04 you must either (a) run on a
> 24.04 host, or (b) build TransformerEngine **from source** against the host
> toolchain — see
> [Section 3.7](#37-install-transformerengine-jax-from-the-prebuilt-rocm-wheel).
> Check your host with `ldd --version`.
>
> **The automated `setup.sh` does (b) for you:** its `te` stage detects the host
> glibc and, when it is < 2.38, transparently builds TransformerEngine from
> source instead of installing the prebuilt wheel — so `bash setup.sh` works on
> both Ubuntu 22.04 and 24.04. Only the prebuilt TE wheel needs glibc ≥ 2.38; the
> ROCm JAX/PJRT wheels load fine on glibc 2.35.

---

## Quick path: automated install scripts

If you just want the environment built for you, use the helper scripts in
[tools/installation-jax/](https://github.com/AMD-AGI/Primus/tree/main/tools/installation-jax).
They automate everything in Section 3 (Python venv) of this guide — venv
creation, the ROCm release tarball, MaxText and its dependencies, TensorFlow
(built from source), JAX + ROCm PJRT/plugin, TransformerEngine, RCCL (built from
source), and Primus itself — and provide a single `env.sh` to activate the
environment before each job. Read the rest of this document if you want to
understand or customize what they do, or if you need the multi-node networking
stack (Section 4), which the scripts do not build.

There are two files:

- **env.sh** — defines the install location and exports every environment
  variable the build and runtime need (ROCm paths, `NVTE_*` flags, `XLA_FLAGS`,
  `MAXTEXT_PATH`, cache locations, etc.). Source it both during the build and
  every time you use the environment.
- **setup.sh** — runs the install in re-runnable **stages**. It sources `env.sh`
  automatically.

### Choose where it installs (important)

Everything lives under `PRIMUS_JAX_BASE` (venv, kept checkouts), with transient
build sources on `SRC_DIR` (defaults to local `/tmp` for fast I/O). The
**default `PRIMUS_JAX_BASE` is site-specific** and may not exist on your host, so
set it to a directory you can write to that has tens of GB free:

```bash
export PRIMUS_JAX_BASE=/path/to/big/disk/primus-jax-env   # venv + checkouts (persistent)
export SRC_DIR=/tmp/primus-jax-build                      # transient sources (optional override)
```

The scripts **auto-detect your GPU architecture** (`env.sh` reads `rocminfo`, or
falls back to the kernel KFD sysfs `gfx_target_version` when no ROCm is
installed yet), and install the matching device wheels — `gfx942` (MI300X/MI325X),
`gfx950` (MI350X/MI355X), or both. To force a target, export it before running:

```bash
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
```

> `PYTORCH_ROCM_ARCH` is the variable name the ROCm SDK and TransformerEngine
> read to select gfx targets — it applies to the JAX build too, despite the name.

**If your default `python3` is older than 3.12** (e.g. Ubuntu 22.04 ships 3.10),
you do **not** need `sudo` or a PPA. The scripts use [`uv`](https://docs.astral.sh/uv/)
to provide Python 3.12:

- If `uv` is already installed, `env.sh` auto-detects a uv-managed
  `>= 3.12` interpreter, and `setup.sh` runs `uv python install 3.12`
  automatically when none is present yet. Just run `bash setup.sh`.
- If `uv` is not installed, install it once (no root) and re-run:

  ```bash
  python3 -m pip install --user uv      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
  bash setup.sh                          # setup.sh will fetch Python 3.12 via uv
  ```

To force a specific interpreter instead, export it before running:

```bash
export PRIMUS_PYTHON=/path/to/python3.12
```

### Build the environment

```bash
cd tools/installation-jax

bash setup.sh            # run all default stages, in order
bash setup.sh --list     # list available stages
bash setup.sh te         # re-run a single stage (e.g. reinstall TransformerEngine)
bash setup.sh venv rocm jax  # run a subset of stages
```

Stages are idempotent and re-runnable, so if a step fails you can fix the cause
and re-run just that stage. On failure the script stops immediately and prints
which stage failed.

Default stages (v26.5):

```
venv → rocm → maxtext → tf_source → jax → te → primus → jaxreqs → rccl → manifest
```

This mirrors the v26.5 image, which **builds TensorFlow (2.21 CPU) and RCCL from
source** (`tf_source`, `rccl`). Those are heavy: the TF bazel build alone is
~30–60 min. Lighter/alternative stages:

- `tf_cpu_fix` — pip `tensorflow-cpu` instead of the `tf_source` bazel build.
- `te_source` — force the from-source TransformerEngine build regardless of
  glibc. You normally don't need to pass this: the default `te` stage
  **auto-falls-back** to a from-source build on glibc < 2.38 hosts (e.g. Ubuntu
  22.04), where the prebuilt wheel won't load (see the host-OS note above and
  Section 3.7). Heavy (~30–60 min).

```bash
# Ubuntu 22.04 (glibc < 2.38): `bash setup.sh` already builds TE from source
# automatically. If you also want to skip the heavy tf_source bazel build, swap
# in the lighter tf_cpu_fix:
bash setup.sh venv rocm maxtext tf_cpu_fix jax te primus jaxreqs rccl manifest
```

### Use the environment for a training job

```bash
# Use the SAME PRIMUS_JAX_BASE you built with
export PRIMUS_JAX_BASE=/path/to/big/disk/primus-jax-env
source tools/installation-jax/env.sh    # activates the venv + sets ROCm/NVTE/XLA vars

python -c "import jax; print('devices:', jax.devices())"

# Primus is checked out under $WORKSPACE_DIR
cd "$WORKSPACE_DIR/Primus"
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-pretrain.yaml
```

### What the scripts do NOT do

- **System (`apt`) packages** (Section 2): skipped — they need root. A C++
  compiler (`g++`/`make`), `git`, and the build basics must already be present,
  along with the small extra set MaxText expects (`numactl`, `curl`, `lsb-release`, …).
- **Multi-node networking** (Section 4: UCX, OpenMPI, AINIC): not built.
  Single-node training works without them; follow Section 4 manually if you need
  distributed-over-RDMA.
- **gcsfuse**: only needed to mount Google Cloud Storage buckets for data. Not
  required for synthetic-data or local-data runs.

The scripts also adapt a few host-specific details beyond the Dockerfile (the
Python part of MaxText's `setup.sh` is run directly to skip its `apt`/interactive
steps, and the ROCm/MaxText/Primus checkouts are collapsed onto a single
`MAXTEXT_PATH`). See
[tools/installation-jax/README.md](https://github.com/AMD-AGI/Primus/blob/main/tools/installation-jax/README.md)
for the full rationale.

---

## 0. The key idea: ROCm comes from a tarball, not from a system install

This build **does not require a system-wide ROCm installation**. In v26.5 ROCm
is delivered as a **release tarball** (AMD "TheRock" multi-arch dist) that is
extracted into a user-writable directory (`$ROCM_DIR`, default
`$PRIMUS_JAX_BASE/rocm`) — no `/opt/rocm`, no root:

- The tarball (`repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-multiarch-7.14.0.tar.gz`)
  provides the full ROCm toolchain (HIP, hipBLASLt, compilers, headers,
  libraries). `ROCM_PATH` points at the extraction dir.
- `jax` + `jaxlib` (upstream) plus the ROCm `jax_rocm7_pjrt` and
  `jax_rocm7_plugin` wheels (from `repo.amd.com/rocm/whl-multi-arch/`) provide
  GPU-accelerated JAX built against that ROCm.

> **Changed from v26.4:** the v26.4 recipe installed ROCm from pip
> `rocm-sdk-devel` **nightly** wheels. v26.5 uses a pinned **release tarball**,
> which is more stable and avoids the pip version-skew that could break
> hipBLASLt GEMMs. RCCL is then rebuilt from source (see Section 3.9) and dropped
> into this ROCm tree.

This means almost the entire stack can be installed **without root** into a
venv. The only host-level requirements from the system administrator are:

- The **AMD GPU kernel driver (amdgpu / ROCm KMD)** must already be installed and
  loaded (`/dev/kfd` and `/dev/dri` must exist, and the user must be in the
  `video` and `render` groups).
- A small set of **build/runtime system libraries** (see Section 2).

---

## 1. Required software stack for JAX MaxText training

The complete environment is composed of the following layers:


| Layer                   | Component                                                          | Source                    | Needs root?               |
| ----------------------- | ----------------------------------------------------------------- | ------------------------- | ------------------------- |
| Kernel / hardware       | AMD GPU driver (amdgpu KMD), GPU device access                    | OS / admin                | Yes (one-time, by admin)  |
| OS libraries            | Build toolchain + runtime libs (`g++`, `git`, `numactl`, RDMA, …) | `apt`                     | Yes (one-time)            |
| ROCm user-space         | TheRock ROCm dist (HIP, hipBLASLt, compilers, libs)              | **release tarball** ($ROCM_DIR) | No (user dir)       |
| Deep learning framework | JAX (`jax`, `jaxlib`) + ROCm `jax_rocm7_pjrt` / `jax_rocm7_plugin`| pip (upstream + repo.amd.com) | No (venv)             |
| Accelerated kernels     | TransformerEngine (JAX) — prebuilt ROCm wheel (or from source)   | pip (staging index) / build | No (venv)               |
| Training framework      | MaxText (ROCm fork) + its Python deps                             | git + pip (`uv`)          | No (venv)                 |
| Collectives fix         | `tensorflow-cpu` 2.21 (built from source; no bundled NCCL/LLVM)   | build from source (bazel) | No (venv)                 |
| Collectives lib         | RCCL (rebuilt from source into the ROCm tree)                    | build from source         | No (user dir)             |
| Multi-node comms        | UCX, OpenMPI, AMD AINIC (libionic)                                | build from source / apt   | Mostly no (AINIC needs root) |
| Primus                  | Primus + `third_party/maxtext` submodule                         | git + pip                 | No (venv)                 |


### 1.1 Version requirements (pins and host prerequisites)

These are the exact versions the v26.5 reference `Dockerfile` pins. The install
scripts use the same pins; change one and you may have to change the others.

| Component                         | Pinned version / source                                                 | Notes |
| --------------------------------- | ----------------------------------------------------------------------- | ----- |
| ROCm (TheRock dist tarball)       | `therock-dist-linux-multiarch-7.14.0.tar.gz`                            | Multi-arch (gfx942 + gfx950). Extracted into `$ROCM_DIR`. |
| JAX / jaxlib                      | `0.10.0`                                                                | Upstream PyPI. |
| ROCm PJRT / plugin                | `jax_rocm7_pjrt` / `jax_rocm7_plugin` `0.10.0+rocm7.14.0`               | From `repo.amd.com/rocm/whl-multi-arch/`. |
| TransformerEngine (JAX)           | `transformer_engine_rocm_jax 2.15.0.dev0+rocm7.15.0a20260707.72d01a0`  | Prebuilt wheel **needs glibc ≥ 2.38**; else build from source (`te_source`). |
| TensorFlow (CPU, from source)     | ROCm `tensorflow-upstream` branch `upstream-v2.21.0`                    | Built with bazelisk `v1.25.0`. Needs host `clang-18`/`lld-18`. |
| RCCL (from source)                | `rocm-systems` @ `9e5e4084a4b8e1e86551b0eb054725c62354a926`            | Installed into `$ROCM_PATH/lib`. Needs host `clang-18`/`lld-18`. |
| MaxText (ROCm fork)               | `release/v26.5`                                                         | 2-value `initialize()`/`run()` API; Primus `main` supports it (fix #912). |
| Primus                            | `main`                                                                  | Includes the MaxText v26.4/v26.5 compatibility shim. |
| scipy                             | `1.16`                                                                  | |
| amdsmi                            | `7.0.2`                                                                 | pip, after the ROCm tarball. |
| Build front-end                   | `cmake 3.31.6`, `ninja 1.11.1.3`, `wheel 0.45.1`, `packaging 25.0`, `setuptools 69.5.1` | Plus `uv` (used by MaxText's dep install). |
| TE deps                           | `pybind11 3.0.4`, `importlib-metadata 8.7.1`, `pydantic 2.13.4`, `flax 0.12.2` | |

**Host prerequisites** (independent of the pins above):

| Requirement            | Minimum / recommended                          | Why |
| ---------------------- | ---------------------------------------------- | --- |
| **Python**             | **≥ 3.12** — **3.12 recommended/pinned**       | MaxText requires ≥ 3.12; the prebuilt TE/JAX wheels are `cp312`, so on a 3.13 venv you must build TE from source. `uv` provides 3.12 with no root. |
| **glibc**              | **≥ 2.38** for the prebuilt TE wheel           | Ubuntu 24.04 = glibc 2.38+. On **Ubuntu 22.04 (glibc 2.35)** the prebuilt TE wheel won't load — the `te` stage auto-falls-back to a from-source build. Check with `ldd --version`. `glibc` cannot be side-loaded via `LD_LIBRARY_PATH`. |
| **libstdc++ (GCC)**    | `GLIBCXX_3.4.32` (GCC 13/14) for prebuilt TE   | Can be side-loaded via `LD_LIBRARY_PATH` if glibc itself is new enough. |
| **C/C++ toolchain**    | `g++` with C++17; `clang-18`/`lld-18` for the TF/RCCL source builds | Source builds (`tf_source`, `rccl`, `te_source`) need LLVM 18. |
| **GPU arch**           | AMD Instinct **gfx942** (MI300/MI325) or **gfx950** (MI350/MI355) | The pinned ROCm tarball + JAX wheels target these. Other archs (e.g. gfx90a/MI250) need matching wheels not pinned here. |
| **GPU driver (KMD)**   | amdgpu / ROCm kernel driver loaded             | `/dev/kfd` + `/dev/dri` present; user in `video`/`render` groups. |

> **Validated:** the scripts' default flow (with `te_source` substituted for the
> prebuilt `te`) has been run end-to-end for single-node MaxText pretraining on
> **gfx942 / Ubuntu 22.04 (glibc 2.35) / Python 3.12**. The prebuilt-`te` flow is
> the path for Ubuntu 24.04 hosts.

For a **distributed (multi-node) JAX MaxText job** specifically, beyond JAX and
ROCm you additionally need:

- **RCCL** (AMD's collective library) — rebuilt from source into the ROCm tree
  (see Section 3.11).
- **UCX + OpenMPI** — point-to-point transport and the launcher.
- **AMD AINIC / RDMA stack** (`libibverbs`, `rdma-core`, `libionic`) — for
  high-performance networking on AMD Pensando NICs.
- Correct GPU/NIC device permissions and (often) hugepages / `ulimit -l unlimited`.

> Unlike the PyTorch stack, the JAX MaxText image does **not** build rocSHMEM.
> JAX distributes over RCCL + the JAX distributed coordinator (`JAX_COORDINATOR_IP`).

---

## 2. System packages (require `sudo` / administrator, one-time)

These are OS-level libraries needed to *build* the rest of the stack and to run
MaxText / RDMA networking. They must be installed by someone with root, but this
is a **one-time** action; everything afterward is done unprivileged in a venv.

> If you genuinely cannot get root at all, these packages must already be present
> on the host. The remainder of the guide (Sections 3+) then runs entirely
> without root.

### 2.1 Build toolchain and core libraries

```bash
sudo apt update
sudo apt install -y \
    gfortran git git-lfs ninja-build g++ pkg-config xxd patchelf \
    automake libtool flex ccache \
    python3-venv python3-dev python3-pip python-is-python3 \
    libegl1-mesa-dev liblzma-dev libdw1 libdrm-dev \
    wget unzip zip
```

> **Source builds (v26.5): LLVM 18 toolchain.** The `tf_source` (TensorFlow 2.21
> bazel) and `rccl` source builds want a host `clang-18`/`lld-18`. The reference
> image adds the LLVM apt repo and installs them:
>
> ```bash
> echo 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main' | sudo tee /etc/apt/sources.list.d/llvm.list
> wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
> sudo apt update && sudo apt install -y clang-18 lld-18 llvm-18-dev llvm-18-tools
> ```
>
> (Use `jammy` for Ubuntu 22.04, `noble` for 24.04.) You can skip this if you use
> the lighter `tf_cpu_fix` stage instead of `tf_source`.

> **Python 3.12+**: MaxText needs Python ≥ 3.12. Ubuntu 24.04 ships 3.12 by
> default. On Ubuntu 22.04 (which ships 3.10) `apt install python3.12` fails
> because jammy has no such package — do **not** rely on it. Two options:
>
> - **Recommended, no sudo — `uv`** (this is what the automated scripts use):
>   ```bash
>   python3 -m pip install --user uv    # or: curl -LsSf https://astral.sh/uv/install.sh | sh
>   uv python install 3.12              # downloads a standalone CPython 3.12 (no root)
>   export PRIMUS_PYTHON="$(uv python find '>=3.12')"
>   ```
>   The manual venv step below then uses `"$PRIMUS_PYTHON" -m venv ...`.
> - **Alternative — deadsnakes PPA** (needs sudo, and the PPA must be reachable
>   from your network):
>   ```bash
>   sudo add-apt-repository ppa:deadsnakes/ppa
>   sudo apt update
>   sudo apt install -y python3.12 python3.12-venv python3.12-dev
>   export PRIMUS_PYTHON=python3.12
>   ```

### 2.2 Extra packages MaxText's setup expects

MaxText's own `setup.sh` installs these via `apt`; on bare metal, pre-install
them once:

```bash
sudo apt install -y \
    numactl lsb-release gnupg curl net-tools iproute2 procps lsof ethtool
```

Optional — only if you read training data from Google Cloud Storage:

```bash
# gcsfuse (mount GCS buckets)
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
    sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install -y gcsfuse
```

### 2.3 RDMA / networking libraries (needed for multi-node training)

```bash
sudo apt install -y \
    rdma-core libibverbs-dev ibverbs-utils infiniband-diags \
    ethtool kmod dpkg-dev jq xz-utils \
    libevent-dev libhwloc-dev libmunge-dev \
    software-properties-common
```

### 2.4 AMD AINIC library (optional, for AMD Pensando NICs)

This pulls a vendor `.deb` from the AMD radeon repository. Skip it if you are
not using AMD AINIC networking.

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
# Pick a stable location, e.g. ~/primus-jax-env. MaxText needs Python >= 3.12.
# On Ubuntu 22.04, get a 3.12 interpreter via uv first (see the Python 3.12+
# note in Section 2.1): export PRIMUS_PYTHON="$(uv python find '>=3.12')"
"${PRIMUS_PYTHON:-python3.12}" -m venv ~/primus-jax-env
source ~/primus-jax-env/bin/activate

# Build/runtime knobs (match the Dockerfile)
export MAX_JOBS=128                              # lower this if you have fewer CPU cores / less RAM
export PYTORCH_ROCM_ARCH="gfx942;gfx950"         # MI300/MI325 = gfx942, MI350/MI355 = gfx950
export ROCM_AMDGPU_TARGETS="gfx942,gfx950"
```

### 3.2 Bootstrap build tooling

```bash
pip install --upgrade pip
pip uninstall -y wheel
pip install \
    cmake==3.31.6 \
    ninja==1.11.1.3 \
    wheel==0.45.1 \
    packaging==25.0 \
    setuptools==69.5.1 \
    uv
```

### 3.3 Workaround environment variables

```bash
# Avoids HSA_STATUS_ERROR_OUT_OF_RESOURCES on some configurations
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1

# Fix the ROCm profiler hang issue
export ROCPROFILER_QUEUE_INTERPOSITION=0
export DEBUG_HIP_DYNAMIC_QUEUES=0
```

### 3.4 Install ROCm from the TheRock release tarball

This step replaces a system ROCm install. v26.5 uses a **release tarball**
(not pip wheels) extracted into a user-writable dir — no `/opt/rocm`, no root.

```bash
# Extract into a user-writable location (the automated env.sh uses
# $PRIMUS_JAX_BASE/rocm). ROCM_PATH will point here.
export ROCM_DIR=~/primus-jax-env/rocm
mkdir -p "$ROCM_DIR"
wget -O /tmp/therock-dist.tar.gz \
    https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-multiarch-7.14.0.tar.gz
tar -xzf /tmp/therock-dist.tar.gz -C "$ROCM_DIR"

# amdsmi (installed via pip in the reference image)
pip install amdsmi==7.0.2
```

> The tarball is multi-arch (gfx942 + gfx950), so there is no per-arch package to
> pick, and there is **no pip version-skew to worry about** (the v26.4 recipe's
> `rocm-sdk-*` nightly wheels could drift out of sync and break hipBLASLt GEMMs;
> the pinned tarball avoids that entirely).

### 3.5 Export ROCm paths

Point the rest of the build/runtime at the extracted ROCm. **These must be set
every time you use the environment** — Section 5 shows how to make them
persistent (the automated `env.sh` does all of this for you).

```bash
export ROCM_PATH=$ROCM_DIR
export ROCM_HOME=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
# The reference Dockerfile puts $ROCM_PATH/lib on PATH too; mirror that.
export PATH="$ROCM_PATH/lib:$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib"
export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
export CPATH=$HIP_INCLUDE_PATH
export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"
```

Quick check before continuing:

```bash
hipcc --version
```

### 3.6 Install MaxText, then JAX + the ROCm PJRT/plugin

> **Order matters (v26.5).** The subsections below are numbered for reference,
> but the correct install order is: **MaxText deps (§3.8) → TensorFlow from
> source (§3.9) → ROCm JAX/PJRT/plugin (§3.6, right here) → TransformerEngine
> (§3.7) → Primus (§3.10) → RCCL from source (§3.11)**. MaxText's `setup.sh`
> pulls in a stock `jax`/`tensorflow`, so the ROCm JAX must be installed *after*
> MaxText (to override it) and *before* TE (or a later step overwrites
> `jaxlib`). The automated `setup.sh` enforces this ordering for you.

MaxText install is in Section 3.8 (deps) below; the JAX packages are:

```bash
JAX_VERSION=0.10.0
JAX_PJRT_VERSION=0.10.0+rocm7.14.0       # https://repo.amd.com/rocm/whl-multi-arch/jax-rocm7-pjrt/
JAX_PLUGIN_VERSION=0.10.0+rocm7.14.0     # https://repo.amd.com/rocm/whl-multi-arch/jax-rocm7-plugin/

pip install jax==${JAX_VERSION} jaxlib==${JAX_VERSION} scipy==1.16
pip install \
    --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    --pre jax_rocm7_pjrt==${JAX_PJRT_VERSION} \
    --pre jax_rocm7_plugin==${JAX_PLUGIN_VERSION}
```

### 3.7 Install TransformerEngine (JAX) from the prebuilt ROCm wheel

TransformerEngine is installed as a prebuilt ROCm JAX wheel. Check
[the staging index](https://rocm.frameworks-nightlies.amd.com/whl-staging/device-all/transformer-engine-rocm-jax/)
for the current pin.

```bash
TE_VERSION=2.15.0.dev0+rocm7.15.0a20260707.72d01a0

pip install \
    pybind11==3.0.4 \
    importlib-metadata==8.7.1 \
    pydantic==2.13.4 \
    flax==0.12.2

pip install \
    --index-url https://rocm.frameworks-nightlies.amd.com/whl-staging/device-all/ \
    --pre \
    --no-build-isolation \
    transformer_engine_rocm_jax==${TE_VERSION}
```

> **The prebuilt wheel needs `glibc ≥ 2.38` (Ubuntu 24.04).** Verify it actually
> loads before continuing:
>
> ```bash
> python -c "import transformer_engine.jax; print('TE JAX OK')"
> ```
>
> If you see `OSError: ... version 'GLIBC_2.38' not found` (typical on Ubuntu
> 22.04, glibc 2.35), the wheel is incompatible with your host. **Build
> TransformerEngine from source instead** so it links against your host's glibc.
> This is exactly what the automated `te_source` stage does
> (`bash setup.sh ... te_source ...`); the manual equivalent is:
>
> ```bash
> pip uninstall -y transformer_engine transformer_engine_rocm_jax
> export USE_ROCM=1 NVTE_FRAMEWORK=jax NVTE_USE_ROCM=1
> export NVTE_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
> git clone --recursive https://github.com/ROCm/TransformerEngine.git
> cd TransformerEngine
> git checkout 635d7c085c39a6d9bfe4881c7d3efab7a46d7129   # last known-good ROCm JAX TE source commit
> git submodule update --init --recursive
> python3 setup.py bdist_wheel && pip install dist/*.whl
> cd ..
> ```
>
> If you only hit a `GLIBCXX_3.4.32` (libstdc++) error but glibc is new enough,
> you can instead side-load a newer `libstdc++.so.6` (e.g. extracted from a
> newer distro's `libstdc++6` package) via `LD_LIBRARY_PATH` — no rebuild needed.

### 3.8 Install MaxText and its dependencies

Clone the ROCm MaxText fork and install its Python dependencies. The reference
image runs MaxText's `src/dependencies/scripts/setup.sh`; on bare metal we run
the **Python portion** of that script directly (the `apt`/`gcsfuse` steps are the
one-time root action from Section 2, and the venv already exists).

> **MaxText `release/v26.5` and the Primus API.** v26.5 switched MaxText's
> `initialize()`/`run()` to a **2-value** API (`config, recorder`); v26.4 and
> earlier used a **3-value** API (adds `diagnostic_config`). Primus `main`
> supports **both**: `MaxTextPretrainTrainer` forwards `initialize()`'s tuple
> verbatim to `run()` (fix #912), so v26.5 trains out of the box. You only need
> `MAXTEXT_BRANCH=release/v26.4 bash setup.sh maxtext` if you deliberately want
> to pin an older MaxText release.

```bash
cd ~/primus-jax-env   # or your $WORKSPACE_DIR
git clone https://github.com/ROCm/maxtext.git
cd maxtext
git checkout release/v26.5   # matches the v26.5 image; Primus main supports its 2-value API

# MaxText installs its deps with uv. The default (tpu) requirements set contains
# the framework-agnostic Python deps WITHOUT any CUDA packages, which is what the
# ROCm image uses.
pip install -U setuptools wheel uv
python -m uv pip install --resolution=lowest \
    -r src/dependencies/requirements/generated_requirements/tpu-requirements.txt
python -m src.dependencies.scripts.install_pre_train_extra_deps
python -m uv pip install --no-deps -e .
cd ..
```

> This pulls in the full `tensorflow` package (via `tensorflow-text`); the next
> step swaps it for the CPU build. It may also nudge `jax`/`jaxlib`/`scipy`
> within their allowed ranges — the ROCm PJRT/plugin installed in 3.6 remain in
> place.

### 3.9 Build TensorFlow (CPU) from source

v26.5 rebuilds TensorFlow 2.21 (CPU) from ROCm's fork. The stock PyPI TF wheel
bundles an LLVM whose symbols collide with ROCm's `libLLVM` in Grain "spawn"
workers → **SIGSEGV on `import tensorflow` after `import jax`**. A CPU build has
correct symbol visibility and no bundled NCCL (so it also preserves the XLA→RCCL
collective fix). This is a **heavy bazel build (~30–60 min)** and needs a host
`clang`/`lld` (LLVM 18) plus `unzip`/`zip` (Section 2).

```bash
# bazelisk (to a user-writable location) auto-picks the bazel version TF pins.
wget -O ~/primus-jax-env/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64
chmod +x ~/primus-jax-env/bin/bazel
export PATH="$HOME/primus-jax-env/bin:$PATH"

git clone --depth 1 --branch upstream-v2.21.0 https://github.com/ROCm/tensorflow-upstream.git
cd tensorflow-upstream
bazel --output_user_root=/tmp/primus-jax-build/bazel build //tensorflow/tools/pip_package:wheel \
    --repo_env=WHEEL_NAME=tensorflow_cpu \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12
pip uninstall -y tensorflow tensorflow-cpu tensorflow_cpu
pip install --no-deps bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.21.0-cp312-cp312-linux_x86_64.whl
cd ..
```

> **Lighter alternative:** if you don't want the bazel build, `pip install
> --no-deps tensorflow-cpu==$(pip show tensorflow | awk '/^Version:/{print $2}')`
> installs a CPU wheel that avoids the bundled-NCCL clash. It may still hit the
> LLVM-symbol SIGSEGV on some ROCm 7.14 configs — the from-source build is the
> robust fix. The automated recipe exposes this as the `tf_cpu_fix` stage.

### 3.10 Install Primus

```bash
cd ~/primus-jax-env   # or your $WORKSPACE_DIR
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout main
git submodule update --init third_party/maxtext/

# The JAX path does not install Primus' torch-oriented requirements.txt.
# Remove stale dataclasses backports that conflict on modern Python:
pip uninstall -y dataclasses dataclasses_json

# Primus' JAX runtime deps (also installed by the MaxText pre-train hook at
# launch time):
pip install -r requirements-jax.txt
```

If you already have a local Primus checkout (e.g. this repository), you can skip
the clone and just run the `git submodule update`, `pip uninstall`, and
`pip install -r requirements-jax.txt` steps from its root.

> **Which MaxText does Primus run?** At launch, Primus resolves the MaxText
> backend from the `MAXTEXT_PATH` environment variable, falling back to its own
> `third_party/maxtext` submodule. Set `MAXTEXT_PATH` to the checkout you
> installed dependencies into (Section 3.8) so the code and the installed deps
> match — the automated `env.sh` does this for you.

### 3.11 Build RCCL from source (into the ROCm tree)

v26.5 rebuilds RCCL from `rocm-systems` and drops the libraries into the ROCm
tree so JAX/XLA collectives use it. Requires the ROCm toolchain (`hipcc`) from
Section 3.4.

```bash
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git checkout 9e5e4084a4b8e1e86551b0eb054725c62354a926
cd projects/rccl
./install.sh -l --prefix build/ --amdgpu_targets="${PYTORCH_ROCM_ARCH}"
cp -r build/release/librccl* "$ROCM_PATH/lib/"
cd ../../..
```

---

## 4. Multi-node communication stack (UCX, OpenMPI)

These are only needed for **multi-node distributed** training. They build from
source and install into user-writable prefixes (no root needed, except the AINIC
`.deb` already handled in Section 2.4).

### 4.1 UCX

```bash
cd ~/primus-jax-env
UCX_VERSION="1.18.0"
wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz
mkdir -p ucx-${UCX_VERSION}
tar -zxf ucx-${UCX_VERSION}.tar.gz -C ucx-${UCX_VERSION} --strip-components=1
cd ucx-${UCX_VERSION}
mkdir build && cd build
../configure --prefix=$HOME/primus-jax-env/ucx-${UCX_VERSION}/install --with-rocm=${ROCM_PATH}
make -j 16 && make install
cd ../..

export UCX_INSTALL_DIR=$HOME/primus-jax-env/ucx-${UCX_VERSION}/install
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
../configure --prefix=$HOME/primus-jax-env/openmpi --with-ucx=${UCX_INSTALL_DIR} \
    --disable-oshmem --disable-mpi-fortran
make -j 16 && make install
cd ../..

export PATH="$HOME/primus-jax-env/openmpi/bin:${PATH}"
export LD_LIBRARY_PATH="$HOME/primus-jax-env/openmpi/lib:${LD_LIBRARY_PATH}"
```

> The Dockerfile installs OpenMPI under `/workspace`. The `$HOME/...` prefixes
> above keep it unprivileged.

---

## 5. Make the environment reproducible (activation script)

Many of the variables above (especially the ROCm paths and the `NVTE_*` / `XLA_*`
runtime flags) must be present in **every** shell that runs training. Append them
to the venv's activation script so they're set whenever you
`source ~/primus-jax-env/bin/activate`:

```bash
cat >> ~/primus-jax-env/bin/activate <<'EOF'

# ---- Primus JAX host environment ----
export PYTORCH_ROCM_ARCH="gfx942;gfx950"
export ROCM_AMDGPU_TARGETS="gfx942,gfx950"
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1
export ROCPROFILER_QUEUE_INTERPOSITION=0
export DEBUG_HIP_DYNAMIC_QUEUES=0

# v26.5: ROCm lives in the extracted tarball dir (Section 3.4), NOT a pip wheel.
export ROCM_PATH=$HOME/primus-jax-env/rocm
export ROCM_HOME=$ROCM_PATH
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
export PATH="$ROCM_PATH/lib:$ROCM_PATH/bin:$HIP_CLANG_PATH:$HOME/primus-jax-env/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$HOME/primus-jax-env/openmpi/lib"
export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib64"
export CPATH=$HIP_INCLUDE_PATH
export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"

# Point Primus at the MaxText checkout whose deps we installed
export MAXTEXT_PATH=$HOME/primus-jax-env/maxtext

# TransformerEngine (ROCm) runtime flags for JAX
export NVTE_ROCM_ARCH="$PYTORCH_ROCM_ARCH"
export NVTE_USE_ROCM=1
export NVTE_USE_HIPBLASLT=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export NVTE_CK_HOW_V3_BF16_CVT=2

# AMD GPU runtime knobs
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=VERSION

# XLA / JAX runtime settings (v26.5 uses .9)
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=0 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_enable_command_buffer=''"
# ---- end Primus JAX host environment ----
EOF
```

> The automated `tools/installation-jax/env.sh` sets all of the above (and
> auto-detects the GPU arch); prefer sourcing it over hand-editing `activate`.

---

## 6. Verify the installation

```bash
source ~/primus-jax-env/bin/activate   # or: source tools/installation-jax/env.sh

# GPUs visible to ROCm?
rocm-smi || ls -l /dev/kfd /dev/dri

# JAX sees the GPUs?
python -c "import jax; print('jax', jax.__version__); \
print('backend:', jax.default_backend()); \
print('devices:', jax.devices())"

# Key libraries import cleanly? (import transformer_engine.jax — this actually
# loads TE's shared lib, which is what fails on glibc < 2.38 with the prebuilt wheel)
python -c "import jax, jaxlib, flax, transformer_engine.jax; print('JAX/flax/TE OK')"

# Run a Primus MaxText training directly (no container)
cd ~/primus-jax-env/Primus   # or your Primus checkout
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-pretrain.yaml
```

`jax.default_backend()` should report `gpu` (ROCm), and `jax.devices()` should
list your AMD GPUs. Use `primus-cli direct` (not `container`) since you are
running on bare metal with everything installed in your environment.

---

## 7. Other important considerations

- **Python version**: MaxText requires Python ≥ 3.12. If your venv is older, the
  build will fail; get a 3.12 interpreter with `uv` (no sudo — see Section 2.1)
  and recreate it. The automated `setup.sh` does this for you (it provisions
  Python 3.12 via `uv` and recreates a too-old venv automatically).
- **GPU device access without root**: the user running training must be able to
  read/write `/dev/kfd` and `/dev/dri/*` — usually via membership in the `video`
  and `render` groups (`sudo usermod -aG video,render $USER`, then re-login).
- **Hugging Face access**: for gated models/tokenizers, export your token
  (`export HF_TOKEN=hf_xxx` and/or `huggingface-cli login`).
- **Install order is load-bearing (v26.5)**: MaxText → TensorFlow (from source)
  → ROCm JAX/PJRT/plugin → TransformerEngine → RCCL (from source). MaxText's
  `setup.sh` pulls in a stock `jax`/`tensorflow`, so the ROCm JAX must be
  installed *after* MaxText (to override it) and *before* TE (or `jaxlib` gets
  clobbered). The automated `setup.sh` stage order enforces this.
- **RDMA / multi-node limits**: high-performance networking typically requires
  `ulimit -l unlimited` and possibly hugepages, configured in
  `/etc/security/limits.conf` (admin help). Verify NICs with `ibv_devinfo` /
  `ibstat`. JAX uses the distributed coordinator (`JAX_COORDINATOR_IP` /
  `JAX_COORDINATOR_PORT`), which Primus sets from `MASTER_ADDR` / `MASTER_PORT`.
- **Version drift**: the ROCm release tarball, the JAX/PJRT/plugin versions, the
  TransformerEngine wheel, the TensorFlow/RCCL source revisions, and the MaxText
  branch are all pinned to one release (see the table in Section 1.1). If you
  change one, you may need to update the others. The Docker image is the
  authoritative, tested combination — match its `Dockerfile` ARGs when in doubt.
- **Automated scripts**: the manual steps in Section 3 are automated by
  [tools/installation-jax/](https://github.com/AMD-AGI/Primus/tree/main/tools/installation-jax)
  (see *Quick path* above). The multi-node networking stack in Section 4 is still
  manual.

---

## 8. Quick reference: minimal vs. full install

If you only need **single-node MaxText pretraining**, you can skip the
multi-node components:


| Component                                              | Needed for                          |
| ------------------------------------------------------ | ----------------------------------- |
| ROCm (tarball), JAX + PJRT/plugin, TransformerEngine (JAX) | Core MaxText training (install these) |
| MaxText + its deps, TensorFlow (from source), RCCL (from source) | Core MaxText training (install these) |
| Primus + `third_party/maxtext`                         | Running MaxText via Primus          |
| gcsfuse                                                | Reading data from GCS buckets only  |
| UCX, OpenMPI, AINIC                                     | Multi-node distributed training     |


Install the core rows first, validate with Section 6, then add the optional
components as your workload requires.
