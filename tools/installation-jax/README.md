# Primus JAX / MaxText environment in a venv (no docker, no sudo) — v26.5

Reproduces the Primus **v26.5 JAX training Dockerfile** in a Python virtual
environment. Same package pins as the Dockerfile, adapted for a bare-metal host
with no root and no containers.

This is the JAX/MaxText counterpart to `tools/installation/` (the PyTorch /
Megatron / TorchTitan stack). Use this one if you want to run the **JAX MaxText**
training backend of Primus.

## Why it differs from the Dockerfile

| Constraint | Dockerfile | Here |
|---|---|---|
| ROCm | release **tarball** → `/opt/rocm` | same tarball → **`$ROCM_DIR`** (`$PRIMUS_JAX_BASE/rocm`, user-writable, **no sudo**) |
| GPU arch | gfx942 + gfx950 | **auto-detected** from the host (`rocminfo`/KFD sysfs) for source builds/TE |
| Build dir | container FS | venv on **`$PRIMUS_JAX_BASE`** (persistent); transient sources on local `/tmp` |
| System deps | `apt install ...` | **skipped** (no sudo); documented in the guide's Section 2 |
| MaxText setup | `setup.sh` (runs `apt`, prompts for a venv) | Python steps only (apt is a one-time root action; venv already made) |

The key reason no sudo is required: the ROCm tarball is extracted into a
user-writable dir (`$ROCM_DIR`) instead of `/opt/rocm`, so we don't depend on
system ROCm or apt for the ROCm toolchain itself.

> **v26.5 vs v26.4:** ROCm moved from pip `rocm-sdk-devel` **nightly** wheels to a
> pinned release **tarball** (`repo.amd.com/.../therock-dist-linux-multiarch-7.14.0.tar.gz`);
> JAX 0.10.2 → **0.10.0** (`repo.amd.com/rocm/whl-multi-arch/`); TE → **2.15.0.dev0**;
> TensorFlow is now **built from source** (2.21 CPU, bazel) and **RCCL** is rebuilt
> from source; MaxText → **release/v26.5** (Primus `main` supports its 2-value API).

> **Python 3.12+ required (3.12 preferred).** MaxText requires Python ≥ 3.12 (the
> PyTorch recipe works on 3.10; this one does not). 3.12 is preferred because the
> prebuilt ROCm wheels are `cp312`. You do **not** need `sudo` or a PPA:
> - The scripts use `python3.12` (preferred) or `python3.13` if on PATH.
> - Otherwise, if [`uv`](https://docs.astral.sh/uv/) is installed, `env.sh`
>   auto-detects a uv-managed `>= 3.12` interpreter and `setup.sh` runs
>   `uv python install 3.12` for you when none exists yet.
> - No `uv`? Install it once (no root) and re-run:
>   `python3 -m pip install --user uv` (or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
> - To force a specific interpreter: `export PRIMUS_PYTHON=/path/to/python3.12`.
>
> On Ubuntu 22.04, `apt install python3.12` fails (jammy has no such package) —
> use the `uv` path above instead.

> **Host OS: Ubuntu 24.04 / glibc ≥ 2.38 recommended, but 22.04 works.** The
> prebuilt `transformer_engine_rocm_jax` wheel is built against the Dockerfile's
> `ubuntu:24.04` base and needs `glibc ≥ 2.38` + `GLIBCXX_3.4.32` (GCC 13/14).
> On Ubuntu 22.04 (glibc 2.35) the prebuilt TE wheel can't load
> (`version 'GLIBC_2.38' not found`). **`setup.sh` handles this automatically:**
> the `te` stage detects the host glibc and builds TransformerEngine from source
> when it is < 2.38, so `bash setup.sh` works on both 22.04 and 24.04. (Only the
> prebuilt TE wheel needs glibc ≥ 2.38; the ROCm JAX/PJRT wheels load on glibc
> 2.35.) Check with `ldd --version`.

## Run it

```bash
cd tools/installation-jax
# Default base is /it-share-4/envs/primus-jax-env (persistent shared disk).
# This default is site-specific; override it for your host:
#   export PRIMUS_JAX_BASE=/some/big/disk/primus-jax-env
bash setup.sh                 # all default stages
```

If any stage fails the script stops immediately and prints which stage failed;
fix the cause and re-run just that stage. Stages are idempotent:

```bash
bash setup.sh --list          # show stages
bash setup.sh te              # reinstall just TransformerEngine
bash setup.sh venv rocm jax   # venv + ROCm + JAX only
```

## Use the environment afterward

```bash
# Use the SAME PRIMUS_JAX_BASE you built with (export it first if you overrode the default)
source tools/installation-jax/env.sh   # activates venv + sets ROCm / NVTE / XLA env vars
python -c "import jax; print(jax.devices())"

# Primus is checked out at $WORKSPACE_DIR/Primus; MaxText at $MAXTEXT_DIR.
cd "$WORKSPACE_DIR/Primus"
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-pretrain.yaml
```

`env.sh` exports `MAXTEXT_PATH=$MAXTEXT_DIR`, so Primus runs the same MaxText
checkout we installed the dependencies for.

## Stages (default order, v26.5)

`venv` → `rocm` → `maxtext` → `tf_source` → `jax` → `te` → `primus`
→ `jaxreqs` → `rccl` → `manifest`

- **venv** — create the venv (Python ≥ 3.12) and bootstrap `cmake`/`ninja`/`uv`.
- **rocm** — download + extract the TheRock ROCm release tarball into `$ROCM_DIR`
  (+ `amdsmi`).
- **maxtext** — clone ROCm/MaxText (`release/v26.5`) and install its deps (the
  Python part of MaxText's `setup.sh`) + the editable MaxText package.
- **tf_source** — build **tensorflow-cpu 2.21 from source** (bazel, ~30–60 min);
  fixes the ROCm-vs-TF LLVM symbol clash (SIGSEGV) and drops bundled NCCL.
- **jax** — `jax`/`jaxlib` 0.10.0 + the ROCm `jax_rocm7_pjrt` / `jax_rocm7_plugin`
  (installed after MaxText to override its stock jax).
- **te** — prebuilt `transformer_engine_rocm_jax` wheel (+ `flax`, `pydantic`, ...);
  auto-falls-back to a from-source build (`te_source`) on glibc < 2.38 hosts.
- **primus** — clone Primus, init the `third_party/maxtext` submodule, drop the
  stale `dataclasses` backports.
- **jaxreqs** — install Primus' `requirements-jax.txt` (loguru, wandb, ...).
- **rccl** — build **RCCL from source** and install it into `$ROCM_PATH/lib`.
- **manifest** — dump `pip list` / `env` for reproducibility.

Optional / alternative stages:

- **te_source** — force the from-source TransformerEngine build regardless of
  glibc. Normally unnecessary: the default `te` stage **auto-detects the host
  glibc** and builds from source when it is < 2.38 (e.g. Ubuntu 22.04), where the
  prebuilt wheel fails to load with `version 'GLIBC_2.38' not found`. Heavy build
  (~30–60 min, compiles CK fused-attention kernels). Only the prebuilt TE wheel
  needs glibc ≥ 2.38 — the ROCm JAX/PJRT wheels load fine on glibc 2.35.
- **tf_cpu_fix** — lighter alternative to `tf_source`: `pip install tensorflow-cpu`
  instead of the bazel build (avoids the bundled-NCCL clash; may still hit the
  LLVM-symbol SIGSEGV on some ROCm 7.14 configs).

```bash
# Ubuntu 22.04 (glibc < 2.38): `bash setup.sh` already builds TE from source
# automatically. To also skip the heavy tf_source bazel build, swap in tf_cpu_fix:
bash setup.sh venv rocm maxtext tf_cpu_fix jax te primus jaxreqs rccl manifest
```

> **MaxText v26.5 and Primus.** v26.5 MaxText uses a **2-value**
> `initialize()`/`run()` API; v26.4 and earlier used a **3-value** API. Primus
> `main` supports **both** — `MaxTextPretrainTrainer` forwards `initialize()`'s
> tuple verbatim to `run()` (fix #912) — so this recipe's pinned `release/v26.5`
> trains out of the box. Use `MAXTEXT_BRANCH=release/v26.4 bash setup.sh maxtext`
> only if you deliberately want to pin an older MaxText release.

## What is SKIPPED (needs sudo / apt — not reproducible here)

- **System packages** (build toolchain, `numactl`, `gcsfuse`, RDMA/verbs libs):
  a one-time root action, documented in Section 2 of the guide.
- **AINIC** (`add-apt-repository`, `libionic-dev`): apt-only. Skipped.
- **UCX + OpenMPI**: autotools source builds needed only for multi-node RDMA.
  Single-node training works without them; see Section 4 of the guide.
- **gcsfuse**: only needed to mount GCS buckets for data; not required for
  synthetic-data or local-data runs.

## Notes / gotchas vs. the Dockerfile

- **Stage order matters (v26.5):** `maxtext` → `tf_source` → `jax` → `te`. MaxText's
  `setup.sh` pulls in a stock `jax`/`tensorflow`; TF is then rebuilt from source and
  the ROCm JAX/plugin is installed after (overriding MaxText's), and TE must come
  after JAX or `jaxlib` gets clobbered. The stage order enforces this.
- **TensorFlow + RCCL are built from source** in the default flow (matching the
  v26.5 image). These are the two long builds; use `tf_cpu_fix` if you want to skip
  the TF bazel build.
- **No FlashAttention/aiter/torch stack.** The JAX MaxText path does not build the
  PyTorch kernel libraries; attention fusion comes from `transformer_engine_rocm_jax`
  + XLA.
- **Two MaxText checkouts in the Docker image** (`/workspace/maxtext` and
  `Primus/third_party/maxtext`) are collapsed here: we install deps from one
  checkout and point `MAXTEXT_PATH` at it.
- **Runtime-validated (single-node).** The default flow — with `te_source`
  substituted for the prebuilt `te` — has been run end-to-end for single-node
  MaxText pretraining on **gfx942 / Ubuntu 22.04 (glibc 2.35) / Python 3.12**.
  On Ubuntu 24.04 (glibc ≥ 2.38) the prebuilt `te` wheel works directly. The
  multi-node networking stack (UCX/OpenMPI/AINIC) is still not exercised here.

Treat the reference `Dockerfile` as the authoritative, tested version
combination; if you bump one pin you may need to bump the others.
