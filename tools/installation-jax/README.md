# Primus JAX / MaxText environment in a venv (no docker, no sudo) — v26.7

Reproduces the Primus **v26.7 JAX training Dockerfile**
([`Dockerfile.jax-v26.7`](../../.github/workflows/docker-release/Dockerfile.jax-v26.7))
in a Python virtual environment. Same package pins as the Dockerfile, adapted for
a bare-metal host with no root and no containers.

This is the JAX/MaxText counterpart to `tools/installation/` (the PyTorch /
Megatron / TorchTitan stack). Use this one if you want to run the **JAX MaxText**
training backend of Primus.

## Why it differs from the Dockerfile

| Constraint | Dockerfile | Here |
|---|---|---|
| ROCm | pip `rocm-sdk-*` 10.0.0 → `/opt/venv/.../_rocm_sdk_devel` | same wheels → in-venv `_rocm_sdk_devel` (**no sudo**) |
| GPU arch | gfx942 + gfx950 | **auto-detected** from the host (`rocminfo`/KFD sysfs) for source builds/TE |
| Build dir | container FS | venv on **`$PRIMUS_JAX_BASE`** (persistent); transient sources on local `/tmp` |
| System deps | `apt install ...` | **skipped** (no sudo); documented in the guide's Section 2 |
| MaxText setup | `setup.sh` (runs `apt`, prompts for a venv) | Python steps only (apt is a one-time root action; venv already made) |
| `LD_LIBRARY_PATH` | blanked at the end of the image | blanked at runtime (`PRIMUS_JAX_KEEP_ROCM_LD=1` for RCCL/TE source builds) |

The key reason no sudo is required: the `rocm-sdk-devel` pip wheel ships a full
ROCm toolchain inside the venv, so we don't depend on system ROCm or apt.

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

> **Host OS: Ubuntu 22.04 or 24.04.** v26.7 lowered the TE glibc floor: the native
> code ships as `transformer_engine_rocm10`, a `manylinux_2_28` wheel, so the
> prebuilt path now works on **glibc ≥ 2.28** — Ubuntu 22.04 (2.35) included. v26.6's
> wheels were `ubuntu:24.04` builds that needed `glibc ≥ 2.38` and failed to load on
> 22.04 with `version 'GLIBC_2.38' not found`. **`setup.sh` still handles this
> automatically:** the `te` stage checks the host glibc and only falls back to the
> from-source build below 2.28. Check with `ldd --version`.

## Run it

```bash
cd tools/installation-jax
# PRIMUS_JAX_BASE is REQUIRED (no default). Point it at a directory you can write
# to with tens of GB free — the venv, ROCm SDK, and checkouts all live here:
export PRIMUS_JAX_BASE=/some/big/disk/primus-jax-env
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
# Set the SAME PRIMUS_JAX_BASE you built with (required — env.sh errors without it)
export PRIMUS_JAX_BASE=/some/big/disk/primus-jax-env
source tools/installation-jax/env.sh   # activates venv + sets ROCm / NVTE / XLA env vars
python -c "import jax; print(jax.devices())"

# Primus is checked out at $WORKSPACE_DIR/Primus; MaxText at $MAXTEXT_DIR.
cd "$WORKSPACE_DIR/Primus"
./primus-cli direct -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama2_7B-bf16-pretrain.yaml
```

`env.sh` exports `MAXTEXT_PATH=$MAXTEXT_DIR`, so Primus runs the same MaxText
checkout we installed the dependencies for.

## Stages (default order, v26.7)

`venv` → `rocm` → `maxtext` → `tf_source` → `jax` → `te` → `primus`
→ `jaxreqs` → `manifest`

- **venv** — create the venv (Python ≥ 3.12) and bootstrap `cmake`/`ninja`/`uv`.
- **rocm** — pip-install TheRock `rocm-sdk-*` 10.0.0 (core/devel/libraries +
  per-arch device wheels) and run `rocm-sdk init`.
- **maxtext** — clone ROCm/MaxText (`release/v26.7`) and install its deps (the
  Python part of MaxText's `setup.sh`) + the editable MaxText package.
- **tf_source** — build **tensorflow-cpu 2.21 from source** (bazel, ~30–60 min);
  fixes the ROCm-vs-TF LLVM symbol clash (SIGSEGV) and drops bundled NCCL.
- **jax** — `jax`/`jaxlib` 0.11.0 + `jax_rocm10_pjrt` / `jax_rocm10_plugin`
  `0.11.0+rocm10.0.0` from PyPI (installed after MaxText to override its stock jax).
- **te** — prebuilt `transformer_engine_rocm_jax==2.17.0+rocm10.0.0`
  (+ `flax==0.12.8`, `pydantic`, ...); auto-falls-back to a from-source build
  (`te_source`) on glibc < 2.28 hosts.
- **primus** — clone Primus, init the `third_party/maxtext` submodule, drop the
  stale `dataclasses` backports.
- **jaxreqs** — install Primus' `requirements-jax.txt` and the v26.7 CVE-fix pins.
- **manifest** — dump `pip list` / `env` for reproducibility.

Optional stages:

- **rccl** — build **RCCL from source** into `$ROCM_PATH/lib`. Not part of the
  default flow from v26.7: the ROCm 10.0.0 pip SDK ships RCCL 2.30.4 and the
  published image no longer overrides it. Run this only to reproduce v26.6, or on
  a host that needs the `rocm-systems` net-ib fix (ROCM-27881).
- **te_source**, **tf_cpu_fix** — see `setup.sh --list`.

Optional / alternative stages:

- **te_source** — force the from-source TransformerEngine build regardless of
  glibc. Normally unnecessary: the default `te` stage **auto-detects the host
  glibc** and builds from source only when it is < 2.28. Heavy build (~30–60 min,
  compiles CK fused-attention kernels). From v26.7 the prebuilt wheel covers Ubuntu
  22.04 as well, so this fallback is rare.
- **tf_cpu_fix** — lighter alternative to `tf_source`: `pip install tensorflow-cpu`
  instead of the bazel build (avoids the bundled-NCCL clash; may still hit the
  LLVM-symbol SIGSEGV on some ROCm 7.14 configs).

```bash
# To skip the heavy tf_source bazel build, swap in tf_cpu_fix:
bash setup.sh venv rocm maxtext tf_cpu_fix jax te primus jaxreqs manifest
```

> **MaxText v26.7 and Primus.** Override `MAXTEXT_BRANCH` only if you deliberately
> need a different MaxText release.

## What is SKIPPED (needs sudo / apt — not reproducible here)

- **System packages** (build toolchain, `numactl`, `gcsfuse`, RDMA/verbs libs):
  a one-time root action, documented in Section 2 of the guide.
- **AINIC** (`add-apt-repository`, `libionic-dev`): apt-only. Skipped.
- **UCX + OpenMPI**: **optional for MaxText** — JAX uses its own distributed
  coordinator + RCCL, not `mpirun`. Carried over from the reference image for
  other/MPI-launched JAX workloads; skipped here. See Section 4 of the guide.
- **gcsfuse**: only needed to mount GCS buckets for data; not required for
  synthetic-data or local-data runs.

## Notes / gotchas vs. the Dockerfile

- **Stage order matters (v26.7):** `maxtext` → `tf_source` → `jax` → `te`. MaxText's
  `setup.sh` pulls in a stock `jax`/`tensorflow`; TF is then rebuilt from source and
  the ROCm JAX/plugin is installed after (overriding MaxText's), and TE must come
  after JAX or `jaxlib` gets clobbered. The stage order enforces this.
- **TensorFlow is built from source** in the default flow, matching the v26.7
  image. It is the one long build; use `tf_cpu_fix` to skip the bazel step. RCCL is
  no longer built — v26.7 takes it from the ROCm 10.0.0 pip SDK.
- **Runtime `LD_LIBRARY_PATH` is empty**, matching the image's TE 2.17 + JAX
  segfault fix. RCCL/TE source builds set `PRIMUS_JAX_KEEP_ROCM_LD=1`.
- **TE from source always compiles gfx942+gfx950.** TE 2.17's HipKittens GEMM
  fails to link if CMake only sees gfx942.
- **No FlashAttention/aiter/torch stack.** The JAX MaxText path does not build the
  PyTorch kernel libraries; attention fusion comes from `transformer_engine_rocm_jax`
  + XLA.
- **Two MaxText checkouts in the Docker image** (`/workspace/maxtext` and
  `Primus/third_party/maxtext`) are collapsed here: we install deps from one
  checkout and point `MAXTEXT_PATH` at it.
- **Runtime-validated.** The default flow — with `te_source` substituted for the
  prebuilt `te` — has been run end-to-end for single-node MaxText pretraining on
  **gfx942 / Ubuntu 22.04 (glibc 2.35) / Python 3.12**, and a **2-node** run has
  also been exercised (over the JAX distributed coordinator + RCCL, no
  UCX/OpenMPI). From v26.7 the prebuilt `te` wheel works directly on 22.04 too.

Treat the reference `Dockerfile` as the authoritative, tested version
combination; if you bump one pin you may need to bump the others.
