# Primus environment in a venv (no docker, no sudo)

Reproduces the Primus training Dockerfile in a Python virtual environment on
this server. Same package pins/commits as the Dockerfile, adapted for the
constraints of this machine.

## Why it differs from the Dockerfile

| Constraint | Dockerfile | Here |
|---|---|---|
| Python | 3.12 (Ubuntu 24.04) | **3.10** (Ubuntu 22.04) — cp310 wheels exist |
| ROCm | pip `rocm-sdk-devel` | same (self-contained, **no sudo needed**) |
| GPU arch | gfx942 + gfx950 | **auto-detected** from the host (`rocminfo`/KFD sysfs); builds only what's present → faster builds |
| Build dir | container FS | venv on **`/it-share-4/envs/primus-env`** (persistent, 1.1 TB free); build sources on local `/tmp` |
| System deps | `apt install ...` | **skipped** (no sudo); pip provides what's needed |

The key reason no sudo is required: the `rocm-sdk-devel` pip wheel ships a full
ROCm toolchain inside the venv, so we don't depend on system ROCm or apt.

## Run it

```bash
cd tools/installation
# Default base is /it-share-4/envs/primus-env (persistent shared disk).
# This default is site-specific; override it for your host:
#   export PRIMUS_BASE=/some/big/disk/primus-env
bash setup.sh                 # all default stages
```

This takes a long time (source builds: TransformerEngine, aiter, Primus-Turbo,
etc.). If any stage fails the script stops immediately and prints which stage
failed; fix the cause and re-run just that stage. Stages are idempotent:

```bash
bash setup.sh --list          # show stages
bash setup.sh te              # rebuild just TransformerEngine
bash setup.sh venv torch      # venv + torch only
```

## Use the environment afterward

```bash
# Use the SAME PRIMUS_BASE you built with (export it first if you overrode the default)
source tools/installation/env.sh   # activates venv + sets all ROCm/NVTE env vars
python -c "import torch; print(torch.cuda.is_available())"
# Primus is checked out at $WORKSPACE_DIR/Primus
```

## Stages (default order)

`venv` → `torch` → `flash_attn` → `te` → `torchtune` → `torchao` → `pydeps`
→ `grouped_gemm` → `causal_conv1d` → `mamba` → `primus` → `aiter` → `turbo`
→ `boto` → `manifest`

Optional: `torchrec` (DLRM/recommendation stack).

## What is SKIPPED (needs sudo / apt — not reproducible here)

- **AINIC** (`add-apt-repository`, `libionic-dev`): apt-only. Skipped.
- **UCX + OpenMPI + rocSHMEM**: autotools source builds need `libtool`
  (missing, apt-only) and RDMA dev headers. Only needed for multi-node
  rocSHMEM/GDA; single-node training works without them. Skipped.
- **DLRM / FBGEMM / Flux**: not part of the default Primus training path.
  `torchrec` is provided as an optional stage; FBGEMM needs more apt deps.
- Misc apt runtime packages (`numactl`, `pciutils`, `libz3-dev`, `ffmpeg`,
  `gfortran`): not installed. Install via sudo later if a specific workload
  needs them.

## Fixes applied (gotchas vs. the Dockerfile)

These were needed because of Python 3.10 / no-sudo and are baked into the scripts:

- **mamba built with pip, not `python setup.py install`.** The legacy
  `easy_install` path ignores pip-installed packages and re-fetches the *latest*
  of every unpinned dep as `.egg`s — it clobbered `transformers` (→5.x), removed
  `accelerate`/`trl`, and pulled NVIDIA CUDA packages. `stage_mamba` now uses
  `pip install --no-build-isolation .` which respects the pins.
- **`apache-tvm-ffi` pinned to 0.1.6.** `mamba_ssm` → `tilelang 0.1.8` declares
  it unpinned; the latest (0.1.12) has a registry bug
  (`attribute '__dict__' of 'type' objects is not writable`). 0.1.6 works.
- **`libz3.so` from pip.** `mamba_ssm` needs it at runtime (Dockerfile uses apt
  `libz3-dev`, unavailable without sudo). `stage_pydeps` installs the `z3-solver`
  pip package (which ships `libz3.so`), and `env.sh` adds its directory to
  `LD_LIBRARY_PATH`.
- **Megatron `helpers_cpp` compiled.** Megatron's dataset indexing imports a
  pybind11 C++ module (`megatron.core.datasets.helpers_cpp`) that must be built,
  or training dies with `ModuleNotFoundError: ...helpers_cpp` (surfacing as
  `MockGPTDataset failed to build`). `stage_primus` runs `make` in
  `megatron/core/datasets` for both the workspace and `~/.cache/Primus` copies.
- **Megatron** is not pip-installed; it's bundled at
  `$WORKSPACE_DIR/Primus/third_party/Megatron-LM` and added to the path by Primus
  at runtime (or set `PYTHONPATH` yourself for standalone `import megatron`).

## Caveats

- **Persistence**: the venv lives on `/it-share-4/envs` (persistent). Transient
  build sources go to local `/tmp` for speed and are deleted after each build.
- **Disk**: full build needs tens of GB; the shared disk has ~1.1 TB free.
- **GPU arch is auto-detected** (`env.sh` reads `rocminfo`, else the kernel KFD
  sysfs `gfx_target_version`), and `stage_torch` installs the matching device
  wheels for whatever it finds (gfx942 and/or gfx950). To force a target — e.g.
  to build a portable env for both — export it before running:
  `export PYTORCH_ROCM_ARCH="gfx942;gfx950"`.
