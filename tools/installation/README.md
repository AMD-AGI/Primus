# Primus environment in a venv (no docker, no sudo)

Reproduces the Primus **v26.9** training image in a Python virtual environment on
a bare-metal host. Derived from
[`.github/workflows/docker-release/Dockerfile.primus-v26.9`](../../.github/workflows/docker-release/Dockerfile.primus-v26.9),
using the same package pins and commits, adapted for the constraints of a machine
where we have no root.

A handful of places deliberately diverge from the Dockerfile, because copying it
exactly produces an environment that does not work outside the container. Each one
is listed under [Fixes applied](#fixes-applied-gotchas-vs-the-dockerfile) with the
failure it avoids — worth reading before "correcting" any of them back.

## Python 3.12

The environment is built on Python 3.12 because that is what the image uses and
what every source build here has been validated against.

Ubuntu 22.04 hosts only have `python3.10`, so `setup.sh` provisions a standalone
CPython 3.12 with [`uv`](https://docs.astral.sh/uv/). No sudo, no apt. Interpreters
land in `$PRIMUS_BASE/python`, keeping the whole environment in one place.

You do not need `uv` beforehand: if it is missing, `setup.sh` downloads it into
`$PRIMUS_BASE/bin` from `astral.sh`. Pre-install it yourself if that download is
blocked, or if you would rather not pipe a script into a shell:

```bash
python3 -m pip install --user uv     # ~/.local/bin, one of the paths setup.sh searches
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

`setup.sh` looks for `uv` on `PATH`, then in `$PRIMUS_BASE/bin`, then `~/.local/bin`.
If you cannot install it at all, supply your own interpreter instead and it is never
used: `PRIMUS_PYTHON=/path/to/python3.12 bash setup.sh`.

> **Migrating from an older venv:** a Python 3.10 venv cannot be
> upgraded in place. `setup.sh` detects the mismatch and stops with instructions;
> remove the old venv and rebuild:
> `rm -rf "$PRIMUS_BASE/venv" && bash setup.sh`

## TransformerEngine and flash-attention are built from source

Both are built from source, as the image has done since v26.8, so there is no
wheel path and no glibc floor to check. Budget for it: these two are the longest
stages after aiter.

- **TransformerEngine** is built at commit `023639b5…` of `ROCm/TransformerEngine`.
  TE's QoLA tool first checks out the aiter commit TE's manifest pins into
  `$WORKSPACE_DIR/deps/te-aiter`, which is kept afterwards (the image keeps
  `/workspace/deps/te-aiter`) and exported as `NVTE_AITER_SOURCE_DIR` by
  `env.sh`. CK fused attention is JIT-compiled at run time (`NVTE_CK_JIT=1`)
  rather than prebuilt into the wheel. `stage_te`
  applies the same four source patches as the Dockerfile, and on gfx950 it
  also assembles the backward-attention payload from `mawad-amd/bwd-attn-asm`
  into that tree; on a gfx942-only host that step is skipped, since nothing
  loads it.
- **flash-attention** is built from the `ROCm/flash-attention` fork at
  `5301a359…` (tag `v2.8.4.1-cktile`, reported as `flash_attn 2.8.4`). The
  PyPI `flash-attn` package that v26.6 and v26.7 used compiles the full CK-tile
  instance set, which the image notes OOM-kills a single translation unit; the
  fork's curated set builds cleanly.

Both builds target only the GPUs detected on the host (see `PYTORCH_ROCM_ARCH`
below), which keeps them considerably faster than the image's two-arch build.
After installing, `stage_te` imports `transformer_engine.pytorch` and fails
loudly if it cannot load, rather than letting the problem surface during
training.

## Why it differs from the Dockerfile

| Constraint | Dockerfile | Here |
|---|---|---|
| Python | 3.12 (Ubuntu 24.04) | 3.12, auto-provisioned by `uv` (host has only 3.10) |
| ROCm | pip `rocm-sdk-devel` | same (self-contained, **no sudo needed**) |
| GPU arch | gfx942 + gfx950 | **auto-detected** from the host (`rocminfo`/KFD sysfs); builds only what's present → faster builds |
| Build dir | container FS | venv under **`$PRIMUS_BASE`** (persistent, you choose it); build sources on local `/tmp` |
| System deps | `apt install ...` | **skipped** (no sudo); pip provides what's needed |
| apex | unpinned | pinned to the exact build the image resolved — see below |
| Installer | `uv pip` | `pip`, with a constraint file that pins the ROCm torch/triton (see [Fixes applied](#fixes-applied-gotchas-vs-the-dockerfile)) |

The key reason no sudo is required: the `rocm-sdk-devel` pip wheel ships a full
ROCm toolchain inside the venv, so we don't depend on system ROCm or apt.

### apex is pinned to what the image resolved

The Dockerfile pins `torch`, `torchaudio` and `torchvision` exactly, but leaves
apex unpinned, so its version is whatever the nightly index offered on build day.
`setup.sh` pins it to the version read out of the `rocm/primus:v26.9` image
(`apex 1.16.0a0+rocm10.3.0a20261029` — a ROCm 10.3 nightly build, next to the
image's ROCm 10.2 nightly torch), so a later index cannot hand bare metal a
different build.

## Run it

`PRIMUS_BASE` is required and has no default — the right location is
site-specific, and quietly falling back to some other host's path only hides the
problem. Point it at a writable directory on a disk with tens of GB free; it
holds the venv, the provisioned interpreter, and the Primus/aiter checkouts.

```bash
cd tools/installation
export PRIMUS_BASE="$HOME/envs/primus-env"   # required, pick your own location
bash setup.sh                                # all default stages
```

The build compiles TransformerEngine, flash-attention, aiter and Primus-Turbo
from source, so budget hours rather than minutes. Running it detached avoids
losing it to a dropped connection:

```bash
nohup bash setup.sh > ~/primus-setup.log 2>&1 &
tail -f ~/primus-setup.log
```

If any stage fails the script stops immediately and prints which stage failed.
Stages are idempotent, so fix the cause and re-run just that one — exporting the
same `PRIMUS_BASE` again, since that is how it finds the venv:

```bash
bash setup.sh --list          # show stages (works without PRIMUS_BASE)
bash setup.sh te              # rebuild just TransformerEngine
bash setup.sh venv torch      # venv + torch only
```

When cherry-picking stages, `te` and `flash_attn` need `torch` to have run first:
both are built against the installed torch with `--no-build-isolation`.

## Use the environment afterward

Export the SAME `PRIMUS_BASE` you built with, then source `env.sh` — it activates
the venv and sets every ROCm/NVTE variable. Without `PRIMUS_BASE` it stops with an
error rather than guessing.

```bash
export PRIMUS_BASE="$HOME/envs/primus-env"
source tools/installation/env.sh
python -c "import torch; print(torch.cuda.is_available())"
# Primus is checked out at $WORKSPACE_DIR/Primus
```

## Stages (default order)

`venv` → `torch` → `te` → `flash_attn` → `torchtune` → `torchao` → `pydeps`
→ `grouped_gemm` → `causal_conv1d` → `mamba` → `primus` → `aiter` → `turbo`
→ `boto` → `cleanup` → `manifest`

Optional: `torchrec` (DLRM/recommendation stack).

`te` runs before `flash_attn`, matching the Dockerfile.

## What changed for v26.9

- **A ROCm 10.2 nightly and PyTorch 2.16.** `rocm-sdk-*` `10.2.0a20261020` and
  torch `2.16.0a0+rocm10.2.0a20261020`, now from the `nightly.repo.amd.com` core
  and pytorch indexes (v26.8 used `stable.repo.amd.com`); torchaudio
  `2.11.0a0+rocm10.2.0a20261020`, torchvision `0.30.0a0+rocm10.2.0a20261020`,
  apex `1.16.0a0+rocm10.3.0a20261029` (see [above](#apex-is-pinned-to-what-the-image-resolved)).
  `patch_torch_cpp_extension` applies the image's fix for torch 2.16's hipify
  warning, which crashes on the `pathlib.Path` sources Primus-Turbo passes.
- **New source patches from the image.** `stage_te` drops the quoted typeinfo
  patterns from QoLA's export script (ROCm LLVM 24 lld rejects them);
  `stage_turbo` removes Primus-Turbo's stale `BLOCK_DMODEL` autotune key and
  fla's stale `BT` key, both of which Triton 3.9 rejects.
- **Updated pins:** Primus `db1a4160…` (on `release/v26.9`, the commit the
  Dockerfile pins), TransformerEngine `023639b5…`, aiter `778d74d5…`,
  Primus-Turbo `de8fa6ca…` (`0.6.1.dev4`), `flydsl==0.3.4.1` (what the image
  resolves; see [Fixes applied](#fixes-applied-gotchas-vs-the-dockerfile)),
  `datasets==5.0.1` and `fsspec==2026.6.0`.
- **CVE pins, as in the image:** `filelock==4.2.1` (re-pinned with `--no-deps`
  after Primus' requirements, which cap it below 4; the image uses
  `uv pip install --override`), `hydra-core==1.3.8`, and `mlflow`,
  `mlflow-skinny` and `mlflow-tracing` `3.17.2` (`--no-deps`).
  `cryptography==50.0.0` is unchanged.
- **Runtime:** `env.sh` also exports `TORCH_DIST_USE_NCCL2=0`, which the image
  sets alongside [Primus #1286](https://github.com/AMD-AGI/Primus/pull/1286) to
  resolve memory issues.

## What is SKIPPED (needs sudo / apt — not reproducible here)

- **AINIC** (`add-apt-repository`, `libionic-dev`): apt-only. Skipped.
- **UCX + OpenMPI**: autotools source builds needing `libtool` and RDMA dev
  headers, both apt-only. Single-node training works without them. Skipped.
  (v26.5 removed rocSHMEM from the image entirely, so nothing else needs them.)
- **DeepEP internode in Primus-Turbo**: needs rocSHMEM, so `stage_turbo` disables
  it by pointing `ROCSHMEM_HOME` at a non-existent path, and the build prints a
  warning saying internode DeepEP is off. This is deliberate — left to
  auto-detect, Primus-Turbo mistakes the pip ROCm SDK directory for a rocSHMEM
  install (it has rocshmem headers and device bitcode, but no host
  `librocshmem.a`) and the link then fails. Intranode DeepEP is unaffected.
  Export a real `ROCSHMEM_HOME` to opt back in.
- **MLPerf `primus_mllog` / `mlperf-common`**: the image installs it; it is not
  part of the core training path, so it is not installed here.
- **DLRM / FBGEMM / Flux**: not part of the default Primus training path.
  `torchrec` is provided as an optional stage; FBGEMM additionally needs apt
  `libtbb-dev`.
- Misc apt runtime packages (`numactl`, `pciutils`, `libz3-dev`, `ffmpeg`,
  `gfortran`): not installed. Install via sudo later if a specific workload
  needs them.

## Fixes applied (gotchas vs. the Dockerfile)

These are needed because of the no-sudo/bare-metal setting and are baked into the
scripts:

- **pip is constrained so it cannot swap the ROCm GPU stack for a CUDA one.**
  After installing torch, `setup.sh` writes `$PRIMUS_BASE/pip-constraints.txt`
  pinning the installed `torch`/`triton`, and every later `pip install` passes
  `-c`. Many packages depend on a bare `torch`; a resolver that decides to
  "upgrade" it silently replaces the whole GPU stack with `nvidia-*` wheels.
  With the constraint, such an attempt fails loudly instead.
- **Primus-Turbo is installed with `--no-deps`, keeping ROCm's triton.** Up to
  v26.7 its `setup.py` hard-pinned upstream `triton==3.7.0`, and letting that wheel
  replace ROCm's triton breaks the environment: ROCm's HIP runtime already loads
  its own `libLLVM.so` from the pip SDK, and the upstream triton wheel bundles a
  second, statically linked LLVM, so importing it segfaults inside LLVM's static
  initialisers — taking down `torch._dynamo`, `aiter`, `torchao` and `mamba_ssm`
  with it. Since v26.8 it asks for `triton>=3.7.0`, which ROCm's
  `3.9.0+git…rocm10.2.0a20261020` satisfies, but `stage_turbo` keeps `--no-deps` and
  supplies the real runtime requirements (`scipy`, `flydsl==0.3.4.1`) explicitly.
  This is a deliberate, tested deviation from the image.
- **mamba built with pip, not `python setup.py install`.** The legacy
  `easy_install` path ignores pip-installed packages and re-fetches the *latest*
  of every unpinned dep as `.egg`s — it clobbered `transformers` (→5.x), removed
  `accelerate`/`trl`, and pulled NVIDIA CUDA packages. `stage_mamba` uses
  `pip install --no-build-isolation .` which respects the pins.
- **`NVTE_CK_IS_V3_ATOMIC_FP32` defaults to `1` on gfx942, not the Dockerfile's `0`.**
  Paired with `NVTE_CK_USES_BWD_V3=1`, turning fp32 atomics off makes the CK v3
  backward attention kernel emit Inf gradients on MI300X/MI325X, killing training at
  the first step. Primus's own tuning guide already prescribes fp32 atomics for these
  GPUs (the MI300X/MI325X block in
  [docs/02-user-guide/end-to-end-training-recipes.md](../../docs/02-user-guide/end-to-end-training-recipes.md)),
  so `env.sh` applies it from the detected architecture and leaves `0` for gfx950,
  which the Dockerfile value targets. `NVTE_CK_USES_BWD_V3` itself stays at the
  Dockerfile's `1`; the atomic mode is what makes it safe.
- **Megatron's `helpers_cpp` is built with an explicit `LIBEXT`, for every
  checkout.** Its Makefile derives the output filename from `python3-config
  --extension-suffix`, but a venv does not ship `python3-config`, so it silently
  falls back to the system interpreter and writes a `cpython-310` name that a 3.12
  venv will never import. `stage_primus` passes `LIBEXT` from the venv's own
  `EXT_SUFFIX`, and does so for the workspace clone, `~/.cache/Primus`, *and* the
  checkout containing these scripts — training is usually launched from the latter.
  It also deletes any pre-existing extension of the target name first: a checkout
  that has been bind-mounted into the training container may hold a root-owned
  Ubuntu 24.04 build that needs `GLIBCXX_3.4.32`, cannot load here, is not
  writable, and otherwise takes precedence. Either problem surfaces as
  `MockGPTDataset failed to build as a mock data generator`.
- **`flydsl` pinned to the version the image resolves (0.3.4.1 for v26.9).** The
  image gets it through Primus-Turbo's requirements, but Turbo is installed here
  with `--no-deps`, so `stage_turbo` restates the pin. aiter only needs
  `flydsl.expr.vector` at runtime, and that survived until 0.3.0 removed it —
  after which aiter prints `ROCm/HIP JIT runtime not available … CK and HIP ops
  are disabled. Triton ops remain available.` and quietly runs Triton-only.
  `stage_turbo` checks the symbol afterwards and warns when it is missing (it does
  not stop: the image makes no such check and ships the same flydsl). With 0.3.4.1
  the warning is expected, and bare metal matches the image. Workloads that lean
  on aiter's CK/HIP kernels are the ones that would notice.
- **The CUDA-only CUTLASS DSL stack is uninstalled after mamba.** `mamba_ssm`
  pins `quack-kernels`, which pulls `nvidia-cutlass-dsl`. Its MLIR Python bindings
  and FlyDSL's (installed with Primus-Turbo) share one process-wide nanobind type
  registry, so whichever loads second aborts
  ([#955](https://github.com/AMD-AGI/Primus/issues/955)). No ROCm path can run
  those kernels, and mamba's `ops/cute/mamba3`, their only importer, degrades
  without them. `stage_mamba` removes `quack-kernels` and `nvidia-cutlass-dsl*`
  and then checks that `import mamba_ssm` still works. The v26.9 image ships
  neither package either.
- **`patchelf` from pip**, since the apt one is unavailable.
- **`libz3.so` from pip** (`z3-solver`, added to `LD_LIBRARY_PATH` by `env.sh`).
  With `tilelang` now uninstalled this is only a safety net, kept so that
  re-installing `tilelang` by hand does not leave a broken environment.
- **Megatron** is not pip-installed; it's bundled at
  `$WORKSPACE_DIR/Primus/third_party/Megatron-LM` and added to the path by Primus
  at runtime (or set `PYTHONPATH` yourself for standalone `import megatron`).

## Caveats

- **Runtime-validated for v26.8.** A clean run of the default stages completed on
  **MI325X (gfx942) / Ubuntu 22.04 (glibc 2.35, GCC 11) / Python 3.12**, followed by
  `examples/megatron/configs/MI325X/llama3.1_8B-BF16-pretrain.yaml` for 10
  iterations on 8 GPUs (with `--micro_batch_size 2`, because the node was shared).
  It ran with no NaN iterations, and its loss matched the same command in
  `rocm/primus:v26.8` at every iteration to five significant digits.
  The installed package set matched the image's except for the optional stages
  skipped here and four minor transitive versions. The v26.9 pins have not been
  through a bare-metal run yet.
- **Persistence**: the venv, the provisioned interpreter and the kept checkouts
  live under `$PRIMUS_BASE` (persistent). Transient build sources go to local
  `/tmp` for speed and are deleted after each build.
- **Disk**: a full build needs tens of GB.
- **Index contents move.** v26.9's ROCm and torch set comes from the
  `nightly.repo.amd.com` indexes, whose builds are dated nightlies rather than the
  `stable.repo.amd.com` releases v26.8 used. If a pin stops resolving,
  update the pin block at the top of `setup.sh` as a set — `torch`,
  `amd-torch-device-*`, `rocm-sdk-*`, `torchaudio`, `torchvision`,
  `amd-torchvision-device-*` must all target the same ROCm version. `apex` is the
  exception: it stays on the ROCm 10.3 build the v26.9 image resolved (see
  [above](#apex-is-pinned-to-what-the-image-resolved)).
- **GPU arch is auto-detected** (`env.sh` reads `rocminfo`, else the kernel KFD
  sysfs `gfx_target_version`), and `stage_torch` installs the matching device
  wheels for whatever it finds (gfx942 and/or gfx950). To force a target — e.g.
  to build a portable env for both — export it before running:
  `export PYTORCH_ROCM_ARCH="gfx942;gfx950"`.
