# Primus environment in a venv (no docker, no sudo)

Reproduces the Primus **v26.5** training image in a Python virtual environment on
a bare-metal host. Derived from
[`.github/workflows/docker-release/Dockerfile.primus-v26.5`](../../.github/workflows/docker-release/Dockerfile.primus-v26.5),
using the same package pins and commits, adapted for the constraints of a machine
where we have no root.

A handful of places deliberately diverge from the Dockerfile, because copying it
exactly produces an environment that does not work outside the container. Each one
is listed under [Fixes applied](#fixes-applied-gotchas-vs-the-dockerfile) with the
failure it avoids — worth reading before "correcting" any of them back.

## Python 3.12 is required

This is a hard requirement from upstream packaging, not a preference. The pinned
torch nightly `2.12.0+rocm7.15.0a20260720` ships a **cp312 Linux wheel only** —
that nightly date published no cp310 Linux build — and the v26.5
TransformerEngine wheels are cp312-only as well.

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

> **Migrating from the v26.4-based scripts:** a Python 3.10 venv cannot be
> upgraded in place. `setup.sh` detects the mismatch and stops with instructions;
> remove the old venv and rebuild:
> `rm -rf "$PRIMUS_BASE/venv" && bash setup.sh`

## TransformerEngine: wheel on glibc ≥ 2.38, otherwise built from source

v26.5 installs TransformerEngine from the ROCm staging index instead of building
it. Those wheels are produced on Ubuntu 24.04, and `libtransformer_engine.so`
requires **glibc ≥ 2.38** plus `GLIBCXX_3.4.32`. Ubuntu 22.04 has glibc 2.35, and
unlike libstdc++, glibc cannot be side-loaded through `LD_LIBRARY_PATH` — so on a
22.04 host the v26.5 wheels install fine but fail at import with:

```
OSError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
```

`stage_te` therefore picks its install path from the host's glibc:

| Host glibc | Path | What happens |
|---|---|---|
| ≥ 2.38 (e.g. Ubuntu 24.04) | wheel | exactly what v26.5 does |
| < 2.38 (e.g. Ubuntu 22.04) | source | clones `ROCm/TransformerEngine` and builds commit `a07e607f14a…`, the same commit the `TE_VERSION` label refers to |

Either way you get the same TE version; the source build just links against the
host toolchain. It takes considerably longer than the wheel install. Override the
choice with `PRIMUS_TE_MODE=wheel|source` (default `auto`). After installing,
`stage_te` imports `transformer_engine` and fails loudly if it cannot load, rather
than letting the problem surface later during training.

## Why it differs from the Dockerfile

| Constraint | Dockerfile | Here |
|---|---|---|
| Python | 3.12 (Ubuntu 24.04) | 3.12, auto-provisioned by `uv` (host has only 3.10) |
| ROCm | pip `rocm-sdk-devel` | same (self-contained, **no sudo needed**) |
| GPU arch | gfx942 + gfx950 | **auto-detected** from the host (`rocminfo`/KFD sysfs); builds only what's present → faster builds |
| Build dir | container FS | venv under **`$PRIMUS_BASE`** (persistent, you choose it); build sources on local `/tmp` |
| System deps | `apt install ...` | **skipped** (no sudo); pip provides what's needed |
| torchaudio / torchvision / apex | floating (`torchvision==0.27`) | pinned to the exact `a20260720` nightly builds — see below |

The key reason no sudo is required: the `rocm-sdk-devel` pip wheel ships a full
ROCm toolchain inside the venv, so we don't depend on system ROCm or apt.

### Companion wheels are pinned, the Dockerfile's floating versions are not reproducible

The Dockerfile pins `torch` exactly but lets `torchaudio` and `apex` float and
resolves torchvision as `==0.27`. That set no longer resolves at all: newer
`rocm10.x` nightlies now publish matching version numbers, so a floating
torchvision selects a build for a different ROCm line, conflicts with the pinned
torch, and pip dies after a long backtracking storm. `setup.sh` pins each
companion wheel to the version the `a20260720` nightly published, i.e. exactly
what the Dockerfile picked up on the day it was built.

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

The build compiles aiter, Primus-Turbo, flash-attention and TransformerEngine
from source, so budget hours rather than minutes — on glibc < 2.38 that means all
of TE, not just its torch glue layer. Running it detached avoids losing it to a
dropped connection:

```bash
nohup bash setup.sh > ~/primus-setup.log 2>&1 &
tail -f ~/primus-setup.log
```

If any stage fails the script stops immediately and prints which stage failed.
Stages are idempotent, so fix the cause and re-run just that one — exporting the
same `PRIMUS_BASE` again, since that is how it finds the venv:

```bash
bash setup.sh --list          # show stages (works without PRIMUS_BASE)
bash setup.sh te              # reinstall just TransformerEngine
bash setup.sh venv torch      # venv + torch only
```

When cherry-picking stages, note that `te` depends on `torch` and `flash_attn`
having run first: the TE staging index serves only TE packages, so every other
dependency must already be installed.

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

`venv` → `torch` → `flash_attn` → `te` → `torchtune` → `torchao` → `pydeps`
→ `grouped_gemm` → `causal_conv1d` → `mamba` → `primus` → `aiter` → `turbo`
→ `boto` → `cleanup` → `manifest`

Optional: `torchrec` (DLRM/recommendation stack).

The order matters for `te`: see the note on the staging index below.

## What changed from the v26.4-based scripts

- **TransformerEngine now comes from the ROCm staging index where it can.**
  `stage_te` installs `transformer_engine_rocm_torch==2.15.0.dev0+rocm7.15.0a20260716.a07e607`
  from `rocm.frameworks-nightlies.amd.com/whl-staging/device-all/`, which pulls a
  prebuilt core library and compiles only the thin torch glue layer. On hosts
  whose glibc is too old for those wheels it builds the same commit from source
  instead — see the section above.
- **`stage_te` must run after `stage_flash_attn`.** That staging index serves
  *only* the transformer-engine packages, so while `--index-url` points at it pip
  cannot reach PyPI for anything else. Every TE dependency has to be installed
  beforehand. The Dockerfile gets `einops` transitively from flash-attention and
  `onnx` from `onnxscript`; `stage_te` installs both explicitly rather than
  relying on another package's transitive deps.
- **`ck_jit_compile.sh` is patched** the same way v26.5 patches it inside the
  image: concurrent CK JIT compiles race to publish the same `.so`, and without
  the patch the loser aborts the run. The script locates the file through
  `sysconfig` (rather than a hardcoded `/opt/venv/lib/python3.12` path) and the
  patch is idempotent.
- **New `cleanup` stage** uninstalls `tilelang`, mirroring v26.5. `mamba-ssm`
  pulls it in as a hard dependency and v26.5 removes it again once everything is
  built.
- **`apache-tvm-ffi` now follows the Dockerfile pin (0.1.11).** The old scripts
  held it at 0.1.6 to dodge a `tilelang` import failure; with `tilelang`
  uninstalled at the end, that workaround is obsolete.
- **`GPU_ARCHS` is `native` at runtime.** v26.5 sets `ENV GPU_ARCHS=native` after
  the last build stage so aiter's JIT compiles for the GPU actually in the box.
  `env.sh` therefore defaults it to `native`, and `setup.sh` overrides it to the
  full arch list for the individual stages that cross-compile.
- **Build-only NVTE vars dropped from the runtime environment.** `NVTE_USE_ROCM`,
  `NVTE_FRAMEWORK`, `NVTE_ROCM_ARCH` and `NVTE_USE_HIPBLASLT` configure the
  from-source TE build and are gone from v26.5, so `env.sh` no longer exports
  them; `stage_te_source` still sets them inline for the duration of its build.
  The runtime performance knobs (`NVTE_CK_*`,
  `NVTE_USE_CAST_TRANSPOSE_TRITON`, `NVTE_FLASH_ATTN=0`, `NVTE_FUSED_ATTN=1`)
  stay.
- **Updated pins:** torch `2.12.0+rocm7.15.0a20260720`, Primus
  `b511d1b66b0068715308ea9bfe8ba147ea1a3860` (`release/v26.5`), Primus-Turbo
  `edc8d2ccb0be4888e80ee7c6e765fd3956026a32`. flash-attention, torchtune,
  torchao, grouped_gemm, causal-conv1d, mamba and aiter are unchanged in v26.5.

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
- **MLPerf `primus_mllog`**: v26.5 installs it from `training_results_v6.0`. Not
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
- **Primus-Turbo is installed with `--no-deps`, keeping ROCm's triton.** Its
  `setup.py` hard-pins upstream `triton==3.7.0`, while torch is built against
  ROCm's `triton==3.7.1+git…rocm…`. The unconstrained Dockerfile lets the upstream
  wheel win, but doing that here breaks the environment: ROCm's HIP runtime
  already loads its own `libLLVM.so.23` from the pip SDK, and the upstream triton
  wheel bundles a second, statically linked LLVM, so importing it segfaults inside
  LLVM's static initialisers — taking down `torch._dynamo`, `aiter`, `torchao` and
  `mamba_ssm` with it. `stage_turbo` therefore installs Primus-Turbo without deps
  and supplies its real runtime requirements (`scipy`, `flydsl`) explicitly. This
  is a deliberate, tested deviation from v26.5.
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
  [docs/02-user-guide/training-recipes.md](../../docs/02-user-guide/training-recipes.md)),
  so `env.sh` applies it from the detected architecture and leaves `0` for gfx950,
  which the Dockerfile value targets. `NVTE_CK_USES_BWD_V3` itself stays at the
  Dockerfile's `1` — it is worth roughly 15% throughput, and the atomic mode is what
  makes it safe.
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
- **`flydsl` pinned to 0.2.4 so aiter keeps its CK/HIP kernels.** aiter pins
  `flydsl==0.1.7` and Primus-Turbo requires `flydsl>=0.2.0`, so one of the two is
  always unsatisfied and Turbo, installing last, wins. aiter only needs
  `flydsl.expr.vector` at runtime, and that survived until 0.3.0 removed it —
  after which aiter prints `ROCm/HIP JIT runtime not available … CK and HIP ops
  are disabled. Triton ops remain available.` and quietly runs Triton-only. 0.2.4
  is the newest release that satisfies Turbo and keeps aiter whole, and is what
  the Dockerfile itself resolved to, since 0.3.0 was published after v26.5 was
  built. `stage_turbo` asserts the symbol is importable afterwards. This is about
  capability parity with the image, not speed: on llama3.1_8B BF16 an A/B of 0.2.4
  against 0.3.0 measured 535.9 vs 534.5 TFLOP/s/GPU, i.e. no difference. Other
  workloads that lean on aiter's CK/HIP kernels are the ones that would notice.
- **`nvidia-cutlass-dsl` pinned to 4.5.3 so `import mamba_ssm` works.**
  `mamba_ssm` pins `quack-kernels==0.3.1`, which requires `nvidia-cutlass-dsl>=4.4.1`
  with no upper bound. 4.6.x restructured the DSL and removed `cute.core.ThrCopy`
  that quack 0.3.1 imports, so the unpinned resolution ends in
  `AttributeError: module 'cutlass.cute.core' has no attribute 'ThrCopy'`. 4.5.3 is
  the newest release quack 0.3.1 still works with, and pip cannot choose it
  unaided because 4.5.3 was published *after* 4.6.0 yet sorts lower. v26.5 does
  not pin this, so the released image is affected by the same break.
- **`patchelf` from pip**, since the apt one is unavailable.
- **`libz3.so` from pip** (`z3-solver`, added to `LD_LIBRARY_PATH` by `env.sh`).
  With `tilelang` now uninstalled this is only a safety net, kept so that
  re-installing `tilelang` by hand does not leave a broken environment.
- **Megatron** is not pip-installed; it's bundled at
  `$WORKSPACE_DIR/Primus/third_party/Megatron-LM` and added to the path by Primus
  at runtime (or set `PYTHONPATH` yourself for standalone `import megatron`).

## Caveats

- **Persistence**: the venv, the provisioned interpreter and the kept checkouts
  live under `$PRIMUS_BASE` (persistent). Transient build sources go to local
  `/tmp` for speed and are deleted after each build.
- **Disk**: a full build needs tens of GB.
- **Nightly indexes are pruned.** The pinned torch/TE nightlies will eventually
  disappear from the ROCm index. When that happens, pick a new nightly date that
  publishes a *complete* cp312 set and update the pin block at the top of
  `setup.sh` — all of `torch`, `amd-torch-device-*`, `rocm-sdk-*`, `torchaudio`,
  `torchvision`, `amd-torchvision-device-*` and `apex` should come from the same
  date.
- **GPU arch is auto-detected** (`env.sh` reads `rocminfo`, else the kernel KFD
  sysfs `gfx_target_version`), and `stage_torch` installs the matching device
  wheels for whatever it finds (gfx942 and/or gfx950). To force a target — e.g.
  to build a portable env for both — export it before running:
  `export PYTORCH_ROCM_ARCH="gfx942;gfx950"`.
