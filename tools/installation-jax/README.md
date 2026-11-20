# Primus JAX / MaxText environment in a venv (no docker, no sudo) — v26.10

Reproduces the Primus **v26.10 JAX training Dockerfile**
([`Dockerfile.jax-v26.10`](../../.github/workflows/docker-release/Dockerfile.jax-v26.10))
in a Python virtual environment. Same package pins as the Dockerfile, adapted for
a bare-metal host with no root and no containers.

This is the JAX/MaxText counterpart to `tools/installation/` (the PyTorch /
Megatron / TorchTitan stack). Use this one if you want to run the **JAX MaxText**
training backend of Primus.

## Why it differs from the Dockerfile

| Constraint | Dockerfile | Here |
|---|---|---|
| ROCm | pip `rocm-sdk-*` 10.3.0a20261109 → `/opt/venv/.../_rocm_sdk_devel` | same wheels → in-venv `_rocm_sdk_devel` (**no sudo**) |
| GPU arch | gfx942 + gfx950 | **auto-detected** from the host (`rocminfo`/KFD sysfs) for the device wheels and TE runtime; TE and Primus-Turbo source builds compile both, as the image does |
| Build dir | container FS | venv on **`$PRIMUS_JAX_BASE`** (persistent); transient sources on local `/tmp` |
| System deps | `apt install ...` | **skipped** (no sudo); documented in the guide's Section 2 |
| MaxText setup | `setup.sh TF=true` (runs `apt`, prompts for a venv) | Python steps only, with the same TF extras (apt is a one-time root action; venv already made) |
| Installer | `uv pip` | `pip` (MaxText's own requirements still go through `uv`) |
| Primus | cloned at commit `31128a29…` (on `release/v26.10`) | same commit, cloned into `$WORKSPACE_DIR/Primus` |
| Primus-Turbo | cloned on `feat/jax-grouped-gemm-flydsl-hipblaslt` and built for JAX | same branch and build, cloned into `$WORKSPACE_DIR/Primus-Turbo` |
| `LD_LIBRARY_PATH` | blanked at the end of the image | blanked at runtime (`PRIMUS_JAX_KEEP_ROCM_LD=1` for RCCL/TE source builds) |

The key reason no sudo is required: the `rocm-sdk-devel` pip wheel ships a full
ROCm toolchain inside the venv, so we don't depend on system ROCm or apt.

> **Python 3.12+ required (3.12 preferred).** MaxText requires Python ≥ 3.12 (the
> PyTorch recipe works on 3.10; this one does not). 3.12 is preferred because it
> is what the image uses and what this environment is validated on. You do
> **not** need `sudo` or a PPA:
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

> **Host OS: Ubuntu 22.04 or 24.04.** The TE native code ships as
> `transformer_engine_rocm10`, a `manylinux_2_28` wheel, and the JAX plugin pair
> as `manylinux_2_27`, so the prebuilt path works on **glibc ≥ 2.28** — Ubuntu
> 22.04 (2.35) included. The `te` stage checks the host glibc and only falls back
> to the from-source build below 2.28. Check with `ldd --version`.

> **The ROCm SDK is a nightly.** v26.10 pins ROCm `10.3.0a20261109` from
> `nightly.repo.amd.com`, and the TE wheel from the frameworks-nightlies staging
> index. Nightly indexes are pruned; if a pin stops resolving, the release
> Dockerfile is the reference for a replacement set — ROCm, the JAX plugin pair
> and the TE wheels all target the same ROCm build (`10.3.0a20261109`).

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
checkout we installed the dependencies for. Configs for MI325X live in
`examples/maxtext/configs/MI325X/`, and for MI350X/MI355X in `MI355X/`.

## Stages (default order, v26.10)

`venv` → `rocm` → `maxtext` → `tf_source` → `jax` → `te` → `primus`
→ `turbo` → `jaxreqs` → `manifest`

- **venv** — create the venv (Python ≥ 3.12) and bootstrap `cmake`/`ninja`/`uv`
  and the `msgpack==1.2.3` / `urllib3==2.8.0` CVE pins.
- **rocm** — pip-install TheRock `rocm-sdk-*` `10.3.0a20261109` (core/devel/libraries
  + per-arch device wheels) from the ROCm nightly index and run `rocm-sdk init`.
- **maxtext** — clone ROCm/MaxText (`release/v26.10`) and install its deps (the
  Python part of MaxText's `setup.sh`, with `--with-tf` as the image's `TF=true`)
  + the editable MaxText package.
- **tf_source** — build **tensorflow-cpu 2.21 from source** (bazel); fixes the
  ROCm-vs-TF LLVM symbol clash (SIGSEGV) and drops bundled NCCL. See the note on
  the LLVM archive below.
- **jax** — `jax`/`jaxlib` 0.12.0 from PyPI + `jax_rocm10_pjrt` / `jax_rocm10_plugin`
  `0.12.0+rocm10.3.0a20261109` from the ROCm nightly jax index (installed after
  MaxText to override its stock jax).
- **te** — prebuilt `transformer_engine_rocm_jax` and `transformer-engine-rocm10`
  `2.21.0.dev0+rocm10.3.0a20261109.4c7ed18` (+ `flax==0.12.10`, `pydantic`, ...);
  auto-falls-back to a from-source build (`te_source`) on glibc < 2.28 hosts.
- **primus** — clone Primus at `31128a29…`, the commit the image pins (on
  `release/v26.10`), init the `third_party/maxtext` submodule, drop the stale
  `dataclasses` backports. `export PRIMUS_COMMIT=release/v26.10` to take the
  commits the branch has gained since the image was built.
- **turbo** — clone Primus-Turbo on the branch the image builds
  (`feat/jax-grouped-gemm-flydsl-hipblaslt`), build its JAX extension for gfx942
  and gfx950 (`--no-deps`, editable, kept in `$WORKSPACE_DIR/Primus-Turbo`), add
  `flydsl==0.3.4.1`, and check `import primus_turbo.jax.lax` as the image does.
  This is MaxText's grouped-GEMM path for dropless MoE
  (`use_turbo_grouped_gemm: true`).
- **jaxreqs** — install Primus' `requirements-jax.txt` and the v26.10 CVE-fix pins
  (v26.10 moves `pyasn1` to 0.6.5 and `oauthlib` to 4.0.1), re-pinning
  `jax`/`jaxlib` as the image does.
- **manifest** — dump `pip list` / `env` for reproducibility.

Optional / alternative stages:

- **rccl** — build **RCCL from source** into `$ROCM_PATH/lib`. Not part of the
  default flow since v26.7: the pip ROCm SDK ships RCCL and the published image
  does not override it. Run this only to
  reproduce v26.6, or on a host that needs the `rocm-systems` net-ib fix
  (ROCM-27881).
- **te_source** — force the from-source TransformerEngine build regardless of
  glibc, at the commit the wheel's local label names (`4c7ed18`). Normally
  unnecessary: the default `te` stage builds from source only when glibc < 2.28.
  Heavy build (~30–60 min, compiles CK fused-attention kernels).
- **tf_cpu_fix** — lighter alternative to `tf_source`: `pip install tensorflow-cpu`
  instead of the bazel build (avoids the bundled-NCCL clash; may still hit the
  LLVM-symbol SIGSEGV in Grain workers).

```bash
# To skip the heavy tf_source bazel build, swap in tf_cpu_fix:
bash setup.sh venv rocm maxtext tf_cpu_fix jax te primus turbo jaxreqs manifest
```

> **MaxText and Primus.** MaxText is on `release/v26.10` and Primus at the
> `release/v26.10` commit the image pins. Override `MAXTEXT_BRANCH` only if you
> deliberately need a different MaxText release.

## What is SKIPPED (needs sudo / apt — not reproducible here)

- **System packages** (`numactl`, `gcsfuse`, RDMA/verbs libs, and the build
  basics): a one-time root action, documented in Section 2 of the guide.
- **AINIC** (`add-apt-repository`, `libionic-dev`): apt-only. Skipped.
- **UCX + OpenMPI**: **optional for MaxText** — JAX uses its own distributed
  coordinator + RCCL, not `mpirun`. Carried over from the reference image for
  other/MPI-launched JAX workloads; skipped here. See Section 4 of the guide.
- **gcsfuse**: only needed to mount GCS buckets for data; not required for
  synthetic-data or local-data runs.
- **MaxDiffusion** (the image's last stage, which also installs `torch`,
  `torchvision` and downgrades `transformers` to 4.57.3): not part of the
  MaxText training path.

## Notes / gotchas vs. the Dockerfile

- **Stage order matters:** `maxtext` → `tf_source` → `jax` → `te`. MaxText's
  `setup.sh` pulls in a stock `jax`/`tensorflow`; TF is then rebuilt from source and
  the ROCm JAX/plugin is installed after (overriding MaxText's), and TE must come
  after JAX or `jaxlib` gets clobbered. The stage order enforces this.
- **The LLVM source archive TF needs is fetched and verified up front.** TF pins
  the LLVM tarball by sha256, but its `mirror.tensorflow.org` copy is gone (404)
  and GitHub generates the `.tar.gz` on the fly with an unstable gzip stream: the
  same commit has come back under two different hashes, with identical contents.
  The image therefore feeds Bazel a frozen copy through `--distdir`. `tf_source`
  does the same where it can: it downloads from GitHub, re-gzips the (identical)
  tar the ways `git archive` does, and retries until the bytes match the pinned
  hash. The image's frozen copy comes from a private registry, so when no public
  copy matches (as for v26.9), `tf_source` fetches the pinned LLVM commit itself (git
  verifies the objects against the commit id), builds the archive with
  `git archive`, and re-pins `LLVM_SHA256` to it in the temporary TensorFlow
  checkout only, logging both hashes. The source stays pinned, by the commit id
  instead of the tarball hash. Before any of this, `tf_source` makes the image's
  own check that the TF branch still pins the expected LLVM archive hash, and
  stops if it does not. To use a known-good copy instead, export
  `PRIMUS_TF_LLVM_ARCHIVE=/path/to/<commit>.tar.gz`, or use `tf_cpu_fix`.
- **The TF build needs no host clang.** TF 2.21 uses Bazel's hermetic C++
  toolchain by default; on Ubuntu 22.04 with no `clang` installed the build ran
  end to end (about 13 minutes on 120 cores). It does need `unzip`/`zip`.
- **Primus and Primus-Turbo are cloned as in the image.** The image clones Primus
  at a pinned `release/v26.10` commit and Primus-Turbo on a branch; the `primus`
  and `turbo` stages clone the same. The image's Primus-Turbo checkout is
  `fe9229ef`; export `PRIMUS_TURBO_BRANCH=fe9229efc1caea65665f74402578d30ff7af88f8`
  to build exactly that commit if the branch has moved.
- **The image's CVE step re-resolves everything; this one does not.** The final
  `uv pip install --upgrade <pins>` in the Dockerfile re-resolves *every*
  installed package (uv's `--upgrade` semantics), so the v26.10 image floats numpy
  to 2.5.5, scipy to 1.18.1 and protobuf to 7.x. `jaxreqs` uses pip, whose
  `--upgrade` touches only the named packages, so those stay at MaxText's own
  pins. Training was unaffected when this was validated for v26.8; it is the
  largest difference between the two package sets.
- **TE's `rocm10` core-package patch is normally a no-op.** The Dockerfile
  seds `transformer-engine-rocm10` into TE's core-package list; wheels since
  `2.18.0.dev0` already name it (the image then lists it twice), so
  `stage_te` patches only when the name is missing.
- **bazelisk is the upstream release binary.** The image rebuilds bazelisk from
  source with a current Go toolchain to clear CVEs in the prebuilt binary's Go
  dependencies; launcher behaviour is identical, and bare metal needs no Go
  toolchain for it.
- **Runtime `LD_LIBRARY_PATH` is empty**, matching the image's TE + JAX segfault
  fix. RCCL/TE source builds set `PRIMUS_JAX_KEEP_ROCM_LD=1`.
- **TE from source always compiles gfx942+gfx950.** TE's HipKittens GEMM
  fails to link if CMake only sees gfx942. The `turbo` stage builds both as well,
  as the image does.
- **MaxDiffusion pins are not applied.** The image's MaxDiffusion stage pins
  `transformers==4.57.3` and `datasets==5.0.1`; bare metal skips that stage (see
  above), so both come from MaxText's and Primus' requirements.
- **No FlashAttention/aiter/torch stack.** The JAX MaxText path does not build the
  PyTorch kernel libraries; attention fusion comes from `transformer_engine_rocm_jax`
  + XLA.
- **Two MaxText checkouts** (`$MAXTEXT_DIR` and `Primus/third_party/maxtext`) are
  collapsed here: we install deps from one checkout and point `MAXTEXT_PATH` at it.
- **Runtime-validated for v26.8.** The default flow (prebuilt `te` wheel,
  `tf_source`) ran end to end on **MI325X (gfx942) / Ubuntu 22.04 (glibc 2.35) /
  Python 3.12**, followed by single-node 8-GPU MaxText pretraining
  (`examples/maxtext/configs/MI325X/llama2_7B-bf16-pretrain.yaml`). Earlier
  releases also exercised a 2-node run over the JAX distributed coordinator +
  RCCL, with no UCX/OpenMPI; that was not repeated for v26.8. The v26.10 pins
  have not been through a bare-metal run yet.

Treat the reference `Dockerfile` as the authoritative, tested version
combination; if you bump one pin you may need to bump the others.
