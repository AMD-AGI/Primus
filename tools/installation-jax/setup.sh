#!/usr/bin/env bash
# setup.sh — Reproduce the Primus JAX/MaxText training environment (v26.8) in a
# Python venv (no sudo, no docker). Mirrors the v26.8 JAX training Dockerfile,
# adapted for bare metal:
#   * ROCm from pip TheRock wheels (`rocm-sdk-*` 10.2.0a20260923, a nightly on
#     nightly.repo.amd.com), same delivery as the image
#   * builds/checkouts on a big disk (home quota is usually tiny)
#   * GPU arch auto-detected (gfx942 and/or gfx950); apt/sudo steps skipped
#   * MaxText's own setup.sh apt/interactive steps skipped (system packages are
#     a one-time root action, documented in the guide's Section 2)
#   * TensorFlow (2.21 CPU) is built from source in the default flow (matching
#     the v26.8 image); RCCL is not — the image uses the ROCm SDK copy
#   * JAX 0.11.1 + jax_rocm10_{pjrt,plugin} 0.11.1+rocm10.2.0a20260923 from the
#     ROCm nightly index
#
# Usage:
#   bash setup.sh                # run all default stages in order
#   bash setup.sh <stage>...     # run only specific stage(s), e.g.
#   bash setup.sh venv rocm jax
#
# Stages are re-runnable. List them with:  bash setup.sh --list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

log()  { echo -e "\n\033[1;36m[setup] $*\033[0m"; }
die()  { echo -e "\033[1;31m[setup][ERROR] $*\033[0m" >&2; exit 1; }
# shellcheck disable=SC1091
reload_env() { source "$SCRIPT_DIR/env.sh"; }

# ---- pinned versions / commits (from Dockerfile.jax-v26.8) ----
# ROCm: TheRock pip wheels. v26.8 takes a ROCm 10.2 nightly, so the SDK comes
# from nightly.repo.amd.com rather than the stable.repo.amd.com index v26.7 used.
# Nightly indexes are pruned; see the README if this pin stops resolving.
# See: https://nightly.repo.amd.com/rocm/core/whl-next/rocm/
ROCM_INDEX="https://nightly.repo.amd.com/rocm/core/whl-next/"
THE_ROCK_VERSION="10.2.0a20260923"

# JAX + ROCm PJRT/plugin (jax_rocm10_*), from the ROCm nightly jax index.
JAX_VERSION="0.11.1"
JAX_ROCM_VERSION="0.11.1+rocm10.2.0a20260923"
JAX_ROCM_INDEX="https://nightly.repo.amd.com/rocm/jax/whl-next/"

# TransformerEngine (prebuilt ROCm JAX wheels). v26.8 installs two distributions,
# both from the frameworks-nightlies staging index: the jax flavour and the
# transformer-engine-rocm10 native core (a manylinux_2_28 wheel). The
# `transformer_engine` meta package v26.7 also installed is gone.
# See: https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/transformer-engine-rocm-jax/
TE_VERSION="2.18.0.dev0+rocm10.2.0a20260929.03afd8f"
TE_INDEX="https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/"
# From-source TransformerEngine (the optional `te_source` stage, used on hosts
# whose glibc is older than the wheels need). The commit is the one the wheel's
# local label names.
TE_REPO="https://github.com/ROCm/TransformerEngine.git"
TE_SOURCE_COMMIT="${TE_SOURCE_COMMIT:-03afd8f813300417ee1a41a4471d89be71c88f2d}"

# TensorFlow (CPU) built from source — replaces the stock PyPI wheel, whose
# bundled LLVM collides with ROCm's libLLVM in Grain workers (SIGSEGV on
# `import tensorflow` after `import jax`). A CPU build also drops the bundled
# NCCL, preserving the XLA->RCCL collective fix.
TF_REPO="https://github.com/ROCm/tensorflow-upstream.git"
TF_BRANCH="upstream-v2.21.0"
BAZELISK_VERSION="v1.29.0"

# RCCL. Not built since v26.7: the pip ROCm SDK ships RCCL (2.31.2 in the v26.8
# image), so the `rccl` stage is not in DEFAULT_STAGES. These pins remain only
# for the optional stage, which reproduces the v26.6 behaviour of overriding the
# SDK copy from rocm-systems.
RCCL_REPO="https://github.com/ROCm/rocm-systems.git"
RCCL_COMMIT="9e5e4084a4b8e1e86551b0eb054725c62354a926"

# MaxText (ROCm fork)
MAXTEXT_REPO="https://github.com/ROCm/maxtext.git"
MAXTEXT_BRANCH="${MAXTEXT_BRANCH:-release/v26.8}"
# Which MaxText requirements set to install. The reference Docker image runs
# MaxText's setup.sh with defaults (DEVICE=tpu), which — on ROCm — pulls the
# framework-agnostic deps WITHOUT any CUDA packages. Override to `cuda12` only
# if you specifically need that set.
MAXTEXT_DEVICE="${MAXTEXT_DEVICE:-tpu}"

# Primus. The v26.8 image no longer ships Primus at all -- it runs from a mounted
# checkout -- so there is no image commit to match; this is the release branch
# the docs pair with the image.
PRIMUS_REPO="https://github.com/AMD-AGI/Primus.git"
PRIMUS_BRANCH="release/v26.8"

PIP="python -m pip"
UVPIP="python -m uv pip"

# Fresh clone helper: clone into transient SRC_DIR, build, then optionally clean.
fresh_clone() {  # fresh_clone <url> <dir> [extra git clone args...]
    local url="$1"; local dir="$2"; shift 2
    rm -rf "${SRC_DIR:?}/$dir"
    git clone "$@" "$url" "$SRC_DIR/$dir"
}

# Return 0 if the host glibc is new enough for the prebuilt TE wheels.
#
# The native code ships as `transformer_engine_rocm10`, a manylinux_2_28 wheel
# (since v26.7), with the jax flavour a small sdist built locally. v26.6's wheels
# were Ubuntu 24.04 builds and did need 2.38.
#
# On failure to parse (unknown libc) we conservatively return non-zero so the
# caller falls back to the always-works from-source build.
TE_WHEEL_MIN_GLIBC_MINOR=28

_glibc_ge_te_min() {
    local v; v="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
    [ -n "$v" ] || return 1
    awk -v v="$v" -v min="$TE_WHEEL_MIN_GLIBC_MINOR" \
        'BEGIN{split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=min))}'
}

# ============================ STAGES ============================

stage_venv() {
    # MaxText requires Python >= 3.12.
    _venv_python_ok() {  # _venv_python_ok <python>  -> 0 if >= 3.12
        local v; v="$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || return 1
        case "$v" in 3.1[2-9]|3.[2-9][0-9]|[4-9].*) return 0 ;; *) return 1 ;; esac
    }

    # If the resolved interpreter is too old, try to provision Python 3.12 with
    # uv (no sudo) — the standard fix on Ubuntu 22.04 where apt has no 3.12.
    if ! _venv_python_ok "$PRIMUS_PYTHON"; then
        if command -v uv >/dev/null 2>&1; then
            log "PRIMUS_PYTHON ($PRIMUS_PYTHON) is < 3.12; provisioning Python 3.12 via uv (no sudo)"
            uv python install 3.12 || die "uv python install 3.12 failed"
            PRIMUS_PYTHON="$(uv python find '>=3.12' 2>/dev/null || true)"
            if [ -z "$PRIMUS_PYTHON" ] || [ ! -x "$PRIMUS_PYTHON" ]; then
                die "could not locate a uv-managed Python >= 3.12 after install"
            fi
        else
            die "MaxText requires Python >= 3.12 but no suitable interpreter was found. Options (no sudo): install uv (pip install --user uv) then re-run — setup.sh will fetch Python 3.12 automatically; or install python3.12 yourself and re-run with PRIMUS_PYTHON=python3.12"
        fi
    fi
    log "Creating venv at $VENV_DIR (interpreter: $PRIMUS_PYTHON -> $("$PRIMUS_PYTHON" --version 2>&1))"
    mkdir -p "$PRIMUS_JAX_BASE" "$SRC_DIR" "$WORKSPACE_DIR" || die "could not create the install dirs under PRIMUS_JAX_BASE=$PRIMUS_JAX_BASE (SRC_DIR=$SRC_DIR, WORKSPACE_DIR=$WORKSPACE_DIR). Set PRIMUS_JAX_BASE to a directory you can write to with tens of GB free, e.g.  export PRIMUS_JAX_BASE=/big/disk/primus-jax-env"
    # Recreate the venv if it exists but was built with a too-old Python.
    if [ -f "$VENV_DIR/bin/activate" ] && ! _venv_python_ok "$VENV_DIR/bin/python"; then
        log "Existing venv at $VENV_DIR is < Python 3.12; recreating it"
        rm -rf "$VENV_DIR"
    fi
    [ -f "$VENV_DIR/bin/activate" ] || "$PRIMUS_PYTHON" -m venv "$VENV_DIR"
    reload_env
    $PIP install --upgrade pip
    # Build front-end tooling (matches the Dockerfile; add uv, used by MaxText).
    $PIP uninstall -y wheel || true
    $PIP install \
        cmake==3.31.6 \
        ninja==1.11.1.3 \
        wheel==0.46.2 \
        packaging==25.0 \
        setuptools==80.10.2 \
        msgpack==1.2.1 \
        urllib3==2.8.0 \
        uv
    rm -rf /root/.cache 2>/dev/null || true
}

stage_rocm() {
    reload_env
    # v26.7: ROCm comes from TheRock pip wheels (same as the JAX image), not a
    # tarball. Device wheels match the host arch detected in env.sh.
    log "Installing ROCm SDK ${THE_ROCK_VERSION} from $ROCM_INDEX (arch: $PYTORCH_ROCM_ARCH)"
    local _arch arch_args=()
    local _arches; IFS=';' read -ra _arches <<< "$PYTORCH_ROCM_ARCH"
    for _arch in "${_arches[@]}"; do
        _arch="${_arch// /}"; [ -z "$_arch" ] && continue
        arch_args+=( "rocm-sdk-device-${_arch}==${THE_ROCK_VERSION}" )
    done
    [ ${#arch_args[@]} -gt 0 ] || die "no GPU arch resolved; export PYTORCH_ROCM_ARCH (e.g. gfx942;gfx950)"

    $PIP install \
        --index-url "$ROCM_INDEX" \
        --pre \
        "rocm==${THE_ROCK_VERSION}" \
        rocm-bootstrap \
        "rocm-sdk-core==${THE_ROCK_VERSION}" \
        "rocm-sdk-devel==${THE_ROCK_VERSION}" \
        "rocm-sdk-libraries==${THE_ROCK_VERSION}" \
        "${arch_args[@]}"
    log "Running rocm-sdk init"
    rocm-sdk init
    reload_env
    [ -n "${ROCM_PATH:-}" ] || die "ROCM_PATH not resolved after rocm-sdk init"
    log "ROCM_PATH=$ROCM_PATH"
    ( "$ROCM_PATH/bin/hipcc" --version || hipcc --version ) 2>/dev/null || log "hipcc not found yet; re-source env.sh"
}

stage_jax() {
    reload_env
    log "Installing JAX ${JAX_VERSION} + ROCm PJRT/plugin ${JAX_ROCM_VERSION} (rocm10 wheels)"
    # Note: JAX and related libraries need to be installed BEFORE TE and AFTER
    # MaxText (whose setup.sh pulls in a stock jax/tensorflow we override here).
    # jax/jaxlib come from PyPI; the jax_rocm10_* plugin pair comes from the ROCm
    # jax index. jaxlib 0.11.1 knows the rocm10 plugin names, so the in-place
    # rename v26.7 applied is no longer needed (the Dockerfile dropped it too).
    $PIP install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" scipy==1.16
    $PIP install --index-url "$JAX_ROCM_INDEX" --pre \
        "jax_rocm10_pjrt==${JAX_ROCM_VERSION}" \
        "jax_rocm10_plugin==${JAX_ROCM_VERSION}"
    python -c "import jax; print('jax', jax.__version__); print('devices:', jax.devices())" || \
        log "WARNING: jax.devices() failed (expected if no GPU is visible on this build host)"
}

stage_te() {
    reload_env
    # The native TE package is a manylinux_2_28 wheel, so it loads on Ubuntu
    # 22.04 (glibc 2.35) as well as 24.04. Only genuinely older hosts fall back to
    # the from-source build, which links against the host glibc.
    if ! _glibc_ge_te_min; then
        log "host glibc < 2.$TE_WHEEL_MIN_GLIBC_MINOR (prebuilt TE needs >= 2.$TE_WHEEL_MIN_GLIBC_MINOR): building TransformerEngine from source instead (te_source)"
        stage_te_source
        return
    fi
    log "Installing TransformerEngine (JAX) ${TE_VERSION} (prebuilt ROCm wheel)"
    $PIP install \
        pybind11==3.0.4 \
        importlib-metadata==8.7.1 \
        pydantic==2.13.4 \
        flax==0.12.9
    # Drop the `transformer_engine` meta package an older (v26.7) install left
    # behind; v26.8 no longer ships it.
    $PIP uninstall -y transformer_engine >/dev/null 2>&1 || true
    $PIP install \
        --index-url "$TE_INDEX" \
        --pre \
        --no-build-isolation \
        "transformer_engine_rocm_jax==${TE_VERSION}" \
        "transformer-engine-rocm10==${TE_VERSION}"
    patch_te_rocm10_core

    # Verify the prebuilt wheel actually loads NOW — otherwise training dies
    # later with a silent exit (the launcher swallows the OSError).
    log "Verifying TransformerEngine loads"
    if ! python -c "import transformer_engine.jax" 2>/tmp/te_import_err; then
        cat /tmp/te_import_err >&2 || true
        if grep -q "GLIBC_2" /tmp/te_import_err 2>/dev/null; then
            die "TransformerEngine failed to import: your host glibc ($(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')) is older than the prebuilt wheel requires (manylinux_2_28, i.e. glibc>=2.28). Build TransformerEngine from source instead: bash setup.sh te_source (see docs Section 3.7). glibc cannot be side-loaded via LD_LIBRARY_PATH."
        fi
        die "TransformerEngine failed to import (see error above). See docs Section 3.7."
    fi
    log "TransformerEngine import OK"
    python - <<'PY' || true
import glob, os, sysconfig
for base in {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}:
    p = os.path.join(base, "flaxlib_src")
    if os.path.isdir(p):
        import shutil; shutil.rmtree(p)
        print("removed", p)
PY
}

# TE locates its native core by distribution name. Dockerfile.jax-v26.8 seds
# `transformer-engine-rocm10` into that list; the 2.18.0.dev0 wheel already
# carries it (the image ends up listing it twice), so only patch when missing.
patch_te_rocm10_core() {
    local f
    f="$(python - <<'PY'
import os, sysconfig
for base in {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}:
    p = os.path.join(base, "transformer_engine", "common", "__init__.py")
    if os.path.exists(p):
        print(p)
        break
PY
)"
    [ -n "$f" ] || die "transformer_engine/common/__init__.py not found after the TE install"
    if grep -qE 'te_core_packages = .*"transformer-engine-rocm10"' "$f"; then
        log "TE already lists transformer-engine-rocm10 as a core package"
        return 0
    fi
    sed -i 's|"transformer-engine-rocm7"|"transformer-engine-rocm7", "transformer-engine-rocm10"|' "$f"
    grep -qE 'te_core_packages = .*"transformer-engine-rocm10"' "$f" \
        || die "could not add transformer-engine-rocm10 to TE's core package list in $f"
    log "Patched $f to recognise transformer-engine-rocm10"
}

stage_te_source() {
    reload_env
    export PRIMUS_JAX_KEEP_ROCM_LD=1
    reload_env
    # Build TransformerEngine (JAX) from source, linking against the HOST glibc.
    # Use this instead of `te` on hosts whose glibc is older than the prebuilt
    # wheel needs (< 2.38, e.g. Ubuntu 22.04). NOTE: this is a heavy build (CK
    # fused-attention kernels for your arch) — expect ~30-60 min.
    log "Building TransformerEngine (JAX) from source @ $TE_SOURCE_COMMIT"
    # TE 2.17 HipKittens grouped MXFP8 GEMM is compiled only when both gfx942
    # and gfx950 appear in CMAKE_HIP_ARCHITECTURES. A gfx942-only configure
    # still defines USE_HIPKITTENS_GEMM and then fails to find
    # kittens_grouped_mxfp8_gemm (CDNA4). The image builds both archs.
    $PIP install pybind11==3.0.4 importlib-metadata==8.7.1 pydantic==2.13.4 flax==0.12.9
    # Remove ALL prebuilt/stale TE variants so TE's install sanity-check sees a
    # single, self-consistent from-source package.
    $PIP uninstall -y transformer_engine transformer-engine transformer_engine_rocm7 \
        transformer_engine_rocm10 transformer_engine_rocm_jax transformer_engine_jax 2>/dev/null || true
    fresh_clone "$TE_REPO" TransformerEngine --recursive
    ( cd "$SRC_DIR/TransformerEngine" \
        && git checkout "$TE_SOURCE_COMMIT" \
        && git submodule update --init --recursive \
        && USE_ROCM=1 NVTE_FRAMEWORK=jax NVTE_USE_ROCM=1 \
           HIP_PATH="$ROCM_PATH" \
           NVTE_ROCM_ARCH="gfx942;gfx950" \
           CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" \
           PYTHONPATH="$SRC_DIR/TransformerEngine/3rdparty/hipify_torch:${PYTHONPATH:-}" \
           python3 setup.py bdist_wheel \
        && $PIP install --no-deps --force-reinstall dist/*.whl ) || die "TransformerEngine source build failed"
    log "Verifying TransformerEngine (from source) loads"
    python -c "import transformer_engine.jax" || die "TransformerEngine (source) import failed"
    log "TransformerEngine (source) import OK"
    rm -rf "$SRC_DIR/TransformerEngine"
    unset PRIMUS_JAX_KEEP_ROCM_LD
    reload_env
}

stage_maxtext() {
    reload_env
    log "Installing MaxText @ $MAXTEXT_BRANCH (kept in $MAXTEXT_DIR)"
    rm -rf "$MAXTEXT_DIR"
    git clone "$MAXTEXT_REPO" "$MAXTEXT_DIR" || die "MaxText clone failed"
    ( cd "$MAXTEXT_DIR" && git checkout "$MAXTEXT_BRANCH" ) || die "MaxText checkout failed"

    # Replicate the Python portion of MaxText's src/dependencies/scripts/setup.sh
    # for MODE=stable, WORKFLOW=pre-training, TF=true (what the image runs).
    # The apt/gcsfuse and interactive-venv steps of that script are skipped:
    # system packages are a one-time root action (see the guide's Section 2)
    # and the venv already exists here. release/v26.8 made the TensorFlow,
    # TFDS, seqio and JetStream extras opt-in (TF=true, i.e. --with-tf); the
    # image passes it, so this does too.
    local req="src/dependencies/requirements/generated_requirements/${MAXTEXT_DEVICE}-requirements.txt"
    [ -f "$MAXTEXT_DIR/$req" ] || die "MaxText requirements not found: $MAXTEXT_DIR/$req"
    ( cd "$MAXTEXT_DIR" \
        && $PIP install -U setuptools wheel uv \
        && $UVPIP install --resolution=lowest -r "$req" \
        && python -m src.dependencies.scripts.install_pre_train_extra_deps --with-tf \
        && $UVPIP install --no-deps -e . ) || die "MaxText dependency install failed"
}

stage_tf_cpu_fix() {
    reload_env
    # Fix: TF 2.20 sets RTLD_GLOBAL which exposes its bundled CUDA-targeting NCCL
    # symbols globally, causing XLA's ncclCommInitRankConfig to resolve to TF's
    # NCCL instead of the system ROCm RCCL. tensorflow-cpu has no bundled NCCL.
    local tfver
    tfver="$($PIP show tensorflow 2>/dev/null | awk '/^Version:/{print $2}')"
    if [ -z "$tfver" ]; then
        log "tensorflow not installed; skipping tensorflow-cpu override (run 'maxtext' stage first)"
        return 0
    fi
    log "Overriding tensorflow with tensorflow-cpu==$tfver (--no-deps) to avoid NCCL symbol clash"
    $PIP install --no-deps "tensorflow-cpu==${tfver}"
}

stage_tf_source() {
    reload_env
    # v26.5: build tensorflow-cpu 2.21 from ROCm's fork (bazel). The stock PyPI
    # TF wheel bundles an LLVM whose symbols collide with ROCm's libLLVM in Grain
    # "spawn" workers -> SIGSEGV on `import tensorflow` after `import jax`. A CPU
    # build has correct symbol visibility and no bundled NCCL. HEAVY: ~30-60 min.
    # Needs a host clang/lld (clang-18) and unzip/zip — see the guide's Section 2.
    # Idempotent: if the CPU build is already installed, skip the ~30-60 min
    # bazel rebuild (pip uninstall tensorflow-cpu first to force a rebuild).
    if $PIP show tensorflow-cpu >/dev/null 2>&1; then
        log "tensorflow-cpu already installed ($($PIP show tensorflow-cpu 2>/dev/null | awk '/^Version:/{print $2}')); skipping source build"
        return 0
    fi
    log "Building tensorflow-cpu ($TF_BRANCH) from source with bazel (~30-60 min)"
    local bz="$PRIMUS_JAX_BASE/bin/bazel"
    mkdir -p "$PRIMUS_JAX_BASE/bin"
    if [ ! -x "$bz" ]; then
        log "Fetching bazelisk $BAZELISK_VERSION -> $bz"
        wget -O "$bz" "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" \
            || die "bazelisk download failed"
        chmod +x "$bz"
    fi
    export PATH="$PRIMUS_JAX_BASE/bin:$PATH"
    fresh_clone "$TF_REPO" tensorflow-upstream --depth 1 --branch "$TF_BRANCH"
    local distdir="$SRC_DIR/bazel_distdir"
    fetch_tf_llvm_archive "$SRC_DIR/tensorflow-upstream" "$distdir"
    # Build the wheel for the venv's ACTUAL Python (the reference image uses 3.12;
    # deriving the version keeps the recipe working on hosts whose interpreter is
    # 3.13). A cp312 wheel will not install into a cp313 venv and vice versa.
    local pyver pytag
    pyver="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    pytag="cp${pyver//./}"   # 3.12 -> cp312, 3.13 -> cp313
    # Keep bazel's (large) output tree off the tiny home quota.
    ( cd "$SRC_DIR/tensorflow-upstream" \
        && "$bz" --output_user_root="$SRC_DIR/bazel" build //tensorflow/tools/pip_package:wheel \
            --distdir="$distdir" \
            --repo_env=WHEEL_NAME=tensorflow_cpu \
            --repo_env=HERMETIC_PYTHON_VERSION="$pyver" ) || die "TensorFlow bazel build failed"
    $PIP uninstall -y tensorflow tensorflow-cpu tensorflow_cpu || true
    $PIP install --no-deps "$SRC_DIR"/tensorflow-upstream/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-*-"${pytag}"-"${pytag}"-linux_x86_64.whl \
        || die "TensorFlow wheel install failed"
    python -c "import tensorflow as tf; print('tensorflow', tf.__version__)" || log "WARNING: tensorflow import failed"
    # bazel marks its external/downloaded trees read-only (dirs lose the write
    # bit), so a plain `rm -rf` fails with "Permission denied". Make them
    # writable first, and never let cleanup abort the install (TF is already
    # built + installed at this point).
    chmod -R u+w "$SRC_DIR/tensorflow-upstream" "$SRC_DIR/bazel" 2>/dev/null || true
    rm -rf "$SRC_DIR/tensorflow-upstream" "$SRC_DIR/bazel" "$distdir" 2>/dev/null || true
}

# TF pins the LLVM source archive by sha256, but its mirror.tensorflow.org copy
# is gone (404) and GitHub generates the .tar.gz on the fly, with a gzip stream
# that is not stable: the same commit has come back under two different hashes.
# The image feeds Bazel a frozen copy through --distdir. Do the same: fetch from
# GitHub until the bytes match the pinned hash, or take a known-good copy from
# PRIMUS_TF_LLVM_ARCHIVE. The checksum is never relaxed.
fetch_tf_llvm_archive() {  # fetch_tf_llvm_archive <tf_src> <distdir>
    local bzl="$1/third_party/xla/third_party/llvm/workspace.bzl"
    local distdir="$2" commit sha target attempt
    commit="$(sed -n 's/^ *LLVM_COMMIT = "\([0-9a-f]*\)".*/\1/p' "$bzl")"
    sha="$(sed -n 's/^ *LLVM_SHA256 = "\([0-9a-f]*\)".*/\1/p' "$bzl")"
    if [ -z "$commit" ] || [ -z "$sha" ]; then
        die "could not read LLVM_COMMIT/LLVM_SHA256 from $bzl"
    fi
    mkdir -p "$distdir"
    target="$distdir/$commit.tar.gz"
    if [ -n "${PRIMUS_TF_LLVM_ARCHIVE:-}" ]; then
        cp "$PRIMUS_TF_LLVM_ARCHIVE" "$target" || die "cannot copy PRIMUS_TF_LLVM_ARCHIVE=$PRIMUS_TF_LLVM_ARCHIVE"
        echo "$sha  $target" | sha256sum -c --status \
            || die "PRIMUS_TF_LLVM_ARCHIVE does not match the sha256 TensorFlow pins ($sha)"
        log "Using LLVM $commit archive from PRIMUS_TF_LLVM_ARCHIVE"
        return 0
    fi
    for attempt in 1 2 3 4 5 6; do
        log "Fetching LLVM $commit archive for bazel --distdir (attempt $attempt)"
        if curl -fsSL --retry 3 -o "$target" "https://github.com/llvm/llvm-project/archive/$commit.tar.gz" \
            && echo "$sha  $target" | sha256sum -c --status; then
            return 0
        fi
        log "LLVM archive hash differs from the pinned $sha; GitHub regenerated it, retrying"
        sleep 30
    done
    die "GitHub did not serve the LLVM $commit archive with the sha256 TensorFlow pins ($sha).
  Export PRIMUS_TF_LLVM_ARCHIVE=/path/to/$commit.tar.gz pointing at a copy that
  matches, re-run 'bash setup.sh tf_source', or use the lighter tf_cpu_fix stage."
}

stage_rccl() {
    reload_env
    export PRIMUS_JAX_KEEP_ROCM_LD=1
    reload_env
    # Optional since v26.7, which takes RCCL straight from the pip ROCm SDK --
    # the published image no longer overrides it, so the default stage list does
    # not run this. Kept for reproducing v26.6, or for hosts that need the
    # rocm-systems net-ib fix (ROCM-27881).
    #
    # Builds RCCL from source (rocm-systems) and drops the libraries into
    # the pip ROCm tree so JAX/XLA's collectives use it. Requires hipcc from
    # rocm-sdk-devel -> run AFTER the `rocm` stage. Existing librccl* entries
    # are symlinks from rocm-sdk-libraries; rm them first (a bare cp onto a
    # symlink hits ELOOP).
    [ -n "${ROCM_PATH:-}" ] || die "ROCM_PATH not set; run the 'rocm' stage first"
    log "Building RCCL from source @ $RCCL_COMMIT -> $ROCM_PATH/lib"
    fresh_clone "$RCCL_REPO" rocm-systems
    ( cd "$SRC_DIR/rocm-systems" \
        && git checkout "$RCCL_COMMIT" \
        && cd projects/rccl \
        && ./install.sh -l --prefix build/ --amdgpu_targets="$PYTORCH_ROCM_ARCH" \
        && rm -f "$ROCM_PATH"/lib/librccl* \
        && cp -r build/release/librccl* "$ROCM_PATH/lib/" ) || die "RCCL build failed"
    rm -rf "$SRC_DIR/rocm-systems"
    unset PRIMUS_JAX_KEEP_ROCM_LD
    reload_env
}

stage_primus() {
    reload_env
    log "Installing Primus @ $PRIMUS_BRANCH (kept in $WORKSPACE_DIR/Primus)"
    rm -rf "$WORKSPACE_DIR/Primus"
    git clone --recurse-submodules "$PRIMUS_REPO" "$WORKSPACE_DIR/Primus" || die "Primus clone failed"
    ( cd "$WORKSPACE_DIR/Primus" \
        && git checkout "$PRIMUS_BRANCH" \
        && git submodule update --init third_party/maxtext/ ) || die "Primus checkout failed"
    # Primus' (torch-oriented) requirements.txt is not installed; the JAX runtime
    # deps live in requirements-jax.txt (the `jaxreqs` stage). Up to v26.7 the
    # image also removed stale dataclasses backports here, which conflict on
    # modern Python; that is kept.
    $PIP uninstall -y dataclasses dataclasses_json || true
}

stage_jaxreqs() {
    reload_env
    # Primus' JAX/MaxText runtime deps (normally installed by the Primus
    # MaxText pre-train hook at launch time). Front-load them here.
    local req="$WORKSPACE_DIR/Primus/requirements-jax.txt"
    if [ -f "$req" ]; then
        log "Installing Primus JAX requirements from $req"
        $PIP install -r "$req"
    else
        log "Primus requirements-jax.txt not found (skipping); run 'primus' stage first"
    fi
    # maxtext/Primus pull these in transitively at older, CVE-affected versions.
    # jax/jaxlib are re-pinned as a guardrail, as in the image, so nothing in the
    # requirements above can move them off JAX_VERSION.
    log "Force-upgrading CVE-fix pins from the v26.8 image"
    $PIP install --upgrade \
        pillow==12.3.0 \
        starlette==1.3.1 \
        pyasn1==0.6.4 \
        cryptography==50.0.0 \
        httplib2==0.32.0 \
        urllib3==2.8.0 \
        virtualenv==21.7.13 \
        black==26.3.1 \
        msgpack==1.2.1 \
        setuptools==80.10.2 \
        flax==0.12.9 \
        keras==3.15.0 \
        nltk==3.10.3 \
        "jax==${JAX_VERSION}" \
        "jaxlib==${JAX_VERSION}"
    python - <<'PY' || true
import os, shutil, sysconfig
for base in {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}:
    p = os.path.join(base, "flaxlib_src")
    if os.path.isdir(p):
        shutil.rmtree(p)
        print("removed", p)
PY
}

stage_manifest() {
    reload_env
    log "Writing manifest to $WORKSPACE_DIR/.manifest"
    mkdir -p "$WORKSPACE_DIR/.manifest"
    env > "$WORKSPACE_DIR/.manifest/env.txt"
    echo "Dockerfile.jax-v26.8" > "$WORKSPACE_DIR/.manifest/derived_from"
    $PIP list > "$WORKSPACE_DIR/.manifest/requirements.txt"
    cp "$SCRIPT_DIR/env.sh" "$WORKSPACE_DIR/.manifest/env.sh"
}

# Order: pip ROCm -> MaxText -> TF-from-source -> JAX (overrides MaxText)
# -> TE -> Primus. JAX/TE stay after MaxText so its setup.sh
# cannot clobber the ROCm plugin / TE wheels.
DEFAULT_STAGES=(venv rocm maxtext tf_source jax te primus jaxreqs manifest)

run_stage() { local s="$1"; local fn="stage_$s"; declare -F "$fn" >/dev/null || die "unknown stage: $s"; "$fn"; }

main() {
    if [ "${1:-}" = "--list" ]; then
        echo "default: ${DEFAULT_STAGES[*]}"
        echo "note:     the 'te' stage auto-falls-back to a from-source build on glibc < 2.28 hosts"
        echo "optional: te_source  (force the from-source TransformerEngine build regardless of glibc)"
        echo "optional: tf_cpu_fix (lighter alternative to tf_source: pip tensorflow-cpu instead of the ~30-60 min bazel build)"
        echo "optional: rccl       (v26.6 behaviour: override the SDK's RCCL with a rocm-systems build; the image uses the ROCm SDK copy)"
        exit 0
    fi
    local stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=("${DEFAULT_STAGES[@]}")
    log "Base dir: $PRIMUS_JAX_BASE | arch: $PYTORCH_ROCM_ARCH | stages: ${stages[*]}"
    df -h "$PRIMUS_JAX_BASE" 2>/dev/null | tail -1 || true
    for s in "${stages[@]}"; do run_stage "$s"; done
    log "DONE. Activate later with:  source $SCRIPT_DIR/env.sh"
}

main "$@"
