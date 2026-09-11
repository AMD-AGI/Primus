#!/usr/bin/env bash
# setup.sh — Reproduce the Primus JAX/MaxText training environment (v26.6) in a
# Python venv (no sudo, no docker). Mirrors the v26.6 JAX training Dockerfile,
# adapted for bare metal:
#   * ROCm from pip TheRock wheels (`rocm-sdk-*` 7.14.0 on repo.amd.com), same
#     delivery as the v26.6 image (v26.5 used a release tarball)
#   * builds/checkouts on a big disk (home quota is usually tiny)
#   * GPU arch auto-detected (gfx942 and/or gfx950); apt/sudo steps skipped
#   * MaxText's own setup.sh apt/interactive steps skipped (system packages are
#     a one-time root action, documented in the guide's Section 2)
#   * TensorFlow (2.21 CPU) and RCCL are built from source in the default flow
#     (matching the v26.6 image)
#   * JAX 0.11.0 + jax_rocm7_{pjrt,plugin} 0.11.0.post1 from PyPI
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

# ---- pinned versions / commits (from Dockerfile.jax-v26.6) ----
# ROCm: TheRock pip wheels (not the v26.5 release tarball).
# See: https://repo.amd.com/rocm/whl-multi-arch/
ROCM_INDEX="https://repo.amd.com/rocm/whl-multi-arch"
THE_ROCK_VERSION="7.14.0"

# JAX + ROCm PJRT/plugin — v26.6 publishes these on PyPI (cp312 plugin).
JAX_VERSION="0.11.0"
JAX_ROCM_VERSION="0.11.0.post1"

# TransformerEngine (prebuilt ROCm JAX wheel)
# See: https://rocm.frameworks-devreleases.amd.com/whl-multi-arch-staging/transformer-engine-rocm-jax/
TE_VERSION="2.17.0+rocm7.14.0.50a84ad"
TE_INDEX="https://rocm.frameworks-devreleases.amd.com/whl-multi-arch-staging/"
# From-source TransformerEngine (used by the `te_source` stage / glibc < 2.38).
TE_REPO="https://github.com/ROCm/TransformerEngine.git"
TE_SOURCE_COMMIT="${TE_SOURCE_COMMIT:-50a84ad}"

# TensorFlow (CPU) built from source — replaces the stock PyPI wheel, whose
# bundled LLVM collides with ROCm's libLLVM in Grain workers (SIGSEGV on
# `import tensorflow` after `import jax`). A CPU build also drops the bundled
# NCCL, preserving the XLA->RCCL collective fix.
TF_REPO="https://github.com/ROCm/tensorflow-upstream.git"
TF_BRANCH="upstream-v2.21.0"
BAZELISK_VERSION="v1.29.0"

# RCCL built from source (rocm-systems), installed into the ROCm tree.
RCCL_REPO="https://github.com/ROCm/rocm-systems.git"
RCCL_COMMIT="9e5e4084a4b8e1e86551b0eb054725c62354a926"

# MaxText (ROCm fork)
MAXTEXT_REPO="https://github.com/ROCm/maxtext.git"
MAXTEXT_BRANCH="${MAXTEXT_BRANCH:-release/v26.6}"
# Which MaxText requirements set to install. The reference Docker image runs
# MaxText's setup.sh with defaults (DEVICE=tpu), which — on ROCm — pulls the
# framework-agnostic deps WITHOUT any CUDA packages. Override to `cuda12` only
# if you specifically need that set.
MAXTEXT_DEVICE="${MAXTEXT_DEVICE:-tpu}"

# Primus
PRIMUS_REPO="https://github.com/AMD-AGI/Primus.git"
PRIMUS_BRANCH="main"

PIP="python -m pip"
UVPIP="python -m uv pip"

# Fresh clone helper: clone into transient SRC_DIR, build, then optionally clean.
fresh_clone() {  # fresh_clone <url> <dir> [extra git clone args...]
    local url="$1"; local dir="$2"; shift 2
    rm -rf "${SRC_DIR:?}/$dir"
    git clone "$@" "$url" "$SRC_DIR/$dir"
}

# Return 0 if the host glibc is >= 2.38 (what the prebuilt TE/JAX wheels need).
# On failure to parse (unknown libc) we conservatively return non-zero so the
# caller falls back to the always-works from-source build.
_glibc_ge_238() {
    local v; v="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
    [ -n "$v" ] || return 1
    awk -v v="$v" 'BEGIN{split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=38))}'
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
        uv
    rm -rf /root/.cache 2>/dev/null || true
}

stage_rocm() {
    reload_env
    # v26.6: ROCm comes from TheRock pip wheels (same as the JAX image), not a
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
    log "Installing JAX ${JAX_VERSION} + ROCm PJRT/plugin ${JAX_ROCM_VERSION} from PyPI (v26.6)"
    # Note: JAX and related libraries need to be installed BEFORE TE and AFTER
    # MaxText (whose setup.sh pulls in a stock jax/tensorflow we override here).
    $PIP install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" scipy==1.16 \
        "jax_rocm7_pjrt==${JAX_ROCM_VERSION}" \
        "jax_rocm7_plugin==${JAX_ROCM_VERSION}"
    python -c "import jax; print('jax', jax.__version__); print('devices:', jax.devices())" || \
        log "WARNING: jax.devices() failed (expected if no GPU is visible on this build host)"
}

stage_te() {
    reload_env
    # The prebuilt wheel targets the Dockerfile's ubuntu:24.04 base (glibc>=2.38).
    # On older hosts (e.g. Ubuntu 22.04, glibc 2.35) it cannot load, so fall back
    # transparently to the from-source build, which links against the host glibc.
    # This keeps `bash setup.sh` (defaults) working on both 22.04 and 24.04.
    if ! _glibc_ge_238; then
        log "host glibc < 2.38 (prebuilt TE needs >= 2.38): building TransformerEngine from source instead (te_source)"
        stage_te_source
        return
    fi
    log "Installing TransformerEngine (JAX) ${TE_VERSION} (prebuilt ROCm wheel)"
    $PIP install \
        pybind11==3.0.4 \
        importlib-metadata==8.7.1 \
        pydantic==2.13.4 \
        flax==0.12.8
    $PIP install \
        --index-url "$TE_INDEX" \
        --pre \
        --no-build-isolation \
        "transformer_engine_rocm_jax==${TE_VERSION}"

    # The prebuilt wheel targets the Dockerfile's ubuntu:24.04 base (glibc>=2.38,
    # GCC 13/14 libstdc++). Verify it actually loads NOW — otherwise training
    # dies later with a silent exit (the launcher swallows the OSError).
    log "Verifying TransformerEngine loads"
    if ! python -c "import transformer_engine.jax" 2>/tmp/te_import_err; then
        cat /tmp/te_import_err >&2 || true
        if grep -q "GLIBC_2" /tmp/te_import_err 2>/dev/null; then
            die "TransformerEngine failed to import: your host glibc ($(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')) is older than the prebuilt wheel requires (needs glibc>=2.38 / Ubuntu 24.04). Options: run on Ubuntu 24.04, or build TransformerEngine from source (see docs Section 3.7). glibc cannot be side-loaded via LD_LIBRARY_PATH."
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
    # kittens_grouped_mxfp8_gemm (CDNA4). The v26.6 image builds both archs.
    $PIP install pybind11==3.0.4 importlib-metadata==8.7.1 pydantic==2.13.4 flax==0.12.8
    # Remove ALL prebuilt/stale TE variants so TE's install sanity-check sees a
    # single, self-consistent from-source package.
    $PIP uninstall -y transformer_engine transformer-engine transformer_engine_rocm7 \
        transformer_engine_rocm_jax transformer_engine_jax 2>/dev/null || true
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
    # for MODE=stable, WORKFLOW=pre-training (the reference image's default).
    # The apt/gcsfuse and interactive-venv steps of that script are skipped:
    # system packages are a one-time root action (see the guide's Section 2)
    # and the venv already exists here.
    local req="src/dependencies/requirements/generated_requirements/${MAXTEXT_DEVICE}-requirements.txt"
    [ -f "$MAXTEXT_DIR/$req" ] || die "MaxText requirements not found: $MAXTEXT_DIR/$req"
    ( cd "$MAXTEXT_DIR" \
        && $PIP install -U setuptools wheel uv \
        && $UVPIP install --resolution=lowest -r "$req" \
        && python -m src.dependencies.scripts.install_pre_train_extra_deps \
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
    # Build the wheel for the venv's ACTUAL Python (the reference image uses 3.12;
    # deriving the version keeps the recipe working on hosts whose interpreter is
    # 3.13). A cp312 wheel will not install into a cp313 venv and vice versa.
    local pyver pytag
    pyver="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    pytag="cp${pyver//./}"   # 3.12 -> cp312, 3.13 -> cp313
    # Keep bazel's (large) output tree off the tiny home quota.
    ( cd "$SRC_DIR/tensorflow-upstream" \
        && "$bz" --output_user_root="$SRC_DIR/bazel" build //tensorflow/tools/pip_package:wheel \
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
    rm -rf "$SRC_DIR/tensorflow-upstream" "$SRC_DIR/bazel" 2>/dev/null || true
}

stage_rccl() {
    reload_env
    export PRIMUS_JAX_KEEP_ROCM_LD=1
    reload_env
    # v26.6: build RCCL from source (rocm-systems) and drop the libraries into
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
    # The JAX Dockerfile does NOT pip install Primus' (torch-oriented)
    # requirements.txt; the JAX runtime deps live in requirements-jax.txt
    # (installed by the `jaxreqs` stage). It also removes stale dataclasses
    # backports that conflict on modern Python.
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
    log "Force-upgrading CVE-fix pins from the v26.6 image"
    $PIP install --upgrade \
        pillow==12.3.0 \
        starlette==1.3.1 \
        pyasn1==0.6.4 \
        cryptography==50.0.0 \
        httplib2==0.32.0 \
        black==26.3.1 \
        msgpack==1.2.1 \
        setuptools==80.10.2 \
        flax==0.12.8 \
        keras==3.15.0
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
    echo "Dockerfile.jax-v26.6" > "$WORKSPACE_DIR/.manifest/derived_from"
    $PIP list > "$WORKSPACE_DIR/.manifest/requirements.txt"
    cp "$SCRIPT_DIR/env.sh" "$WORKSPACE_DIR/.manifest/env.sh"
}

# v26.6 order: pip ROCm -> MaxText -> TF-from-source -> JAX (overrides MaxText)
# -> TE -> Primus -> RCCL-from-source. JAX/TE stay after MaxText so its setup.sh
# cannot clobber the ROCm plugin / TE 2.17 wheels.
DEFAULT_STAGES=(venv rocm maxtext tf_source jax te primus jaxreqs rccl manifest)

run_stage() { local s="$1"; local fn="stage_$s"; declare -F "$fn" >/dev/null || die "unknown stage: $s"; "$fn"; }

main() {
    if [ "${1:-}" = "--list" ]; then
        echo "default: ${DEFAULT_STAGES[*]}"
        echo "note:     the 'te' stage auto-falls-back to a from-source build on glibc < 2.38 hosts (e.g. Ubuntu 22.04)"
        echo "optional: te_source  (force the from-source TransformerEngine build regardless of glibc)"
        echo "optional: tf_cpu_fix (lighter alternative to tf_source: pip tensorflow-cpu instead of the ~30-60 min bazel build)"
        exit 0
    fi
    local stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=("${DEFAULT_STAGES[@]}")
    log "Base dir: $PRIMUS_JAX_BASE | arch: $PYTORCH_ROCM_ARCH | stages: ${stages[*]}"
    df -h "$PRIMUS_JAX_BASE" 2>/dev/null | tail -1 || true
    for s in "${stages[@]}"; do run_stage "$s"; done
    log "DONE. Activate later with:  source $SCRIPT_DIR/env.sh"
}

main "$@"
