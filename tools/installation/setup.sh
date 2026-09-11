#!/usr/bin/env bash
# setup.sh — Reproduce the Primus training environment in a Python venv
# (no sudo, no docker). Mirrors the pins of
#   .github/workflows/docker-release/Dockerfile.primus-v26.6
# adapted for:
#   * Python 3.12, auto-provisioned with `uv` (the pinned torch nightly is
#     cp312-only on Linux; see README.md)
#   * ROCm provided by the pip `rocm-sdk-devel` wheel (no system ROCm needed)
#   * Transient build sources on the local /tmp disk (fast, and freed afterwards)
#   * GPU arch auto-detected (gfx942 and/or gfx950); apt/sudo steps skipped
#
# Usage:
#   export PRIMUS_BASE=/big/disk/primus-env   # REQUIRED, no default
#   bash setup.sh                # run all default stages in order
#   bash setup.sh <stage>...     # run only specific stage(s), e.g.
#   bash setup.sh venv torch flash_attn
#
# Stages are re-runnable. List them with:  bash setup.sh --list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The Primus checkout these scripts live in. Training is usually launched from
# here rather than from the workspace clone, so it needs the same treatment.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_STAGES=(venv torch flash_attn te torchtune torchao pydeps \
                grouped_gemm causal_conv1d mamba primus aiter turbo boto \
                cleanup manifest)
OPTIONAL_STAGES=(torchrec)

usage() {
    cat <<EOF
setup.sh — build the Primus v26.6 training environment in a venv.

PRIMUS_BASE must be exported first; it has no default because the right location
is site-specific. Point it at a writable dir on a disk with tens of GB free:

    export PRIMUS_BASE="\$HOME/envs/primus-env"
    bash setup.sh

Run selected stages only (they are idempotent, so this is how you resume after a
failure):

    bash setup.sh te
    bash setup.sh venv torch

default stages: ${DEFAULT_STAGES[*]}
optional stages: ${OPTIONAL_STAGES[*]}
EOF
}

# Answer informational flags before sourcing env.sh, so listing the stages does
# not require PRIMUS_BASE.
case "${1:-}" in
    --list|-h|--help) usage; exit 0 ;;
esac

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

log()  { echo -e "\n\033[1;36m[setup] $*\033[0m"; }
die()  { echo -e "\033[1;31m[setup][ERROR] $*\033[0m" >&2; exit 1; }
# shellcheck disable=SC1091
reload_env() { source "$SCRIPT_DIR/env.sh"; }

# ---- pinned versions / commits (from Dockerfile.primus-v26.6) ----
TORCH_INDEX="https://rocm.nightlies.amd.com/whl-multi-arch"
# The Dockerfile pins only `torch` and lets torchaudio/apex float and
# torchvision resolve as `==0.27`. That no longer resolves: newer rocm10.x
# nightlies now publish matching version numbers, so a floating torchvision
# drags in a build for a different ROCm line and the resolve dies on a
# backtracking storm. These are the versions the a20260727 nightly published,
# i.e. the exact set the Dockerfile picked up when it was built.
PYTORCH_VERSION="2.12.0+rocm7.15.0a20260727"
ROCM_SDK_VERSION="7.15.0a20260727"
TORCHAUDIO_VERSION="2.11.0+rocm7.15.0a20260727"
TORCHVISION_VERSION="0.27.0+rocm7.15.0a20260727"
APEX_VERSION="1.12.0+rocm7.15.0a20260727"
FLASH_ATTN_VERSION="2.8.1"
# v26.6 installs TransformerEngine from the ROCm multi-arch staging index.
# Those wheels are built on Ubuntu 24.04 though, and libtransformer_engine.so
# needs glibc >= 2.38, so they cannot load on a 22.04 host. stage_te falls back
# to building the same TE commit from source; see PRIMUS_TE_MODE and the README.
TE_INDEX="https://rocm.frameworks-nightlies.amd.com/whl-multi-arch-staging/"
TE_VERSION="2.17.0+rocm7.15.0a20260727.e028a6c"
TE_WHEEL_MIN_GLIBC="2.38"
TE_REPO="https://github.com/ROCm/TransformerEngine.git"
# The commit the TE_VERSION local label refers to (…a20260727.e028a6c).
TE_COMMIT="e028a6c"
TORCHTUNE_REPO="https://github.com/pytorch/torchtune.git"
TORCHTUNE_BRANCH="b4c98ac2a37f0397d64c22579aed415ce7264db6"
TORCHAO_REPO="https://github.com/pytorch/ao.git"
TORCHAO_BRANCH="e9c7bead90b840b280f97374308255957108ce47"
GROUPED_GEMM_REPO="https://github.com/caaatch22/grouped_gemm.git"
GROUPED_GEMM_BRANCH="rocm"
CAUSAL_CONV1D_REPO="https://github.com/Dao-AILab/causal-conv1d"
CAUSAL_CONV1D_BRANCH="e940ead2fd962c56854455017541384909ca669f"
MAMBA_REPO="https://github.com/AndreasKaratzas/mamba.git"
MAMBA_BRANCH="enable-primus-hybrid-models"
TVM_FFI_VERSION="0.1.11"
PRIMUS_REPO="https://github.com/AMD-AGI/Primus.git"
# Latest commit on `release/v26.6` branch. Committed on 2026-08-25.
PRIMUS_BRANCH="2aa05ead3401708cf1a1e2958c1def18d7aecf92"
AITER_REPO="https://github.com/ROCm/aiter.git"
AITER_COMMIT="0f3c58e6edb6754940bcf9fd5f09ccb6f389f52e"
TURBO_REPO="https://github.com/AMD-AGI/Primus-Turbo.git"
# Latest commit on `main` branch. Committed on 2026-08-21.
TURBO_COMMIT="a6a16cdce46bd2235a248b41f501fb363c215b9b"
# aiter pins `flydsl==0.1.7` and Primus-Turbo wants `flydsl>=0.2.0`, so one of
# them is always unsatisfied; Turbo installs last and wins. aiter only needs
# `flydsl.expr.vector` at runtime, which survived until 0.3.0 removed it -- and
# with it aiter's whole CK/HIP JIT path ("CK and HIP ops are disabled. Triton ops
# remain available."). 0.2.4 is the newest release that satisfies Turbo and still
# keeps aiter whole.
FLYDSL_VERSION="0.2.4"
NVDISASM_VERSION="13.3.73"

PIP="python -m pip"

# pip install that honours the torch constraint file once stage_torch has
# written it, so no later resolve can swap the ROCm stack for a CUDA one.
pipi() {
    local args=()
    [ -f "${PRIMUS_PIP_CONSTRAINTS:-}" ] && args+=( -c "$PRIMUS_PIP_CONSTRAINTS" )
    $PIP install "${args[@]}" "$@"
}

# Fresh clone helper: clone into transient SRC_DIR, build, then optionally clean.
fresh_clone() {  # fresh_clone <url> <dir> [extra git clone args...]
    local url="$1"; local dir="$2"; shift 2
    rm -rf "${SRC_DIR:?}/$dir"
    git clone "$@" "$url" "$SRC_DIR/$dir"
}

# ======================== PYTHON PROVISIONING ========================
# v26.5 needs cp312 wheels but Ubuntu 22.04 ships only python3.10 and we have no
# sudo, so fetch a standalone CPython with uv (bootstrapping uv if needed).

UV_BIN=""

py_is_target() {  # py_is_target <interpreter>
    local p="${1:-}"
    [ -n "$p" ] && [ -x "$p" ] || return 1
    "$p" -c "import sys; raise SystemExit(0 if '%d.%d' % sys.version_info[:2] == '$PRIMUS_PYTHON_VERSION' else 1)" 2>/dev/null
}

find_uv() {
    local c
    for c in "$(command -v uv 2>/dev/null || true)" "$PRIMUS_BASE/bin/uv" "$HOME/.local/bin/uv"; do
        [ -n "$c" ] && [ -x "$c" ] && { UV_BIN="$c"; return 0; }
    done
    return 1
}

bootstrap_uv() {
    log "uv not found; installing it into $PRIMUS_BASE/bin (no sudo required)"
    mkdir -p "$PRIMUS_BASE/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$PRIMUS_BASE/bin" INSTALLER_NO_MODIFY_PATH=1 sh \
        || die "could not install uv. Provide Python $PRIMUS_PYTHON_VERSION yourself and re-run with PRIMUS_PYTHON=/path/to/python$PRIMUS_PYTHON_VERSION"
    UV_BIN="$PRIMUS_BASE/bin/uv"
}

resolve_python() {
    if [ -n "${PRIMUS_PYTHON:-}" ]; then
        py_is_target "$PRIMUS_PYTHON" \
            || die "PRIMUS_PYTHON=$PRIMUS_PYTHON is not Python $PRIMUS_PYTHON_VERSION"
        return 0
    fi
    local cand
    cand="$(command -v "python$PRIMUS_PYTHON_VERSION" 2>/dev/null || true)"
    py_is_target "$cand" && { PRIMUS_PYTHON="$cand"; return 0; }
    for cand in "$UV_PYTHON_INSTALL_DIR"/cpython-"$PRIMUS_PYTHON_VERSION"*/bin/"python$PRIMUS_PYTHON_VERSION"; do
        py_is_target "$cand" && { PRIMUS_PYTHON="$cand"; return 0; }
    done
    return 1
}

provision_python() {
    resolve_python && return 0
    find_uv || bootstrap_uv
    log "Provisioning CPython $PRIMUS_PYTHON_VERSION via uv into $UV_PYTHON_INSTALL_DIR"
    "$UV_BIN" python install "$PRIMUS_PYTHON_VERSION" || die "uv python install $PRIMUS_PYTHON_VERSION failed"
    resolve_python && return 0
    PRIMUS_PYTHON="$("$UV_BIN" python find "$PRIMUS_PYTHON_VERSION" 2>/dev/null || true)"
    py_is_target "${PRIMUS_PYTHON:-}" || die "could not provision Python $PRIMUS_PYTHON_VERSION"
}

# Pin the GPU stack so a later `pip install` cannot quietly replace the ROCm
# torch/triton builds with upstream ones (several deps ask for a bare `torch`,
# and Primus-Turbo asks for an exact upstream `triton`).
write_pip_constraints() {
    python - "$PRIMUS_PIP_CONSTRAINTS" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

pins = []
for pkg in ("torch", "triton", "pytorch-triton-rocm"):
    try:
        pins.append(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        pass
with open(sys.argv[1], "w") as fh:
    fh.write("# Generated by tools/installation/setup.sh.\n")
    fh.write("# Keeps pip from replacing the ROCm GPU stack with upstream builds.\n")
    fh.write("\n".join(pins) + "\n")
print("constraints: " + ", ".join(pins))
PY
}

# Regenerate the constraint file if it is missing or predates the torch install,
# so resuming at a single late stage still gets the guard.
ensure_pip_constraints() {
    grep -q '^torch==' "${PRIMUS_PIP_CONSTRAINTS:-/nonexistent}" 2>/dev/null && return 0
    write_pip_constraints
}

# ============================ STAGES ============================

stage_venv() {
    mkdir -p "$PRIMUS_BASE" "$SRC_DIR" "$WORKSPACE_DIR"
    provision_python
    log "Creating venv at $VENV_DIR (interpreter: $PRIMUS_PYTHON, $("$PRIMUS_PYTHON" --version))"
    if [ -f "$VENV_DIR/bin/python" ]; then
        local have
        have="$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
        [ "$have" = "$PRIMUS_PYTHON_VERSION" ] || die \
"existing venv at $VENV_DIR is Python $have, but v26.6 requires $PRIMUS_PYTHON_VERSION.
  The pinned torch nightly ships a cp312 Linux wheel only, so an older venv
  cannot be upgraded in place. Remove it and re-run:
      rm -rf '$VENV_DIR' && bash setup.sh"
    else
        "$PRIMUS_PYTHON" -m venv "$VENV_DIR" || die "venv creation failed"
    fi
    reload_env
    $PIP install --upgrade pip
    # Build front-end tooling. patchelf via pip (system one is missing/no sudo).
    $PIP install \
        pybind11 \
        typeguard \
        wheel==0.46.2 \
        cmake==3.31.6 \
        ninja==1.11.1.3 \
        packaging==25.0 \
        setuptools==80.10.2 \
        patchelf
}

stage_torch() {
    reload_env
    log "Installing PyTorch ${PYTORCH_VERSION} + ROCm SDK (arch: $PYTORCH_ROCM_ARCH) from nightly index"
    # Early pip deps (Dockerfile block before torch). apex declares cxxfilt and
    # pytest requirements that the nightly index does not serve, so these have
    # to land before the torch install or its resolve fails.
    $PIP install \
        cxxfilt==0.3.0 \
        tqdm==4.67.3 \
        pyyaml==6.0.3 \
        pytest==9.0.3 \
        matplotlib==3.10.9 \
        pandas==2.3.3 \
        py-cpuinfo==9.0.0 \
        build==1.5.0

    $PIP uninstall -y torch || true

    # Per-arch device wheels, derived from PYTORCH_ROCM_ARCH (auto-detected in
    # env.sh, ";"-separated). Installs the matching gfx942 and/or gfx950 sets.
    local _arch arch_args=()
    local _arches; IFS=';' read -ra _arches <<< "$PYTORCH_ROCM_ARCH"
    for _arch in "${_arches[@]}"; do
        _arch="${_arch// /}"; [ -z "$_arch" ] && continue
        arch_args+=( "amd-torch-device-${_arch}==${PYTORCH_VERSION}" \
                     "rocm-sdk-device-${_arch}==${ROCM_SDK_VERSION}" \
                     "amd-torchvision-device-${_arch}==${TORCHVISION_VERSION}" )
    done
    [ ${#arch_args[@]} -gt 0 ] || die "no GPU arch resolved; export PYTORCH_ROCM_ARCH (e.g. gfx942;gfx950)"
    log "Installing device wheels for: $PYTORCH_ROCM_ARCH"

    $PIP install \
        --index-url "$TORCH_INDEX" \
        --pre \
        "rocm==${ROCM_SDK_VERSION}" \
        rocm-bootstrap \
        "rocm-sdk-core==${ROCM_SDK_VERSION}" \
        "rocm-sdk-devel==${ROCM_SDK_VERSION}" \
        "rocm-sdk-libraries==${ROCM_SDK_VERSION}" \
        "torch==${PYTORCH_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "apex==${APEX_VERSION}" \
        "${arch_args[@]}"

    log "Running rocm-sdk init"
    rocm-sdk init
    relink_comgr
    patch_rocprofv3
    reload_env
    [ -n "${ROCM_PATH:-}" ] || die "ROCM_PATH not resolved after rocm-sdk init"
    log "ROCM_PATH=$ROCM_PATH"
    write_pip_constraints
    python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())"
}

# rocm-sdk-core and rocm-sdk-devel both ship libamd_comgr.so.3 as separate
# inodes, causing duplicate LLVM static constructor registration (SIGABRT).
# Same workaround as Dockerfile.primus-v26.6 (AIMA-248).
relink_comgr() {
    python - <<'PY' || true
import os, pathlib
try:
    import _rocm_sdk_core, _rocm_sdk_devel
except Exception as e:
    print("comgr relink skipped:", e)
    raise SystemExit(0)
src = pathlib.Path(_rocm_sdk_devel.__file__).parent / "lib" / "libamd_comgr.so.3"
dst = pathlib.Path(_rocm_sdk_core.__file__).parent / "lib" / "libamd_comgr.so.3"
if src.exists() and dst.parent.exists():
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)
    print(f"linked {dst} -> {src}")
PY
}

patch_rocprofv3() {
    local f
    f="$(python - <<'PY'
import os
try:
    import _rocm_sdk_devel
    p = os.path.join(os.path.dirname(_rocm_sdk_devel.__file__), "bin", "rocprofv3")
    print(p if os.path.exists(p) else "")
except Exception:
    pass
PY
)"
    [ -n "$f" ] || return 0
    sed -i 's/update_env("ROCPROFILER_LIBRARY_CTOR", True)/update_env("ROCPROFILER_LIBRARY_CTOR", False)/' "$f" || true
}

stage_flash_attn() {
    reload_env
    # v26.6 switched from the ROCm/flash-attention git fork to the pip package.
    log "Installing flash-attn==${FLASH_ATTN_VERSION} (--no-build-isolation)"
    MAX_JOBS="$MAX_JOBS" GPU_ARCHS="$PYTORCH_ROCM_ARCH" pipi \
        --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}" \
        || die "flash-attn install failed"
    python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" \
        || die "flash_attn installed but cannot be imported"
}

# Concurrent CK JIT compiles race to publish the same .so; without this the
# loser aborts the run. Same patch v26.5 applies inside the image.
# The single quotes below are deliberate: $_TMP_SO and $OUTPUT are literal text
# in the target script, not variables to expand here.
# shellcheck disable=SC2016
patch_te_ck_jit() {
    local f
    f="$(python - <<'PY'
import os, sysconfig
for base in {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}:
    p = os.path.join(base, "transformer_engine", "lib", "ck_jit", "ck_jit_compile.sh")
    if os.path.exists(p):
        print(p)
        break
PY
)"
    if [ -z "$f" ]; then
        # Both install paths ship this today; tolerate a layout change upstream
        # rather than failing the build over a race-condition mitigation.
        log "ck_jit_compile.sh not present, nothing to patch"
        return 0
    fi
    if grep -qF 'mv -n "$_TMP_SO" "$OUTPUT" 2>/dev/null' "$f"; then
        log "ck_jit_compile.sh already patched"
        return 0
    fi
    sed -i 's|    mv -n "$_TMP_SO" "$OUTPUT"$|    mv -n "$_TMP_SO" "$OUTPUT" 2>/dev/null \|\| true|' "$f"
    grep -qF 'mv -n "$_TMP_SO" "$OUTPUT" 2>/dev/null' "$f" \
        || die "ck_jit_compile.sh patch did not apply; upstream changed $f"
    log "Patched $f"
}

host_glibc() { ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$'; }

# version_ge <a> <b> -> true when a >= b
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# The v26.5 wheels only load where glibc is new enough; otherwise build the same
# commit locally. Force either path with PRIMUS_TE_MODE=wheel|source.
resolve_te_mode() {
    local mode="${PRIMUS_TE_MODE:-auto}"
    if [ "$mode" != auto ]; then echo "$mode"; return 0; fi
    local glibc; glibc="$(host_glibc)"
    if [ -n "$glibc" ] && version_ge "$glibc" "$TE_WHEEL_MIN_GLIBC"; then
        echo wheel
    else
        echo source
    fi
}

# TE's own dependencies, needed by both install paths. TE_INDEX serves only the
# transformer-engine packages, so pip cannot fall back to PyPI for these while
# --index-url points there; they must already be installed. The Dockerfile gets
# einops transitively from flash-attention and onnx from onnxscript -- install
# both explicitly rather than relying on that.
install_te_deps() {
    pipi \
        pybind11==3.0.4 \
        importlib-metadata==8.7.1 \
        onnxscript==0.7.0 \
        pydantic==2.13.4 \
        nvdlfw_inspect==0.2.2 \
        einops==0.9.0.dev0 \
        onnx
}

stage_te_wheel() {
    log "Installing TransformerEngine $TE_VERSION from the ROCm staging index"
    install_te_deps
    # transformer_engine_rocm_torch ships as an sdist: the torch glue layer is
    # compiled here against the installed torch (hence --no-build-isolation),
    # and it pulls the prebuilt transformer_engine_rocm7 core wheel.
    MAX_JOBS="$MAX_JOBS" GPU_ARCHS="$PYTORCH_ROCM_ARCH" pipi \
        --index-url "$TE_INDEX" \
        --pre \
        --no-build-isolation \
        "transformer_engine_rocm_torch==${TE_VERSION}" \
        || die "TransformerEngine install failed"
}

stage_te_source() {
    log "Building TransformerEngine from source @ $TE_COMMIT (glibc $(host_glibc) < $TE_WHEEL_MIN_GLIBC, so the v26.6 wheels cannot load)"
    # Drop any previously wheel-installed TE, otherwise the unusable prebuilt
    # core library stays behind and keeps winning the import.
    $PIP uninstall -y transformer_engine_rocm_torch transformer_engine_rocm7 \
        transformer_engine transformer_engine_torch >/dev/null 2>&1 || true
    install_te_deps
    pipi psutil
    fresh_clone "$TE_REPO" TransformerEngine --recursive
    # NVTE_FRAMEWORK/NVTE_USE_ROCM/NVTE_ROCM_ARCH/NVTE_USE_HIPBLASLT only matter
    # for a from-source build, which is why env.sh no longer exports them.
    ( cd "$SRC_DIR/TransformerEngine" \
        && git checkout "$TE_COMMIT" \
        && git submodule update --init --recursive \
        && MAX_JOBS="$MAX_JOBS" \
           NVTE_FRAMEWORK=pytorch \
           NVTE_USE_ROCM=1 \
           NVTE_USE_HIPBLASLT=1 \
           NVTE_ROCM_ARCH="$PYTORCH_ROCM_ARCH" \
           GPU_ARCHS="$PYTORCH_ROCM_ARCH" \
           pipi --no-build-isolation . ) || die "TransformerEngine source build failed"
    rm -rf "$SRC_DIR/TransformerEngine"
}

stage_te() {
    reload_env
    local mode; mode="$(resolve_te_mode)"
    case "$mode" in
        wheel)  stage_te_wheel ;;
        source) stage_te_source ;;
        *) die "PRIMUS_TE_MODE must be one of auto, wheel, source (got '$mode')" ;;
    esac
    patch_te_ck_jit
    python -c "import transformer_engine" 2>/dev/null \
        || die "TransformerEngine installed but cannot be imported. Re-run with
  PRIMUS_TE_MODE=source to build it against this host's toolchain:
      bash setup.sh te"
}

stage_torchtune() {
    reload_env
    log "Installing torchtune @ $TORCHTUNE_BRANCH (with use_grouped_mm patch)"
    fresh_clone "$TORCHTUNE_REPO" torchtune
    ( cd "$SRC_DIR/torchtune" \
        && git checkout "$TORCHTUNE_BRANCH" \
        && sed -i 's/use_grouped_mm = True/use_grouped_mm = False/g' torchtune/modules/moe/utils.py \
        && pipi . ) || die "torchtune install failed"
    rm -rf "$SRC_DIR/torchtune"
}

stage_torchao() {
    reload_env
    log "Building torchao @ $TORCHAO_BRANCH (with pad_inner_dim + swizzle patches)"
    fresh_clone "$TORCHAO_REPO" ao
    ( cd "$SRC_DIR/ao" \
        && git checkout "$TORCHAO_BRANCH" \
        && sed -i 's/pad_inner_dim: bool = False/pad_inner_dim: bool = True/g' torchao/float8/config.py \
        && sed -i 's/if defined(HIPBLASLT_VEC_EXT)/if false/g' torchao/csrc/rocm/swizzle/swizzle.cpp \
        && pipi --no-build-isolation . ) || die "torchao build failed"
    rm -rf "$SRC_DIR/ao"
}

stage_pydeps() {
    reload_env
    log "Installing main pip dependency set"
    pipi \
        datasets==3.6.0 \
        av==16.0.1 \
        transformers==5.5.0 \
        optree==0.18.0 \
        sympy \
        accelerate==1.9.0 \
        trl==0.21.0 \
        tensorboard==2.20.0 \
        peft \
        scipy \
        einops==0.9.0.dev0 \
        flask-restful \
        nltk \
        pytest \
        pytest-cov \
        pytest_mock \
        pytest-csv \
        pytest-random-order \
        sentencepiece \
        wrapt \
        zarr==2.18.7 \
        numcodecs==0.12.1 \
        xarray \
        wandb==0.28.2 \
        tensorstore==0.1.45 \
        pybind11 \
        tiktoken \
        pynvml \
        z3-solver \
        "huggingface_hub[cli]"
    python -m nltk.downloader punkt_tab || true
}

stage_grouped_gemm() {
    reload_env
    log "Building grouped_gemm @ $GROUPED_GEMM_BRANCH"
    fresh_clone "$GROUPED_GEMM_REPO" grouped_gemm
    ( cd "$SRC_DIR/grouped_gemm" \
        && git checkout "$GROUPED_GEMM_BRANCH" \
        && git submodule update --init --recursive \
        && pipi --no-build-isolation . ) || die "grouped_gemm build failed"
    rm -rf "$SRC_DIR/grouped_gemm"
}

stage_causal_conv1d() {
    reload_env
    log "Building causal-conv1d @ $CAUSAL_CONV1D_BRANCH"
    fresh_clone "$CAUSAL_CONV1D_REPO" causal-conv1d
    ( cd "$SRC_DIR/causal-conv1d" \
        && git checkout "$CAUSAL_CONV1D_BRANCH" \
        && pipi --no-build-isolation . ) || die "causal-conv1d build failed"
    rm -rf "$SRC_DIR/causal-conv1d"
}

# mamba_ssm pins quack-kernels, which pulls nvidia-cutlass-dsl. Its MLIR Python
# bindings and FlyDSL's (installed with Primus-Turbo) share one process-wide
# nanobind type registry, so whichever loads second aborts (AMD-AGI/Primus#955).
# No ROCm path can run these CUDA-only kernels and mamba's ops/cute/mamba3, their
# only importer, degrades silently without them: mamba_370M ran 50 iterations to
# bit-identical loss and grad norm after removal.
purge_cutlass_dsl() {
    local pkgs
    pkgs="$(python - <<'PY'
import importlib.metadata as md

names = set()
for dist in md.distributions():
    name = dist.metadata["Name"] or ""
    if name.lower().startswith(("quack-kernels", "nvidia-cutlass-dsl")):
        names.add(name)
print(" ".join(sorted(names)))
PY
)"
    [ -n "$pkgs" ] || { log "No CUTLASS DSL packages installed; nothing to purge"; return 0; }
    log "Uninstalling CUDA-only CUTLASS DSL stack: $pkgs"
    # shellcheck disable=SC2086  # deliberate word splitting: one package per arg
    $PIP uninstall -y $pkgs || die "failed to uninstall: $pkgs"
}

stage_mamba() {
    reload_env
    log "Building mamba @ $MAMBA_BRANCH"
    fresh_clone "$MAMBA_REPO" mamba --branch "$MAMBA_BRANCH"
    # IMPORTANT: use pip, NOT `python setup.py install`. The legacy easy_install
    # path does not recognize pip-installed packages and re-fetches the LATEST
    # of every unpinned dep as .egg files, clobbering our pins (it pulled
    # transformers 5.x, removed accelerate/trl, and dragged in NVIDIA CUDA
    # packages). pip respects already-installed versions.
    ( cd "$SRC_DIR/mamba" \
        && pipi "apache-tvm-ffi==${TVM_FFI_VERSION}" \
           "nvidia-cuda-nvdisasm==${NVDISASM_VERSION}" \
        && sed -i '/^\s*namespace std\s*=\s*::std\s*;/d' csrc/selective_scan/uninitialized_copy.cuh \
        && sed -i '/namespace cuda {/,/^    }/{/namespace cuda {/d;/namespace std = ::std;/d;/^    }$/d;}' \
               csrc/selective_scan/uninitialized_copy.cuh \
        && sed -i 's/::cuda::std::/::std::/g' csrc/selective_scan/uninitialized_copy.cuh \
        && pipi --no-build-isolation . ) || die "mamba build failed"
    rm -rf "$SRC_DIR/mamba"
    # Ahead of the check below, which then doubles as proof mamba survives it.
    purge_cutlass_dsl
    python -c "import mamba_ssm" 2>/dev/null \
        || die "mamba_ssm cannot be imported after purge_cutlass_dsl removed
  quack-kernels. Gate the import instead of putting the packages back --
  restoring nvidia-cutlass-dsl reintroduces Primus#955."
}

# Megatron's dataset indexing imports a pybind11 extension that has to be
# compiled, or training dies with `MockGPTDataset failed to build as a mock data
# generator`. Two traps:
#  * The Makefile takes the output filename from `python3-config
#    --extension-suffix`, but a venv does not ship `python3-config`, so it falls
#    through to the system interpreter and emits a `cpython-310` name that this
#    3.12 venv will never import. Pass LIBEXT explicitly.
#  * A checkout that has been bind-mounted into the training container may already
#    hold a root-owned .so built on Ubuntu 24.04. That one needs GLIBCXX_3.4.32
#    and cannot load here, yet it takes precedence over anything we build. It is
#    also not writable by us, so delete it before rebuilding.
compile_megatron_helpers() {  # compile_megatron_helpers <primus_checkout>
    local root="$1"
    local dir="$root/third_party/Megatron-LM/megatron/core/datasets"
    [ -f "$dir/Makefile" ] || return 0
    local ext
    ext="$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    [ -n "$ext" ] || die "could not determine this interpreter's extension suffix"
    log "Compiling Megatron helpers_cpp in $dir"
    rm -f "$dir/helpers_cpp$ext"
    make -C "$dir" LIBEXT="$ext" || die "helpers_cpp build failed in $dir"
    python - "$dir/helpers_cpp$ext" <<'PY' || die "helpers_cpp built but will not load in $dir"
import importlib.util, sys

spec = importlib.util.spec_from_file_location("helpers_cpp", sys.argv[1])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
PY
}

stage_primus() {
    reload_env
    log "Installing Primus @ $PRIMUS_BRANCH (kept in $WORKSPACE_DIR/Primus)"
    rm -rf "$WORKSPACE_DIR/Primus"
    git clone --recurse-submodules "$PRIMUS_REPO" "$WORKSPACE_DIR/Primus" || die "Primus clone failed"
    ( cd "$WORKSPACE_DIR/Primus" \
        && git checkout "$PRIMUS_BRANCH" \
        && git submodule update --init --recursive \
        && pipi -r requirements.txt ) || die "Primus install failed"
    # Build the Megatron dataset extension for every checkout that might be used.
    local seen=" " root
    for root in "$WORKSPACE_DIR/Primus" "$HOME/.cache/Primus" "$REPO_ROOT"; do
        case "$seen" in *" $root "*) continue ;; esac
        seen="$seen$root "
        compile_megatron_helpers "$root"
    done
}

stage_aiter() {
    reload_env
    log "Building aiter @ $AITER_COMMIT (kept in $WORKSPACE_DIR/aiter)"
    $PIP uninstall aiter amd-aiter -y || true
    rm -rf "$WORKSPACE_DIR/aiter"
    git clone --recursive "$AITER_REPO" "$WORKSPACE_DIR/aiter" || die "aiter clone failed"
    ( cd "$WORKSPACE_DIR/aiter" \
        && git checkout "$AITER_COMMIT" \
        && git submodule update --init --recursive \
        && PREBUILD_KERNELS=3 GPU_ARCHS="$PYTORCH_ROCM_ARCH" \
           pipi --no-cache-dir --use-pep517 . ) || die "aiter build failed"
}

stage_turbo() {
    reload_env
    ensure_pip_constraints
    log "Building Primus-Turbo @ $TURBO_COMMIT"
    # Installed with --no-deps on purpose. Primus-Turbo's setup.py hard-pins
    # upstream `triton==3.7.0`, but torch is built against ROCm's
    # `triton==3.7.1+git...rocm...`. Letting the upstream wheel win (as the
    # unconstrained Dockerfile does) breaks this environment: ROCm's HIP runtime
    # already loads its own libLLVM.so.23 from the pip SDK, and upstream triton
    # bundles a second, statically linked LLVM, so importing it segfaults inside
    # LLVM's static initialisers. That takes down torch._dynamo, aiter, torchao
    # and mamba_ssm. Keeping ROCm's triton fixes all of them, so install without
    # deps and supply the real runtime requirements explicitly.
    #
    # Disable the DeepEP internode path. Primus-Turbo probes for rocSHMEM and,
    # finding none, falls back to the pip ROCm SDK dir, which does exist and even
    # ships rocshmem headers and device bitcode but no host `librocshmem.a`. With
    # a system OpenMPI on PATH that false positive turns the internode link on,
    # which then fails on the missing librocshmem.a. v26.5 dropped rocSHMEM from
    # the image entirely, so make the probe fail cleanly instead: it treats a
    # non-existent ROCSHMEM_HOME as "not found" and warns that internode DeepEP is
    # disabled. Export ROCSHMEM_HOME yourself if you do have a real rocSHMEM.
    local rocshmem_home="${ROCSHMEM_HOME:-$PRIMUS_BASE/.rocshmem-not-installed}"
    if [ -z "${ROCSHMEM_HOME:-}" ] && [ -e "$rocshmem_home" ]; then
        die "the placeholder path $rocshmem_home exists, so rocSHMEM detection
  would not be disabled as intended. Delete it, or export ROCSHMEM_HOME to point
  at a real rocSHMEM install."
    fi
    fresh_clone "$TURBO_REPO" Primus-Turbo --recursive
    ( cd "$SRC_DIR/Primus-Turbo" \
        && git checkout "$TURBO_COMMIT" \
        && git submodule update --init --recursive \
        && pipi -r requirements.txt \
        && pipi scipy "flydsl==${FLYDSL_VERSION}" \
        && GPU_ARCHS="$PYTORCH_ROCM_ARCH" \
           ROCSHMEM_HOME="$rocshmem_home" \
           pipi --no-build-isolation --no-deps . -v ) || die "Primus-Turbo build failed"
    rm -rf "$SRC_DIR/Primus-Turbo"
    # aiter silently downgrades itself to Triton-only ops if this symbol is gone,
    # so fail here instead of losing the CK/HIP kernels at training time.
    python -c "from flydsl.expr import vector" 2>/dev/null \
        || die "flydsl ${FLYDSL_VERSION} has no flydsl.expr.vector, so aiter's CK/HIP
  ops would be disabled. Pick a flydsl that still exports it and satisfies
  Primus-Turbo's flydsl>=0.2.0."
}

stage_boto() {
    reload_env
    log "Installing boto3/botocore and CVE-fix pins from the v26.6 image"
    pipi boto3==1.35.42 botocore==1.35.99
    pipi cryptography==50.0.0 diffusers==0.38.0 "jaraco.context==6.1.0" pyarrow==23.0.1
    # mlflow caps cryptography<50; --no-deps keeps the 50.0.0 pin above.
    $PIP install --no-deps mlflow==3.15.1
}

stage_cleanup() {
    reload_env
    # mamba-ssm depends on tilelang, which v26.5 removes again once everything
    # is built to keep it out of the dependency graph at runtime.
    log "Uninstalling tilelang (pulled in by mamba-ssm)"
    $PIP uninstall -y tilelang || true
}

stage_manifest() {
    reload_env
    log "Writing manifest to $WORKSPACE_DIR/.manifest"
    mkdir -p "$WORKSPACE_DIR/.manifest"
    env > "$WORKSPACE_DIR/.manifest/env.txt"
    $PIP list > "$WORKSPACE_DIR/.manifest/requirements.txt"
    python --version > "$WORKSPACE_DIR/.manifest/python_version"
    echo "Dockerfile.primus-v26.6" > "$WORKSPACE_DIR/.manifest/derived_from"
    cp "$SCRIPT_DIR/env.sh" "$WORKSPACE_DIR/.manifest/env.sh"
    [ -f "$PRIMUS_PIP_CONSTRAINTS" ] && cp "$PRIMUS_PIP_CONSTRAINTS" "$WORKSPACE_DIR/.manifest/"
    log "Environment ready. torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '??')"
}

# ---- Optional stages (DLRM / recommendation stack); not in default run ----
stage_torchrec() {
    reload_env
    log "Installing torchrec stack (optional)"
    pipi --no-deps torchrec
    pipi tensordict iopath torchmetrics==1.0.3 \
        "git+https://github.com/mlperf/logging.git" \
        --extra-index-url "$TORCH_INDEX"
}

run_stage() { local s="$1"; local fn="stage_$s"; declare -F "$fn" >/dev/null || die "unknown stage: $s"; "$fn"; }

main() {
    local stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=("${DEFAULT_STAGES[@]}")
    log "Base dir: $PRIMUS_BASE | python: $PRIMUS_PYTHON_VERSION | arch: $PYTORCH_ROCM_ARCH | stages: ${stages[*]}"
    # Tolerate a base dir that does not exist yet: under `pipefail` a failing df
    # would otherwise abort the whole run before the first stage.
    df -h "$PRIMUS_BASE" 2>/dev/null | tail -1 || true
    for s in "${stages[@]}"; do run_stage "$s"; done
    log "DONE. Activate later with:  source $SCRIPT_DIR/env.sh"
}

main "$@"
