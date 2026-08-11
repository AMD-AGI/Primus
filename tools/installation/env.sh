#!/usr/bin/env bash
# env.sh — Primus venv environment (Python 3.12, ROCm via pip rocm-sdk-devel)
# Derived from .github/workflows/docker-release/Dockerfile.primus-v26.5
#
# Source this both during the build (setup.sh does it) and every time you
# want to USE the environment:   source env.sh
#
# PRIMUS_BASE is REQUIRED and intentionally has no default: the right location is
# site-specific, and silently falling back to some other host's path just moves
# the failure somewhere less obvious. Point it at a writable directory on a disk
# with tens of GB free; it holds the venv, the provisioned interpreter and the
# Primus/aiter checkouts.

if [ -z "${PRIMUS_BASE:-}" ]; then
    echo "[env] ERROR: PRIMUS_BASE is not set." >&2
    echo "[env]   Export it first, pointing at a writable dir on a big disk:" >&2
    echo "[env]       export PRIMUS_BASE=\"\$HOME/envs/primus-env\"" >&2
    echo "[env]   Use the same value for building and for every later 'source env.sh'." >&2
    # This file is meant to be sourced, where `return` is the only way to stop
    # without killing the caller's shell. The exit covers running it directly.
    # shellcheck disable=SC2317
    return 1 2>/dev/null || exit 1
fi

case "$PRIMUS_BASE" in
    /*) ;;
    *)
        echo "[env] ERROR: PRIMUS_BASE must be an absolute path, got '$PRIMUS_BASE'." >&2
        echo "[env]   Build stages cd into subdirectories, so a relative base breaks them." >&2
        # shellcheck disable=SC2317
        return 1 2>/dev/null || exit 1
        ;;
esac

export PRIMUS_BASE
export VENV_DIR="${VENV_DIR:-$PRIMUS_BASE/venv}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$PRIMUS_BASE/workspace}"  # kept checkouts (Primus, etc.)
# Transient build sources: put on fast LOCAL /tmp (NFS is slow for compile I/O;
# these dirs are deleted after each build anyway).
export SRC_DIR="${SRC_DIR:-/tmp/primus-build}"

# Build parallelism.
export MAX_JOBS="${MAX_JOBS:-128}"

# ---- Python version ----
# 3.12 is REQUIRED, not a preference. The pinned torch nightly
# (2.12.0+rocm7.15.0a20260720) published a cp312 Linux wheel and nothing else,
# so this holds even where TransformerEngine is built from source. In wheel mode
# TE reinforces it: its prebuilt core library `transformer_engine_rocm7` is also
# cp312-only, with no sdist and no other cp3xx build. See README.md.
export PRIMUS_PYTHON_VERSION="${PRIMUS_PYTHON_VERSION:-3.12}"
# Interpreters provisioned by `uv python install` land under PRIMUS_BASE rather
# than uv's default in ~/.local/share, so the whole environment stays in one
# place and off whatever quota the home directory has.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PRIMUS_BASE/python}"

# ---- Target GPU architecture (auto-detected) ----
# PYTORCH_ROCM_ARCH controls which gfx targets we build and install device
# wheels for. If you set it yourself it is respected as-is, e.g.:
#     export PYTORCH_ROCM_ARCH="gfx942;gfx950"
# Otherwise it is auto-detected from the GPUs on this host. Detection needs no
# sudo and no system ROCm: it uses `rocminfo` when available (e.g. after the
# pip ROCm SDK is installed) and otherwise falls back to the kernel KFD sysfs
# topology, so it works even on a fresh machine before any pip install.

# Decode a KFD gfx_target_version integer (e.g. 90402) into a gfx name (gfx942).
_primus_gfxver_to_arch() {
    local v="$1"
    [ -n "$v" ] && [ "$v" != "0" ] || return 0
    printf 'gfx%d%x%x\n' "$(( v / 10000 ))" "$(( (v / 100) % 100 ))" "$(( v % 100 ))"
}

# Echo a ";"-separated, de-duplicated list of detected gfx architectures.
_primus_detect_gpu_arch() {
    local arches=""
    if command -v rocminfo >/dev/null 2>&1; then
        arches="$(rocminfo 2>/dev/null \
            | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{print $2}' \
            | sort -u | paste -sd';' -)"
    fi
    if [ -z "$arches" ] && [ -d /sys/class/kfd/kfd/topology/nodes ]; then
        local f v a list=""
        for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            [ -f "$f" ] || continue
            v="$(awk '/^gfx_target_version/{print $2}' "$f" 2>/dev/null)"
            a="$(_primus_gfxver_to_arch "$v")"
            [ -n "$a" ] && list="${list}${a}"$'\n'
        done
        arches="$(printf '%s' "$list" | sort -u | sed '/^$/d' | paste -sd';' -)"
    fi
    printf '%s' "$arches"
}

if [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
    _DETECTED_ARCH="$(_primus_detect_gpu_arch)"
    if [ -n "$_DETECTED_ARCH" ]; then
        export PYTORCH_ROCM_ARCH="$_DETECTED_ARCH"
        echo "[env] detected GPU arch: PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH" >&2
    else
        export PYTORCH_ROCM_ARCH="gfx942;gfx950"
        echo "[env] WARNING: no GPU detected; defaulting PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH (override by exporting it)" >&2
    fi
fi

# Comma-separated variant for the tools/vars that expect commas.
_ARCH_CSV="${PYTORCH_ROCM_ARCH//;/,}"
export ROCM_AMDGPU_TARGETS="${ROCM_AMDGPU_TARGETS:-$_ARCH_CSV}"
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-$_ARCH_CSV}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-$_ARCH_CSV}"

# GPU_ARCHS is deliberately `native` here, matching the `ENV GPU_ARCHS=native`
# that v26.5 sets after the last build stage: aiter's JIT has to compile for the
# GPU it actually runs on. setup.sh overrides it to the full arch list for the
# individual build stages that cross-compile.
export GPU_ARCHS="${GPU_ARCHS:-native}"

# Keep build caches off the home directory (they reach tens of GB). Local /tmp is
# both roomier and much faster than NFS for this.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/primus-cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/primus-cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/primus-cache/triton}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/primus-cache/uv}"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$UV_CACHE_DIR" 2>/dev/null || true

# Activate the venv if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Constrain pip so nothing can silently swap the ROCm torch/triton builds for
# upstream ones. setup.sh writes this file after installing torch; several
# packages here depend on a bare `torch`, and a resolver that decides to
# "upgrade" it replaces the whole GPU stack (see README, mamba egg incident).
# The triton pin matters just as much: ROCm's HIP runtime loads its own
# libLLVM.so, so an upstream triton (which bundles a second, statically linked
# LLVM) segfaults on import and takes torch._dynamo/aiter/torchao with it.
export PRIMUS_PIP_CONSTRAINTS="${PRIMUS_PIP_CONSTRAINTS:-$PRIMUS_BASE/pip-constraints.txt}"

# Workaround for HSA_STATUS_ERROR_OUT_OF_RESOURCES
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1

# ---- ROCm path: comes from the pip-installed _rocm_sdk_devel package ----
# Computed dynamically so it works for any Python minor version.
if command -v python >/dev/null 2>&1; then
    _ROCM_SDK="$(python - <<'PY' 2>/dev/null
try:
    import _rocm_sdk_devel, os
    print(os.path.dirname(_rocm_sdk_devel.__file__))
except Exception:
    pass
PY
)"
fi

if [ -n "${_ROCM_SDK:-}" ]; then
    export ROCM_PATH="$_ROCM_SDK"
    export ROCM_HOME="$_ROCM_SDK"   # Primus uses ROCM_HOME; set both
    export HIP_PLATFORM=amd
    export HIP_PATH="$ROCM_PATH"
    export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
    export HIP_INCLUDE_PATH="$ROCM_PATH/include"
    export HIP_LIB_PATH="$ROCM_PATH/lib"
    export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
    export PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"
    export LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"
    export CPATH="$HIP_INCLUDE_PATH"
    export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"
fi

# ---- TransformerEngine tuning ----
# Only the runtime performance knobs v26.5 keeps. The NVTE_USE_ROCM /
# NVTE_FRAMEWORK / NVTE_ROCM_ARCH / NVTE_USE_HIPBLASLT vars that v26.4 exported
# are build-time switches, which is why v26.5 dropped them and why they are not
# exported here; stage_te_source sets them inline for the duration of its build.
export NVTE_USE_CAST_TRANSPOSE_TRITON="${NVTE_USE_CAST_TRANSPOSE_TRITON:-1}"
export NVTE_CK_USES_FWD_V3="${NVTE_CK_USES_FWD_V3:-1}"
export NVTE_CK_USES_BWD_V3="${NVTE_CK_USES_BWD_V3:-1}"
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT="${CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT:-2}"
export NVTE_CK_HOW_V3_BF16_CVT="${NVTE_CK_HOW_V3_BF16_CVT:-2}"

# The Dockerfile sets NVTE_CK_IS_V3_ATOMIC_FP32=0, which is correct for gfx950.
# On gfx942 (MI300X/MI325X) the CK v3 *backward* attention kernel needs fp32
# atomics: with 0 it produces Inf gradients and llama3.1_8B BF16 pretrain dies at
# iteration 1 with "found Inf in local grad norm for bucket #0 in backward pass".
# Primus's own tuning guide prescribes 1 for these GPUs
# (docs/02-user-guide/training-recipes.md), so default it per detected arch.
# Measured on MI325X: BWD_V3=1 + fp32 atomics = 614 TFLOP/s/GPU, versus 535 with
# BWD_V3 disabled, so this is worth ~15% as well as being the correct fix.
case ";${PYTORCH_ROCM_ARCH};" in
    *";gfx942;"*) _PRIMUS_CK_ATOMIC_FP32=1 ;;
    *)            _PRIMUS_CK_ATOMIC_FP32=0 ;;
esac
export NVTE_CK_IS_V3_ATOMIC_FP32="${NVTE_CK_IS_V3_ATOMIC_FP32:-$_PRIMUS_CK_ATOMIC_FP32}"
# Required post-v26.2 to resolve Primus attention backend issues
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

# ---- causal-conv1d / mamba ----
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE

# libz3.so safety net. v26.5 uninstalls tilelang (the mamba_ssm dep that needs
# z3) at the end of the build, so this is normally unused; it stays because the
# Dockerfile still ships apt libz3-dev and re-installing tilelang by hand should
# not leave a broken environment behind. pip `z3-solver` supplies the library.
if command -v python >/dev/null 2>&1; then
    _Z3_LIB="$(python - <<'PY' 2>/dev/null
import glob, os
for p in glob.glob(os.path.join(os.path.dirname(__import__("site").getsitepackages()[0]), "site-packages", "**", "libz3.so"), recursive=True):
    print(os.path.dirname(p)); break
PY
)"
    if [ -n "${_Z3_LIB:-}" ]; then
        export LD_LIBRARY_PATH="$_Z3_LIB:${LD_LIBRARY_PATH:-}"
    fi
fi

# Keep the final status clean. setup.sh runs under `set -e` and re-sources this
# file at the start of every stage, so a falsy test on the last line would abort
# the build.
:
