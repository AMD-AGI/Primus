# env.sh — Primus venv environment (Python 3.10, ROCm via pip rocm-sdk-devel)
# Source this both during the build (setup.sh does it) and every time you
# want to USE the environment:   source env.sh
#
# NOTE: choose a BASE on the big disk. Home quota here is tiny (~8 GB free),
# while /tmp lives on a 28 TB device. Override by exporting PRIMUS_BASE first.

export PRIMUS_BASE="${PRIMUS_BASE:-/it-share-4/envs/primus-env}"   # persistent shared disk
export VENV_DIR="${VENV_DIR:-$PRIMUS_BASE/venv}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$PRIMUS_BASE/workspace}"  # kept checkouts (Primus, etc.)
# Transient build sources: put on fast LOCAL /tmp (NFS is slow for compile I/O;
# these dirs are deleted after each build anyway).
export SRC_DIR="${SRC_DIR:-/tmp/primus-build}"

# Build parallelism.
export MAX_JOBS="${MAX_JOBS:-128}"

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
export GPU_ARCHS="${GPU_ARCHS:-$PYTORCH_ROCM_ARCH}"
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-$_ARCH_CSV}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-$_ARCH_CSV}"

# Keep caches OFF the tiny home quota (~8 GB). Use local /tmp.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/primus-cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/primus-cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/primus-cache/triton}"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" 2>/dev/null || true

# Activate the venv if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Workaround for HSA_STATUS_ERROR_OUT_OF_RESOURCES
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1

# ---- ROCm path: comes from the pip-installed _rocm_sdk_devel package ----
# Computed dynamically so it works for any Python minor version (3.10 here).
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

# ---- TransformerEngine ROCm settings ----
export NVTE_USE_HIPBLASLT=1
export NVTE_FRAMEWORK=pytorch
export NVTE_ROCM_ARCH="$PYTORCH_ROCM_ARCH"
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_USE_ROCM=1
# Required post-v26.2 to resolve Primus attention backend issues
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1

# ---- causal-conv1d / mamba ----
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE

# mamba_ssm needs libz3.so at runtime (Dockerfile gets it from apt libz3-dev;
# no sudo here, so use the copy shipped by the pip `z3-solver` package).
if command -v python >/dev/null 2>&1; then
    _Z3_LIB="$(python - <<'PY' 2>/dev/null
import glob, os
for p in glob.glob(os.path.join(os.path.dirname(__import__("site").getsitepackages()[0]), "site-packages", "**", "libz3.so"), recursive=True):
    print(os.path.dirname(p)); break
PY
)"
    [ -n "${_Z3_LIB:-}" ] && export LD_LIBRARY_PATH="$_Z3_LIB:${LD_LIBRARY_PATH:-}"
fi
