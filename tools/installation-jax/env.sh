#!/usr/bin/env bash
# env.sh — Primus JAX/MaxText venv environment (v26.6).
# Source this both during the build (setup.sh does it) and every time you
# want to USE the environment:   source env.sh
#
# This mirrors the v26.6 JAX training Dockerfile, adapted for a no-docker /
# no-sudo bare-metal install. In v26.6 ROCm is delivered as pip TheRock wheels
# (`rocm-sdk-devel` 7.14.0), same as the image — not the v26.5 release tarball.
# No system ROCm is required.
#
# NOTE: PRIMUS_JAX_BASE is REQUIRED (no default) — export it to a big-disk dir
#       before sourcing this file. MaxText requires Python >= 3.12; see
#       PRIMUS_PYTHON below.

# ---- Install location (persistent) + transient build sources ----
# PRIMUS_JAX_BASE is REQUIRED (there is intentionally no default): it is where
# Everything lives under PRIMUS_JAX_BASE (venv, kept checkouts). Point it at
# a directory you can write to with tens of GB free, then (re-)source this file:
#     export PRIMUS_JAX_BASE=/big/disk/primus-jax-env
if [ -z "${PRIMUS_JAX_BASE:-}" ]; then
    echo "[env] ERROR: PRIMUS_JAX_BASE is not set (no default). Set it to a directory" >&2
    echo "[env]        you can write to with tens of GB free, then re-run:" >&2
    echo "[env]            export PRIMUS_JAX_BASE=/big/disk/primus-jax-env" >&2
    # env.sh is always sourced (by setup.sh and to use the env); stop sourcing
    # without killing an interactive shell. Fall back to exit if run directly.
    # shellcheck disable=SC2317  # 'exit 1' is reachable when this file is executed, not sourced
    return 1 2>/dev/null || exit 1
fi
export PRIMUS_JAX_BASE
export VENV_DIR="${VENV_DIR:-$PRIMUS_JAX_BASE/venv}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$PRIMUS_JAX_BASE/workspace}"  # kept checkouts (maxtext, Primus)
# Transient build sources: put on fast LOCAL /tmp (NFS is slow for compile I/O;
# these dirs are deleted after each build anyway).
export SRC_DIR="${SRC_DIR:-/tmp/primus-jax-build}"

# MaxText clone (deps + editable install live here). Primus resolves the
# MaxText backend from MAXTEXT_PATH (else its own third_party/maxtext), so we
# point it at the checkout we actually installed the deps for.
export MAXTEXT_DIR="${MAXTEXT_DIR:-$WORKSPACE_DIR/maxtext}"
export MAXTEXT_PATH="${MAXTEXT_PATH:-$MAXTEXT_DIR}"

# Python interpreter used to CREATE the venv. MaxText needs >= 3.12.
# 3.12 is PREFERRED: it is the version the reference image (ubuntu:24.04) uses,
# and the prebuilt ROCm wheels (transformer_engine_rocm_jax, jax_rocm7_*) are
# built for cp312 — a 3.13 venv would need those built from source instead.
# Resolution order:
#   1. An explicit python3.12 / python3.13 on PATH (3.12 first).
#   2. A uv-managed interpreter (>= 3.12), if `uv` is installed. This is the
#      no-sudo path on distros whose apt has no python3.12 (e.g. Ubuntu 22.04):
#      `uv python install 3.12` provides one without root. setup.sh will run
#      that install automatically if none is found yet.
#   3. Fall back to python3 (setup.sh errors clearly if that is < 3.12).
if [ -z "${PRIMUS_PYTHON:-}" ]; then
    for _py in python3.12 python3.13; do
        if command -v "$_py" >/dev/null 2>&1; then export PRIMUS_PYTHON="$_py"; break; fi
    done
    if [ -z "${PRIMUS_PYTHON:-}" ] && command -v uv >/dev/null 2>&1; then
        _uv_py="$(uv python find '>=3.12' 2>/dev/null || true)"
        [ -n "$_uv_py" ] && [ -x "$_uv_py" ] && export PRIMUS_PYTHON="$_uv_py"
    fi
    export PRIMUS_PYTHON="${PRIMUS_PYTHON:-python3}"
fi

# Build parallelism.
export MAX_JOBS="${MAX_JOBS:-128}"

# ---- Target GPU architecture (auto-detected) ----
# PYTORCH_ROCM_ARCH controls the gfx targets used for the source builds (TE,
# RCCL) and TE's runtime NVTE_ROCM_ARCH. The name is kept for parity with the
# Dockerfile / TE which read it. If you set it yourself it is respected as-is:
#     export PYTORCH_ROCM_ARCH="gfx942;gfx950"
# Otherwise it is auto-detected from the GPUs on this host. Detection needs no
# sudo and no system ROCm: it uses `rocminfo` when available (e.g. after the
# pip ROCm SDK is installed) and otherwise falls back to the kernel KFD sysfs
# topology, so it works even on a fresh machine before any install.

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

# Keep caches OFF the tiny home quota. Use local /tmp.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/primus-jax-cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/primus-jax-cache/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/primus-jax-cache/uv}"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" 2>/dev/null || true

# Activate the venv if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Every VENV_DIR/WORKSPACE_DIR/MAXTEXT_* export above honours a pre-existing
# value, so sourcing this file in a shell that already has another Primus env
# (e.g. tools/installation/env.sh, or a different PRIMUS_JAX_BASE) silently
# keeps the OLD paths. The failure then shows up much later as a confusing
# "Backend path not found for 'maxtext'" or as pip installing into the wrong
# venv, so say something now. Start from a fresh shell to clear these.
for _v in VENV_DIR WORKSPACE_DIR MAXTEXT_PATH; do
    eval "_val=\${$_v}"
    case "$_val" in
        "$PRIMUS_JAX_BASE"/*) ;;
        *) echo "[env] WARNING: $_v=$_val is outside PRIMUS_JAX_BASE=$PRIMUS_JAX_BASE" >&2 ;;
    esac
done
unset _v _val
if [ "${VIRTUAL_ENV:-}" != "$VENV_DIR" ]; then
    echo "[env] WARNING: venv not active: expected $VENV_DIR, got ${VIRTUAL_ENV:-<none>}" >&2
    echo "[env]          (before the 'venv' stage of setup.sh this is expected)" >&2
fi
if [ ! -d "$MAXTEXT_PATH" ]; then
    echo "[env] WARNING: MAXTEXT_PATH does not exist: $MAXTEXT_PATH" >&2
    echo "[env]          Primus will fail with \"Backend path not found for 'maxtext'\"." >&2
    echo "[env]          (before the 'maxtext' stage of setup.sh this is expected)" >&2
fi

# NOTE: the two HSA_* vars below are NOT in the v26.6 Dockerfile — they are a
# carryover workaround for HSA_STATUS_ERROR_OUT_OF_RESOURCES. Harmless, but
# remove them if you want an exact match to the image.
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
export HSA_NO_SCRATCH_RECLAIM=1

# Flags to fix the ROCm profiler hang issue (from the JAX Dockerfile base stage)
export ROCPROFILER_QUEUE_INTERPOSITION=0
export DEBUG_HIP_DYNAMIC_QUEUES=0

# ---- ROCm path: from the pip-installed _rocm_sdk_devel package (v26.6) ----
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
    export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"
    export CPATH="$HIP_INCLUDE_PATH"
    export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"
    # v26.6 blanks LD_LIBRARY_PATH at runtime so TE 2.17 + JAX resolve ROCm via
    # RPATH instead of loading a mismatched HIP copy from rocm-sdk-devel
    # (CK-JIT fmha_bwd segfault on gfx950). Source builds that need the devel
    # tree can export PRIMUS_JAX_KEEP_ROCM_LD=1.
    _ROCM_LD="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
    if [ "${PRIMUS_JAX_KEEP_ROCM_LD:-0}" = "1" ]; then
        export LD_LIBRARY_PATH="${_ROCM_LD}:${LD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH=""
    fi
fi

# ---- TransformerEngine (ROCm) runtime settings for JAX ----
# The transformer_engine_rocm_jax wheel is prebuilt, so these are runtime knobs.
export NVTE_ROCM_ARCH="$PYTORCH_ROCM_ARCH"
export NVTE_USE_ROCM=1
export NVTE_USE_HIPBLASLT=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_FUSED_ATTN=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export NVTE_CK_HOW_V3_BF16_CVT=2

# ---- AMD GPU runtime knobs (from the JAX Dockerfile) ----
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=VERSION

# Bare-metal-only: force RCCL to use its built-in ROCm IB/RoCE transport instead
# of any host NCCL net plugin. The rocm/jax-training image ships no
# librccl-net.so and sets no NCCL_NET_PLUGIN, but on a host the system
# /usr/local/lib/librccl-net.so is on the default loader path and is
# ABI-incompatible with the from-source RCCL (undefined symbol
# ncclNetPlugin_v11/_v10 -> falls back to v9 -> segfault at clique init).
# NCCL_NET_PLUGIN=none makes RCCL ignore that host plugin, matching the image.
export NCCL_NET_PLUGIN=none

# NOT in the v26.6 Dockerfile: avoids a NaN-loss issue when training on gfx950
# (no-op on gfx942). Keep it if you train on MI350/MI355; drop for exact parity.
export RCCL_WARP_SPEED_AUTO=0

# ---- XLA / JAX runtime settings ----
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=0 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_enable_command_buffer=''}"
