#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# =============================================================================
# Base Environment Configuration
# =============================================================================
# This file provides all environment configurations for Primus:
#   - Logging functions (LOG_INFO, LOG_INFO_RANK0, LOG_ERROR, etc.)
#   - Distributed training cluster information (MASTER_ADDR, NNODES, etc.)
#   - Python path setup and data paths
#   - NCCL and network settings
#   - RCCL communication library settings
#   - AMD GPU optimizations
#   - General performance tuning
#   - Transformer Engine optimizations
#
# GPU-specific settings can override these in GPU model files (e.g., MI300X.sh)
# =============================================================================

# ---------------------------------------------------------------------------
# Guard: avoid duplicate exports/logging on multiple sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_BASE_ENV_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_BASE_ENV_SOURCED=1

# ---------------------------------------------------------------------------
# Load common library for consistent logging
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../../lib/common.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../../lib/common.sh"
else
    # Fallback logging functions if common.sh not available
    HOSTNAME="$(hostname)"
    export HOSTNAME

    LOG_INFO() {
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] $*"
        fi
    }

    LOG_INFO_RANK0() {
        if [ "${NODE_RANK:-0}" -eq 0 ]; then
            if [ "$*" = "" ]; then
                echo ""
            else
                echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] $*"
            fi
        fi
    }

    LOG_ERROR() {
        echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] [ERROR] $*" >&2
    }

    LOG_WARN() {
        echo "[NODE-${NODE_RANK:-0}($HOSTNAME)] [WARN] $*" >&2
    }

    log_exported_vars() {
        LOG_INFO_RANK0 "========== $1 =========="
        for var in "${@:2}"; do
            LOG_INFO_RANK0 "    $var=${!var-}"
        done
    }
fi

# Load path helpers so ROCm libraries keep highest priority.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/path_utils.sh"

# ---------------------------------------------------------------------------
# Distributed Training Cluster Configuration
# ---------------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}

log_exported_vars "Training Cluster Info" \
    MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

# ---------------------------------------------------------------------------
# Core Dumps
# ---------------------------------------------------------------------------
# Disabled for every rank the launcher goes on to start. A large training
# process maps on the order of 160-170 GB of host memory per rank, so a crash
# under the container default of `ulimit -c unlimited` writes one core file of
# that size per crashing rank -- ~45 GB each, observed on Flux 12B. On a shared
# node that is enough to fill the disk out from under every other tenant, not
# just this job, and it has happened: several such files were found still
# present days after the run that produced them.
#
# `ulimit -c` is a shell resource limit and is inherited by every process
# forked from here, so setting it once covers all ranks regardless of backend.
# Opt out for a single invocation when a dump is actually wanted:
#   PRIMUS_ENABLE_CORE_DUMPS=1 primus-cli train pretrain ...
if [ "${PRIMUS_ENABLE_CORE_DUMPS:-0}" != "1" ]; then
    ulimit -c 0
fi

# ---------------------------------------------------------------------------
# Python Path Setup
# ---------------------------------------------------------------------------
PRIMUS_PATH=$(cd "$SCRIPT_DIR/../../.." && pwd)
export PRIMUS_PATH

# Determine the directory that must be on PYTHONPATH for `import primus` to work.
# - git checkout: PRIMUS_PATH is the repo root that *contains* the `primus/` package.
# - pip/site-packages install: PRIMUS_PATH is the `primus` package dir itself, so the
#   import root is its parent (e.g. .../site-packages).
if [[ -f "${PRIMUS_PATH}/__init__.py" ]]; then
    PRIMUS_IMPORT_ROOT="$(cd "${PRIMUS_PATH}/.." && pwd)"
else
    PRIMUS_IMPORT_ROOT="${PRIMUS_PATH}"
fi
export PRIMUS_IMPORT_ROOT

# Set data paths
export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}

# ---------------------------------------------------------------------------
# Persistent kernel/JIT cache layout
# ---------------------------------------------------------------------------
# The default lives under /workspace because that is where the training image
# keeps its writable persistent volume. Bare-metal runs (primus-cli direct)
# source this file too, and there /workspace is typically absent or root-owned,
# so probe the default before committing to it: otherwise the first Triton JIT
# compile is what surfaces the unwritable directory. An explicitly configured
# PRIMUS_CACHE_ROOT is always honoured as-is.
PRIMUS_CACHE_ROOT_DEFAULT=/workspace/cache_persist
if [[ -z "${PRIMUS_CACHE_ROOT:-}" ]]; then
    PRIMUS_CACHE_ROOT="${PRIMUS_CACHE_ROOT_DEFAULT}"
    if ! { mkdir -p "${PRIMUS_CACHE_ROOT}" 2>/dev/null && [[ -w "${PRIMUS_CACHE_ROOT}" ]]; }; then
        PRIMUS_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/primus"
        LOG_WARN "[env] ${PRIMUS_CACHE_ROOT_DEFAULT} is not writable; falling back to ${PRIMUS_CACHE_ROOT} for the kernel/JIT cache. Export PRIMUS_CACHE_ROOT=<dir> to pick a different location."
    fi
fi
export PRIMUS_CACHE_ROOT
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/torchinductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${PRIMUS_CACHE_ROOT}/torch_extensions}
export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH:-${PRIMUS_CACHE_ROOT}/pytorch_kernel}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-${PRIMUS_CACHE_ROOT}/miopen_db}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/miopen_cache}
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-2}
export MIOPEN_DISABLE_CACHE=${MIOPEN_DISABLE_CACHE:-0}
export AOTRITON_CACHE_DIR=${AOTRITON_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/aotriton}
export AMD_COMGR_CACHE_DIR=${AMD_COMGR_CACHE_DIR:-${PRIMUS_CACHE_ROOT}/comgr}
export AMD_COMGR_CACHE=${AMD_COMGR_CACHE:-1}
export HIPBLASLT_TUNING_FILE=${HIPBLASLT_TUNING_FILE:-${PRIMUS_CACHE_ROOT}/hipblaslt_tuning.json}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}

mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TORCH_EXTENSIONS_DIR}" "${PYTORCH_KERNEL_CACHE_PATH}" \
         "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
         "${AOTRITON_CACHE_DIR}" "${AMD_COMGR_CACHE_DIR}" \
         "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" 2>/dev/null || true

log_exported_vars "Persistent Cache Layout" \
    PRIMUS_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
    TORCH_EXTENSIONS_DIR PYTORCH_KERNEL_CACHE_PATH MIOPEN_USER_DB_PATH \
    MIOPEN_CUSTOM_CACHE_DIR AOTRITON_CACHE_DIR AMD_COMGR_CACHE_DIR \
    HIPBLASLT_TUNING_FILE HF_HUB_CACHE HF_DATASETS_CACHE

site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || echo "")
if [[ -n "$site_packages" ]]; then
    export PYTHONPATH="${PRIMUS_IMPORT_ROOT}:${site_packages}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${PRIMUS_IMPORT_ROOT}:${PYTHONPATH:-}"
fi

log_exported_vars "Python Path and Data Paths" \
    PRIMUS_PATH PRIMUS_IMPORT_ROOT DATA_PATH HF_HOME PYTHONPATH

# =============================================================================
# NCCL and Network Configuration
# =============================================================================

# Set visible GPUs for the current node (0 to GPUS_PER_NODE-1)
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES

# Keep ROCm libraries ahead of any system-provided HSA runtime.
ensure_rocm_ld_library_path

# ----------------- NCCL and Network Settings -----------------

# NCCL logging level: VERSION, WARN, INFO, DEBUG, TRACE
# Set to empty for default behavior, or specify level for debugging
export NCCL_DEBUG=${NCCL_DEBUG:-}

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}

# Set InfiniBand GID index for NCCL communication
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# Disable cross NIC communication for NCCL
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}

# Dynamically get InfiniBand Host Channel Adapter index for NCCL if not set
if [ -z "${NCCL_IB_HCA:-}" ]; then
    NCCL_IB_HCA=$(bash "${SCRIPT_DIR}/get_nccl_ib_hca.sh" 2>/dev/null || echo "")
fi
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"

# Dynamically get network interface IP address for socket communication if not set
if [ -z "${IP_INTERFACE:-}" ]; then
    # No fallback to an IP: this value feeds *_SOCKET_IFNAME, which must name an
    # interface. Leaving it empty lets NCCL and Gloo pick one themselves.
    IP_INTERFACE=$(bash "${SCRIPT_DIR}/get_ip_interface.sh" 2>/dev/null || true)
fi
export IP_INTERFACE="${IP_INTERFACE:-}"

# Set network interfaces for NCCL and Gloo, fallback to detected IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

# ----------------- RCCL Topology and GPU-Direct RDMA -----------------

export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}

# Without a topology file RCCL discovers the fabric itself and plans 2 channels
# inter-node, which is 46 GB/s; a vendor XML declaring which rail is local to
# which GPU widens the plan to 8, which is 118.
#
# Resolved by GPU PCI device id, never by filename. The mi350x and mi355x files
# differ only there (0x75a0 against 0x75a3) and a mismatched file still parses,
# so RCCL applies it and silently loses the affinity the file exists to declare
# -- a wrong filename is worse than no file, because it looks configured. Read
# from sysfs rather than lspci, which is a package and not present in every
# image. The vendor directory is not mounted in every container, so a staged
# copy is accepted as a fallback.
#
# PRIMUS_RCCL_TOPO_DISABLE=1 restores the previous behaviour exactly: no
# topology file and no GDR level of our choosing. It exists so the A/B stays
# measurable from outside the launcher, and as an escape hatch if a future node
# ships a file that plans worse than RCCL's own discovery.
if [ "${PRIMUS_RCCL_TOPO_DISABLE:-0}" = "1" ]; then
    LOG_INFO_RANK0 "RCCL topology: disabled by PRIMUS_RCCL_TOPO_DISABLE=1"
elif [ -z "${NCCL_TOPO_FILE:-}" ]; then
    _gpu_devid=""
    for _d in /sys/bus/pci/devices/*; do
        [ "$(cat "$_d/vendor" 2>/dev/null)" = "0x1002" ] || continue
        _gpu_devid="$(cat "$_d/device" 2>/dev/null)"
        [ -n "$_gpu_devid" ] && break
    done
    for _f in /etc/crusoe/rccl_topo/*.xml /opt/rccl_topo_node.xml; do
        [ -f "$_f" ] || continue
        if [ -n "$_gpu_devid" ] && grep -q "device=\"${_gpu_devid}\"" "$_f"; then
            export NCCL_TOPO_FILE="$_f"
            break
        fi
    done
    if [ -n "${NCCL_TOPO_FILE:-}" ]; then
        LOG_INFO_RANK0 "RCCL topology: $NCCL_TOPO_FILE (GPU device $_gpu_devid)"
    else
        LOG_INFO_RANK0 "RCCL topology: no matching file for GPU device ${_gpu_devid:-unknown}"
    fi
fi

# GPU-Direct RDMA is one switch because the settings below only work together,
# and the failure mode of getting it partly right is that every rank dies during
# connection setup rather than falling back.
#
# dmabuf is the only registration path available on the target fabric: the
# peerdirect client is absent from ib_core, and RCCL's own dmabuf gate reads a
# kernel config file the image does not ship, hence the force flag. Registration
# also has to come off the VMM allocator -- with the default allocator the
# buffers are not dmabuf-exportable, RCCL falls back to ibv_reg_mr_iova2, and
# that returns EINVAL for every buffer. That EINVAL reads as the fabric refusing
# GDR outright, but it is the allocator: NCCL_CUMEM_ENABLE=1 removes it.
#
# Measured on two same-pod nodes at MBS=32/GBS=512: 537.0 -> 502.8 ms/step, a
# 34 ms saving against a 3.3 ms spread between repeats, final loss identical.
# Needs the topology file above, which is what makes RCCL attempt GDR at all.
#
# Off by default because it is only safe once the dataloader can avoid fork: a
# process holding dmabuf-registered GPU memory segfaults inside os.fork(). That
# needs an Energon that does not fork and ENERGON_MP_CONTEXT set away from fork,
# so refuse rather than hand back a segfault with no explanation.
if [ "${PRIMUS_RCCL_GDR:-0}" = "1" ]; then
    if [ "${ENERGON_MP_CONTEXT:-fork}" = "fork" ]; then
        LOG_ERROR "PRIMUS_RCCL_GDR=1 requires ENERGON_MP_CONTEXT=forkserver (or spawn)."
        LOG_ERROR "Under fork the dataloader forks with GPU memory registered for GDR"
        LOG_ERROR "and its workers segfault immediately."
        exit 1
    fi
    # Set outright, not with :-, because NCCL_DMABUF_ENABLE is defaulted to 0
    # above and a :- default would silently keep that 0 and break registration.
    export NCCL_NET_GDR_LEVEL="${PRIMUS_RCCL_GDR_LEVEL:-SYS}"
    export NCCL_DMABUF_ENABLE=1
    export RCCL_FORCE_ENABLE_DMABUF=1
    export NCCL_CUMEM_ENABLE=1
    LOG_INFO_RANK0 "RCCL GDR: enabled, level $NCCL_NET_GDR_LEVEL (dmabuf via cumem)"
fi

# ----------------- RCCL Settings (AMD ROCm Communication Library) -----------------

# Disable MSCCL (RCCL multi-connection feature) for better stability
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
export RCCL_MSCCLPP_ENABLE=${RCCL_MSCCLPP_ENABLE:-0}
export RCCL_MSCCLPP_FORCE_ENABLE=${RCCL_MSCCLPP_FORCE_ENABLE:-0}
export RCCL_MSCCLPP_THRESHOLD=${RCCL_MSCCLPP_THRESHOLD:-$((1*1024*1024*1024))} # default 1GB

# https://github.com/microsoft/mscclpp/blob/main/include/mscclpp/env.hpp#L82-L87
export MSCCLPP_DISABLE_CHANNEL_CACHE=${MSCCLPP_DISABLE_CHANNEL_CACHE:-FALSE}

# PyTorch needs this env to enable register comm
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=${TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK:-0}

log_exported_vars "NCCL and Network Settings" \
    HIP_VISIBLE_DEVICES NCCL_DEBUG NCCL_CHECKS_DISABLE NCCL_IB_GID_INDEX \
    NCCL_CROSS_NIC NCCL_IB_HCA IP_INTERFACE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

log_exported_vars "RCCL Settings" \
    RCCL_MSCCL_ENABLE RCCL_MSCCLPP_ENABLE RCCL_MSCCLPP_FORCE_ENABLE RCCL_MSCCLPP_THRESHOLD \
    MSCCLPP_DISABLE_CHANNEL_CACHE TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK

# =============================================================================
# Performance Tuning Configuration
# =============================================================================

# ----------------- AMD-specific GPU optimizations -----------------
# Enable system DMA engine (SDMA) on AMD GPUs for better IO throughput
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-1}

# Prevent scratch memory from being reclaimed to stabilize large memory usage
# NOTE: Must disable scratch reclaim to avoid MoE training crash on AMD GPUs
# Setting this to 0 prevents core dumps when using Mixture-of-Experts (MoE) models
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}

log_exported_vars "AMD GPU Optimizations" \
    HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM

# ----------------- General Performance Tuning -----------------
# Limit GPU hardware queues to 2 for performance stability
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}

# Increase HSA kernarg pool size to 12MB for models with many kernels (optional, can be set by GPU-specific configs)
# export HSA_KERNARG_POOL_SIZE=${HSA_KERNARG_POOL_SIZE:-12582912}

# Enable NUMA binding for better memory locality (may increase stability for large models)
export ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-0}

# Limit max CUDA device connections to reduce PCIe traffic
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}

# ----------------- NCCL Performance Settings -----------------
# In multi-node training, PXN can be enabled to improve inter-node all-to-all
# communication efficiency, but it will increase GPU memory usage.
# Default: disable PXN for NCCL
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-524288}

log_exported_vars "General Performance Tuning" \
    GPU_MAX_HW_QUEUES ENABLE_NUMA_BINDING HSA_KERNARG_POOL_SIZE CUDA_DEVICE_MAX_CONNECTIONS \
    TORCH_NCCL_HIGH_PRIORITY NCCL_PXN_DISABLE NCCL_P2P_NET_CHUNKSIZE

# ----------------- Transformer Engine Optimizations -----------------
# Optimize NVTE fp8 cast transpose
export NVTE_USE_CAST_TRANSPOSE_TRITON=${NVTE_USE_CAST_TRANSPOSE_TRITON:-1}
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=${NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE:-0}

# enable mxfp8 on ROCm Transformer Engine
export NVTE_ROCM_ENABLE_MXFP8=1

export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}

# Note: Disable fp32 atomic if you find any accuracy issue
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-0}

# NVTE debug envs
export NVTE_DEBUG=${NVTE_DEBUG:-0}              # 0, 1
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}  # 0, 1, 2
export NVTE_FUSED_ATTN_LOG_CONFIG=${NVTE_FUSED_ATTN_LOG_CONFIG:-0}  # 0, 1
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

# ----------------- Deterministic Mode -----------------
# PRIMUS_DETERMINISTIC=1 forces deterministic-related envs.
if [[ "${PRIMUS_DETERMINISTIC:-0}" == "1" ]]; then
    export NCCL_ALGO="Ring"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export ROCBLAS_DEFAULT_ATOMICS_MODE=0
    # Disable torch compile to avoid race issues in some triton versions.
    export TORCH_COMPILE_DISABLE=1
    export PRIMUS_TURBO_AUTO_TUNE=0
fi
# turbo deepep timeout
export PRIMUS_TURBO_DEEPEP_TIMEOUT=${PRIMUS_TURBO_DEEPEP_TIMEOUT:-600}

log_exported_vars "Transformer Engine Optimizations" \
    NVTE_USE_CAST_TRANSPOSE_TRITON NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE \
    NVTE_CK_USES_BWD_V3 PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32 \
    NVTE_DEBUG NVTE_DEBUG_LEVEL NVTE_FUSED_ATTN_LOG_CONFIG PATCH_TE_FLASH_ATTN \
    PRIMUS_DETERMINISTIC NCCL_ALGO NVTE_ALLOW_NONDETERMINISTIC_ALGO \
    ROCBLAS_DEFAULT_ATOMICS_MODE TORCH_COMPILE_DISABLE \
    PRIMUS_TURBO_DEEPEP_TIMEOUT
