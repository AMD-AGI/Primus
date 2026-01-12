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

# ---------------------------------------------------------------------------
# Distributed Training Cluster Configuration
# ---------------------------------------------------------------------------
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

log_exported_vars "Training Cluster Info" \
    MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

# ---------------------------------------------------------------------------
# Python Path Setup
# ---------------------------------------------------------------------------
PRIMUS_PATH=$(cd "$SCRIPT_DIR/../../.." && pwd)
export PRIMUS_PATH

# Set data paths
export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}

site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null || echo "")
if [[ -n "$site_packages" ]]; then
    export PYTHONPATH="${PRIMUS_PATH}:${site_packages}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${PRIMUS_PATH}:${PYTHONPATH:-}"
fi

log_exported_vars "Python Path and Data Paths" \
    PRIMUS_PATH DATA_PATH HF_HOME PYTHONPATH

# =============================================================================
# NCCL and Network Configuration
# =============================================================================

# Set visible GPUs for the current node (0 to GPUS_PER_NODE-1)
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES

# ----------------- NCCL and Network Settings -----------------

# NCCL logging level: VERSION, WARN, INFO, DEBUG, TRACE
# Set to empty for default behavior, or specify level for debugging
export NCCL_DEBUG=${NCCL_DEBUG:-}

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=1

# Set InfiniBand GID index for NCCL communication
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# Disable cross NIC communication for NCCL
export NCCL_CROSS_NIC=0

# Dynamically get InfiniBand Host Channel Adapter index for NCCL if not set
if [ -z "${NCCL_IB_HCA}" ]; then
    NCCL_IB_HCA=$(bash "${SCRIPT_DIR}/get_nccl_ib_hca.sh" 2>/dev/null || echo "")
fi
export NCCL_IB_HCA

# Dynamically get network interface IP address for socket communication if not set
if [ -z "${IP_INTERFACE}" ]; then
    IP_INTERFACE=$(bash "${SCRIPT_DIR}/get_ip_interface.sh" 2>/dev/null || hostname -I | awk '{print $1}')
fi
export IP_INTERFACE

# Set network interfaces for NCCL and Gloo, fallback to detected IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

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
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}

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
    GPU_MAX_HW_QUEUES ENABLE_NUMA_BINDING CUDA_DEVICE_MAX_CONNECTIONS \
    TORCH_NCCL_HIGH_PRIORITY NCCL_PXN_DISABLE NCCL_P2P_NET_CHUNKSIZE

# ----------------- Transformer Engine Optimizations -----------------
# Optimize NVTE fp8 cast transpose
export NVTE_USE_CAST_TRANSPOSE_TRITON=${NVTE_USE_CAST_TRANSPOSE_TRITON:-1}
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=${NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE:-0}

# enable mxfp8 on ROCm Transformer Engine
export NVTE_ROCM_ENABLE_MXFP8=1

# Note: Disable v3 due to accuracy issues. Will fix after TE version 2.1.
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

# Note: Disable fp32 atomic if you find any accuracy issue
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-0}

# NVTE debug envs
export NVTE_DEBUG=${NVTE_DEBUG:-0}              # 0, 1
export NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}  # 0, 1, 2
export NVTE_FUSED_ATTN_LOG_CONFIG=${NVTE_FUSED_ATTN_LOG_CONFIG:-0}  # 0, 1
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

log_exported_vars "Transformer Engine Optimizations" \
    NVTE_USE_CAST_TRANSPOSE_TRITON NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE \
    NVTE_CK_USES_BWD_V3 PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32 \
    NVTE_DEBUG NVTE_DEBUG_LEVEL NVTE_FUSED_ATTN_LOG_CONFIG PATCH_TE_FLASH_ATTN
