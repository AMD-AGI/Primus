#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# =============================================================================
# NCCL and Network Configuration
# =============================================================================
# This file contains all network-related environment variable settings
# for distributed training with NCCL and communication libraries.
# =============================================================================

# Dependency check: ensure base_env.sh has been loaded
if [[ -z "${GPUS_PER_NODE}" ]]; then
    echo "[ERROR] GPUS_PER_NODE not set. base_env.sh must be loaded first." >&2
    exit 1
fi

if ! declare -f log_exported_vars >/dev/null 2>&1; then
    echo "[ERROR] log_exported_vars function not found. base_env.sh must be loaded first." >&2
    exit 1
fi

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
export NCCL_IB_GID_INDEX=3

# Disable cross NIC communication for NCCL
export NCCL_CROSS_NIC=0

# Dynamically get InfiniBand Host Channel Adapter index for NCCL if not set
if [ -z "${NCCL_IB_HCA}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    NCCL_IB_HCA=$(bash "${SCRIPT_DIR}/get_nccl_ib_hca.sh" 2>/dev/null || echo "")
fi
export NCCL_IB_HCA

# Dynamically get network interface IP address for socket communication if not set
if [ -z "${IP_INTERFACE}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
