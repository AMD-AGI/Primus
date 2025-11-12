#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# ---------------------------------------------------------------------------
# Guard: avoid duplicate exports/logging on multiple sourcing
# ---------------------------------------------------------------------------
if [[ -n "${__PRIMUS_ENV_SOURCED:-}" ]]; then
  return 0
fi
export __PRIMUS_ENV_SOURCED=1

# Hostname is useful for logs in any script that sources this file
HOSTNAME="$(hostname)"
export HOSTNAME

LOG_INFO() {
    if [ "$*" = "" ]; then
        echo ""
    else
        echo "[NODE-$NODE_RANK($HOSTNAME)] $*"
    fi
}

LOG_INFO_RANK0() {
    if [ "$NODE_RANK" -eq 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] $*"
        fi
    fi
}

LOG_ERROR() {
    echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*";
}

log_exported_vars() {
    LOG_INFO_RANK0 "========== $1 =========="
    for var in "${@:2}"; do
        LOG_INFO_RANK0 "    $var=${!var-}"
    done
}

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
log_exported_vars "Training cluster info" \
    MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

# -------------------- NCCL and Communication Setup --------------------
# Set visible GPUs for the current node (0 to GPUS_PER_NODE-1)
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES

# ----------------- NCCL and Network Settings -----------------
# VERSION, WARN, INFO, DEBUG, TRACE
export NCCL_DEBUG=

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=1

# Set InfiniBand GID index for NCCL communication
export NCCL_IB_GID_INDEX=3

# Disable cross NIC communication for NCCL
export NCCL_CROSS_NIC=0

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

# Dynamically get InfiniBand Host Channel Adapter index for NCCL if not set
if [ -z "${NCCL_IB_HCA}" ]; then
    NCCL_IB_HCA=$(bash "$SCRIPT_DIR/helpers/get_nccl_ib_hca.sh")
fi
export NCCL_IB_HCA

# Dynamically get network interface IP address for socket communication if not set
if [ -z "${IP_INTERFACE}" ]; then
    IP_INTERFACE=$(bash "$SCRIPT_DIR/helpers/get_ip_interface.sh")
fi
export IP_INTERFACE
# Set network interfaces for NCCL and Gloo, fallback to detected IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

log_exported_vars "NCCL and Network Settings" \
    HIP_VISIBLE_DEVICES NCCL_DEBUG NCCL_CHECKS_DISABLE NCCL_IB_GID_INDEX \
    NCCL_IB_HCA IP_INTERFACE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

# ----------------- GPU Model Detection and Configuration Loading -----------------
# Detect GPU model and load model-specific optimizations
GPU_MODEL=${GPU_MODEL:-}

if [[ -z "$GPU_MODEL" ]]; then
    # Try to detect GPU model using rocm-smi
    if command -v rocm-smi &> /dev/null; then
        GPU_MODEL=$(bash "$SCRIPT_DIR/helpers/detect_gpu_model.sh" 2>/dev/null || echo "")
    fi
fi

# Load GPU model-specific configuration
ENV_CONFIG_DIR="$SCRIPT_DIR/helpers/envs"
GPU_CONFIG_FILE=""

if [[ -n "$GPU_MODEL" ]]; then
    LOG_INFO_RANK0 "Detected GPU model: $GPU_MODEL"

    # Check for exact match first
    if [[ -f "$ENV_CONFIG_DIR/${GPU_MODEL}.sh" ]]; then
        GPU_CONFIG_FILE="$ENV_CONFIG_DIR/${GPU_MODEL}.sh"
    # Check for MI300 variants (MI300X, MI300A)
    elif [[ "$GPU_MODEL" =~ ^MI300 ]] && [[ -f "$ENV_CONFIG_DIR/MI300X.sh" ]]; then
        GPU_CONFIG_FILE="$ENV_CONFIG_DIR/MI300X.sh"
    # Check for MI250 variants
    elif [[ "$GPU_MODEL" =~ ^MI250 ]] && [[ -f "$ENV_CONFIG_DIR/MI250X.sh" ]]; then
        GPU_CONFIG_FILE="$ENV_CONFIG_DIR/MI250X.sh"
    # Fallback to default
    else
        LOG_WARN "No specific configuration found for GPU model: $GPU_MODEL"
        GPU_CONFIG_FILE="$ENV_CONFIG_DIR/default.sh"
    fi
else
    LOG_WARN "Unable to detect GPU model, using default configuration"
    GPU_CONFIG_FILE="$ENV_CONFIG_DIR/default.sh"
fi

# Source the GPU-specific configuration
if [[ -f "$GPU_CONFIG_FILE" ]]; then
    LOG_INFO_RANK0 "Loading GPU configuration: $GPU_CONFIG_FILE"
    # shellcheck disable=SC1090
    source "$GPU_CONFIG_FILE"
else
    LOG_ERROR "GPU configuration file not found: $GPU_CONFIG_FILE"
    LOG_ERROR "Please create $ENV_CONFIG_DIR/default.sh or specify GPU_MODEL manually"
    exit 1
fi

# ----------------- Common Performance Tuning -----------------
# These settings apply to all GPU models

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}

log_exported_vars "Common Performance Tuning" \
    TORCH_NCCL_HIGH_PRIORITY

# -------------------- setup_pythonpath -------------------
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH="${PRIMUS_PATH}:${site_packages}:${PYTHONPATH:-}"
log_exported_vars "pythonpath" PYTHONPATH
