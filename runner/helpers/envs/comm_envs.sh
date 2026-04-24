#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Communication library env defaults (UCCL / MoRI).
#
# Both UCCL and MoRI network env vars default to the corresponding NCCL
# values so they work out of the box. Users can override any variable by
# exporting it before invoking the launcher.
#
# Prerequisite: this file must be sourced AFTER the following NCCL vars
# have been resolved (dynamic probing in base_env.sh / run_pretrain.sh):
#   NCCL_IB_GID_INDEX, NCCL_IB_HCA, NCCL_SOCKET_IFNAME,
#   NCCL_IB_TC (optional), NCCL_IB_SL (optional)
#
# This script is idempotent: sourcing it multiple times is safe.
###############################################################################

# Guard against duplicate sourcing / logging.
if [[ -n "${__PRIMUS_COMM_ENVS_SOURCED:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
export __PRIMUS_COMM_ENVS_SOURCED=1

# ----------------- UCCL network defaults (derive from NCCL) -----------------
export UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX:-${NCCL_IB_GID_INDEX:-}}
export UCCL_IB_HCA=${UCCL_IB_HCA:-${NCCL_IB_HCA:-}}
export UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}
export UCCL_IB_TC=${UCCL_IB_TC:-${NCCL_IB_TC:-}}
export UCCL_IB_SL=${UCCL_IB_SL:-${NCCL_IB_SL:-}}
# ep performance tuning for uccl
export UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC=${UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC:-1}
export UCCL_EP_FORCE_CURRENT_STREAM=${UCCL_EP_FORCE_CURRENT_STREAM:-1}

# UCCL inflight tuning: NIC-specific defaults to avoid hang on AMD Pollara
# AI NIC (USING_AINIC=1) and Broadcom Thor-2 (REBUILD_BNXT=1). When neither
# is set, leave the variables unset so UCCL uses its own defaults.
if [[ "${USING_AINIC:-0}" == "1" ]]; then
    export UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
    export UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
    export UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-4194304} # 4MB
elif [[ "${REBUILD_BNXT:-0}" == "1" ]]; then
    export UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
    export UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
    export UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-1572864} # 1.5MB
fi

# ----------------- MoRI network defaults (derive from NCCL) -----------------
export MORI_IB_GID_INDEX=${MORI_IB_GID_INDEX:-${NCCL_IB_GID_INDEX:-}}
export MORI_RDMA_DEVICES=${MORI_RDMA_DEVICES:-${NCCL_IB_HCA:-}}
export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}
export MORI_RDMA_TC=${MORI_RDMA_TC:-${NCCL_IB_TC:-}}
export MORI_RDMA_SL=${MORI_RDMA_SL:-${NCCL_IB_SL:-}}
export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-}

# ----------------- Logging (best effort) -----------------
if declare -F log_exported_vars >/dev/null 2>&1; then
    log_exported_vars "UCCL Network Settings" \
        UCCL_IB_GID_INDEX UCCL_IB_HCA UCCL_SOCKET_IFNAME UCCL_IB_TC UCCL_IB_SL \
        UCCL_IB_MAX_INFLIGHT_NORMAL UCCL_IB_MAX_INFLIGHT_LOW_LATENCY UCCL_IB_MAX_INFLIGHT_BYTES
    log_exported_vars "MoRI Network Settings" \
        MORI_IB_GID_INDEX MORI_RDMA_DEVICES MORI_SOCKET_IFNAME MORI_RDMA_TC MORI_IB_SL \
        MORI_SHMEM_HEAP_SIZE
fi
