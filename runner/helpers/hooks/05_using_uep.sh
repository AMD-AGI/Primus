#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: enable using UEP settings.
#
# Trigger:
#   export USING_UEP=1
#
###############################################################################


if [ "$USING_UEP" == "1" ]; then
    LOG_INFO "USING_UEP is enabled, checking required packages..."

    if ! pip show uccl &>/dev/null || ! pip show deep_ep &>/dev/null; then
        LOG_ERROR "uccl is not installed! Please use pre-installed primus image or set REBUILD_UCCL=1."
        exit 1
    fi
    LOG_INFO "uccl package is installed: $(pip show uccl | grep Version)"
    LOG_INFO "deep_ep package is installed: $(pip show deep_ep | grep Version)"

    if [ "$ENABLE_NUMA_BINDING" != "1" ]; then
        LOG_WARN "ENABLE_NUMA_BINDING is not enabled! Please set ENABLE_NUMA_BINDING=1 to avoid dataloader worker exited unexpectedly."
    fi

    export PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP
    LOG_INFO "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to DEEP_EP"

    # network settings for UCCL
    export UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX:-$NCCL_IB_GID_INDEX}
    export UCCL_IB_HCA=${UCCL_IB_HCA:-$NCCL_IB_HCA}
    export UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}

    # set low latency and normal inflight and bytes to avoid hang on AMD Pollara AI NIC and Broadcom Thor-2
    if [ "$USING_AINIC" == "1" ]; then
        export UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
        export UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
        export UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-4194304} # 4MB
    elif [ "$REBUILD_BNXT" == "1" ]; then # Broadcom Thor-2
        # FIXME(zhuang12): use `USING_BNXT` for Broadcom Thor-2 maybe better than `REBUILD_BNXT`
        export UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
        export UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
        export UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-1572864}
    fi


    LOG_INFO "==========UCCL Network Settings=========="
    LOG_INFO "UCCL_IB_GID_INDEX: $UCCL_IB_GID_INDEX"
    LOG_INFO "UCCL_IB_HCA: $UCCL_IB_HCA"
    LOG_INFO "UCCL_SOCKET_IFNAME: $UCCL_SOCKET_IFNAME"
    LOG_INFO "UCCL_IB_MAX_INFLIGHT_NORMAL: $UCCL_IB_MAX_INFLIGHT_NORMAL"
    LOG_INFO "UCCL_IB_MAX_INFLIGHT_LOW_LATENCY: $UCCL_IB_MAX_INFLIGHT_LOW_LATENCY"
    LOG_INFO "UCCL_IB_MAX_INFLIGHT_BYTES: $UCCL_IB_MAX_INFLIGHT_BYTES"
    LOG_INFO ""
else
    export PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO
    LOG_INFO "USING_UEP is disabled. PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to TURBO"
fi
