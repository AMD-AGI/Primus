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

    if ! python3 -m pip show uccl &>/dev/null || ! python3 -m pip show deep_ep &>/dev/null; then
        LOG_ERROR "uccl is not installed! Please use pre-installed primus image or set REBUILD_UEP=1."
        exit 1
    fi
    LOG_INFO "uccl package is installed: $(python3 -m pip show uccl | grep Version)"
    LOG_INFO "deep_ep package is installed: $(python3 -m pip show deep_ep | grep Version)"

    echo "env.PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP"
    LOG_INFO "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to DEEP_EP"

    UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX:-$NCCL_IB_GID_INDEX}
    UCCL_IB_HCA=${UCCL_IB_HCA:-$NCCL_IB_HCA}
    UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
    UCCL_IB_TC=${UCCL_IB_TC:-$NCCL_IB_TC}
    UCCL_IB_SL=${UCCL_IB_SL:-$NCCL_IB_SL}


    # set low latency and normal inflight and bytes to avoid hang on AMD Pollara AI NIC and Broadcom Thor-2
    if [ "$USING_AINIC" == "1" ]; then
        UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
        UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
        UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-4194304} # 4MB
    elif [ "$REBUILD_BNXT" == "1" ]; then # Broadcom Thor-2
        # FIXME(zhuang12): use `USING_BNXT` for Broadcom Thor-2 maybe better than `REBUILD_BNXT`
        UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
        UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
        UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-1572864} # 1.5MB
    fi

    # network settings for UCCL
    echo "env.UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX}"
    echo "env.UCCL_IB_HCA=${UCCL_IB_HCA}"
    echo "env.UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME}"
    echo "env.UCCL_IB_TC=${UCCL_IB_TC}"
    echo "env.UCCL_IB_SL=${UCCL_IB_SL}"
    echo "env.UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL}"
    echo "env.UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY}"
    echo "env.UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES}" # 4MB


    LOG_INFO "==========UCCL Network Settings=========="
    LOG_INFO "UCCL_IB_GID_INDEX: $UCCL_IB_GID_INDEX"
    LOG_INFO "UCCL_IB_HCA: $UCCL_IB_HCA"
    LOG_INFO "UCCL_SOCKET_IFNAME: $UCCL_SOCKET_IFNAME"
    LOG_INFO "UCCL_IB_MAX_INFLIGHT_NORMAL: $UCCL_IB_MAX_INFLIGHT_NORMAL"
    LOG_INFO "UCCL_IB_MAX_INFLIGHT_LOW_LATENCY: $UCCL_IB_MAX_INFLIGHT_LOW_LATENCY"
    LOG_INFO "UCCL_IB_MAX_INFLIGHT_BYTES: $UCCL_IB_MAX_INFLIGHT_BYTES"
    LOG_INFO "UCCL_IB_TC: $UCCL_IB_TC"
    LOG_INFO "UCCL_IB_SL: $UCCL_IB_SL"
    LOG_INFO "====================================="
else
    echo "env.PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO"
    LOG_INFO "USING_UEP is disabled. PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to TURBO"
fi
