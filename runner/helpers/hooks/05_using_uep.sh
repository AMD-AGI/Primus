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

set -euo pipefail

is_turbo_deepep_requested() {
    local prev=""
    local arg
    for arg in "$@"; do
        if [[ "$prev" == "--use_turbo_deepep" ]]; then
            case "${arg,,}" in
                1|true|yes|on)
                    return 0
                    ;;
            esac
        fi
        prev="$arg"
    done

    return 1
}

if [[ "${USING_UEP:-0}" == "1" ]]; then
    LOG_INFO "USING_UEP is enabled, checking required packages..."

    if ! python3 -m pip show uccl &>/dev/null || ! python3 -m pip show deep_ep &>/dev/null; then
        LOG_ERROR "uccl is not installed! Please use pre-installed primus image or set REBUILD_UEP=1."
        exit 1
    fi
    LOG_INFO "uccl package is installed: $(python3 -m pip show uccl | grep Version)"
    LOG_INFO "deep_ep package is installed: $(python3 -m pip show deep_ep | grep Version)"

    echo "env.PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP"
    LOG_INFO "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to DEEP_EP"
    if [[ -n "${UCCL_REF:-}" || -n "${UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC:-}" || -n "${UCCL_EP_FORCE_CURRENT_STREAM:-}" ]]; then
        LOG_INFO "UCCL-EP rocSHMEM fallback env configured: UCCL_REF=${UCCL_REF:-<unset>}, UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC=${UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC:-<unset>}, UCCL_EP_FORCE_CURRENT_STREAM=${UCCL_EP_FORCE_CURRENT_STREAM:-<unset>}."
    fi

    UCCL_IB_GID_INDEX=${UCCL_IB_GID_INDEX:-${NCCL_IB_GID_INDEX:-}}
    UCCL_IB_HCA=${UCCL_IB_HCA:-${NCCL_IB_HCA:-}}
    UCCL_SOCKET_IFNAME=${UCCL_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}
    UCCL_IB_TC=${UCCL_IB_TC:-${NCCL_IB_TC:-}}
    UCCL_IB_SL=${UCCL_IB_SL:-${NCCL_IB_SL:-}}

    is_broadcom_turbo_deepep=false
    if [[ "${UCCL_IB_HCA}" == *"rocep"* ]] && is_turbo_deepep_requested "$@"; then
        is_broadcom_turbo_deepep=true
        LOG_INFO "Broadcom RoCE + Turbo DeepEP detected; applying communication and dataloader compatibility settings only."
    fi

    # defaults for inflight settings; may be overridden for specific NICs below
    UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-}
    UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-}
    UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-}

    # Recommended inflight defaults for specific NIC families.
    if [[ "${USING_AINIC:-0}" == "1" ]]; then
        UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL:-1}
        UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY:-1}
        UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES:-4194304} # 4MB
    elif [[ "${REBUILD_BNXT:-0}" == "1" ]]; then # Broadcom Thor-2
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
    if [[ -n "${UCCL_IB_MAX_INFLIGHT_NORMAL}" ]]; then
        echo "env.UCCL_IB_MAX_INFLIGHT_NORMAL=${UCCL_IB_MAX_INFLIGHT_NORMAL}"
    fi
    if [[ -n "${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY}" ]]; then
        echo "env.UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=${UCCL_IB_MAX_INFLIGHT_LOW_LATENCY}"
    fi
    if [[ -n "${UCCL_IB_MAX_INFLIGHT_BYTES}" ]]; then
        echo "env.UCCL_IB_MAX_INFLIGHT_BYTES=${UCCL_IB_MAX_INFLIGHT_BYTES}" # 4MB
    fi


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

    if [[
        "${PRIMUS_DISABLE_BATCH_P2P_FOR_DEEPEP_ROCEP:-0}" != "1"
        && "${is_broadcom_turbo_deepep}" == "true"
    ]]; then
        echo "extra.overlap_p2p_comm=false"
        echo "extra.communication_method=batch_p2p"
        LOG_INFO "Broadcom RoCE + Turbo DeepEP detected; requesting overlap_p2p_comm=false and communication_method=batch_p2p."

        if [[ "${PRIMUS_DISABLE_NUM_WORKERS0_FOR_DEEPEP_ROCEP:-0}" != "1" ]]; then
            echo "extra.num_workers=0"
            LOG_INFO "Broadcom RoCE + Turbo DeepEP detected; requesting num_workers=0."
        fi
    fi
else
    echo "env.PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO"
    LOG_INFO "USING_UEP is disabled. PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to TURBO"
fi
