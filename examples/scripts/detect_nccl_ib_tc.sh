#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Auto-detect correct NCCL_IB_TC and NCCL_IB_FIFO_TC for Pensando AINIC clusters.
# Reads QoS DSCP-to-priority mapping from nicctl and finds the PFC-protected DSCP.
#
# Usage:
#   source detect_nccl_ib_tc.sh        # sets NCCL_IB_TC and NCCL_IB_FIFO_TC
#   eval $(./detect_nccl_ib_tc.sh)     # alternative: export from subshell

set -euo pipefail

is_pensando() {
    local ib_dev=""

    for dev in /sys/class/infiniband/*; do
        [ -e "$dev" ] || continue
        ib_dev=$(basename "$dev")
        break
    done
    [ -z "$ib_dev" ] && return 1

    if echo "$ib_dev" | grep -qi "ionic"; then
        return 0
    fi

    local ca_type
    ca_type=$(ibstat "$ib_dev" 2>/dev/null | grep "CA type:" | head -1 || true)
    echo "$ca_type" | grep -qi "Pensando"
}

detect_pensando_tc() {
    if ! command -v nicctl &>/dev/null; then
        echo "WARN: nicctl not found, using known Pensando defaults" >&2
        echo "104 192"
        return
    fi

    local qos_output
    qos_output=$(nicctl show qos 2>/dev/null) || {
        echo "WARN: nicctl show qos failed, using defaults" >&2
        echo "104 192"
        return
    }

    local pfc_prio
    pfc_prio=$(echo "$qos_output" | grep "PFC no-drop priorities" | head -1 | awk '{print $NF}')

    if [ -z "$pfc_prio" ]; then
        echo "WARN: Could not determine PFC priority, using defaults" >&2
        echo "104 192"
        return
    fi

    # nicctl output lines look like:
    #   DSCP                      : 26 ==> priority : 3
    #   DSCP bitmap               : 0x0000000004000000 ==> priority : 3
    #   DSCP                      : 0-25, 27-47, 49-63 ==> priority : 0
    # We want the single DSCP (not bitmap, not range) that maps to each priority.

    # Helper: extract single-value DSCP for a given priority
    extract_dscp_for_priority() {
        echo "$qos_output" \
            | grep -v "bitmap" \
            | grep "DSCP" \
            | grep "==> priority : ${1}$" \
            | head -1 \
            | sed 's/.*DSCP[^:]*: *//' \
            | sed 's/ *==> .*//' \
            | tr -d ' '
    }

    # NCCL_IB_TC: use DSCP that maps to PFC-protected (no-drop) priority
    local data_dscp
    data_dscp=$(extract_dscp_for_priority "$pfc_prio")

    if ! echo "$data_dscp" | grep -qE '^[0-9]+$'; then
        echo "WARN: Could not parse DSCP for PFC priority $pfc_prio, using defaults" >&2
        echo "104 192"
        return
    fi

    # NCCL_IB_FIFO_TC: use DSCP that maps to the strict-priority queue
    # (scheduling output: priority N has "strict" type)
    local strict_prio
    strict_prio=$(echo "$qos_output" | grep -i "strict" | head -1 | awk '{print $1}')
    local fifo_dscp=""
    if [ -n "$strict_prio" ] && echo "$strict_prio" | grep -qE '^[0-9]+$'; then
        fifo_dscp=$(extract_dscp_for_priority "$strict_prio")
    fi

    if ! echo "$fifo_dscp" | grep -qE '^[0-9]+$'; then
        echo "WARN: Could not find strict-priority DSCP, using same as data" >&2
        fifo_dscp="$data_dscp"
    fi

    echo "$((data_dscp * 4)) $((fifo_dscp * 4))"
}

if ! is_pensando; then
    echo "# Not a Pensando AINIC cluster, no NCCL_IB_TC override needed" >&2
    exit 0
fi

result=$(detect_pensando_tc)
echo "$result"
