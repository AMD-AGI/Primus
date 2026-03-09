#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: auto-detect NCCL_IB_TC and NCCL_IB_FIFO_TC for Pensando AINIC.
#
# Runs BEFORE 03_enable_ainic.sh (sort order: 03_d < 03_e).
# On Pensando clusters this hook queries nicctl to find the actual PFC-protected
# DSCP and sets NCCL_IB_TC / NCCL_IB_FIFO_TC accordingly.
#
# Priority chain:  manual env > this hook (hardware detect) > 03_enable_ainic defaults
#
# If the user already exported NCCL_IB_TC / NCCL_IB_FIFO_TC, this hook
# respects them and exits immediately.
#
# On non-Pensando / non-AINIC clusters this hook is a no-op.
#
# This hook emits env.* lines which will be exported by execute_hooks.sh.
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Guard: only relevant when AINIC is enabled
# ---------------------------------------------------------------------------
if [[ "${USING_AINIC:-0}" != "1" ]]; then
    exit 0
fi

# ---------------------------------------------------------------------------
# Guard: respect user-specified values (manual > detect)
# ---------------------------------------------------------------------------
if [[ -n "${NCCL_IB_TC:-}" && -n "${NCCL_IB_FIFO_TC:-}" ]]; then
    echo "NCCL_IB_TC and NCCL_IB_FIFO_TC already set, skipping auto-detect" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Detect whether the first IB device is a Pensando/ionic NIC
# ---------------------------------------------------------------------------
_is_pensando() {
    local ib_dev=""
    for dev in /sys/class/infiniband/*; do
        [ -e "$dev" ] || continue
        ib_dev=$(basename "$dev")
        break
    done
    [ -z "$ib_dev" ] && return 1

    # Quick path: device name contains "ionic"
    if echo "$ib_dev" | grep -qi "ionic"; then
        return 0
    fi

    # Fallback: check ibstat CA type
    local ca_type
    ca_type=$(ibstat "$ib_dev" 2>/dev/null | grep "CA type:" | head -1 || true)
    echo "$ca_type" | grep -qi "Pensando"
}

if ! _is_pensando; then
    # Not a Pensando AINIC cluster — nothing to do.
    exit 0
fi

# ---------------------------------------------------------------------------
# Ensure nicctl is available (try to install if missing)
# ---------------------------------------------------------------------------
if ! command -v nicctl &>/dev/null; then
    echo "nicctl not found, attempting to install..." >&2
    if ! apt-get install -y nicctl &>/dev/null; then
        echo "WARN: Failed to install nicctl, keeping current defaults" >&2
        exit 0
    fi
fi

# Double-check after install attempt
if ! command -v nicctl &>/dev/null; then
    echo "WARN: nicctl still not available after install, keeping current defaults" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Query nicctl for the correct DSCP values
# ---------------------------------------------------------------------------
qos_output=$(nicctl show qos 2>/dev/null) || {
    echo "WARN: nicctl show qos failed, keeping current defaults" >&2
    exit 0
}

pfc_prio=$(echo "$qos_output" | grep "PFC no-drop priorities" | head -1 | awk '{print $NF}' || true)
if [ -z "$pfc_prio" ]; then
    echo "WARN: Could not determine PFC priority, keeping current defaults" >&2
    exit 0
fi

# Extract single-value DSCP for a given priority (skip bitmap and range lines)
_extract_dscp_for_priority() {
    echo "$qos_output" \
        | grep -v "bitmap" \
        | grep "DSCP" \
        | grep "==> priority : ${1}$" \
        | head -1 \
        | sed 's/.*DSCP[^:]*: *//' \
        | sed 's/ *==> .*//' \
        | tr -d ' ' || true
}

# NCCL_IB_TC: DSCP mapped to PFC-protected (no-drop) priority × 4
data_dscp=$(_extract_dscp_for_priority "$pfc_prio")
if ! echo "$data_dscp" | grep -qE '^[0-9]+$'; then
    echo "WARN: Could not parse DSCP for PFC priority $pfc_prio, keeping current defaults" >&2
    exit 0
fi

# NCCL_IB_FIFO_TC: DSCP mapped to strict-priority queue × 4
strict_prio=$(echo "$qos_output" | grep -i "strict" | head -1 | awk '{print $1}' || true)
fifo_dscp=""
if [ -n "$strict_prio" ] && echo "$strict_prio" | grep -qE '^[0-9]+$'; then
    fifo_dscp=$(_extract_dscp_for_priority "$strict_prio")
fi
if ! echo "$fifo_dscp" | grep -qE '^[0-9]+$'; then
    fifo_dscp="$data_dscp"
fi

ib_tc=$((data_dscp * 4))
fifo_tc=$((fifo_dscp * 4))

echo "env.NCCL_IB_TC=${ib_tc}"
echo "env.NCCL_IB_FIFO_TC=${fifo_tc}"
