#!/bin/bash

LOG_INFO_NODE_RANK0() {
    if [ "$NODE_RANK" -eq 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
        fi
    fi
}

LOG_INFO_RANK0() {
    if [ "$RANK" -eq 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
        fi
    fi
}

# Capture the output from amd-smi and process it without quotes
mapfile -t BUS_ID < <(amd-smi list --csv | awk -F, 'NR>1 && $2!="" {print $2}')
LOG_INFO_RANK0 "BUS_IDs: ${BUS_ID[*]}"

# Access the NUMA node information for the specific bus ID
NODE=$(cat /sys/bus/pci/devices/"${BUS_ID[$LOCAL_RANK]}"/numa_node)

LOG_INFO_NODE_RANK0 "Starting binding local rank ${LOCAL_RANK} to numa_node ${NODE}..."
numactl --cpunodebind="$NODE" --membind="$NODE" "$@"
