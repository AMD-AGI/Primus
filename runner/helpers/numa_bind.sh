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

# This rank's GPU in amd-smi's list: LOCAL_RANK indexes the visible devices, so map it through
# HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES), then ROCR_VISIBLE_DEVICES, which select from the GPUs
# below them. Without the mapping a rank on GPU 2 alone (HIP_VISIBLE_DEVICES=2) took GPU 0's node.
# Lists of UUIDs are not mapped.
GPU=${LOCAL_RANK}
for VISIBLE in "${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}" "${ROCR_VISIBLE_DEVICES:-}"; do
    if [[ "${VISIBLE}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        IFS=, read -r -a VISIBLE_IDS <<<"${VISIBLE}"
        GPU=${VISIBLE_IDS[$GPU]:-$GPU}
    fi
done

# Access the NUMA node information for the specific bus ID
NODE=$(cat /sys/bus/pci/devices/"${BUS_ID[$GPU]}"/numa_node)

LOG_INFO_NODE_RANK0 "Starting binding local rank ${LOCAL_RANK} (GPU ${GPU}) to numa_node ${NODE}..."

# If the first argument is a Python entrypoint (e.g. primus/cli/main.py) or a
# module invocation (e.g. -m primus.cli.main), it is not directly executable.
# In that case, run it via python3.
if [[ $# -gt 0 && ( "${1}" == *.py || "${1}" == "-m" ) ]]; then
    set -- python3 "$@"
fi

numactl --cpunodebind="$NODE" --membind="$NODE" "$@"
