#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# Get current script dir for resolving downstream scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_NODELIST:-}" ]]; then
    echo "[primus-slurm-entry][ERROR] SLURM_NODELIST not set. Are you running inside a Slurm job?"
    exit 2
fi

# Pick master node address from SLURM_NODELIST, or fallback
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_NODELIST:-}" ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
else
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi
MASTER_PORT="${MASTER_PORT:-1234}"

# Get all node hostnames (sorted, as needed)
readarray -t NODE_ARRAY < <(scontrol show hostnames "$SLURM_NODELIST")
# (Optional: sort by IP if needed, e.g., for deterministic rank mapping)
# Uncomment if you need IP sort
# readarray -t NODE_ARRAY < <(
#     for node in $(scontrol show hostnames "$SLURM_NODELIST"); do
#         getent hosts "$node" | awk '{print $1, $2}'
#     done | sort -k1,1n | awk '{print $2}'
# )

NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-${NNODES:-1}}}"
NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-${NODE_RANK:-0}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"


echo "[primus-run-slurm] MASTER_ADDR=$MASTER_ADDR"
echo "[primus-run-slurm] MASTER_PORT=$MASTER_PORT"
echo "[primus-run-slurm] NNODES=$NNODES"
echo "[primus-run-slurm] NODE_RANK=$NODE_RANK"
echo "[primus-run-slurm] GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[primus-run-slurm] NODE_LIST: ${NODE_ARRAY[*]}"

# ------------- Dispatch based on mode ---------------

# Default: 'container' mode, unless user overrides
MODE="container"
if [[ $# -gt 0 && "$1" =~ ^(container|direct|native|host)$ ]]; then
    MODE="$1"
    shift
fi

PATCH_ARGS=(
    --env MASTER_ADDR="$MASTER_ADDR"
    --env MASTER_PORT="$MASTER_PORT"
    --env NNODES="$NNODES"
    --env NODE_RANK="$NODE_RANK"
    --env GPUS_PER_NODE="$GPUS_PER_NODE"
    --log_file "logs/log_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S).txt"
)

case "$MODE" in
    container)
        script_path="$SCRIPT_DIR/primus-cli-container.sh"
        if [[ "$NODE_RANK" == "0" ]]; then
            PATCH_ARGS=(--verbose "${PATCH_ARGS[@]}")
        else
            PATCH_ARGS=(--no-verbose "${PATCH_ARGS[@]}")
        fi
        ;;
    direct/native/host)
        script_path="$SCRIPT_DIR/primus-cli-entrypoint.sh"
        ;;
    *)
        echo "Unknown mode: $MODE. Use 'container' or 'native'."
        exit 2
        ;;
esac

if [[ ! -f "$script_path" ]]; then
    echo "[primus-slurm-entry][ERROR] Script not found: $script_path"
    exit 2
fi

echo "[primus-slurm-entry] Executing: bash $script_path ${PATCH_ARGS[*]} $*"
exec bash "$script_path" "${PATCH_ARGS[@]}" "$@"
