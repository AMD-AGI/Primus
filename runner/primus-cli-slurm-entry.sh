#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# Resolve script directory robustly (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"


if [[ -z "${SLURM_NODELIST:-}" ]]; then
    echo "[primus-slurm-entry][ERROR] SLURM_NODELIST not set. Are you running inside a Slurm job?"
    exit 2
fi

# Pick master node address from SLURM_NODELIST, or fallback
if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1 || echo localhost)"
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


echo "[primus-cli-slurm-entry] MASTER_ADDR=$MASTER_ADDR"
echo "[primus-cli-slurm-entry] MASTER_PORT=$MASTER_PORT"
echo "[primus-cli-slurm-entry] NNODES=$NNODES"
echo "[primus-cli-slurm-entry] NODE_RANK=$NODE_RANK"
echo "[primus-cli-slurm-entry] GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[primus-cli-slurm-entry] NODE_LIST: ${NODE_ARRAY[*]}"

# ------------- Dispatch based on mode ---------------

PATCH_ARGS=(
    --env MASTER_ADDR="$MASTER_ADDR"
    --env MASTER_PORT="$MASTER_PORT"
    --env NNODES="$NNODES"
    --env NODE_RANK="$NODE_RANK"
    --env GPUS_PER_NODE="$GPUS_PER_NODE"
    --log_file "logs/log_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S).txt"
)

# Default: 'container' mode, unless user overrides
MODE="container"
if [[ $# -gt 0 && "$1" =~ ^(container|direct|native|host)$ ]]; then
    MODE="$1"
    shift
fi
# MODE="${1:-container}"
# shift || true
case "$MODE" in
    container)
        script_path="$SCRIPT_DIR/primus-cli-container.sh"
        if [[ "$NODE_RANK" == "0" ]]; then
            PATCH_ARGS=(--verbose "${PATCH_ARGS[@]}")
        else
            PATCH_ARGS=(--no-verbose "${PATCH_ARGS[@]}")
        fi
        ;;
    direct)
        script_path="$SCRIPT_DIR/primus-cli-entrypoint.sh"
        ;;
    *)
        echo "[primus-cli-slurm-entry][ERROR] Unknown mode: $MODE. Use 'container' or 'direct'." >&2
        exit 2
        ;;
esac

if [[ ! -f "$script_path" ]]; then
    echo "[primus-slurm-entry][ERROR] Script not found: $script_path" >&2
    exit 2
fi

echo "[primus-slurm-entry] Executing: bash $script_path ${PATCH_ARGS[*]} $*"
exec bash "$script_path" "${PATCH_ARGS[@]}" "$@"
