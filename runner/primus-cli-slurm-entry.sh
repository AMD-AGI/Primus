#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# Resolve script directory robustly (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

# Parse --config, --debug first if present
# Note: --dry-run is handled by primus-cli and won't be passed here
CONFIG_FILE=""
DEBUG_MODE=0
for ((i=1; i<=$#; i++)); do
    case "${!i}" in
        --config)
            ((i++))
            CONFIG_FILE="${!i}"
            ;;
        --debug)
            DEBUG_MODE=1
            ;;
    esac
done

# Load common library and validation
if [[ -f "$SCRIPT_DIR/lib/common.sh" ]]; then
    source "$SCRIPT_DIR/lib/common.sh"
fi

if [[ -f "$SCRIPT_DIR/lib/validation.sh" ]]; then
    source "$SCRIPT_DIR/lib/validation.sh"
fi

# Load config library and mode-specific config
if [[ -f "$SCRIPT_DIR/lib/config.sh" ]]; then
    source "$SCRIPT_DIR/lib/config.sh" 2>/dev/null || true
    # If config file is provided via --config, load slurm-specific config
    if [[ -n "$CONFIG_FILE" ]] && [[ -f "$CONFIG_FILE" ]]; then
        load_yaml_config "$CONFIG_FILE" 2>/dev/null || true
        load_mode_config "slurm" 2>/dev/null || true
    fi
fi

# Validate Slurm environment
if [[ -z "${SLURM_NODELIST:-}" ]]; then
    if type LOG_ERROR &>/dev/null; then
        LOG_ERROR "SLURM_NODELIST not set. Are you running inside a Slurm job?"
    else
        echo "[primus-slurm-entry][ERROR] SLURM_NODELIST not set. Are you running inside a Slurm job?" >&2
    fi
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

export NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-${NNODES:-1}}}"
export NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-${NODE_RANK:-0}}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export MASTER_ADDR
export MASTER_PORT

# Validate distributed parameters
if type validate_distributed_params &>/dev/null; then
    validate_distributed_params
fi

# Log configuration
if type LOG_INFO &>/dev/null; then
    LOG_INFO "MASTER_ADDR=$MASTER_ADDR"
    LOG_INFO "MASTER_PORT=$MASTER_PORT"
    LOG_INFO "NNODES=$NNODES"
    LOG_INFO "NODE_RANK=$NODE_RANK"
    LOG_INFO "GPUS_PER_NODE=$GPUS_PER_NODE"
    LOG_INFO "NODE_LIST: ${NODE_ARRAY[*]}"
else
    echo "[primus-cli-slurm-entry] MASTER_ADDR=$MASTER_ADDR"
    echo "[primus-cli-slurm-entry] MASTER_PORT=$MASTER_PORT"
    echo "[primus-cli-slurm-entry] NNODES=$NNODES"
    echo "[primus-cli-slurm-entry] NODE_RANK=$NODE_RANK"
    echo "[primus-cli-slurm-entry] GPUS_PER_NODE=$GPUS_PER_NODE"
    echo "[primus-cli-slurm-entry] NODE_LIST: ${NODE_ARRAY[*]}"
fi

# ------------- Dispatch based on mode ---------------

PATCH_ARGS=(
    --env MASTER_ADDR="$MASTER_ADDR"
    --env MASTER_PORT="$MASTER_PORT"
    --env NNODES="$NNODES"
    --env NODE_RANK="$NODE_RANK"
    --env GPUS_PER_NODE="$GPUS_PER_NODE"
    --log_file "logs/log_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S).txt"
)

# Pass --config, --debug to next script if present
if [[ -n "$CONFIG_FILE" ]]; then
    PATCH_ARGS+=(--config "$CONFIG_FILE")
fi
if [[ "$DEBUG_MODE" == "1" ]]; then
    PATCH_ARGS+=(--debug)
fi

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
        if type LOG_ERROR &>/dev/null; then
            LOG_ERROR "Unknown mode: $MODE. Use 'container' or 'direct'."
        else
            echo "[primus-cli-slurm-entry][ERROR] Unknown mode: $MODE. Use 'container' or 'direct'." >&2
        fi
        exit 2
        ;;
esac

if [[ ! -f "$script_path" ]]; then
    if type LOG_ERROR &>/dev/null; then
        LOG_ERROR "Script not found: $script_path"
    else
        echo "[primus-slurm-entry][ERROR] Script not found: $script_path" >&2
    fi
    exit 2
fi

if type LOG_INFO &>/dev/null; then
    LOG_INFO "Executing: bash $script_path ${PATCH_ARGS[*]} $*"
else
    echo "[primus-slurm-entry] Executing: bash $script_path ${PATCH_ARGS[*]} $*"
fi
exec bash "$script_path" "${PATCH_ARGS[@]}" "$@"
