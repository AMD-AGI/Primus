#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# Resolve runner directory robustly (handles symlinks)
RUNNER_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

# Load common library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Load validation library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/validation.sh" || {
    LOG_ERROR "[slurm-entry] Failed to load validation library: $RUNNER_DIR/lib/validation.sh"
    exit 1
}

# Load config library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/config.sh" || {
    LOG_ERROR "[slurm-entry] Failed to load config library: $RUNNER_DIR/lib/config.sh"
    exit 1
}

# Parse --config, --debug, --dry-run first if present
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0
PRE_PARSE_ARGS=()
# Pre-parse only until mode separator or mode keyword
while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            PRE_PARSE_ARGS+=("$@")
            break
            ;;
        container|direct|native|host)
            PRE_PARSE_ARGS+=("$@")
            break
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            export DEBUG_MODE=1
            shift
            ;;
        --dry-run)
            # Only treat as entry dry-run if before mode keyword
            DRY_RUN_MODE=1
            shift
            ;;
        *)
            PRE_PARSE_ARGS+=("$1")
            shift
            ;;
    esac
done
# Restore arguments
set -- "${PRE_PARSE_ARGS[@]}"

# Load configuration (specified or defaults)
load_config_auto "$CONFIG_FILE" "slurm-entry" || {
    LOG_ERROR "[slurm-entry] Configuration loading failed"
    exit 1
}

# Extract slurm.* config parameters
declare -A slurm_config
extract_config_section "slurm" slurm_config || {
    LOG_ERROR "[slurm-entry] Failed to extract slurm config section"
    exit 1
}

# Apply slurm config values if not set via CLI
[[ "$DEBUG_MODE" == "0" && ("${slurm_config[debug]:-false}" == "true" || "${slurm_config[debug]:-false}" == "1") ]] && DEBUG_MODE=1
[[ "$DRY_RUN_MODE" == "0" && ("${slurm_config[dry_run]:-false}" == "true" || "${slurm_config[dry_run]:-false}" == "1") ]] && DRY_RUN_MODE=1

# Enable debug mode if set
# if [[ "$DEBUG_MODE" == "1" ]]; then
#     set -x
# fi

# Validate Slurm environment
if [[ -z "${SLURM_NODELIST:-}" ]]; then
    LOG_ERROR "[slurm-entry] SLURM_NODELIST not set. Are you running inside a Slurm job?"
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
validate_distributed_params || LOG_WARN "[slurm-entry] Failed to validate distributed parameters"

# Log configuration
LOG_INFO_RANK0 "[slurm-entry] MASTER_ADDR=$MASTER_ADDR"
LOG_INFO_RANK0 "[slurm-entry] MASTER_PORT=$MASTER_PORT"
LOG_INFO_RANK0 "[slurm-entry] NNODES=$NNODES"
LOG_INFO_RANK0 "[slurm-entry] NODE_RANK=$NODE_RANK"
LOG_INFO_RANK0 "[slurm-entry] GPUS_PER_NODE=$GPUS_PER_NODE"
LOG_INFO_RANK0 "[slurm-entry] NODE_LIST: ${NODE_ARRAY[*]}"

# ------------- Dispatch based on mode ---------------

# Export environment variables for next script
export MASTER_ADDR
export MASTER_PORT
export NNODES
export NODE_RANK
export GPUS_PER_NODE

# Default: 'container' mode, unless user overrides
MODE="container"
# If the first arg is just a separator, skip it
if [[ "$1" == "--" ]]; then
    shift
fi
if [[ $# -gt 0 && "$1" =~ ^(container|direct|native|host)$ ]]; then
    MODE="$1"
    shift
fi

# Build arguments based on mode
SCRIPT_ARGS=()

# Pass --config, --debug, --dry-run to next script if present
if [[ -n "$CONFIG_FILE" ]]; then
    SCRIPT_ARGS+=(--config "$CONFIG_FILE")
fi
if [[ "$DEBUG_MODE" == "1" ]]; then
    SCRIPT_ARGS+=(--debug)
fi

case "$MODE" in
    container)
        script_path="$RUNNER_DIR/primus-cli-container.sh"
        # For container mode, pass environment variables as docker --env options
        SCRIPT_ARGS+=(
            --env "MASTER_ADDR=$MASTER_ADDR"
            --env "MASTER_PORT=$MASTER_PORT"
            --env "NNODES=$NNODES"
            --env "NODE_RANK=$NODE_RANK"
            --env "GPUS_PER_NODE=$GPUS_PER_NODE"
        )
        ;;
    direct)
        script_path="$RUNNER_DIR/primus-cli-direct.sh"
        # For direct mode, environment variables are already exported above
        ;;
    *)
        LOG_ERROR "[slurm-entry] Unknown mode: $MODE. Use 'container' or 'direct'."
        exit 2
        ;;
esac

if [[ ! -f "$script_path" ]]; then
    LOG_ERROR "[slurm-entry] Script not found: $script_path"
    exit 2
fi


# Execute or dry-run
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    LOG_INFO "[slurm-entry] [DRY-RUN] Would execute: bash $script_path ${SCRIPT_ARGS[*]} $*"
else
    LOG_INFO "[slurm-entry] Executing: bash $script_path ${SCRIPT_ARGS[*]} $*"
    exec bash "$script_path" "${SCRIPT_ARGS[@]}" "$@"
fi
