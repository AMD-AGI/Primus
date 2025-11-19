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
DEBUG_MODE=false
DRY_RUN_MODE=false
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
            DEBUG_MODE=true
            shift
            ;;
        --dry-run)
            # Only treat as entry dry-run if before mode keyword
            DRY_RUN_MODE=true
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
if [[ "$DEBUG_MODE" == "false" ]]; then
    debug_value="${slurm_config[debug]:-false}"
    if [[ "$debug_value" == "true" ]]; then
        DEBUG_MODE=true
        LOG_INFO "[slurm] Debug mode enabled via config (PRIMUS_LOG_LEVEL=DEBUG)"
    fi
fi
# Enable debug mode if set
if [[ "$DEBUG_MODE" == "true" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO "[slurm] Debug mode enabled (PRIMUS_LOG_LEVEL=DEBUG)"
fi

if [[ "$DRY_RUN_MODE" == "false" ]]; then
    dry_run_value="${slurm_config[dry_run]:-false}"
    if [[ "$dry_run_value" == "true" || "$dry_run_value" == "1" ]]; then
        DRY_RUN_MODE=true
        LOG_INFO "[slurm] Dry-run mode enabled via config"
    fi
fi

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

# Log configuration
LOG_DEBUG_RANK0 "[slurm-entry] MASTER_ADDR=$MASTER_ADDR"
LOG_DEBUG_RANK0 "[slurm-entry] MASTER_PORT=$MASTER_PORT"
LOG_DEBUG_RANK0 "[slurm-entry] NNODES=$NNODES"
LOG_DEBUG_RANK0 "[slurm-entry] NODE_RANK=$NODE_RANK"
LOG_DEBUG_RANK0 "[slurm-entry] GPUS_PER_NODE=$GPUS_PER_NODE"
LOG_DEBUG_RANK0 "[slurm-entry] NODE_LIST: ${NODE_ARRAY[*]}"

# Validate distributed parameters
validate_distributed_params || LOG_WARN "[slurm-entry] Failed to validate distributed parameters"

# ------------- Dispatch based on mode ---------------

# Parse mode (default: container)
[[ "${1:-}" == "--" ]] && shift
MODE="container"
if [[ "${1:-}" =~ ^(container|direct)$ ]]; then
    MODE="$1"
    shift
fi

# Build arguments based on mode
SCRIPT_ARGS=()
if [[ -n "$CONFIG_FILE" ]]; then
    SCRIPT_ARGS+=(--config "$CONFIG_FILE")
fi
if [[ "$DEBUG_MODE" == "true" ]]; then
    SCRIPT_ARGS+=(--debug)
fi
SCRIPT_ARGS+=(
    --env "MASTER_ADDR=$MASTER_ADDR"
    --env "MASTER_PORT=$MASTER_PORT"
    --env "NNODES=$NNODES"
    --env "NODE_RANK=$NODE_RANK"
    --env "GPUS_PER_NODE=$GPUS_PER_NODE"
)

# Build script path based on mode
script_path="$RUNNER_DIR/primus-cli-${MODE}.sh"
require_file "$script_path" "[slurm-entry] Script not found: $script_path"

# Build full command
CMD=(bash "$script_path" "${SCRIPT_ARGS[@]}" "$@")
LOG_INFO "[slurm-entry] Would execute: ${CMD[*]}"

if [[ "$DRY_RUN_MODE" == "true" ]]; then
    LOG_INFO "[slurm-entry] Dry-run mode: command not executed"
    exit 0
fi

LOG_INFO "[slurm-entry] Executing command..."
exec "${CMD[@]}"
