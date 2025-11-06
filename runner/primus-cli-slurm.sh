#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

print_usage() {
cat <<'EOF'
Primus Slurm Launcher

Usage:
    primus-cli slurm [--config FILE] [--debug] [--dry-run] [srun|sbatch] [SLURM_FLAGS...] -- <entry> [ENTRY_ARGS...] -- [PRIMUS_ARGS...]

Description:
    Launch distributed Primus jobs via Slurm.
    - Everything before the first '--' is passed to Slurm (srun/sbatch and flags).
    - <entry> specifies Primus execution mode: container | direct | preflight (see below).
    - The second '--' (if any) separates Primus entry args from Primus CLI arguments.

Options:
    --config FILE    Load configuration from specified file
    --debug          Enable debug mode (verbose logging)
    --dry-run        Show what would be executed without actually running

Examples:
    # Launch 4 nodes using srun and container mode
    primus-cli slurm srun -N 4 -p AIG_Model -- container -- train pretrain --config exp.yaml

    # Launch with sbatch, log to file, run benchmark
    primus-cli slurm sbatch --output=run.log -N 2 -- container -- benchmark gemm -M 4096 -N 4096 -K 4096

    # Run preflight environment check across 4 nodes
    primus-cli slurm srun -N 4 -- preflight

    # Dry-run to see what would be executed
    primus-cli slurm --dry-run srun -N 4 -- container -- train

    # Use configuration file with dry-run
    primus-cli slurm --config slurm.yaml --dry-run sbatch -- container -- benchmark

Notes:
    - [srun|sbatch] is optional; defaults to srun if not specified.
    - All SLURM_FLAGS before '--' are passed directly to Slurm (supports both --flag=value and --flag value).
    - Everything after the first '--' is passed to Primus entry (e.g. container, direct, etc.), and then to Primus CLI.
    - For unsupported or extra Slurm options, just pass them after '--' (they'll be ignored by this wrapper).

Debug:
    - Collected SLURM flags and primus arguments will be printed before launch.

EOF
}

# Show help if requested or if no args are given
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# 0. Parse --config, --debug, --dry-run first if present
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0
CONFIG_ARGS=()
PRE_PARSE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            CONFIG_ARGS+=(--config "$CONFIG_FILE")
            shift 2
            ;;
        --debug)
            export DEBUG_MODE=1
            CONFIG_ARGS+=(--debug)
            shift
            ;;
        --dry-run)
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

# Load common library first (required by config.sh)
if [[ -f "$SCRIPT_DIR/lib/common.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/lib/common.sh" 2>/dev/null || true
fi

# Load config library and slurm-specific config
if [[ -f "$SCRIPT_DIR/lib/config.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/lib/config.sh" 2>/dev/null || true
    # If config file is provided via --config, load slurm-specific config
    if [[ -n "$CONFIG_FILE" ]] && [[ -f "$CONFIG_FILE" ]]; then
        load_yaml_config "$CONFIG_FILE" 2>/dev/null || true
        load_mode_config "slurm" 2>/dev/null || true
    fi
fi

# 1. Detect srun/sbatch mode
LAUNCH_CMD="srun"   # Default launcher
if [[ "${1:-}" == "sbatch" || "${1:-}" == "srun" ]]; then
    LAUNCH_CMD="$1"
    shift
fi

# 2. Collect SLURM_FLAGS before '--'
# Fully generic passthrough mechanism: all CLI args override config
# No hardcoded Slurm parameters - everything is passed through
declare -A CLI_OPTS_SET
CLI_ARGS=()

# Collect all CLI arguments and track which options are set
while [[ $# -gt 0 && "$1" != "--" ]]; do
    CLI_ARGS+=("$1")
    # Track option flags (e.g., -p, -N, --partition, --nodes, etc.)
    if [[ "$1" =~ ^- ]]; then
        CLI_OPTS_SET["$1"]=1
    fi
    shift
done

# Build config args that are NOT overridden by CLI
CONFIG_ARGS=()
if [[ -n "${SLURM_PARTITION:-}" ]] && [[ -z "${CLI_OPTS_SET[-p]:-}" ]] && [[ -z "${CLI_OPTS_SET[--partition]:-}" ]]; then
    CONFIG_ARGS+=(-p "$SLURM_PARTITION")
fi
if [[ -n "${SLURM_NODES:-}" ]] && [[ -z "${CLI_OPTS_SET[-N]:-}" ]] && [[ -z "${CLI_OPTS_SET[--nodes]:-}" ]]; then
    CONFIG_ARGS+=(-N "$SLURM_NODES")
fi

# Final SLURM_FLAGS: config args first, then CLI args (CLI has priority)
SLURM_FLAGS=("${CONFIG_ARGS[@]}" "${CLI_ARGS[@]}")

# Skip '--'
if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
fi

# 3. Check for primus-run args
if [[ $# -eq 0 ]]; then
    echo "[primus-cli-slurm][ERROR] Missing Primus entry (container|direct|preflight)" >&2
    print_usage >&2
    exit 2
fi

# 4. Logging and launch
ENTRY="$SCRIPT_DIR/primus-cli-slurm-entry.sh"

# Prepend global options to entry args
ENTRY_ARGS=()
if [[ -n "$CONFIG_FILE" ]]; then
    ENTRY_ARGS+=(--config "$CONFIG_FILE")
fi
if [[ "$DEBUG_MODE" == "1" ]]; then
    ENTRY_ARGS+=(--debug)
fi
ENTRY_ARGS+=("$@")

echo "[primus-cli-slurm] Executing: $LAUNCH_CMD ${SLURM_FLAGS[*]:-} $ENTRY ${ENTRY_ARGS[*]:-}"

# Handle dry-run mode
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    echo "[DRY-RUN] Would execute: $LAUNCH_CMD ${SLURM_FLAGS[*]:-} $ENTRY ${ENTRY_ARGS[*]:-}"
    exit 0
fi

exec "$LAUNCH_CMD" "${SLURM_FLAGS[@]}" "$ENTRY" "${ENTRY_ARGS[@]}"
