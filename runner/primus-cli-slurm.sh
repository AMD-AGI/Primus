#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

print_usage() {
cat <<'EOF'
Primus Slurm Launcher

Usage:
    primus-cli slurm [--config FILE] [--debug] [--dry-run] [srun|sbatch] [SLURM_FLAGS...] -- <entry> [ENTRY_ARGS...] -- [PRIMUS_ARGS...]

Description:
    Launch distributed Primus jobs via Slurm.
    - Everything before the first '--' is passed to Slurm (srun/sbatch and flags).
    - <entry> specifies Primus execution mode: container | direct  (default: container).
    - The second '--' (if any) separates Primus entry args from Primus CLI arguments.
    - If <entry> is anything other than 'container' or 'direct' (e.g. a primus
      subcommand like 'preflight'), it is routed through the default container
      chain unchanged.

Options:
    --config FILE    Load configuration from specified file
    --debug          Enable debug mode (verbose logging)
    --dry-run        Show what would be executed without actually running

Examples:
    # Launch 4 nodes using srun and container mode
    primus-cli slurm srun -N 4 -p AIG_Model -- container -- train pretrain --config exp.yaml

    # Launch with sbatch, log to file, run benchmark
    primus-cli slurm sbatch --output=run.log -N 2 -- container -- benchmark gemm -M 4096 -N 4096 -K 4096

    # Run preflight environment check across 4 nodes (terse form: defaults to container entry)
    primus-cli slurm srun -N 4 -- preflight --quick

    # Run preflight via the direct entry (no container; uses host python or VENV_ACTIVATE)
    primus-cli slurm srun -N 4 -- direct -- preflight --quick

    # Run node smoke test via direct entry (auto-selects single mode for node_smoke)
    primus-cli slurm srun -N 4 -- direct -- node_smoke --tier2-perf

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

# Resolve runner directory
RUNNER_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

# Load common library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Load config library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/config.sh" || {
    LOG_ERROR "[slurm] Failed to load config library: $RUNNER_DIR/lib/config.sh"
    exit 1
}

LOG_INFO "-----------------------------------------------"
LOG_INFO "primus-cli-slurm.sh"
LOG_INFO "-----------------------------------------------"


# 0. Parse --config, --debug, --dry-run first if present (before first --)
CONFIG_FILE=""
DEBUG_MODE=false
DRY_RUN_MODE=false
ENTRY_ARGS=()
PRE_PARSE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            # Stop parsing, preserve -- and all remaining args
            PRE_PARSE_ARGS+=("$@")
            break
            ;;
        --config)
            CONFIG_FILE="$2"
            ENTRY_ARGS+=(--config "$CONFIG_FILE")
            shift 2
            ;;
        --debug)
            export DEBUG_MODE=true
            ENTRY_ARGS+=(--debug)
            shift
            ;;
        --dry-run)
            DRY_RUN_MODE=true
            ENTRY_ARGS+=(--dry-run)
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

# Step 1: Load config and extract slurm.* parameters
declare -A slurm_config

# Load configuration (specified or defaults)
load_config_auto "$CONFIG_FILE" "slurm" || {
    LOG_ERROR "[slurm] Configuration loading failed"
    exit 1
}

# Extract all slurm.* config parameters using the generic function
extract_config_section "slurm" slurm_config || {
    LOG_ERROR "[slurm] Failed to extract slurm config section"
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

if [[ "$DRY_RUN_MODE" == "false" ]]; then
    dry_run_value="${slurm_config[dry_run]:-false}"
    if [[ "$dry_run_value" == "true" || "$dry_run_value" == "1" ]]; then
        DRY_RUN_MODE=true
        LOG_INFO "[slurm] Dry-run mode enabled via config"
    fi
fi

# Enable debug mode if set
if [[ "$DEBUG_MODE" == "true" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO "[slurm] Debug mode enabled (PRIMUS_LOG_LEVEL=DEBUG)"
fi


# Step 2: Detect srun/sbatch mode
LAUNCH_CMD="srun"   # Default launcher
if [[ "${1:-}" == "sbatch" || "${1:-}" == "srun" ]]; then
    LAUNCH_CMD="$1"
    shift
fi

# Step 3: Collect CLI arguments and track which parameters are overridden
# Map long options to their short option equivalents for override detection
declare -A LONG_TO_SHORT=(
    ["partition"]="p"
    ["nodes"]="N"
    ["ntasks"]="n"
    ["cpus-per-task"]="c"
    ["time"]="t"
    ["output"]="o"
    ["error"]="e"
    ["job-name"]="J"
)


declare -A CLI_OVERRIDES  # Track which config params are overridden by CLI
CLI_ARGS=()  # Store original CLI arguments

while [[ $# -gt 0 && "$1" != "--" ]]; do
    arg="$1"
    shift

    # Store original CLI arg
    CLI_ARGS+=("$arg")

    # Track what parameter is being overridden
    if [[ "$arg" =~ ^-- ]]; then
        # Long options support both `--nodes 2` and `--nodes=2`. Strip any
        # inline value before override detection, while preserving the original
        # token in CLI_ARGS for Slurm itself.
        param_name="${arg#--}"
        has_inline_value=false
        if [[ "$param_name" == *=* ]]; then
            param_name="${param_name%%=*}"
            has_inline_value=true
        fi
        CLI_OVERRIDES["$param_name"]=1
        # Also mark the short form as overridden
        if [[ -n "${LONG_TO_SHORT[$param_name]:-}" ]]; then
            CLI_OVERRIDES["${LONG_TO_SHORT[$param_name]}"]=1
        fi

        # Store a separate value when this was not the --option=value form.
        if [[ "$has_inline_value" == "false" && $# -gt 0 && ! "$1" =~ ^- ]]; then
            CLI_ARGS+=("$1")
            shift
        fi
    elif [[ "$arg" =~ ^- ]]; then
        # Short options support both `-N 2` and the Slurm-native attached form
        # `-N2`. For attached values, the option name is the first character.
        short_token="${arg#-}"
        param_name="$short_token"
        has_inline_value=false
        if [[ ${#short_token} -gt 1 ]]; then
            short_name="${short_token:0:1}"
            for long in "${!LONG_TO_SHORT[@]}"; do
                if [[ "${LONG_TO_SHORT[$long]}" == "$short_name" ]]; then
                    param_name="$short_name"
                    has_inline_value=true
                    break
                fi
            done
        fi
        CLI_OVERRIDES["$param_name"]=1
        # Also mark the long form as overridden by checking reverse mapping
        for long in "${!LONG_TO_SHORT[@]}"; do
            if [[ "${LONG_TO_SHORT[$long]}" == "$param_name" ]]; then
                CLI_OVERRIDES["$long"]=1
                break
            fi
        done

        # If option has a separate value, store it too.
        if [[ "$has_inline_value" == "false" && $# -gt 0 && ! "$1" =~ ^- ]]; then
            CLI_ARGS+=("$1")
            shift
        fi
    fi
done

# Step 4: Build SLURM_FLAGS from config (only non-overridden params) + CLI args
SLURM_FLAGS=()

# Add config parameters that were not overridden by CLI
for param_name in "${!slurm_config[@]}"; do
    # Skip internal primus-cli parameters (not slurm flags)
    if [[ "$param_name" == "debug" || "$param_name" == "dry_run" || "$param_name" == "gpus_per_node" ]]; then
        continue
    fi

    # Skip if this parameter was provided via CLI
    if [[ -n "${CLI_OVERRIDES[$param_name]:-}" ]]; then
        continue
    fi

    param_value="${slurm_config[$param_name]}"

    # Use short form if available, otherwise long form
    if [[ ${#param_name} -eq 1 ]]; then
        # Already a short option
        if [[ -n "$param_value" ]]; then
            SLURM_FLAGS+=("-$param_name" "$param_value")
        else
            SLURM_FLAGS+=("-$param_name")
        fi
    elif [[ -n "${LONG_TO_SHORT[$param_name]:-}" ]]; then
        # Has a known short form, use it
        if [[ -n "$param_value" ]]; then
            SLURM_FLAGS+=("-${LONG_TO_SHORT[$param_name]}" "$param_value")
        else
            SLURM_FLAGS+=("-${LONG_TO_SHORT[$param_name]}")
        fi
    else
        # Use long form
        if [[ -n "$param_value" ]]; then
            SLURM_FLAGS+=("--$param_name" "$param_value")
        else
            SLURM_FLAGS+=("--$param_name")
        fi
    fi
done

# Append all CLI arguments (preserving their original format)
SLURM_FLAGS+=("${CLI_ARGS[@]}")

# Skip '--'
if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
fi

# 3. Check for primus-run args
if [[ $# -eq 0 ]]; then
    LOG_ERROR "[slurm] Missing Primus entry (container|direct)"
    print_usage >&2
    exit 2
fi

# 4. Logging and launch
ENTRY="$RUNNER_DIR/primus-cli-slurm-entry.sh"
require_file "$ENTRY" "[slurm] Entry script not found: $ENTRY"

# On the sbatch path the scheduler copies the batch script into its spool dir, so
# the entry cannot locate its own lib/ from $0. Hand it the real runner directory
# (propagated by sbatch --export=ALL, which is the default on both flavors).
export PRIMUS_RUNNER_DIR="$RUNNER_DIR"

# Build full command. Spur is detected by the presence of the `spur` command; its
# srun takes a narrower flag set than standard Slurm's and counts tasks
# differently -- see below.
IS_SPUR=false
command -v spur >/dev/null 2>&1 && IS_SPUR=true

# Normalize the srun flags that spur rejects, then make sure a task count is
# present. Only the two flags we have observed failing on spur are touched:
#
#   --export           spur's srun aborts with "unexpected argument '--export'".
#                      Dropping it is safe because both flavors export the full
#                      environment by default (see the note above).
#   --ntasks-per-node  spur's srun aborts the same way. `-N <nodes>
#                      --ntasks-per-node=<k>` means the same thing as
#                      `-n <nodes * k>`, so translate rather than drop.
#
# Beyond that, standard Slurm's srun derives one task per node from -N while
# spur's srun defaults to --ntasks=1: `srun -N 4 ENTRY` dispatches a single task,
# so only one node runs the entry and the other ranks never start. Request one
# task per node unless the caller already chose a task count.
#
# The other flags this repo passes to srun (--exclusive, --cpus-per-task,
# --time, --nodelist, --partition) have no reported spur failures and are left
# alone. sbatch is likewise untouched: the reports are srun-only, and spur runs
# the batch script on every allocated node regardless of --ntasks.
if [[ "$IS_SPUR" == "true" && "$LAUNCH_CMD" == "srun" ]]; then
    spur_nodes=""
    spur_has_ntasks=false
    for ((i = 0; i < ${#SLURM_FLAGS[@]}; i++)); do
        case "${SLURM_FLAGS[i]}" in
            -N|--nodes) spur_nodes="${SLURM_FLAGS[i + 1]:-}" ;;
            -N?*) spur_nodes="${SLURM_FLAGS[i]#-N}" ;;
            --nodes=*) spur_nodes="${SLURM_FLAGS[i]#--nodes=}" ;;
            -n|--ntasks|-n?*|--ntasks=*) spur_has_ntasks=true ;;
        esac
    done

    # Second pass: rewrite the flag list now that the node count is known, since
    # -N may appear after --ntasks-per-node.
    spur_flags=()
    spur_tasks=""
    for ((i = 0; i < ${#SLURM_FLAGS[@]}; i++)); do
        case "${SLURM_FLAGS[i]}" in
            --export=*) ;;
            --export)
                # Also swallow the detached value of `--export ALL`.
                if [[ $((i + 1)) -lt ${#SLURM_FLAGS[@]} && "${SLURM_FLAGS[i + 1]}" != -* ]]; then
                    i=$((i + 1))
                fi
                ;;
            --ntasks-per-node|--ntasks-per-node=*)
                spur_per_node=""
                spur_value_detached=false
                if [[ "${SLURM_FLAGS[i]}" == --ntasks-per-node=* ]]; then
                    spur_per_node="${SLURM_FLAGS[i]#--ntasks-per-node=}"
                elif [[ $((i + 1)) -lt ${#SLURM_FLAGS[@]} && "${SLURM_FLAGS[i + 1]}" != -* ]]; then
                    spur_per_node="${SLURM_FLAGS[i + 1]}"
                    spur_value_detached=true
                fi
                if [[ "$spur_nodes" =~ ^[0-9]+$ && "$spur_per_node" =~ ^[0-9]+$ ]]; then
                    spur_tasks=$((spur_nodes * spur_per_node))
                    if [[ "$spur_value_detached" == "true" ]]; then
                        i=$((i + 1))
                    fi
                else
                    # Without a plain integer node count there is no equivalent
                    # task count to compute. Keep the flag: dropping it would
                    # silently fall back to spur's one-task default and leave the
                    # remaining ranks unstarted, which is a far more expensive
                    # failure than srun rejecting the argument outright. The
                    # detached value, if any, is preserved by the next iteration.
                    LOG_WARN "[slurm] Spur srun: cannot translate ${SLURM_FLAGS[i]} without an integer node count (got '${spur_nodes:-unset}'); pass -n explicitly if srun rejects it"
                    spur_flags+=("${SLURM_FLAGS[i]}")
                fi
                ;;
            *) spur_flags+=("${SLURM_FLAGS[i]}") ;;
        esac
    done
    SLURM_FLAGS=("${spur_flags[@]}")

    if [[ "$spur_has_ntasks" == "true" ]]; then
        : # The caller picked a task count; leave it alone.
    elif [[ -n "$spur_tasks" ]]; then
        SLURM_FLAGS+=(-n "$spur_tasks")
        LOG_INFO "[slurm] Spur srun: translated --ntasks-per-node to -n $spur_tasks"
    elif [[ "$spur_nodes" =~ ^[0-9]+$ ]]; then
        SLURM_FLAGS+=(-n "$spur_nodes")
        LOG_INFO "[slurm] Spur srun: requesting one task per node (-n $spur_nodes)"
    fi
fi

CMD=("$LAUNCH_CMD" "${SLURM_FLAGS[@]}" "$ENTRY" "${ENTRY_ARGS[@]}" -- "$@")

# Display command
if [[ "$DRY_RUN_MODE" == "true" ]]; then
    LOG_INFO "[slurm] [DRY RUN] Would execute: ${CMD[*]}"
    LOG_INFO "[slurm] Dry-run mode: command not executed"
    exit 0
fi

LOG_INFO "[slurm] Executing: ${CMD[*]}"
exec "${CMD[@]}"
