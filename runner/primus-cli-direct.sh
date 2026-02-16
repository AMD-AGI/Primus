#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

print_usage() {
cat << EOF
Primus Direct Launcher

Usage:
    primus-cli direct [--env KEY=VALUE ...] [--single] [--script <file.py>] [--patch <file>] [-- primus-args]

Description:
    Launch Primus training, benchmarking, or preflight directly on the host (or inside a container).
    Distributed settings can be controlled by either exporting environment variables in advance,
    or by specifying them inline using --env KEY=VALUE.

    This script automatically detects your GPU model (e.g., MI300X, MI250X) and loads GPU-specific
    environment variable optimizations from runner/helpers/envs/<GPU_MODEL>.sh for best performance.

Options:
    --config <file>      Specify configuration file (default: .primus.yaml or system default)
    --debug              Enable debug mode with verbose logging
    --dry-run            Show configuration and command without executing
    --single             Run with python3 instead of torchrun (single process only)
    --script <file.py>   Python script to execute (default: primus/cli/main.py)
    --env KEY=VALUE      Set environment variable before execution
    --patch <file>       Apply a patch script before execution (can specify multiple times)
    --log_file PATH      Save log to a specific file (default: logs/log_TIMESTAMP.txt)
    --numa               Force enable NUMA binding for processes
    --no-numa            Force disable NUMA binding for processes

Distributed Environment Variables:
    NNODES        Number of nodes participating in distributed run        [default: 1]
    NODE_RANK     Rank of the current node (unique integer per node)      [default: 0]
    GPUS_PER_NODE Number of GPUs to use per node                          [default: 8]
    MASTER_ADDR   Hostname or IP of master node                           [default: localhost]
    MASTER_PORT   Port of master node                                     [default: 1234]

You can set these variables in either of the following ways:
    # (1) Export variables before launch (recommended for scripts or single-node runs)
      export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=host1
      primus-cli direct -- train pretrain --config exp.yaml

    # (2) Inject via CLI with --env (useful for launchers and multi-node jobs)
      primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=1 --env MASTER_ADDR=host1 -- \\
        benchmark gemm -M 4096 -N 4096 -K 4096

Examples:
    # Pretrain with a config file (single node)
      primus-cli direct -- train pretrain --config examples/megatron/exp_pretrain.yaml

    # Benchmark GEMM (single node)
      primus-cli direct -- benchmark gemm

    # Distributed GEMM benchmark, 2 nodes, 8 GPUs per node (rank 0)
      primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=0 --env MASTER_ADDR=host1 -- \\
        benchmark gemm -M 4096 -N 4096 -K 4096

    # Launch as rank 1 (2-node distributed)
      primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=1 --env MASTER_ADDR=host1 -- \\
        benchmark gemm -M 4096 -N 4096 -K 4096

    # Run a custom script directly (no torchrun)
      primus-cli direct --single --script examples/debug_run.py -- --arg1 val1

    # Run a custom script with torchrun
      primus-cli direct --script examples/run_distributed.py -- --config conf.yaml

    # Apply patch scripts before execution
      primus-cli direct --patch patches/fix_env.sh --patch patches/setup.py -- train pretrain --config exp.yaml

    # Force enable NUMA binding for better performance
      primus-cli direct --numa -- benchmark gemm -M 8192 -N 8192 -K 8192

Notes:
    - If --single is specified, Primus skips torchrun and uses python3 directly.
    - If --script is not specified, defaults to primus/cli/main.py.
    - Always separate Primus arguments from launcher options using '--'.
    - Environment variables can be mixed: 'export' takes precedence unless overridden by '--env'.
    - Multi-node jobs require MASTER_ADDR set to the master node's hostname/IP.
    - Patch scripts are executed in order before running the main script (useful for env setup, hot fixes, etc.).
    - NUMA binding is auto-disabled by default; use --numa to enable for better memory locality.
    - GPU-specific optimizations: The script automatically sources primus-env.sh, which detects your
      GPU model and loads optimized environment variables from runner/helpers/envs/<GPU_MODEL>.sh
      (e.g., MI300X.sh, MI250X.sh). If no GPU-specific config is found, it uses base_env.sh only.

EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# Resolve runner directory
# Use RUNNER_DIR instead of SCRIPT_DIR to avoid conflicts with sourced scripts
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/common.sh" || {
    echo "[ERROR] Failed to load common library: $RUNNER_DIR/lib/common.sh" >&2
    exit 1
}

# Load config library (required)
# shellcheck disable=SC1091
source "$RUNNER_DIR/lib/config.sh" || {
    LOG_ERROR "[direct] Failed to load config library: $RUNNER_DIR/lib/config.sh"
    exit 1
}

LOG_INFO_RANK0 "-----------------------------------------------"
LOG_INFO_RANK0 "primus-cli-direct.sh"
LOG_INFO_RANK0 "-----------------------------------------------"



###############################################################################
# STEP 1: Pre-parse global options (--config, --debug, --dry-run, --help)
###############################################################################
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0
PRE_PARSE_ARGS=()
PRIMUS_ARGS=()

# If the first argument is the subcommand "direct", skip it
if [[ "${1:-}" == "direct" ]]; then
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --dry-run)
            DRY_RUN_MODE=1
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        --env)
            if [[ "$2" == *=* ]]; then
                # Inline KEY=VALUE form: export immediately and keep for later so that
                # direct.env / --env can be re-applied with highest priority.
                export "${2%%=*}"="${2#*=}"
                PRE_PARSE_ARGS+=("$1" "$2")
                shift 2
            else
                # Non KEY=VALUE form: treat as env file path and defer sourcing until
                # after hooks and patches (STEP 9). Use a synthetic --env_file option
                # so that it is tracked via direct_config[env_file].
                PRE_PARSE_ARGS+=("--env_file" "$2")
                LOG_INFO_RANK0 "[direct] CLI: --env_file $2"
                shift 2
            fi
            ;;
        --script|--log_file|--patch)
            # Runner options that take a value
            PRE_PARSE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --numa|--no-numa|--single)
            # Runner boolean flags (no value)
            PRE_PARSE_ARGS+=("$1")
            shift
            ;;
        --)
            # Explicit separator: remaining args are for primus Python module
            shift  # skip the '--'
            PRIMUS_ARGS+=("$@")
            break
            ;;
        *)
            # First unrecognized argument (typically a subcommand like 'train'):
            # Stop parsing runner options, pass everything from here to primus Python module.
            # This prevents runner from intercepting options meant for Python (e.g., --config).
            PRIMUS_ARGS+=("$@")
            break
            ;;
    esac
done
# Restore arguments for second pass. Use `set --` so that options parsing stops
# and all following tokens (including those starting with '-') become positional
# parameters for the next parsing stage.
set -- "${PRE_PARSE_ARGS[@]}" -- "${PRIMUS_ARGS[@]}"
LOG_INFO_RANK0 "[direct] PRE_PARSE_ARGS (runner options): ${PRE_PARSE_ARGS[*]}"
LOG_INFO_RANK0 "[direct] PRIMUS_ARGS (python module args): ${PRIMUS_ARGS[*]}"
LOG_INFO_RANK0 "[direct] Combined args for second pass: $*"

# Enable debug mode early if set via CLI
if [[ "$DEBUG_MODE" == "1" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO_RANK0 "[direct] Debug mode enabled via CLI (PRIMUS_LOG_LEVEL=DEBUG)"
fi

###############################################################################
# STEP 2: Load configuration files
###############################################################################
load_config_auto "$CONFIG_FILE" "direct" || {
    LOG_ERROR "[direct] Configuration loading failed"
    exit 1
}

###############################################################################
# STEP 3: Extract direct.* config and apply defaults
###############################################################################
declare -A direct_config
extract_config_section "direct" direct_config || true

# Check debug/dry-run from config (CLI takes precedence, already set in STEP 1)
if [[ "$DEBUG_MODE" == "0" ]]; then
    debug_value="${direct_config[debug]:-false}"
    if [[ "$debug_value" == "true" || "$debug_value" == "1" ]]; then
        DEBUG_MODE=1
        export PRIMUS_LOG_LEVEL="DEBUG"
        LOG_INFO_RANK0 "[direct] Debug mode enabled via config (PRIMUS_LOG_LEVEL=DEBUG)"
    fi
fi

if [[ "$DRY_RUN_MODE" == "0" ]]; then
    dry_run_value="${direct_config[dry_run]:-false}"
    if [[ "$dry_run_value" == "true" || "$dry_run_value" == "1" ]]; then
        DRY_RUN_MODE=1
        LOG_INFO_RANK0 "[direct] Dry-run mode enabled via config"
    fi
fi

# Set default values for parameters not in config
direct_config[run_mode]="${direct_config[run_mode]:-torchrun}"
direct_config[script]="${direct_config[script]:-primus/cli/main.py}"
direct_config[numa]="${direct_config[numa]:-auto}"
direct_config[log_file]="${direct_config[log_file]:-}"

# Clean up empty array markers ("[]")
for key in "${!direct_config[@]}"; do
    if [[ "${direct_config[$key]}" == "[]" ]]; then
        unset "direct_config[$key]"
    fi
done

LOG_DEBUG_RANK0 "[direct] Configuration loaded, ready for CLI argument override"

###############################################################################
# STEP 4: Parse CLI arguments (all stored as newline-separated strings)
# Priority: CLI args > Config file > Defaults
###############################################################################
primus_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            primus_args+=("$@")
            break
            ;;
        --single)
            # Special case: --single maps to run_mode=single
            direct_config[run_mode]="single"
            LOG_DEBUG_RANK0 "[direct] CLI: run_mode=single"
            shift
            ;;
        --no-numa)
            # Special case: --no-numa maps to numa=false
            direct_config[numa]="false"
            LOG_DEBUG_RANK0 "[direct] CLI: numa=false"
            shift
            ;;
        --*)
            # Generic option handler
            opt_name="${1#--}"
            opt_value="${2:-}"

            # Check if next argument is a value or another flag
            if [[ -z "$opt_value" ]] || [[ "$opt_value" == --* ]]; then
                # Boolean flag (no value or next is a flag)
                direct_config[$opt_name]="true"
                LOG_INFO_RANK0 "[direct] CLI: $opt_name=true"
                shift
            else
                # Key-value option: append with newline
                if [[ -z "${direct_config[$opt_name]:-}" ]]; then
                    direct_config[$opt_name]="$opt_value"
                else
                    direct_config[$opt_name]+=$'\n'"$opt_value"
                fi
                LOG_INFO_RANK0 "[direct] CLI: $opt_name += $opt_value"
                shift 2
            fi
            ;;
        *)
            primus_args+=("$1")
            shift
            ;;
    esac
done
set -- "${primus_args[@]}"

###############################################################################
# STEP 4.5: Process non-cumulative parameters (use last value only)
###############################################################################

# For non-cumulative parameters with multiple values (config + CLI), use only the last value
for param in "script" "log_file"; do
    if [[ -n "${direct_config[$param]:-}" && "${direct_config[$param]}" == *$'\n'* ]]; then
        # Has newlines, take last value (CLI overrides config)
        last_value=$(echo "${direct_config[$param]}" | tail -1)
        direct_config[$param]="$last_value"
        LOG_DEBUG_RANK0 "[direct] Using last value for $param: $last_value"
    fi
done

###############################################################################
# STEP 4.6: Setup log file path
###############################################################################
if [[ -z "${direct_config[log_file]}" ]]; then
    direct_config[log_file]="logs/log_$(date +%Y%m%d_%H%M%S).txt"
fi
mkdir -p "$(dirname "${direct_config[log_file]}")"


###############################################################################
# STEP 5: Install dependencies
###############################################################################
# Detect the backend framework from the experiment YAML (--config in PRIMUS_ARGS)
# so we can install the correct requirements file:
#   maxtext -> requirements-jax.txt
#   others  -> requirements.txt
_detect_framework() {
    local cfg_path=""
    local args=("${primus_args[@]}")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--config" && -n "${args[$((i+1))]:-}" ]]; then
            cfg_path="${args[$((i+1))]}"
            break
        fi
    done
    if [[ -z "$cfg_path" || ! -f "$cfg_path" ]]; then
        echo ""
        return
    fi
    python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$cfg_path'))
    print(cfg.get('modules',{}).get('pre_trainer',{}).get('framework',''))
except Exception:
    print('')
" 2>/dev/null
}

DETECTED_FRAMEWORK="$(_detect_framework)"
LOG_INFO_RANK0 "[direct] Detected framework: ${DETECTED_FRAMEWORK:-unknown}"

# Skip pip install in dry-run mode
if [[ "$DRY_RUN_MODE" != "1" ]]; then
    if [[ "$DETECTED_FRAMEWORK" == "maxtext" ]]; then
        LOG_INFO_RANK0 "[direct] Installing JAX dependencies (requirements-jax.txt)"
        pip install -qq -r requirements-jax.txt
    else
        LOG_INFO_RANK0 "[direct] Installing PyTorch dependencies (requirements.txt)"
        pip install -qq -r requirements.txt
    fi
fi

###############################################################################
# STEP 6: Source GPU environment and helper modules
###############################################################################

# Source primus-env.sh (it will set its own SCRIPT_DIR, which is fine)
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/envs/primus-env.sh"

###############################################################################
# STEP 7: Execute hooks and capture extra arguments / env
###############################################################################
# Hooks can return:
#   - Extra Primus CLI arguments by printing lines:  extra.<name>=<value>
#   - Environment variables by printing lines:       env.VAR_NAME=VALUE
# The detailed parsing logic lives in execute_hooks.sh, which fills the
# global HOOK_EXTRA_PRIMUS_ARGS array and exports env.* entries.
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/execute_hooks.sh"
HOOK_EXTRA_PRIMUS_ARGS=()
if ! execute_hooks "$1" "$2" "$@"; then
    LOG_ERROR "[direct] Hooks execution failed"
    exit 1
fi

# Prepend collected extra args to the current Primus arguments ($@) so that
# they are automatically picked up when building the final CMD below.
if [[ ${#HOOK_EXTRA_PRIMUS_ARGS[@]} -gt 0 ]]; then
    set -- "$@" "${HOOK_EXTRA_PRIMUS_ARGS[@]}"
fi

###############################################################################
# STEP 8: Execute patch scripts
###############################################################################
# Execute patch scripts from config + CLI.
# Note: direct_config[patch] is stored as a newline-separated list.
if [[ -n "${direct_config[patch]:-}" ]]; then
    # shellcheck disable=SC1091
    source "${RUNNER_DIR}/helpers/execute_patches.sh"
    # Reset collected extra args from patches for this run
    PATCH_EXTRA_PRIMUS_ARGS=()
    if ! execute_patches "${direct_config[patch]}"; then
        LOG_ERROR "[direct] Patch execution failed"
        exit 1
    fi
fi

###############################################################################
# STEP 8.5: Apply extra Primus args from patches (extra.* protocol)
###############################################################################
if [[ ${#PATCH_EXTRA_PRIMUS_ARGS[@]} -gt 0 ]]; then
    set -- "$@" "${PATCH_EXTRA_PRIMUS_ARGS[@]}"
    LOG_INFO_RANK0 "[direct] Applied extra args from patches: ${PATCH_EXTRA_PRIMUS_ARGS[*]}"
fi

###############################################################################
# STEP 9: Build and export environment variables (highest priority)
###############################################################################

# First, source any env files specified via --env <file> (tracked as env_file).
if [[ -n "${direct_config[env_file]:-}" ]]; then
    while IFS= read -r env_file; do
        [[ -n "$env_file" ]] || continue
        if [[ ! -f "$env_file" || ! -r "$env_file" ]]; then
            LOG_ERROR "[direct] Env file not found or not readable: $env_file"
            exit 1
        fi
        # shellcheck disable=SC1090
        source "$env_file"
        LOG_INFO_RANK0 "[direct] Sourced env file (final): $env_file"
    done <<< "${direct_config[env_file]}"
fi

# Then, build primus_env_kv array and export inline KEY=VALUE envs. This happens
# after hooks, patches, and env files so that CLI/config-provided envs
# (direct.env / --env KEY=VALUE) take final precedence for the actual Primus run.
primus_env_kv=()
if [[ -n "${direct_config[env]:-}" ]]; then
    while IFS= read -r env_entry; do
        [[ -n "$env_entry" ]] || continue
        # Validate env format (KEY=VALUE)
        if ! [[ "$env_entry" == *=* ]]; then
            LOG_ERROR "[direct] Invalid env format: $env_entry"
            LOG_ERROR "  Expected format: KEY=VALUE (e.g., NCCL_DEBUG=INFO)"
            exit 1
        fi
        primus_env_kv+=("$env_entry")
        export "${env_entry%%=*}"="${env_entry#*=}"
        LOG_INFO_RANK0 "[direct] Exported env (final): ${env_entry%%=*}=${env_entry#*=}"
    done <<< "${direct_config[env]}"
fi

###############################################################################
# STEP 10: Build launch command
###############################################################################

# Allow RUN_MODE to be overridden by environment variable
RUN_MODE="${RUN_MODE:-${direct_config[run_mode]}}"

CMD="${direct_config[script]} $* 2>&1 | tee ${direct_config[log_file]}"
if [[ "$RUN_MODE" == "single" ]]; then
    CMD="python3 ${CMD}"
    LOG_INFO_RANK0 "[direct] Using python launcher (single mode)"
elif [[ "$RUN_MODE" == "torchrun" ]]; then
    # Step 2: Add NUMA binding prefix if enabled
    if [[ "${direct_config[numa]}" == "true" ]]; then
        CMD="--no-python ${RUNNER_DIR}/helpers/numa_bind.sh ${CMD}"
        LOG_INFO_RANK0 "[direct] NUMA binding: ENABLED (forced by CLI)"
    else
        LOG_INFO_RANK0 "[direct] NUMA binding: AUTO (default OFF)"
    fi

    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE:-8}"
        --nnodes "${NNODES:-1}"
        --node_rank "${NODE_RANK:-0}"
        --master_addr "${MASTER_ADDR:-localhost}"
        --master_port "${MASTER_PORT:-1234}"
    )

    LAST_NODE=$((NNODES - 1))
    FILTERS=()
    # Add local rank 0 on the first node
    if [ "$NODE_RANK" -eq 0 ]; then
        FILTERS+=(0)
    fi

    # Add the last local rank on the last node
    if [ "$NODE_RANK" -eq "$LAST_NODE" ]; then
        FILTERS+=($((GPUS_PER_NODE - 1)))
    fi

    # Build filter argument (only if FILTERS is non-empty)
    if [ "${#FILTERS[@]}" -gt 0 ]; then
        LOCAL_FILTER=$(IFS=,; echo "${FILTERS[*]}")
        FILTER_ARG=(--local-ranks-filter "$LOCAL_FILTER")
    else
        FILTER_ARG=()
    fi

    CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} ${CMD}"
fi

###############################################################################
# STEP 11: Display configuration (always)
###############################################################################
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    print_section "[DRY RUN] Direct Launch Configuration"
else
    print_section "Primus Direct Launch Configuration"
fi

PRINT_INFO_RANK0 "  Run Mode        : ${direct_config[run_mode]}"
PRINT_INFO_RANK0 "  Script Path     : ${direct_config[script]}"
PRINT_INFO_RANK0 "  Config File     : ${CONFIG_FILE:-<none>}"
PRINT_INFO_RANK0 "  Log File        : ${direct_config[log_file]}"
PRINT_INFO_RANK0 "  NUMA Binding    : ${direct_config[numa]}"
if [[ -n "${direct_config[patch]:-}" ]]; then
    PRINT_INFO_RANK0 "  Patch Scripts   : $(echo "${direct_config[patch]}" | tr '\n' ' ')"
else
    PRINT_INFO_RANK0 "  Patch Scripts   : <none>"
fi
PRINT_INFO_RANK0 "  Primus Args     : $*"
PRINT_INFO_RANK0 ""

if [[ ${#primus_env_kv[@]} -gt 0 ]]; then
    PRINT_INFO_RANK0 "  Environment Variables:"
    for kv in "${primus_env_kv[@]}"; do
        PRINT_INFO_RANK0 "    $kv"
    done
    PRINT_INFO_RANK0 ""
fi

if [[ "${direct_config[run_mode]}" == "torchrun" ]]; then
    PRINT_INFO_RANK0 "  Distributed Settings:"
    PRINT_INFO_RANK0 "    NNODES          : ${NNODES:-1}"
    PRINT_INFO_RANK0 "    NODE_RANK       : ${NODE_RANK:-0}"
    PRINT_INFO_RANK0 "    GPUS_PER_NODE   : ${GPUS_PER_NODE:-8}"
    PRINT_INFO_RANK0 "    MASTER_ADDR     : ${MASTER_ADDR:-localhost}"
    PRINT_INFO_RANK0 "    MASTER_PORT     : ${MASTER_PORT:-1234}"
    PRINT_INFO_RANK0 ""
fi

print_system_info

PRINT_INFO_RANK0 "  Full Command:"
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    PRINT_INFO_RANK0 "    Would Execute: $CMD"
else
    PRINT_INFO_RANK0 "    Executing: $CMD"
fi

# In dry-run mode, exit after displaying the command
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    print_section "End of Dry Run"
    LOG_INFO_RANK0 "[direct] Dry-run mode: command not executed"
    exit 0
fi

print_section ""

###############################################################################
# STEP 12: Execute command
###############################################################################
eval "$CMD"
exit_code=${PIPESTATUS[0]}

# Print result based on exit code
if [[ $exit_code -ge 128 ]]; then
    LOG_ERROR "[direct] torchrun crashed due to signal $((exit_code - 128))"
elif [[ $exit_code -ne 0 ]]; then
    LOG_ERROR "[direct] torchrun exited with code $exit_code"
else
    LOG_INFO "[direct] torchrun finished successfully (code 0)"
fi

exit "$exit_code"
