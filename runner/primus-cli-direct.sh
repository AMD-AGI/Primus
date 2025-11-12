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

run_mode="torchrun"
script_path="primus/cli/main.py"
primus_env_kv=()
primus_args=()
patch_scripts=()
log_file=""
enable_numa="auto"  # auto / 1 / 0
CONFIG_FILE=""
DEBUG_MODE=0
DRY_RUN_MODE=0

# Track which parameters were explicitly set via CLI
CLI_RUN_MODE_SET=0
CLI_SCRIPT_PATH_SET=0
CLI_NUMA_SET=0
CLI_LOG_FILE_SET=0
CLI_DEBUG_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=1
            CLI_DEBUG_SET=1
            shift
            ;;
        --dry-run)
            DRY_RUN_MODE=1
            shift
            ;;
        --single)
            run_mode="single"
            CLI_RUN_MODE_SET=1
            shift
            ;;
        --numa)
            enable_numa="1"
            CLI_NUMA_SET=1
            shift
            ;;
        --no-numa)
            enable_numa="0"
            CLI_NUMA_SET=1
            shift
            ;;
        --script)
            script_path="$2"
            CLI_SCRIPT_PATH_SET=1
            shift 2
            ;;
        --env)
            if [[ "$2" == *=* ]]; then
                export "${2%%=*}"="${2#*=}"
                primus_env_kv+=("${2}")
                shift 2
            else
                LOG_ERROR "[direct] --env requires KEY=VALUE"
                exit 2
            fi
            ;;
        --patch)
            patch_scripts+=("$2")
            shift 2
            ;;
        --patch=*)
            patch_scripts+=("${1#*=}")
            shift
            ;;
        --log_file)
            log_file="$2"
            CLI_LOG_FILE_SET=1
            shift 2
            ;;
        --log_file=*)
            log_file="${1#*=}"
            CLI_LOG_FILE_SET=1
            shift
            ;;
        --)
            shift
            primus_args+=("$@")
            break
            ;;
        *)
            primus_args+=("$1")
            shift
            ;;
    esac
done
set -- "${primus_args[@]}"

if [[ "$*" == *"--help"* ]] || [[ "$*" == *"-h"* ]]; then
    exec python3 -m primus.cli.main "$@"
fi

# Step 0: Setup log directory and generate log file path
if [[ -z "$log_file" ]]; then
    log_file="logs/log_$(date +%Y%m%d_%H%M%S).txt"
fi
mkdir -p "$(dirname "$log_file")"


# Step 1: Load configuration and apply mode-specific settings
# Load configuration (specified or defaults)
load_config_auto "$CONFIG_FILE" "entrypoint" || {
    LOG_ERROR "[direct] Configuration loading failed"
    exit 1
}

# Extract all direct.* config parameters
declare -A direct_config
extract_config_section "direct" direct_config || true

# Apply direct config values if not set via CLI
[[ "$CLI_DEBUG_SET" == "0" && ("${direct_config[debug]:-false}" == "true" || "${direct_config[debug]:-false}" == "1") ]] && DEBUG_MODE=1

# Enable debug mode if set
if [[ "$DEBUG_MODE" == "1" ]]; then
    export PRIMUS_LOG_LEVEL="DEBUG"
    LOG_INFO "[direct] Debug mode enabled (PRIMUS_LOG_LEVEL=DEBUG)"
    print_config_section "direct" direct_config
fi

# Apply config values for non-overridden parameters (CLI args take precedence)
# Priority: CLI args > Config file > Default values
[[ "$CLI_RUN_MODE_SET" == "0" && "${direct_config[run_mode]}" == "single" ]] && run_mode="single"
[[ "$CLI_SCRIPT_PATH_SET" == "0" && -n "${direct_config[script_path]}" ]] && script_path="${direct_config[script_path]}"
[[ "$CLI_NUMA_SET" == "0" && -n "${direct_config[numa]}" ]] && enable_numa="${direct_config[numa]}"
[[ "$CLI_LOG_FILE_SET" == "0" && -n "${direct_config[log_file]}" ]] && log_file="${direct_config[log_file]}"

# Handle patch scripts array (patches.0, patches.1, ...) if not set via CLI
if [[ ${#patch_scripts[@]} -eq 0 ]]; then
    patch_idx=0
    while [[ -n "${direct_config[patches.$patch_idx]:-}" ]]; do
        patch_scripts+=("${direct_config[patches.$patch_idx]}")
        ((patch_idx++))
    done
fi

# Handle env array (env.0, env.1, ...) - always append from config (config + CLI)
env_idx=0
while [[ -n "${direct_config[env.$env_idx]:-}" ]]; do
    env_value="${direct_config[env.$env_idx]}"
    if [[ "$env_value" == *=* ]]; then
        export "${env_value%%=*}"="${env_value#*=}"
        primus_env_kv+=("$env_value")
    fi
    ((env_idx++))
done


# Source primus-env.sh (it will set its own SCRIPT_DIR, which is fine)
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/envs/primus-env.sh"

# Source helper modules
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/execute_hooks.sh"
# shellcheck disable=SC1091
source "${RUNNER_DIR}/helpers/execute_patches.sh"

# Export environment variables passed via --env
for kv in "${primus_env_kv[@]}"; do
    export "${kv%%=*}"="${kv#*=}"
    LOG_INFO_RANK0 "[direct] Exported env: ${kv%%=*}=${kv#*=}"
done

# Step 2: Auto-run hooks based on $1 $2 (e.g., train pretrain → hooks/train/pretrain/*) (skip in dry-run mode)
if [[ $# -ge 2 && "$DRY_RUN_MODE" == "0" ]]; then
    if ! execute_hooks "$1" "$2" "${primus_args[@]}"; then
        LOG_ERROR "[direct] Hooks execution failed"
        exit 1
    fi
elif [[ "$DRY_RUN_MODE" == "0" ]]; then
    LOG_INFO_RANK0 "[direct] No hook target detected (missing \$1 \$2)."
fi


# Step 3: Run patch scripts if specified (skip in dry-run mode)
if [[ ${#patch_scripts[@]} -gt 0 && "$DRY_RUN_MODE" == "0" ]]; then
    if ! execute_patches "${patch_scripts[@]}"; then
        LOG_ERROR "[direct] Patch execution failed"
        exit 1
    fi
fi

# Install dependencies (skip in dry-run mode)
if [[ "$DRY_RUN_MODE" == "0" ]]; then
    pip install -qq -r requirements.txt
    if [[ "$enable_numa" == "1" ]]; then
        apt-get install numactl -y > /dev/null 2>&1
    fi
fi

# Build launch arguments.
if [[ "$run_mode" == "single" ]]; then
    if [[ "$enable_numa" == "1" ]]; then
        CMD="bash ${RUNNER_DIR}/helpers/numa_bind.sh python3 $script_path $*"
        LOG_INFO "[direct] Launching single-process script with NUMA binding:"
    else
        CMD="python3 $script_path $*"
        LOG_INFO "[direct] Launching single-process script:"
    fi
else
    # NOTE: These variables use environment variables from config file (via primus-cli --config)
    # Priority: Environment (from config/export) > Script defaults
    DISTRIBUTED_ARGS=(
        --nproc_per_node "${GPUS_PER_NODE:-8}"
        --nnodes "${NNODES:-1}"
        --node_rank "${NODE_RANK:-0}"
        --master_addr "${MASTER_ADDR:-localhost}"
        --master_port "${MASTER_PORT:-1234}"
    )

    # Build local rank filter argument.
    # Only local rank 0 on first node and last local rank on last node are filtered for special logging.
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

    NUMA_LAUNCHER_ARGS=()
    if [[ "$enable_numa" == "1" ]]; then
        NUMA_LAUNCHER_ARGS=(--no-python "${RUNNER_DIR}/helpers/numa_bind.sh" python3)
        LOG_INFO_RANK0 "[direct] NUMA binding: ENABLED (forced by CLI)"
    elif [[ "$enable_numa" == "0" ]]; then
        LOG_INFO_RANK0 "[direct] NUMA binding: DISABLED (forced by CLI)"
    else
        LOG_INFO_RANK0 "[direct] NUMA binding: AUTO (default OFF)"
    fi

    if [[ "$DEBUG_MODE" == "1" ]]; then
        print_section "Primus Entrypoint Summary"
        LOG_INFO_RANK0 "  Run Mode     : $run_mode"
        LOG_INFO_RANK0 "  Script Path  : $script_path"
        LOG_INFO_RANK0 "  Config File  : ${CONFIG_FILE:-<none>}"
        LOG_INFO_RANK0 "  Log File     : $log_file"
        LOG_INFO_RANK0 "  NUMA Binding : $enable_numa"
        LOG_INFO_RANK0 "  Patches      : ${patch_scripts[*]:-<none>}"
        LOG_INFO_RANK0 "  Env Vars     : ${primus_env_kv[*]:-<none>}"
    fi

    # Step 4: Build the final command.
    CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} ${NUMA_LAUNCHER_ARGS[*]}  $script_path $* "
    LOG_INFO "[direct] Launching distributed training with command: $CMD 2>&1 | tee $log_file"
fi

# Dry-run mode: display configuration and exit
if [[ "$DRY_RUN_MODE" == "1" ]]; then
    print_section "[DRY RUN] Direct Launch Configuration"
    PRINT_INFO_RANK0 "  Run Mode        : $run_mode"
    PRINT_INFO_RANK0 "  Script Path     : $script_path"
    PRINT_INFO_RANK0 "  Config File     : ${CONFIG_FILE:-<none>}"
    PRINT_INFO_RANK0 "  Log File        : $log_file"
    PRINT_INFO_RANK0 "  NUMA Binding    : $enable_numa"
    PRINT_INFO_RANK0 "  Patch Scripts   : ${patch_scripts[*]:-<none>}"
    PRINT_INFO_RANK0 "  Primus Args     : $*"
    PRINT_INFO_RANK0 ""
    if [[ ${#primus_env_kv[@]} -gt 0 ]]; then
        PRINT_INFO_RANK0 "  Environment Variables:"
        for kv in "${primus_env_kv[@]}"; do
            PRINT_INFO_RANK0 "    $kv"
        done
        PRINT_INFO_RANK0 ""
    fi
    if [[ "$run_mode" == "torchrun" ]]; then
        PRINT_INFO_RANK0 "  Distributed Settings:"
        PRINT_INFO_RANK0 "    NNODES          : ${NNODES:-1}"
        PRINT_INFO_RANK0 "    NODE_RANK       : ${NODE_RANK:-0}"
        PRINT_INFO_RANK0 "    GPUS_PER_NODE   : ${GPUS_PER_NODE:-8}"
        PRINT_INFO_RANK0 "    MASTER_ADDR     : ${MASTER_ADDR:-localhost}"
        PRINT_INFO_RANK0 "    MASTER_PORT     : ${MASTER_PORT:-1234}"
        PRINT_INFO_RANK0 ""
    fi
    PRINT_INFO_RANK0 "  Full Command:"
    PRINT_INFO_RANK0 "    $CMD 2>&1 | tee $log_file"
    print_section "End of Dry Run"
    exit 0
fi

eval "$CMD" 2>&1 | tee "$log_file"
exit_code=${PIPESTATUS[0]}

# Print log based on exit code
if [[ $exit_code -ge 128 ]]; then
    LOG_ERROR "[direct] torchrun crashed due to signal $((exit_code - 128))"
elif [[ $exit_code -ne 0 ]]; then
    LOG_ERROR "[direct] torchrun exited with code $exit_code"
else
    LOG_INFO "[direct] torchrun finished successfully (code 0)"
fi

exit "$exit_code"
