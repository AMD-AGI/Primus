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
    primus-cli direct [--env KEY=VALUE ...] [--single] [--script <file.py>] [-- primus-args]

Description:
    Launch Primus training, benchmarking, or preflight directly on the host (or inside a container).
    Distributed settings can be controlled by either exporting environment variables in advance,
    or by specifying them inline using --env KEY=VALUE.

Options:
    --single             Run with python3 instead of torchrun (single process only)
    --script <file.py>   Python script to execute (default: primus/cli/main.py)
    --env KEY=VALUE      Set environment variable before execution
    --log_file PATH      Save log to a specific file (default: logs/log_TIMESTAMP.txt)

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

Notes:
    - If --single is specified, Primus skips torchrun and uses python3 directly.
    - If --script is not specified, defaults to primus/cli/main.py.
    - Always separate Primus arguments from launcher options using '--'.
    - Environment variables can be mixed: 'export' takes precedence unless overridden by '--env'.
    - Multi-node jobs require MASTER_ADDR set to the master node's hostname/IP.

EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

run_mode="torchrun"
script_path="primus/cli/main.py"
primus_env_kv=()
primus_args=()
log_file=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --single)
            run_mode="single"
            shift
            ;;
        --script)
            script_path="$2"
            shift 2
            ;;
        --env)
            if [[ "$2" == *=* ]]; then
                export "${2%%=*}"="${2#*=}"
                primus_env_kv+=("${2}")
                shift 2
            else
                echo "[primus-entry][ERROR] --env requires KEY=VALUE"
                exit 2
            fi
            ;;
        --log_file)
            log_file="$2"
            shift 2
            ;;
        --log_file=*)
            log_file="${1#*=}"
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


# Source the environment setup script (centralizes all exports and helper functions).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/primus-env.sh"

for kv in "${primus_env_kv[@]}"; do
    export "${kv%%=*}"="${kv#*=}"
    LOG_INFO_RANK0 "[Primus Entrypoint] Exported env: ${kv%%=*}=${kv#*=}"
done


# pip install -qq -r requirements.txt

# Build launch arguments.
if [[ "$run_mode" == "single" ]]; then
    CMD="python3 $script_path $*"
    LOG_INFO "Launching single-process script: $CMD"
else
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

    # Step 4: Build the final command.
    CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} -- $script_path $* "
    LOG_INFO "Launching distributed training with command: $CMD 2>&1 | tee $log_file"
fi

eval "$CMD" 2>&1 | tee "$log_file"
exit_code=${PIPESTATUS[0]}

# Print log based on exit code
if [[ $exit_code -ge 128 ]]; then
    LOG_ERROR "torchrun crashed due to signal $((exit_code - 128))"
elif [[ $exit_code -ne 0 ]]; then
    LOG_ERROR "torchrun exited with code $exit_code"
else
    LOG_INFO "torchrun finished successfully (code 0)"
fi

exit "$exit_code"
