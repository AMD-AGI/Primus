#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2034
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

# Default configuration
EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}
export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}

# Slurm configuration (common options)
# Set any of these via environment variables to customize the job:
export NODES_LIST=${NODES_LIST:-""}          # e.g. "node[01-04]" (optional)
export PARTITION=${PARTITION:-""}            # e.g. "AIG_Model" (optional)
export RESERVATION=${RESERVATION:-""}        # e.g. "my_resv" (optional)
export CONSTRAINT=${CONSTRAINT:-""}          # e.g. "mi300x" (optional)
export ACCOUNT=${ACCOUNT:-""}                # e.g. "my_account" (optional)
export QOS=${QOS:-""}                        # e.g. "normal" (optional)
export GPUS_PER_NODE=${GPUS_PER_NODE:-""}    # e.g. "8" (optional)
export NTASKS_PER_NODE=${NTASKS_PER_NODE:-""} # e.g. "8" (optional)
export CPUS_PER_TASK=${CPUS_PER_TASK:-""}    # e.g. "8" (optional)
export EXCLUSIVE=${EXCLUSIVE:-0}             # 1 to add --exclusive

SLURM_ARGS=("-N" "$NNODES")
[[ -n "$NODES_LIST" ]] && SLURM_ARGS+=("--nodelist" "$NODES_LIST")
[[ -n "$PARTITION" ]] && SLURM_ARGS+=("-p" "$PARTITION")
[[ -n "$RESERVATION" ]] && SLURM_ARGS+=("--reservation" "$RESERVATION")
[[ -n "$CONSTRAINT" ]] && SLURM_ARGS+=("--constraint" "$CONSTRAINT")
[[ -n "$ACCOUNT" ]] && SLURM_ARGS+=("--account" "$ACCOUNT")
[[ -n "$QOS" ]] && SLURM_ARGS+=("--qos" "$QOS")
[[ -n "$GPUS_PER_NODE" ]] && SLURM_ARGS+=("--gpus-per-node" "$GPUS_PER_NODE")
[[ -n "$NTASKS_PER_NODE" ]] && SLURM_ARGS+=("--ntasks-per-node" "$NTASKS_PER_NODE")
[[ -n "$CPUS_PER_TASK" ]] && SLURM_ARGS+=("--cpus-per-task" "$CPUS_PER_TASK")
[[ "$EXCLUSIVE" == "1" ]] && SLURM_ARGS+=("--exclusive")


# Log configuration
export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

# Scenario 1: Single node test with default config
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun "${SLURM_ARGS[@]}" \
-- container \
  --image "${DOCKER_IMAGE:-rocm/primus:v25.10}" \
-- \
  --env "USING_AINIC=${USING_AINIC:-0}" \
  --env "PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}" \
  --env "REBUILD_PRIMUS_TURBO=${REBUILD_PRIMUS_TURBO:-0}" \
  --env "REBUILD_BNXT=${REBUILD_BNXT:-0}" \
-- train pretrain --config "$EXP" "$@" 2>&1 | tee "$LOG_FILE"
