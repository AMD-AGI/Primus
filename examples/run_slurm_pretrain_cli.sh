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
export NODES_LIST=${NODES_LIST:-""}            # e.g. "node[01-04]" (optional)
export RESERVATION=${RESERVATION:-""}          # e.g. "my_resv" (optional)
export GPUS_PER_NODE=${GPUS_PER_NODE:-"8"}     # e.g. "8" (optional)
export CPUS_PER_TASK=${CPUS_PER_TASK:-"64"}    # e.g. "8" (optional)

SLURM_ARGS=("-N" "$NNODES")
[[ -n "$NODES_LIST" ]] && SLURM_ARGS+=("--nodelist" "$NODES_LIST")
[[ -n "$RESERVATION" ]] && SLURM_ARGS+=("--reservation" "$RESERVATION")
[[ -n "$GPUS_PER_NODE" ]] && SLURM_ARGS+=("--gpus-per-node" "$GPUS_PER_NODE")
[[ -n "$CPUS_PER_TASK" ]] && SLURM_ARGS+=("--cpus-per-task" "$CPUS_PER_TASK")


# Log configuration
export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

# NOTE: The --env entries below are passed into the container and will be visible
# to the Primus training process (and system hooks) inside the container.
bash "$PRIMUS_PATH/runner/primus-cli" slurm "${SLURM_ARGS[@]}" \
-- --image "${DOCKER_IMAGE:-rocm/primus:v26.1}" \
-- \
  --env "USING_AINIC=${USING_AINIC:-0}" \
  --env "PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}" \
  --env "REBUILD_PRIMUS_TURBO=${REBUILD_PRIMUS_TURBO:-0}" \
  --env "REBUILD_BNXT=${REBUILD_BNXT:-0}" \
-- train pretrain --config "$EXP" "$@" 2>&1 | tee "$LOG_FILE"
