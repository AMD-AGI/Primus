#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

# Default configuration
EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}
export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}

export USING_AINIC=${USING_AINIC:-0}
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}
export REBUILD_PRIMUS_TURBO=${REBUILD_PRIMUS_TURBO:-0}
export REBUILD_BNXT=${REBUILD_BNXT:-0}


# Log configuration
export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

# Scenario 1: Single node test with default config
bash "$PRIMUS_PATH/runner/primus-cli" slurm srun -N "$NNODES" \
-- container \
  --image "${DOCKER_IMAGE:-rocm/primus:v25.10}" \
-- train pretrain --config "$EXP" "$@" 2>&1 | tee "$LOG_FILE"
