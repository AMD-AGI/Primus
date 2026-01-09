#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Primus Slurm Mode Training Script
#
# Usage Examples:
#
# Scenario 1: Single node test with default config
#   bash $PRIMUS_PATH/runner/primus-cli slurm srun \
#     -N 1 --nodelist "node01" \
#   -- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
#
# Scenario 2: Multi-node training (4 nodes)
#   bash $PRIMUS_PATH/runner/primus-cli slurm srun \
#     -N 4 --nodelist "node[01-04]" \
#   -- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
#
# Scenario 3: Pass extra training arguments
#   bash $PRIMUS_PATH/runner/primus-cli slurm srun \
#     -N 2 --nodelist "node[01-02]" \
#   -- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml \
#     --train_iters 10 \
#     --micro_batch_size 4 \
#     --global_batch_size 128
#
# Scenario 4: Add environment variables for debugging
#   bash $PRIMUS_PATH/runner/primus-cli slurm srun \
#     -N 4 --nodelist "node[01-04]" \
#   -- \
#     --env NCCL_DEBUG=INFO \
#     --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
#   -- train pretrain --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
#
###############################################################################

# shellcheck disable=SC2086,SC2048,SC2034

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

# Default configuration
EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}
export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}
export NODES_LIST=${NODES_LIST:-"node[02,03,10,14,15,34,38]"}

# Log configuration
export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

echo "PRIMUS_PATH: $PRIMUS_PATH"
echo "EXP: $EXP"
echo "NNODES: $NNODES"
echo "NODES_LIST: $NODES_LIST"

bash $PRIMUS_PATH/runner/primus-cli slurm srun \
  -N $NNODES \
  --nodelist "$NODES_LIST" \
-- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE
