#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2086,SC2048,SC2034

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

# Default configuration
EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}
export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}

# Log configuration
export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

# Scenario 1: Single node test with default config
bash $PRIMUS_PATH/runner/primus-cli slurm srun -N $NNODES --nodelist "node[02,03,10,14,15,34,38]" \
-- --env "./runner/helpers/envs/enable_ainic.sh" \
-- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE

# Scenario 2: Pass extra arguments
# bash $PRIMUS_PATH/runner/primus-cli slurm srun -N $NNODES \
# -- train pretrain \
#   --config $EXP \
#   --micro_batch_size 4 \
#   --global_batch_size 128 \
#   --train_iters 10 \
#   $* 2>&1 | tee $LOG_FILE

# Scenario 3: Multi-node training (2 nodes)
#   bash $PRIMUS_PATH/runner/primus-cli slurm -N 2 \
#   -- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE

# Scenario 3: Use custom Docker image
#   bash $PRIMUS_PATH/runner/primus-cli slurm -N 2 --nodelist "node[01-02]" \
#   -- --image docker.io/rocm/primus:v25.10 --clean \
#   -- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE

# Scenario 4: Add environment variables
#   bash $PRIMUS_PATH/runner/primus-cli slurm -N 4 --nodelist "node[01-04]" \
#   -- container \
#     --image docker.io/rocm/primus:v25.10 \
#   -- \
#     --env NCCL_DEBUG=INFO \
#     --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
#   -- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE
