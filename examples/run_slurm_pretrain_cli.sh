#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2086,SC2048,SC2034

EXP=${EXP:-"examples/megatron/exp_pretrain.yaml"}

export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
echo "PRIMUS_PATH: $PRIMUS_PATH"

bash $PRIMUS_PATH/runner/primus-cli slurm -N $NNODES \
-- train pretrain --config $EXP $* 2>&1 | tee $LOG_FILE
