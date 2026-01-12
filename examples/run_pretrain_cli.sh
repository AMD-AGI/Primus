#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2086,SC2048

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}

# Scenario 1: Use default config (Llama3.1 8B BF16)
bash $PRIMUS_PATH/runner/primus-cli direct -- train pretrain --config $EXP $*

# Scenario 2: Pass extra arguments
# bash $PRIMUS_PATH/runner/primus-cli direct -- train pretrain --config $EXP \
#     --train_iters 10 \
#     --micro_batch_size 4 \
#     --global_batch_size 128 $* 

# Scenario 2: Pass extra arguments
# bash $PRIMUS_PATH/runner/primus-cli direct \
# --env "MASTER_PORT=12345" \
# -- train pretrain --config $EXP \
#     --train_iters 10 \
#     --micro_batch_size 4 \
#     --global_batch_size 128 $* 