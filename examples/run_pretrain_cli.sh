#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Primus Direct Mode Training Script
#
# Usage Examples:
#
# Scenario 1: Use default config (Llama3.1 8B BF16)
#   bash examples/run_pretrain_cli.sh
#
# Scenario 2: Use custom config
#   export EXP=my_experiments/custom_config.yaml
#   bash examples/run_pretrain_cli.sh
#
# Scenario 3: Pass extra arguments
#   bash examples/run_pretrain_cli.sh \
#     --train_iters 10 \
#     --micro_batch_size 4 \
#     --global_batch_size 128 \
#
###############################################################################

# shellcheck disable=SC2086,SC2048

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"}
bash $PRIMUS_PATH/runner/primus-cli direct -- train pretrain --config $EXP $*
