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
# Scenario 1: Use default config (Llama3.1 8B FP8)
#   bash examples/run_pretrain_cli.sh
#
# Scenario 2: Use custom config
#   export EXP=my_experiments/custom_config.yaml
#   bash examples/run_pretrain_cli.sh
#
# Scenario 3: Pass extra arguments
#   export EXP=examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
#   bash examples/run_pretrain_cli.sh \
#     --checkpoint-interval 500 \
#     --log-level DEBUG \
#     --enable-profiling
#
###############################################################################

# shellcheck disable=SC2086,SC2048

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

# Scenario 1: Use default config (Llama3.1 8B FP8)
EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml"}
bash $PRIMUS_PATH/runner/primus-cli direct -- train pretrain --config $EXP $*


# Scenario 4: Advanced CLI call (NUMA binding + custom logging)
#   PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
#   bash $PRIMUS_PATH/runner/primus-cli-direct.sh \
#     --numa \
#     --log_file /tmp/training.log \
#     -- train pretrain \
#     --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml \
#     --train_iters 10
