#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Primus Direct Mode Training Script
#
# Usage Examples:
#
# 场景 1: 使用默认配置（Llama3.1 8B FP8）
#   bash examples/run_pretrain_cli.sh
#
# 场景 2: 使用自定义配置
#   export EXP=my_experiments/custom_config.yaml
#   bash examples/run_pretrain_cli.sh
#
# 场景 3: 传递额外参数
#   export EXP=examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
#   bash examples/run_pretrain_cli.sh \
#     --checkpoint-interval 500 \
#     --log-level DEBUG \
#     --enable-profiling
#
# 场景 4: 完整命令行调用（NUMA 绑定 + 自定义日志）
#   PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
#   bash $PRIMUS_PATH/runner/primus-cli-direct.sh \
#     --numa \
#     --log_file /tmp/training.log \
#     -- train pretrain \
#     --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml \
#     --checkpoint-interval 1000
#
###############################################################################

# shellcheck disable=SC2086,SC2048

LOG_ERROR() {
    echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*";
}

# Default configuration: Llama3.1 8B with FP8 training
EXP=${EXP:-"examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml"}

if [ -z "${EXP:-}" ]; then
    LOG_ERROR "EXP must be specified (e.g., examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml)." \
              "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# Ensure EXP file exists, otherwise exit with error
if [ ! -f "${EXP}" ]; then
    LOG_ERROR "The specified EXP file does not exist: ${EXP}" \
              "Primus will use the configuration in EXP to train the model."
    exit 1
fi

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
bash $PRIMUS_PATH/runner/primus-cli direct -- train pretrain --config $EXP $*
