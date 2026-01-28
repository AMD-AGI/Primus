#!/bin/bash
# Quick start script for Qwen3-32B SFT post-training with mock dataset
# Avoids HuggingFace dataset compatibility issues

cd "$(dirname "$0")" || exit 1
bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain_mock.yaml "$@"
