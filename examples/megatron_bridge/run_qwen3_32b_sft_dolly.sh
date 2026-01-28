#!/bin/bash
# Quick start script for Qwen3-32B SFT with Databricks Dolly dataset
# 15K diverse instruction-response pairs

cd "$(dirname "$0")" || exit 1
bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_dolly.yaml "$@"
