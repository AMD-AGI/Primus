#!/bin/bash
# Quick start script for Qwen3-32B SFT with Stanford Alpaca dataset
# 51K high-quality instruction-following samples

cd "$(dirname "$0")" || exit 1
bash run.sh examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_alpaca.yaml "$@"
