#!/bin/bash

# Default config if no argument is provided
CONFIG_FILE=${1:-"./examples/megatron/configs/MI355X/llama3_8B-BF16-sft.yaml"}

# Extract the base name for the experiment
BASENAME=$(basename "$CONFIG_FILE" .yaml)

# Add timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export PRIMUS_EXP_NAME="${BASENAME}_${TIMESTAMP}"

# --- Quiet noisy C++ / NCCL logs that flood the console when training crashes ---
# Suppresses WARNING-level c10 messages such as the
# "symbolizing C++ stack trace for exception ..." line that PyTorch prints
# every time it captures a C++ exception. Real ERROR-level messages and Python
# tracebacks are still emitted.
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
# Skip addr2line during C++ symbolization. addr2line is the main reason the
# stack-trace path hangs / floods. Set to 0 if you ever need full file:line info.
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
# Lower NCCL/RCCL verbosity. Set NCCL_DEBUG=INFO only when chasing comm bugs.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "Starting training with config: $CONFIG_FILE"
echo "Experiment Name: $PRIMUS_EXP_NAME"

PRIMUS_TRAIN_RUNTIME=core ./primus-cli --debug direct -- train posttrain --config "$CONFIG_FILE"
