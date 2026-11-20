#!/usr/bin/env bash

export DGXSYSTEM=MI355X_4x8x32
export NNODES=4
export DGXNNODES=${NNODES}
export MLLOG_SUBMISSION_PLATFORM=MI355X
export LOCAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$LOCAL_BATCH_SIZE}
export GLOBAL_BATCH_SIZE=1024
export LR=${LR:-0.00025}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
export FLUX_FPROP_SWITCH_STEP=${FLUX_FPROP_SWITCH_STEP-7325}  # 7.5M samples
# HIP IPC cannot gather across hosts; use NCCL and keep low-precision gathers.
export FSDP_ALL_GATHER_BACKEND=${FSDP_ALL_GATHER_BACKEND:-default}
export FLUX_FP4_ALL_GATHER=${FLUX_FP4_ALL_GATHER:-1}

# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/config_common.sh"
