#!/usr/bin/env bash

# Utilization profile: keep six GPUs busy and leave physical GPUs 6-7 free
# for intermittent kernel-optimization work on the same node.
# This is not an MLPerf submission profile.
export FLUX_A6W4_RECIPE=1
export NNODES=1
export DP_REPLICATE=1
export LOCAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=192
export LR=${LR:-0.00025}
export WARMUP_STEPS=${WARMUP_STEPS:-800}
export FSDP_SDMA_MODULES=${FSDP_SDMA_MODULES:-single_blocks}
export FLUX_FPROP_SWITCH_STEP=${FLUX_FPROP_SWITCH_STEP-}
export TORCHINDUCTOR_CPP_WRAPPER=${TORCHINDUCTOR_CPP_WRAPPER:-0}
export FLUX_A6W4_FC1_GELU_DUAL=${FLUX_A6W4_FC1_GELU_DUAL:-0}
export FLUX_A6W4_PACK_HARVEST=${FLUX_A6W4_PACK_HARVEST:-0}
export FLUX_A6W4_GELU_PACK_DOUBLE=${FLUX_A6W4_GELU_PACK_DOUBLE:-0}
export FLUX_A6W4_LNMOD_PACK=${FLUX_A6W4_LNMOD_PACK:-0}
export FLUX_A6W4_LEGS_ONE_LAUNCH=${FLUX_A6W4_LEGS_ONE_LAUNCH:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

# shellcheck source=examples/mlperf/flux1/config_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config_common.sh"
export GPUS_PER_NODE=6
export GLOBAL_BATCH_SIZE=192
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$LOCAL_BATCH_SIZE}
