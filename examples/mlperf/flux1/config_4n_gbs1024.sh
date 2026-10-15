#!/usr/bin/env bash

# Four-node handoff values are established before common defaults so callers
# may override the tunable entries while the fixed topology remains explicit.
export FLUX_A6W4_RECIPE=1
export NNODES=4
export DP_REPLICATE=1
export LOCAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$LOCAL_BATCH_SIZE}
export GLOBAL_BATCH_SIZE=1024
export LR=${LR:-0.00025}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
export FSDP_SDMA_MODULES=${FSDP_SDMA_MODULES:-single_blocks}
export TORCHINDUCTOR_CPP_WRAPPER=${TORCHINDUCTOR_CPP_WRAPPER:-1}
export FLUX_A6W4_DENSE_ACT=${FLUX_A6W4_DENSE_ACT:-0}
export FLUX_A6W4_FC1_GELU_DUAL=${FLUX_A6W4_FC1_GELU_DUAL:-1}
export FLUX_A6W4_GELU_PACK_DOUBLE=${FLUX_A6W4_GELU_PACK_DOUBLE:-1}
export FLUX_FWD_GELU_PACK=${FLUX_FWD_GELU_PACK:-1}
export FLUX_FPROP_SWITCH_STEP=${FLUX_FPROP_SWITCH_STEP-7325}
export FLUX_FPROP_SWITCH_PRECOMPILE=${FLUX_FPROP_SWITCH_PRECOMPILE:-1}

source "$(dirname -- "${BASH_SOURCE[0]}")/config_common.sh"

export GRADIENT_CHECKPOINTING_RATIO=0

# Pollara GDR uses DMA-BUF registration; the legacy ibv_reg_mr path fails on
# these hosts even though the ABI-4 Ionic provider is installed.
export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}
export NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ:-1}
