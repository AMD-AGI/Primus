#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/config_common.sh"

export NNODES=4
export DP_REPLICATE=4
export LOCAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=1024
export LR=0.00025
export WARMUP_STEPS=800
export GRADIENT_CHECKPOINTING_RATIO=0
export TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE:-}
if [[ -n "${TORCHINDUCTOR_CACHE_SEED:-}" && -z "$TORCH_COMPILE_MODE" ]]; then
    export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
fi
export FLUX_FP8_GEMM_BACKEND=selective_flydsl
export TORCHINDUCTOR_BENCHMARK_FUSION=1
export PRIMUS_FLUX_AITER_ATOMIC_FP32=0
export PRIMUS_FLUX_REUSE_FP8_INPUT=1

# Pollara GDR uses DMA-BUF registration; the legacy ibv_reg_mr path fails on
# these hosts even though the ABI-4 Ionic provider is installed.
export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-1}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}
export NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ:-1}
