#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Run the DeepSeek-V4 Flash proxy (1 GPU, EP=1, 32 experts) with the MI455X block-FP8 preset.
#
# usage: bash run_opt3_1gpu.sh [eager|graph] [iters]
#
# Run inside the Primus container. Environment:
#   PRIMUS_ROOT        Primus checkout (default /workspace/Primus)
#   PRIMUS_TURBO_ROOT  Primus-Turbo checkout put ahead of the image copy (default /workspace/Primus-Turbo)
#   TRITON_MAIN        optional Triton main python dir (Triton >= 3.9, enables the Gluon kernels); without it the
#                      image Triton is used and the non-Gluon fallbacks are selected
#   OPT3_EXTRA_ARGS    more comma-separated trainer args (e.g. --save,<dir>,--save_interval,10)
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-graph}; ITERS=${2:-20}
PRIMUS_ROOT=${PRIMUS_ROOT:-/workspace/Primus}
PRIMUS_TURBO_ROOT=${PRIMUS_TURBO_ROOT:-/workspace/Primus-Turbo}
# shellcheck disable=SC1091
source "$HERE/opt3_best.env"
export PYTHONPATH=${TRITON_MAIN:+$TRITON_MAIN:}$PRIMUS_TURBO_ROOT${PYTHONPATH:+:$PYTHONPATH}
# MegaMoE fp8_block experts; no dense attention mask built on the host (V4 never reads it); Turbo grouped GEMM
# without padding, so the MoE permute size is static and the dispatcher needs no per-layer device-to-host sync.
EXTRA="--turbo_mega_moe_precision,fp8_block,--create_attention_mask_in_dataloader,False"
EXTRA="$EXTRA,--turbo_grouped_gemm_without_padding,True,--moe_router_padding_for_quantization,False"
EXTRA="$EXTRA${OPT3_EXTRA_ARGS:+,$OPT3_EXTRA_ARGS}"
[ "$MODE" = graph ] && EXTRA="--cuda_graph_impl,full_iteration,--check_for_nan_in_loss_and_grad,False,$EXTRA"
export PRIMUS_EXTRA_TRAIN_ARGS=$EXTRA
export PRIMUS_LAUNCHER=direct GPUS_PER_NODE=1 NNODES=1 PRIMUS_TP=1 PRIMUS_PP=1 PRIMUS_EP=1 MBS=1 GBS=1
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-32} USE_TURBO_DEEPEP=False ENABLE_PRIMUS_TURBO=True
export PRECISION_TYPE=BF16 FP8_RECIPE=tensorwise TRAIN_ITERS=$ITERS PRIMUS_GPU_MODEL=MI455X HSA_COREDUMP_PATTERN=/dev/null PYTORCH_ROCM_ARCH=gfx1250
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261109}
cd "$PRIMUS_ROOT"
exec bash examples/megatron/models/deepseek-v4/run_deepseek_v4_flash_proxy.sh
