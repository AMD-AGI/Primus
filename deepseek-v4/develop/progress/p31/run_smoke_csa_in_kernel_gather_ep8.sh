#!/bin/bash
# Plan-5 P31 — smoke gate for CSA in-kernel top-K gather/scatter.
#
# Runs the V4-Flash production-shape proxy for 10 iterations with the
# post-P30 knobs still ON. P31 is code-side: cr=4 CSA layers route
# through v4_csa_attention_from_pool and no longer materialise the
# wrapper-side [B, S, K_topk, head_dim] gathered tensor.
set -euo pipefail
set -x

cd /shared/amdgpu/home/wen_xie_qle/workspace/Primus-deepseek-v4

export TRAIN_ITERS=${TRAIN_ITERS:-10}
export PROFILE=False

export USE_V4_TRITON_ATTENTION=True
export USE_V4_TRITON_CSA_ATTENTION=True
export USE_V4_COMPILED_SINKHORN=True
export USE_TURBO_DEEPEP=True
export TURBO_USE_GROUPED_MLP=True
export USE_TURBO_ATTENTION=False

export PRIMUS_EXP_NAME=p31_smoke_csa_in_kernel_gather_ep8

./run_deepseek_v4_flash_proxy.sh
