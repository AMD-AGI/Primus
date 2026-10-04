#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# DeepSeek-V3 BF16 4-layer MoE proxy pretrain on ONE MI455X (gfx1250).
#
# Config: examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain-1gpu-proxy.yaml
# (its header lists the model shape and why each feature is disabled). This script
# supplies the part of that recipe that has to be environment: it must be in place
# before Python starts.
#
# Run inside a gfx1250 training container (the Primus checkout mounted, the image's
# torch / TE / Primus-Turbo installed):
#
#   bash examples/moe_package/run_deepseek_v3_proxy_1gpu_mi455x.sh [extra primus args]
#
# Extra arguments are appended to `train pretrain`, e.g. `--train_iters 10` or
# `--profile false --use_pytorch_profiler false`.
#
# Optional environment:
#   MASTER_PORT        default: random in [20000, 40000), so back-to-back runs never
#                      collide on a port still in TIME_WAIT
#   E2E_RUN_MARKER     default: <exp name>-<timestamp>; exported so the training
#                      processes carry it in their environment (lets a supervisor
#                      tell this run's GPU clients from any other)
#   PRIMUS_WORKSPACE, PRIMUS_EXP_NAME   output location (<workspace>/amd/<user>/<exp>)
#   TRAIN_LOG          launcher log (default: <workspace>/<exp>.log)
#   EXTRA_PYTHONPATH   prepended to PYTHONPATH, e.g. instrumentation hooks
#   PRIMUS_TURBO_ATTN_BACKEND   default triton
###############################################################################
set -euo pipefail

PRIMUS_ROOT=$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)
cd "${PRIMUS_ROOT}"

EXP=${EXP:-examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain-1gpu-proxy.yaml}

# ---------- world size 1 ----------
export NNODES=1 NODE_RANK=0 GPUS_PER_NODE=1
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
# Load runner/helpers/envs/MI455X.sh without probing the device (no SMI call). It
# derives the world-size-1 NCCL settings: P2P off, IB off, no inherited IB HCA list.
export PRIMUS_GPU_MODEL=MI455X
# The image already carries every dependency; never pip-install over it.
export PRIMUS_SKIP_PIP=1

# ---------- code paths that must not run on gfx1250 ----------
# Megatron's jit_fuser is torch.compile and decorates at import time (it also JIT-warms
# the fused bias/activation functions at init): inductor's Triton autotune has taken a
# gfx1250 card down, so torch.compile is disabled process-wide before any import.
export TORCH_COMPILE_DISABLE=1
# TE's Triton norms autotune num_warps up to 16 in process; num_warps=16 hangs gfx1250.
# Pin TE to its HIP norm kernels.
export NVTE_USE_RMSNORM_TRITON=0
export NVTE_USE_LAYERNORM_TRITON=0
# No Primus-Turbo backend autotuning inside the training process.
export PRIMUS_TURBO_AUTO_TUNE=0
# MLA core attention (qk 192 / v 128) through Turbo's Triton flash attention.
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-triton}

# ---------- all_reduce(AVG) hangs at world size 1 on this RCCL build ----------
# The sitecustomize in this directory rewrites AVG as SUM / world_size in the
# training worker (Megatron's MoE aux-loss tracker reduces with AVG).
export PYTHONPATH="${EXTRA_PYTHONPATH:+${EXTRA_PYTHONPATH}:}${PRIMUS_ROOT}/examples/deepseek-v4/rccl_avg_workaround${PYTHONPATH:+:${PYTHONPATH}}"

export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v3-BF16-pretrain-1gpu-proxy}
export E2E_RUN_MARKER=${E2E_RUN_MARKER:-${PRIMUS_EXP_NAME}-$(date +%Y%m%d_%H%M%S)}
WORKSPACE_DIR=${PRIMUS_WORKSPACE:-./output}
TRAIN_LOG=${TRAIN_LOG:-${WORKSPACE_DIR}/${PRIMUS_EXP_NAME}.log}
mkdir -p "$(dirname "${TRAIN_LOG}")"

echo "[dsv3-proxy-1gpu] exp=${PRIMUS_EXP_NAME} marker=${E2E_RUN_MARKER} port=${MASTER_PORT} log=${TRAIN_LOG}"
exec bash ./primus-cli direct --log_file "${TRAIN_LOG}" -- train pretrain --config "${EXP}" "$@"
