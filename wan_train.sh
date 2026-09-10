#!/usr/bin/env bash
###############################################################################
# WAN 2.1 / 2.2 training launcher (Primus-Megatron).
#
# Usage:
#   bash wan_train.sh                                  # default EXP below
#   EXP=<path/to/config.yaml> bash wan_train.sh        # pick an arm
#   GPUS_PER_NODE=8 bash wan_train.sh                  # 8-GPU single node
#
# Expects PRIMUS_DIFFUSION_DATA_PATH (prepared Energon dataset) and HF_HOME
# (cache holding the VAE and UMT5 weights) to be set in the environment.
###############################################################################

set -euo pipefail

PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

export EXP="${EXP:-examples/megatron/configs/MI355X/diffusion/wan2.1_t2v_1.3b_pretrain_pusa_te_spec.yaml}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export PYTHONPATH="${PRIMUS_ROOT}:${PYTHONPATH:-}"

# Long video sequences fragment the allocator badly without this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Force AITER for the FP4 matmuls. The HipBLASLt/rocRoller FP4 path runs a
# per-shape CPU-bound solution search that stalls on WAN's long-sequence shapes.
# Harmless for non-FP4 runs, since only the FP4 dispatcher consults it.
export PRIMUS_TURBO_GEMM_BACKEND="${PRIMUS_TURBO_GEMM_BACKEND:-fp4:AITER}"

# run_pretrain.sh pins MIOPEN_FIND_MODE=2 (Fast), which is inert for LLMs but
# ruinous here: WAN's Conv3d patch embed is the only convolution in the model,
# and Fast mode resolves its backward to a naive im2col path -- a per-tile GEMM
# launch per patch -- instead of the fused Composable Kernel wgrad solver, which
# on its own dominates the step. Mode 5 (DynamicHybrid) is MIOpen's own default;
# it searches once per new conv shape, so only the first iteration pays.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-5}"

# TE pins its hipBLASLt workspace at 64 MiB, which is not enough for the
# 1536 -> 1536 wgrad GEMM once WAN's token count passes ~70k (micro_batch_size
# 7+): hipBLASLt picks split-K, the splits outgrow the workspace, and the call
# fails with "HIPBLASLT Error: 6" instead of falling back. 128 MiB clears every
# WAN shape. Read by primus/backends/megatron/patches/te_patches/
# hipblaslt_workspace_patches.py; unset it and TE's own default applies.
export PRIMUS_TE_GEMM_WORKSPACE_MIB="${PRIMUS_TE_GEMM_WORKSPACE_MIB:-128}"

# Set HF_HOME and PRIMUS_DIFFUSION_DATA_PATH in the environment to point at the
# HuggingFace cache holding the VAE / UMT5 weights and at the prepared Energon
# dataset; the configs read the latter with a placeholder default.

if [[ "${WAN_DEBUG:-0}" == "1" ]]; then
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
fi

mkdir -p "${PRIMUS_ROOT}/output"
LOG_FILE="${LOG_FILE:-${PRIMUS_ROOT}/output/wan_train_$(hostname).log}"

echo "EXP=${EXP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "LOG_FILE=${LOG_FILE}"

cd "${PRIMUS_ROOT}"
exec bash examples/run_pretrain.sh "$@" |& tee -a "${LOG_FILE}"
