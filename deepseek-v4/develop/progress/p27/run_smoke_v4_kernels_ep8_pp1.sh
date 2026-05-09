#!/bin/bash
# Plan-4 P27 G30 — DeepSeek-V4 smoke gate, EP=8, both V4 Triton kernels **on**.
#
# Runs a 10-iter smoke at TP=1 PP=1 EP=8 with:
#   * USE_V4_TRITON_ATTENTION=True       (cr ∈ {0, 128} layers route
#                                         through the in-tree Primus Triton
#                                         kernel from P25)
#   * USE_V4_TRITON_CSA_ATTENTION=True   (cr == 4 layers route through
#                                         the in-tree Primus CSA Triton
#                                         kernel from P26)
#   * USE_TURBO_ATTENTION=False          (Turbo would take precedence over
#                                         the dense / HCA Triton kernel;
#                                         keep it off so the Triton kernel
#                                         actually runs)
#   * USE_TURBO_DEEPEP=True              (per plan-4 03-test-strategy
#                                         G30: validates the V4 Triton
#                                         kernels coexist cleanly with the
#                                         Turbo DeepEP MoE dispatcher,
#                                         which is the V4-Flash release
#                                         configuration the smoke is
#                                         supposed to reproduce)
#
# Pass criteria (G30):
#   1. The model boots — every layer emits its [V4-attn] kernel-choice
#      log line under deepseek_v4_attention.py:_log_kernel_choice.
#      cr=0 / cr=128 layers must say "v4_attention (Triton, ...)" and
#      cr=4 layers must say "v4_csa_attention (Triton)".
#   2. 10 iter losses are stable, no NaN / Inf in the log.
#   3. No banned warnings: the deepseek_v4_attention "build_module
#      failed" warning (retired in P21) and the c10d::allreduce_
#      warning (retired in P19) MUST NOT reappear.
#
# Smoke log lands at:
#   deepseek-v4/develop/progress/p27/log_smoke_v4_kernels_ep8_pp1.txt
# (gitignored — see .gitignore in this folder; smoke logs MUST NOT be
# uploaded to GitHub per the user's plan-3 directive.)
set -euo pipefail
set -x

cd /shared/amdgpu/home/wen_xie_qle/workspace/Primus-deepseek-v4

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"
export NNODES=${PET_NNODES:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1}"
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}

# TP=1 PP=1 EP=8 — see plan-4 02-phase-details Phase 27 task 4.
export MBS=1
export GBS=16
export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=8

export PRIMUS_TOTAL_LAYERS=8
export PRIMUS_SEQ_LENGTH=128
export PRIMUS_MAX_POSITION_EMBEDDINGS=128
export PRIMUS_NUM_EXPERTS=8
export PRIMUS_MOE_TOPK=2
export PRIMUS_MOE_FFN_HIDDEN_SIZE=512
export PRIMUS_INDEX_TOPK=8
# Mixed compress_ratios so all three V4 kernel paths fire:
#   cr=0   -> v4_attention (Triton, dense path)
#   cr=128 -> v4_attention (Triton, HCA path)  (none in this list, but
#             the smoke covers cr={0, 4} which is the two of three that
#             matter for the Triton kernels in the V4-Flash YAML default)
#   cr=4   -> v4_csa_attention (Triton)
# The default YAML uses [0,0,4,4,4,4,4,0]; we keep that so the smoke
# matches V4-Flash production.
export PRIMUS_COMPRESS_RATIOS="[0,0,4,4,4,4,4,0]"
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=False
export PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=True
export PROFILE=False

# Plan-4 P27 G30: enable BOTH V4 Triton kernels.
export USE_V4_TRITON_ATTENTION=True
export USE_V4_TRITON_CSA_ATTENTION=True

# Turbo attention OFF so the dense / HCA Triton kernel actually fires
# (Turbo > V4 Triton precedence in DeepseekV4Attention.forward).
export USE_TURBO_ATTENTION=False
# Turbo DeepEP ON — per plan-4 03-test-strategy G30 the smoke runs the
# full V4-Flash release configuration (DeepEP routes the MoE all-to-all).
export USE_TURBO_DEEPEP=True
export TURBO_USE_GROUPED_MLP=False
export LEGACY_GG=False
# enable_primus_turbo gates the before_train patches; auto-flipped to
# True by run_deepseek_v4.sh when USE_TURBO_DEEPEP=True.
export ENABLE_PRIMUS_TURBO=True

export PRECISION_TYPE=BF16
export FP8=null
export FP8_RECIPE=null

export EXP=examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml
PWD_DIR="$(pwd)"
export BACKEND_PATH="${PWD_DIR}/third_party/Megatron-LM"
export PRIMUS_TEAM=amd
TODAY="$(date +%Y%m%d)"
export PRIMUS_USER="tas-mi355x-${TODAY}"
export PRIMUS_EXP_NAME=p27_smoke_v4_kernels_ep8_pp1

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

# Tee a copy of the run output into the (gitignored) progress folder so
# the smoke log lives next to the script for easy review without
# polluting the PR diff. The output/<team>/<user>/<exp>/log_node_0.txt
# tee already happens inside run_deepseek_v4.sh; we mirror the
# stdout/stderr here so debugging the smoke does not require chasing
# the output/ tree.
SMOKE_LOG="deepseek-v4/develop/progress/p27/log_smoke_v4_kernels_ep8_pp1.txt"
mkdir -p "$(dirname "$SMOKE_LOG")"

bash run_deepseek_v4.sh 2>&1 | tee "$SMOKE_LOG"
