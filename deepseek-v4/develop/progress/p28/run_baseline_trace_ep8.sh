#!/bin/bash
# Plan-5 P28 — DeepSeek-V4 Flash perf-baseline torch profile trace, EP=8.
#
# Captures iter 6 -> iter 7 (1 active step) under the V4-Flash production-
# shape proxy: 8 layers, full V4-Flash widths (hidden=4096, H=64, head_dim=
# 512, num_experts=256, moe_router_topk=6, moe_ffn=2048, index_topk=512),
# compress_ratios=[0,0,4,128,4,128,4,0] (every layer kind exercised), all
# four plan-5 perf knobs on:
#   * USE_V4_TRITON_ATTENTION=True       (cr ∈ {0, 128} -> Primus Triton)
#   * USE_V4_TRITON_CSA_ATTENTION=True   (cr == 4       -> Primus Triton CSA)
#   * USE_TURBO_DEEPEP=True              (PrimusTurboDeepEPTokenDispatcher)
#   * TURBO_USE_GROUPED_MLP=True         (Turbo grouped-GEMM expert path)
#
# This is the BASELINE for plan-5: every subsequent perf phase reports
# its delta against this trace.  The bottleneck-analysis report at
# `deepseek-v4/develop/profile/profile-baseline-ep8-<YYYYMMDD>.{md,html}`
# consumes the trace JSON emitted by this script.
#
# Calibrated seq length is set via the PRIMUS_SEQ_LENGTH env var.  The
# default below is 4096 — confirmed to fit on a single MI355X node at
# EP=8 by the P28 calibration probe (peak rocm HBM 195 GiB / 287 GiB ≈
# 68 %, no OOM, 5/5 iters clean at 77.5 TFLOP/s/GPU steady).  If a
# future config change pushes HBM over budget, fall back to 2048 / 1024
# / 512 via PRIMUS_SEQ_LENGTH and document the chosen value in the
# report.
#
# Trace lands at
#   output/<team>/<user>/p28_profile_baseline_pp1_ep8_seq<S>/tensorboard/
#       primus-megatron-exp[...]-rank[0].<id>.pt.trace.json
# Raw trace JSON is gitignored under deepseek-v4/develop/progress/p28/
# (the .gitignore in this dir matches the plan-3 P23 / plan-4 P25 pattern).
#
# Self-contained (mirrors deepseek-v4/develop/progress/p25/
# run_profile_v4_triton_attention_ep8.sh) because run_deepseek_v4.sh
# hard-codes --disable_tensorboard True; the torch profiler writes its
# trace JSON next to TB events.out.tfevents.*, so we need to flip that.
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

# ---------- V4-Flash production widths (8-layer proxy slice) ----------------
export MBS=${MBS:-1}
export GBS=${GBS:-8}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}

export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-8}
# Calibrated baseline seq.  Default 4096 — confirmed fitting (HBM
# peak ≈ 68 %).  Pass PRIMUS_SEQ_LENGTH=2048 / 1024 / 512 if a future
# config change OOMs at 4096.
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-256}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-2048}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-512}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-"[0,0,4,128,4,128,4,0]"}
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=${PRIMUS_MOE_ENABLE_EXPERT_BIAS:-False}
export PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=True
export PROFILE=True

# ---------- Plan-4 P25 / P26: in-tree Primus Triton kernels (BOTH ON) -------
export USE_V4_TRITON_ATTENTION=True
export USE_V4_TRITON_CSA_ATTENTION=True

# Turbo attention OFF — would take precedence over the V4 Triton dense
# path in DeepseekV4Attention.forward (plan-4 P27 dispatch precedence).
export USE_TURBO_ATTENTION=False

# ---------- Plan-3 P23 + plan-5 perf path: Turbo DeepEP + grouped GEMM ON ---
export USE_TURBO_DEEPEP=True
export TURBO_USE_GROUPED_MLP=True
export LEGACY_GG=False
# enable_primus_turbo gates the before_train patches that re-bind the spec
# provider — required when ANY of {USE_TURBO_DEEPEP, USE_TURBO_ATTENTION,
# TURBO_USE_GROUPED_MLP} is True.
export ENABLE_PRIMUS_TURBO=True

# Best-practice DeepEP knobs for EP=8 (mirrors the gated block in
# run_deepseek_v4.sh; this profile script bypasses that wrapper because
# we need --disable_tensorboard False for the torch profiler to write
# its trace next to events.out.tfevents.*).
export TURBO_DEEPEP_NUM_CU=${TURBO_DEEPEP_NUM_CU:-80}
export TURBO_DEEPEP_USE_COMM_STREAM=${TURBO_DEEPEP_USE_COMM_STREAM:-False}
export MOE_ROUTER_DTYPE=${MOE_ROUTER_DTYPE:-fp32}
export MOE_SHARED_EXPERT_OVERLAP=${MOE_SHARED_EXPERT_OVERLAP:-False}

export PRECISION_TYPE=BF16
export FP8=null
export FP8_RECIPE=null

# ---------- Bookkeeping -----------------------------------------------------
export EXP=examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml
PWD_DIR="$(pwd)"
export BACKEND_PATH="${PWD_DIR}/third_party/Megatron-LM"
export PRIMUS_TEAM=amd
TODAY="$(date +%Y%m%d)"
export PRIMUS_USER="tas-mi355x-${TODAY}"
export PRIMUS_EXP_NAME=p28_profile_baseline_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

./primus-cli direct \
  -- train pretrain --config "$EXP" \
  --backend_path "$BACKEND_PATH" \
  --num_layers "$PRIMUS_TOTAL_LAYERS" \
  --train_iters "$TRAIN_ITERS" \
  --lr_warmup_iters 0 \
  --lr_decay_iters "$TRAIN_ITERS" \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --seq_length "$PRIMUS_SEQ_LENGTH" \
  --max_position_embeddings "$PRIMUS_MAX_POSITION_EMBEDDINGS" \
  --rope_type rope \
  --tensor_model_parallel_size "$PRIMUS_TP" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  --num_experts "$PRIMUS_NUM_EXPERTS" \
  --moe_router_topk "$PRIMUS_MOE_TOPK" \
  --moe_router_enable_expert_bias "$PRIMUS_MOE_ENABLE_EXPERT_BIAS" \
  --moe_ffn_hidden_size "$PRIMUS_MOE_FFN_HIDDEN_SIZE" \
  --index_topk "$PRIMUS_INDEX_TOPK" \
  --v4_grouped_experts_support_clamped_swiglu "$PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU" \
  --compress_ratios "$PRIMUS_COMPRESS_RATIOS" \
  --mtp_num_layers 0 \
  --mock_data True \
  --enable_primus_turbo "$ENABLE_PRIMUS_TURBO" \
  --use_turbo_attention "$USE_TURBO_ATTENTION" \
  --use_v4_triton_attention "$USE_V4_TRITON_ATTENTION" \
  --use_v4_triton_csa_attention "$USE_V4_TRITON_CSA_ATTENTION" \
  --use_turbo_deepep "$USE_TURBO_DEEPEP" \
  --turbo_deepep_num_cu "$TURBO_DEEPEP_NUM_CU" \
  --turbo_deepep_use_comm_stream "$TURBO_DEEPEP_USE_COMM_STREAM" \
  --moe_router_dtype "$MOE_ROUTER_DTYPE" \
  --moe_shared_expert_overlap "$MOE_SHARED_EXPERT_OVERLAP" \
  --use_turbo_grouped_mlp "$TURBO_USE_GROUPED_MLP" \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --fp8 "$FP8" \
  --fp8_recipe "$FP8_RECIPE" \
  --recompute_num_layers 0 \
  --recompute_granularity full \
  --recompute_method block \
  --overlap_grad_reduce False \
  --overlap_param_gather False \
  --disable_last_saving True \
  --disable_wandb True \
  --disable_tensorboard False \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6
