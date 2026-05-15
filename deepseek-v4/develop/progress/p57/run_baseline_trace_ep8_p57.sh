#!/bin/bash
# Plan-8 P57 — Triton V4 attention perf-push trace.
#
# Same proxy + profiler window as the P32 final trace
# (TP=1 PP=1 EP=8, Sq=4096, 8 layers, iter 6 -> 7).
# Captures the post-R2 P57 state across all four targets:
#
#   * cr=0 BWD: 7.65 -> 2.08 ms (3.68x, hit <= 3 ms target)
#   * cr=4 FWD: 3.18 -> 1.43 ms (2.22x, hit <= 1.5 ms target)
#   * cr=4 BWD: 16.31 -> 5.11 ms (3.19x, 2.2% over 5.0 ms target)
#   * cr=128 BWD: 11.91 -> 2.81 ms (4.23x, hit <= 3 ms target)
#
# Expected steady-state iter time vs the P40 final 510.6 ms baseline:
# the four V4 attention BWD wins should surface as a ~50-100 ms / iter
# drop assuming the kernel time aligns with bench (P32 final already
# closed the bench/proxy gap via the RoPE bf16 fix).
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

# ---------- V4 attention BWD path knobs (defaults ON since P32 final) -------
export PRIMUS_V4_ATTN_BWD_USE_SPLIT=${PRIMUS_V4_ATTN_BWD_USE_SPLIT:-1}
export PRIMUS_V4_CSA_BWD_SEGREDUCE=${PRIMUS_V4_CSA_BWD_SEGREDUCE:-1}

# ---------- Plan-4 / Plan-5 perf knobs --------------------------------------
export USE_V4_TRITON_ATTENTION=True
export USE_V4_TRITON_CSA_ATTENTION=True
export USE_V4_COMPILED_SINKHORN=True
export USE_TURBO_ATTENTION=False

export USE_TURBO_DEEPEP=True
export TURBO_USE_GROUPED_MLP=True
export LEGACY_GG=False
export ENABLE_PRIMUS_TURBO=True
export TURBO_DEEPEP_NUM_CU=${TURBO_DEEPEP_NUM_CU:-80}
export TURBO_DEEPEP_USE_COMM_STREAM=${TURBO_DEEPEP_USE_COMM_STREAM:-False}
export MOE_ROUTER_DTYPE=${MOE_ROUTER_DTYPE:-fp32}
export MOE_SHARED_EXPERT_OVERLAP=${MOE_SHARED_EXPERT_OVERLAP:-False}

# Plan-6 Triton fusions (default ON since P40 final).
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-1}
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-1}
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-1}
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-1}
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-0}
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-0}

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
export PRIMUS_EXP_NAME=p57_profile_triton_attn_perf_push_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}

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
  --use_v4_compiled_sinkhorn "$USE_V4_COMPILED_SINKHORN" \
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
