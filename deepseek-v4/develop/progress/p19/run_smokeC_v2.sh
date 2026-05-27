#!/bin/bash
# Smoke C v2: PP=4 EP=2 (no VPP) with the V4 hash-router input_ids
# pre-broadcast moved to a primus-patch on get_forward_backward_func +
# pretrain_gpt.get_batch. Confirms the upfront-broadcast path still
# matches smoke C's earlier loss curve.
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

export MBS=1
export GBS=16
export PRIMUS_TP=1
export PRIMUS_PP=4
export PRIMUS_EP=2

export PRIMUS_TOTAL_LAYERS=8
export PRIMUS_SEQ_LENGTH=128
export PRIMUS_MAX_POSITION_EMBEDDINGS=128
export PRIMUS_NUM_EXPERTS=8
export PRIMUS_MOE_TOPK=2
export PRIMUS_MOE_FFN_HIDDEN_SIZE=512
export PRIMUS_INDEX_TOPK=8
export PRIMUS_COMPRESS_RATIOS="[0,0,4,4,4,4,4,0]"
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=False
export PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=True
export PROFILE=False
export USE_TURBO_ATTENTION=False
export TURBO_USE_GROUPED_MLP=False
export LEGACY_GG=False

export PRECISION_TYPE=BF16
export FP8=null
export FP8_RECIPE=null

export EXP=examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml
export BACKEND_PATH="$(pwd)/third_party/Megatron-LM"
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas-mi355x-$(date +%Y%m%d)
export PRIMUS_EXP_NAME=p19_smokeC_pp4_ep2_v2

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
  --use_turbo_attention "$USE_TURBO_ATTENTION" \
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
  --disable_tensorboard True \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6
