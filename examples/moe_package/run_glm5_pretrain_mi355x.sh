#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -x

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"

export NNODES=${NNODES:-8}

export TRAIN_ITERS=${TRAIN_ITERS:-10}

export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0

export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600

export MBS=${MBS:-1}
export GBS=$((32 * NNODES))
export PRIMUS_TOTAL_LAYERS=78
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_PP=${PRIMUS_PP:-8}
export PRIMUS_VPP=${PRIMUS_VPP:-1}
export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-8}

export PROFILE=False
export TURBO_ATTENTION=${TURBO_ATTENTION:-True}
export TURBO_DEEPEEP=${TURBO_DEEPEEP:-True}
export LEGACY_GG=${LEGACY_GG:-True}
export TURBO_GROUPED_MLP=${TURBO_GROUPED_MLP:-False}
export TURBO_RMS_NORM=${TURBO_RMS_NORM:-True}
# MLA does not support rope fusion
export APPLY_ROPE_FUSION=False
export PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-0}

export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

# GLM-5: 78 layers, MLA + MoE (256 experts, top-8, first 3 dense)
# Pipeline layout for PP*VPP stages
STAGE=$(( PRIMUS_PP * PRIMUS_VPP ))
FEATURE_ARGS=()
PIPELINE_ARGS=()
if [ "$PRIMUS_VPP" -gt 1 ]; then
  case $STAGE in
    8)
      # 78 layers / 8 stages: 10*6+9*2=78
      FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*9|t*10|t*10|t*10|t*10|t*10|t*10|t*9,L'")
      ;;
    16)
      # 78 layers / 16 stages: 5*14+4*2=78
      FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*4|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*5|t*4,L'")
      ;;
    32)
      # 78 layers / 32 stages: 3*14+2*18=78 → actually 3*10+2*22=74 nope
      # 78/32=2.4375 → 3*14+2*18=78 nope. Let's do: 2*16+3*16=2*16+3*16 nope.
      # Correct: 78=3*24+2*8=72+6 nope. Actually 78/32: 2 per stage + 14 stages with 3 = 2*18+3*14=36+42=78 ✓
      FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*2|t*3|t*3|t*3|t*2|t*3|t*3|t*3|t*2|t*3|t*3|t*3|t*2|t*3|t*3|t*2|t*3|t*3|t*2|t*3|t*3|t*2|t*3|t*3|t*2|t*3|t*3|t*2|t*3|t*3|t*2|t*2,L'")
      ;;
    *)
      echo "Unsupported STAGE=${STAGE} (PRIMUS_PP=${PRIMUS_PP}, PRIMUS_VPP=${PRIMUS_VPP})." >&2
      exit 1
      ;;
  esac
else
  if [ -z "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]; then
    if [ "$PRIMUS_PP" -eq 4 ]; then
      DECODER_LAST_PIPELINE_NUM_LAYERS=18
    elif [ "$PRIMUS_PP" -eq 8 ]; then
      DECODER_LAST_PIPELINE_NUM_LAYERS=8
    else
      MIDDLE_LAYERS_EACH=$(( (PRIMUS_TOTAL_LAYERS + PRIMUS_PP - 1) / PRIMUS_PP ))
      DECODER_LAST_PIPELINE_NUM_LAYERS=$(( PRIMUS_TOTAL_LAYERS - (PRIMUS_PP - 1) * MIDDLE_LAYERS_EACH ))
    fi
  fi
  export DECODER_LAST_PIPELINE_NUM_LAYERS
  PIPELINE_ARGS+=("--decoder_last_pipeline_num_layers" "$DECODER_LAST_PIPELINE_NUM_LAYERS")
fi

if [ -n "$RECOMP_IDS" ]; then
  export RECOMP_IDS
  RECOMP_ARGS=(--recompute_layer_ids "$RECOMP_IDS" --recompute_granularity full)
else
  RECOMP_ARGS=(--recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" --recompute_granularity full --recompute_method block)
fi

export PRETRAIN_TYPE=${PRETRAIN_TYPE:-BF16}

export EXP=examples/megatron/configs/MI355X/glm5-${PRETRAIN_TYPE}-pretrain.yaml
PRIMUS_TEAM="amd-$(date +%Y%m%d)"
export PRIMUS_TEAM
PRIMUS_USER="${WORKLOAD_ID:-tas}"
export PRIMUS_USER
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document
export PRIMUS_EXP_NAME=glm5-type_$PRETRAIN_TYPE-legacygg_$LEGACY_GG-turbogg_$TURBO_GROUPED_MLP-turbodeepep_$TURBO_DEEPEEP-turboattn_$TURBO_ATTENTION-autotune_$PRIMUS_TURBO_AUTO_TUNE

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
./primus-cli direct --numa \
  -- train pretrain --config "$EXP" \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters "$TRAIN_ITERS" \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --use_turbo_attention "$TURBO_ATTENTION" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --use_turbo_grouped_mlp "$TURBO_GROUPED_MLP" \
  --use_turbo_rms_norm "$TURBO_RMS_NORM" \
  --lr 2.2e-4 \
  --min_lr 2.2e-5 \
  --lr_warmup_iters 200 \
  --lr_decay_iters 5000 \
  --lr_decay_style cosine \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  "${PIPELINE_ARGS[@]}" \
  "${FEATURE_ARGS[@]}" \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  "${RECOMP_ARGS[@]}" \
  --disable_last_saving True \
  --moe_layer_freq "$PRIMUS_MOE_LAYER_FREQ" \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --pp_warmup True  \
  --mtp_num_layers 0 \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb True \
  --disable_tensorboard True \
  --turbo_deepep_num_cu 80 \
  --overlap_grad_reduce False \
  --overlap_param_gather False \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
