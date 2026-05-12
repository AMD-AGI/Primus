#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

export HF_TOKEN="your_hf_token"  # make it your own hf token
export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
# export SLURM_TREE_WIDTH=128
export NNODES=32
export TRAIN_ITERS=10

export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export CLEAN_DOCKER_CONTAINER=1

export MBS=${MBS:-4}
export GBS=8192
export PRIMUS_TOTAL_LAYERS=94
export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-3}
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_PP=${PRIMUS_PP:-4}
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_VPP=${PRIMUS_VPP:-4}
export PROFILE=False
export TURBO_DEEPEEP=True
export TURBO_ATTENTION=${TURBO_ATTENTION:-False}
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
export TURBO_GROUPED_MLP=${TURBO_GROUPED_MLP:-False}
export TURBO_RMS_NORM=${TURBO_RMS_NORM:-False}
export PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-0}
export APPLY_ROPE_FUSION=True
export LEGACY_GG=${LEGACY_GG:-True}
export PRIMUS_DETERMINISTIC=0
# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912


FEATURE_ARGS=()
PIPELINE_ARGS=()
if [ "$PRIMUS_VPP" -gt 1 ]; then
  case "$PRIMUS_VPP" in
    2)
      if [ "$PRIMUS_PP" -eq 4 ]; then
        # Qwen3-235B has 94 decoder layers. For PP4+VPP2 (8 chunks): 12*6 + 11*2 = 94
        FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*11|t*12|t*12|t*12|t*12|t*12|t*12|t*11,L'")
      elif [ "$PRIMUS_PP" -eq 8 ]; then
        # Qwen3-235B has 94 decoder layers. For PP8+VPP2 (16 chunks): 6*14 + 5*2 = 94
        FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*5|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*5,L'")
      else
        echo "Unsupported PRIMUS_PP=${PRIMUS_PP} for PRIMUS_VPP=2. Supported PP values: 4, 8." >&2
        exit 1
      fi
      ;;
    4)
      # Qwen3-235B has 94 decoder layers. For PP4+VPP4 (16 chunks): 6*14 + 5*2 = 94
      FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*5|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*6|t*5,L'")
      ;;
    *)
      echo "Unsupported PRIMUS_VPP=${PRIMUS_VPP}. Supported values in this script: 1, 2, 4." >&2
      exit 1
      ;;
  esac
else
  if [ -z "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]; then
    if [ "$PRIMUS_PP" -eq 4 ]; then
      # 94 layers, PP4: 24,24,24,22
      DECODER_LAST_PIPELINE_NUM_LAYERS=22
    elif [ "$PRIMUS_PP" -eq 8 ]; then
      # 94 layers, PP8: 12,12,12,12,12,12,12,10
      DECODER_LAST_PIPELINE_NUM_LAYERS=10
    else
      # Auto-compute: each middle stage gets ceil(total/PP) layers, last stage gets the remainder
      MIDDLE_LAYERS_EACH=$(( (PRIMUS_TOTAL_LAYERS + PRIMUS_PP - 1) / PRIMUS_PP ))
      DECODER_LAST_PIPELINE_NUM_LAYERS=$(( PRIMUS_TOTAL_LAYERS - (PRIMUS_PP - 1) * MIDDLE_LAYERS_EACH ))
    fi
  fi
  export DECODER_LAST_PIPELINE_NUM_LAYERS
  MIDDLE_PP_SIZE=$((PRIMUS_PP - 1))
  if [ "$MIDDLE_PP_SIZE" -le 0 ]; then
    echo "Invalid PRIMUS_PP=${PRIMUS_PP}. PRIMUS_PP must be >= 2 when PRIMUS_VPP <= 1." >&2
    exit 1
  fi
  MIDDLE_STAGE_LAYERS=$((PRIMUS_TOTAL_LAYERS - DECODER_LAST_PIPELINE_NUM_LAYERS))
  if [ $((MIDDLE_STAGE_LAYERS % MIDDLE_PP_SIZE)) -ne 0 ]; then
    echo "Invalid split: PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS}, DECODER_LAST_PIPELINE_NUM_LAYERS=${DECODER_LAST_PIPELINE_NUM_LAYERS}, PRIMUS_PP=${PRIMUS_PP}. (PRIMUS_TOTAL_LAYERS - DECODER_LAST_PIPELINE_NUM_LAYERS) must be divisible by (PRIMUS_PP - 1)." >&2
    exit 1
  fi
  PIPELINE_ARGS+=("--decoder_last_pipeline_num_layers" "$DECODER_LAST_PIPELINE_NUM_LAYERS")
fi


export PRETRAIN_TYPE=${PRETRAIN_TYPE:-BF16}

export EXP=examples/megatron/configs/MI355X/qwen3_235B_A22B-${PRETRAIN_TYPE}-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=qwen3_235B_A22B-pretrain-${PRETRAIN_TYPE}-node_$NNODES-mbs_$MBS-gbs_$GBS-PP_$PRIMUS_PP-EP_$PRIMUS_EP-VPP_$PRIMUS_VPP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-profile_$PROFILE-recompute_$PRIMUS_RECOMPUTE_LAYERS


mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
# mkdir -p "$CKPT_DIR"
#bash ./examples/run_slurm_pretrain.sh \
./primus-cli direct --numa \
  -- train pretrain --config "$EXP" \
  --num_layers "$PRIMUS_TOTAL_LAYERS" \
  --train_iters "$TRAIN_ITERS" \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --use_turbo_attention "$TURBO_ATTENTION" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --use_turbo_grouped_mlp "$TURBO_GROUPED_MLP" \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  "${PIPELINE_ARGS[@]}" \
  "${FEATURE_ARGS[@]}" \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" \
  --recompute_granularity full \
  --recompute_method block \
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
  --enable_experimental "$APPLY_ROPE_FUSION" \
  --apply_rope_fusion "$APPLY_ROPE_FUSION" \
  --use_turbo_rms_norm "$TURBO_RMS_NORM" \
  --turbo_deepep_num_cu 80 \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt"
