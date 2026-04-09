#!/bin/bash

set -x

export HF_TOKEN="${HF_TOKEN:-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}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-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}"  # make it your own wandb api key

export NNODES=${NNODES:-4}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export SLURM_TIME=01:00:00
export SLURM_PARTITION=amd-aig
# export SLURM_NODELIST="uswslocpm2m-106-079"

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0

export MBS=${MBS:-2}
export GBS=$((128 * NNODES))
export PRIMUS_TOTAL_LAYERS=4
export PRIMUS_RECOMPUTE_LAYERS=0
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_PP=1
export PRIMUS_EP=8
export PRIMUS_VPP=1

export PROFILE=${PROFILE:-False}
export TURBO_ATTENTION=${TURBO_ATTENTION:-True}
export TURBO_DEEPEEP=${TURBO_DEEPEEP:-True}
export ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-True}
export LEGACY_GG=${LEGACY_GG:-True}
export TURBO_GROUPED_MLP=${TURBO_GROUPED_MLP:-True}
export TURBO_RMS_NORM=${TURBO_RMS_NORM:-False}
export APPLY_ROPE_FUSION=True
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
export PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-0}
export PRIMUS_DETERMINISTIC=0

# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

export PRETRAIN_TYPE=${PRETRAIN_TYPE:-FP8} # BF16 or FP8

export EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-pretrain.yaml
PRIMUS_TEAM="amd-$(date +%Y%m%d)"
export PRIMUS_TEAM

PRIMUS_USER="${WORKLOAD_ID:-tas}"
export PRIMUS_USER
export PRIMUS_EXP_NAME=debug_4layers-type_$PRETRAIN_TYPE-legacygg_$LEGACY_GG-turbogg_$TURBO_GROUPED_MLP

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
./primus-cli direct --numa \
  -- train pretrain --config "$EXP" \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --use_turbo_attention "$TURBO_ATTENTION" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --enable_primus_turbo "$ENABLE_PRIMUS_TURBO" \
  --use_turbo_grouped_mlp "$TURBO_GROUPED_MLP" \
  --use_turbo_rms_norm "$TURBO_RMS_NORM" \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --enable_experimental "$APPLY_ROPE_FUSION" \
  --apply_rope_fusion "$APPLY_ROPE_FUSION" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
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
  --turbo_deepep_num_cu 80 \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
