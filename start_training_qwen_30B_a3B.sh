#!/bin/bash

set -x

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/tasimage/primus:pr-563-ainic"}

# export SLURM_TREE_WIDTH=128
export NNODES=1
export TRAIN_ITERS=20
export SLURM_TIME=48:00:00
export SLURM_PARTITION=amd-aig-2
export SLURM_NODELIST="uswslocpm2m-106-1962"

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export CLEAN_DOCKER_CONTAINER=1

export MBS=8
export GBS=$((512 * NNODES))
export PRIMUS_RECOMPUTE_LAYERS=0
export PRIMUS_PP=1
export PRIMUS_EP=8
export PRIMUS_VPP=1
export PROFILE=False
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export APPLY_ROPE_FUSION=True
export PRIMUS_DETERMINISTIC=0
# Enable NUMA binding for better memory locality (increase stability for large models)
# export ENABLE_NUMA_BINDING=1
# export HSA_KERNARG_POOL_SIZE=12582912

export PRETRAIN_TYPE=${PRETRAIN_TYPE:-FP8}
export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-${PRETRAIN_TYPE}-pretrain.yaml

PRIMUS_TEAM="amd-$(date +%Y%m%d)"
export PRIMUS_TEAM
export PRIMUS_USER="${WORKLOAD_ID:-tas}"
export PRIMUS_USER
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=qwen3_30B_A3B-pretrain-${PRETRAIN_TYPE}-node_$NNODES-mbs_$MBS-gbs_$GBS-PP_$PRIMUS_PP-EP_$PRIMUS_EP-VPP_$PRIMUS_VPP-legacygg_$LEGACY_GG
# export PRIMUS_EXP_NAME=debug


mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
# ./primus-cli slurm -N $NNODES \
#   ${SLURM_TIME:+--time="${SLURM_TIME}"} \
#   ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
#   ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
#   -- --image "${DOCKER_IMAGE}" --clean -- --numa \

./primus-cli direct --numa \
  -- train pretrain --config "$EXP" \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --use_turbo_deepep $TURBO_DEEPEEP \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --pipeline_model_parallel_size $PRIMUS_PP \
  --expert_model_parallel_size $PRIMUS_EP \
  --enable_experimental "$APPLY_ROPE_FUSION" \
  --apply_rope_fusion "$APPLY_ROPE_FUSION" \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  --disable_last_saving True \
  --mock_data True \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  --turbo_deepep_num_cu 80 \
  --disable_wandb True \
  --disable_tensorboard True \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
  # --manual_gc True \
  # --manual_gc_interval 1 \
