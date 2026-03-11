#!/bin/bash

export HF_TOKEN="your_hf_token"  # make it your own hf token
export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
# export SLURM_TREE_WIDTH=128
export NNODES=1
export TRAIN_ITERS=5
export SLURM_TIME=48:00:00
export SLURM_PARTITION=amd-aig-2
export SLURM_NODELIST="uswslocpm2m-106-1962"
# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export CLEAN_DOCKER_CONTAINER=1
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}

export MBS=8
# export GBS=$((128 * NNODES))
export GBS=$((512 * NNODES))
export PRIMUS_RECOMPUTE_LAYERS=5 # 5
export PRIMUS_PP=1
export PRIMUS_EP=8
export PRIMUS_VPP=1
export PROFILE=False
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export PRIMUS_DETERMINISTIC=0
# Enable NUMA binding for better memory locality (increase stability for large models)
# export ENABLE_NUMA_BINDING=1
# export HSA_KERNARG_POOL_SIZE=12582912

export PRETRAIN_TYPE=BF16
# export PRETRAIN_TYPE=FP8

# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-${PRETRAIN_TYPE}-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=qwen3_30B_A3B-pretrain-${PRETRAIN_TYPE}-node_$NNODES-mbs_$MBS-gbs_$GBS-PP_$PRIMUS_PP-EP_$PRIMUS_EP-VPP_$PRIMUS_VPP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-profile_$PROFILE-recompute_$PRIMUS_RECOMPUTE_LAYERS
# export PRIMUS_EXP_NAME=debug

#CKPT_DIR=output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/checkpoints


mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
# mkdir -p "$CKPT_DIR"
bash ./examples/run_slurm_pretrain.sh \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --use_turbo_deepep $TURBO_DEEPEEP \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --pipeline_model_parallel_size $PRIMUS_PP \
  --expert_model_parallel_size $PRIMUS_EP \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  --disable_last_saving True \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --pp_warmup True  \
  --mtp_num_layers 0 \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb True \
  --disable_tensorboard True \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
