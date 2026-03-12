#!/bin/bash

export HF_TOKEN="your_hf_token"  # make it your own hf token
export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key

export PLATFORM="MI355X" # "B200" "GB200"
if [ "$PLATFORM" = "MI355X" ]; then
  export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
elif [ "$PLATFORM" = "B200" ] || [ "$PLATFORM" = "GB200" ]; then
  export DOCKER_IMAGE="nvcr.io/nvidia/nemo:25.09"
  EXTRA_ARGS="--use_rocm_mem_info_iters None"
else
  echo "Error: unsupported PLATFORM '$PLATFORM'. Must be MI355X, B200, or GB200." >&2
  exit 1
fi

export NNODES=1
export TRAIN_ITERS=10
export SLURM_TIME=01:00:00
export SLURM_PARTITION=amd-aig
# export SLURM_NODELIST="uswslocpm2m-106-079"

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export CLEAN_DOCKER_CONTAINER=1

export MBS=1
export GBS=16
export PRIMUS_TOTAL_LAYERS=4
export PRIMUS_RECOMPUTE_LAYERS=0
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_PP=1
export PRIMUS_EP=8
export PRIMUS_VPP=1
export PROFILE=True
export TURBO_DEEPEEP=True
export LEGACY_GG=True # True: legacy GG, False: TE GG
export PRIMUS_DETERMINISTIC=0
# Enable NUMA binding for better memory locality (increase stability for large models)
# export ENABLE_NUMA_BINDING=1
# export HSA_KERNARG_POOL_SIZE=12582912


export PRETRAIN_TYPE=BF16 # BF16 # FP8
export EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_EXP_NAME=dsv3-pretrain-platform_$PLATFORM-layers_$PRIMUS_TOTAL_LAYERS-type_$PRETRAIN_TYPE-mbs_$MBS-gbs_$GBS-legacygg_$LEGACY_GG
# export PRIMUS_EXP_NAME=debug


mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
# bash ./examples/run_pretrain.sh \
bash ./examples/run_slurm_pretrain.sh \
  --num_layers $PRIMUS_TOTAL_LAYERS \
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
  --moe_layer_freq $PRIMUS_MOE_LAYER_FREQ \
  --mock_data True \
  --pp_warmup True  \
  --mtp_num_layers 0 \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb True \
  --disable_tensorboard True \
  "$EXTRA_ARGS" \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt

  # --manual_gc True \
  # --manual_gc_interval 1 \
