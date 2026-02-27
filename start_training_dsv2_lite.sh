#!/bin/bash

export HF_TOKEN="your_hf_token"  # make it your own hf token
export WANDB_API_KEY="your_wandb_api_key"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"

export NNODES=8
export SLURM_TIME=07:00:00
export SLURM_PARTITION=amd-aig

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export CLEAN_DOCKER_CONTAINER=1

export MBS=12
export GBS=$((96 * NNODES))
export PROFILE=False
export TURBO_GROUPED_MLP=True
export TURBO_DEEPEEP=True
export LEGACY_GG=True

# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_EXP_NAME=dsv2_lite-pretrain-mbs_$MBS-gbs_$GBS-turbogg_$TURBO_GROUPED_MLP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-profile_$PROFILE
export PRIMUS_EXP_NAME=debug

mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
bash ./examples/run_slurm_pretrain.sh \
  --train_iters 10 \
  --disable_wandb True \
  --disable_tensorboard True \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --seq_length 4096 \
  --max_position_embeddings 4096 \
  --use_turbo_grouped_mlp $TURBO_GROUPED_MLP \
  --use_turbo_deepep $TURBO_DEEPEEP \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
