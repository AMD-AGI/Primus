#!/bin/bash

set -x

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key

export USING_AINIC=1
export NCCL_IB_HCA="rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0,rocep25s0,rocep9s0"
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export NCCL_IB_GID_INDEX=1
export NCCL_IB_TC=96
export NCCL_IB_FIFO_TC=184
export NCCL_IB_MERGE_NICS=1
export NCCL_CROSS_NIC=0

# export NCCL_DEBUG=INFO

export NNODES=${NNODES:-4}
export DOCKER_IMAGE="docker.io/rocm/primus:v26.2"

export SLURM_TIME="01:00:00"
export SLURM_ACCOUNT="odf"
export SLURM_MEM="0"
export SLURM_NODELIST="mi355-gpu-7,mi355-gpu-8,mi355-gpu-12,mi355-gpu-26"
export SLURM_RESERVATION="mi355-gpu-7_gpu-8_gpu-12_gpu-26_reservation"

export EXP="examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml"
export GBS=$((512 * NNODES))
export PRIMUS_RECOMPUTE_LAYERS=0

./primus-cli slurm -N "$NNODES" \
  ${SLURM_TIME:+--time="${SLURM_TIME}"} \
  ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
  ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
  ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
  ${SLURM_ACCOUNT:+--account="${SLURM_ACCOUNT}"} \
  ${SLURM_MEM:+--mem="${SLURM_MEM}"} \
  --exclusive \
  -- --image "${DOCKER_IMAGE}" --clean -- --numa \
  -- train pretrain --config "${EXP}" \
  --train_iters=10 \
  --global_batch_size "$GBS" \
  --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" \
  2>&1 | tee log.txt
