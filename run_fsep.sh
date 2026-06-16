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

export NNODES=${NNODES:-1}
# export DOCKER_IMAGE="docker.io/rocm/primus:v26.2"
# export DOCKER_IMAGE="docker.io/tasimage/primus:pr-764"
export DOCKER_IMAGE="docker.io/tasimage/primus:latest"

export SLURM_TIME="01:00:00"
export SLURM_ACCOUNT="odf"
export SLURM_MEM="0"
export SLURM_NODELIST="mi355-gpu-7,mi355-gpu-8,mi355-gpu-12,mi355-gpu-26"
export SLURM_RESERVATION="mi355-gpu-7_gpu-8_gpu-12_gpu-26_reservation"

export MBS=8
export GA=4
export GBS=$((MBS * NNODES * 8 * GA))
export PRIMUS_RECOMPUTE_LAYERS=0
export PROFILE=True

# FSDP
export USE_MEGATRON_FSDP=False
export MEGATRON_FSDP_ZERO_LEVEL=2 # 2, 3
if [ "$USE_MEGATRON_FSDP" = "True" ]; then
  export EP=1
  export EXP="examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-FSDP-pretrain.yaml"
  if [ "$MEGATRON_FSDP_ZERO_LEVEL" = "2" ]; then
    export DATA_PARALLEL_SHARDING_STRATEGY="optim_grads"
  elif [ "$MEGATRON_FSDP_ZERO_LEVEL" = "3" ]; then
    export DATA_PARALLEL_SHARDING_STRATEGY="optim_grads_params"
  else
    echo "Invalid MEGATRON_FSDP_ZERO_LEVEL: $MEGATRON_FSDP_ZERO_LEVEL"
    exit 1
  fi
else
  export EP=8
  export EXP="examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml"
fi

# DeepEP flex token dispatcher requires TPxEP > 1; disable it when EP=1
if [ "$EP" = "1" ]; then
  export USE_TURBO_DEEPEP=False
fi

export USE_TURBO_ATTENTION=True
export USE_TURBO_GROUPED_GEMM=False
export USE_TURBO_GEMM=False

if [[ "${USE_TURBO_GROUPED_GEMM}" == "True" ]]; then
  export USE_LEGACY_GROUPED_GEMM=False
else
  export USE_LEGACY_GROUPED_GEMM=True
fi

export PRIMUS_TEAM=amd
PRIMUS_USER="fsep-$(date +%Y%m%d)"
export PRIMUS_USER
export PRIMUS_EXP_NAME=dpskv2lite-BF16-mbs${MBS}-gbs${GBS}-ep${EP}-fsdp${USE_MEGATRON_FSDP}-fsdpzl${MEGATRON_FSDP_ZERO_LEVEL}-turbogg${USE_TURBO_GROUPED_GEMM}-turboa${USE_TURBO_ATTENTION}
# export PRIMUS_EXP_NAME=debug
mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

# ./primus-cli slurm -N "$NNODES" \
#   ${SLURM_TIME:+--time="${SLURM_TIME}"} \
#   ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
#   ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
#   ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
#   ${SLURM_ACCOUNT:+--account="${SLURM_ACCOUNT}"} \
#   ${SLURM_MEM:+--mem="${SLURM_MEM}"} \
#   --exclusive \
#   -- --image "${DOCKER_IMAGE}" --clean -- --numa \
./primus-cli container \
  --image "${DOCKER_IMAGE}" --clean -- --numa \
  -- train pretrain --config "${EXP}" \
  --train_iters=10 \
  --micro_batch_size="$MBS" \
  --global_batch_size "$GBS" \
  --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" \
  --use_turbo_attention "$USE_TURBO_ATTENTION" \
  --use_turbo_gemm "$USE_TURBO_GEMM" \
  --use_turbo_grouped_gemm "$USE_TURBO_GROUPED_GEMM" \
  --moe_use_legacy_grouped_gemm "$USE_LEGACY_GROUPED_GEMM" \
  --expert_model_parallel_size "$EP" \
  ${USE_TURBO_DEEPEP:+--use_turbo_deepep "$USE_TURBO_DEEPEP"} \
  ${DATA_PARALLEL_SHARDING_STRATEGY:+--data_parallel_sharding_strategy "$DATA_PARALLEL_SHARDING_STRATEGY"} \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt"
