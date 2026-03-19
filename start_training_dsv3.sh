#!/bin/bash

#======================================ENV CONFIG======================================
export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key

export LAUNCHER=${LAUNCHER:-direct}
export PRETRAIN_TYPE=${PRETRAIN_TYPE:-FP8}

#======================================LAUNCHER CONFIG======================================
LAUNCH_CMD=()
if [ "$LAUNCHER" = "slurm" ]; then
  # slurm launcher mean the job is running on the slurm cluster.
  export NNODES=8
  export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/tasimage/primus:pr-609-ainic"}
  export SLURM_TIME=1:00:00
  export SLURM_PARTITION=amd-aig
  export SLURM_NODELIST="uswslocpm2m-106-[030-031,038-039,050,063,069,225]"

  LAUNCH_CMD=(./primus-cli slurm --numa -N "$NNODES")
  [ -n "$SLURM_TIME" ] && LAUNCH_CMD+=(--time "$SLURM_TIME")
  [ -n "$SLURM_PARTITION" ] && LAUNCH_CMD+=(--partition "$SLURM_PARTITION")
  [ -n "$SLURM_NODELIST" ] && LAUNCH_CMD+=(--nodelist "$SLURM_NODELIST")
  LAUNCH_CMD+=(-- --image "$DOCKER_IMAGE" --clean)
elif [ "$LAUNCHER" = "direct" ]; then
  # direct launcher mean the job is running inside the container
  export MASTER_ADDR=${MASTER_ADDR:-localhost}
  export MASTER_PORT=${MASTER_PORT:-1234}
  export NNODES=${NNODES:-1}
  export NODE_RANK=${NODE_RANK:-0}
  export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

  LAUNCH_CMD=(./primus-cli direct --numa)
else
  echo "Unsupported LAUNCHER=${LAUNCHER}. Supported: slurm, direct." >&2
  exit 1
fi

#======================================TRAIN CONFIG======================================
export TRAIN_ITERS=10
export MBS=4
export GBS=2048
export PRIMUS_EP=8
export PRIMUS_PP=8
export PRIMUS_VPP=2
export PRIMUS_RECOMPUTE_LAYERS=4

#======================================ENV CONFIG======================================
# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0

export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

#======================================EXP CONFIG======================================
export EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-pretrain.yaml
export PRIMUS_TEAM=amd
PRIMUS_USER="tas-$(date +%Y%m%d)"
export PRIMUS_USER
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-dsv3-pretrain-type_$PRETRAIN_TYPE}

# the log files are saved in the output directory
mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

#======================================TRAIN JOB======================================
"${LAUNCH_CMD[@]}" \
  -- train pretrain --config "$EXP" \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --pipeline_model_parallel_size $PRIMUS_PP \
  --expert_model_parallel_size $PRIMUS_EP \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
