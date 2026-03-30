#!/bin/bash

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-624-ainic"

export NNODES=${NNODES:-4}
export SLURM_TIME=48:00:00
export SLURM_PARTITION=amd-aig
# export SLURM_NODELIST="uswslocpm2m-106-[2184,2194,2215,2230]"
# export SLURM_NODELIST="uswslocpm2m-106-[2185,2194,2215,2230]"
export SLURM_NODELIST="uswslocpm2m-106-[2185,2195,2215,2230]"

export TRAIN_ITERS=10

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0

export MBS=4
export GBS=512
# export GBS=64
export SEQ_LENGTH=16384
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_VPP=${PRIMUS_VPP:-1}
export PRIMUS_CP=${PRIMUS_CP:-4}
export PRIMUS_ETP=${PRIMUS_ETP:-1}
export TURBO_DEEPEEP=${TURBO_DEEPEEP:-False}
export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-0}
export DISABLE_PROFILER_ACTIVITY_CPU=${DISABLE_PROFILER_ACTIVITY_CPU:-True}

export PROFILE=False
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
export PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-0}


# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

export USE_MEGATRON_FSDP=False
export MEGATRON_FSDP_ZERO_LEVEL=2 # 2, 3
export PRETRAIN_TYPE=${PRETRAIN_TYPE:-BF16}
if [ "$USE_MEGATRON_FSDP" = "True" ]; then
  export EXP=examples/megatron/configs/MI355X/deepseek_proxy-${PRETRAIN_TYPE}-FSDP-pretrain.yaml
  if [ "$MEGATRON_FSDP_ZERO_LEVEL" = "2" ]; then
      export DATA_PARALLEL_SHARDING_STRATEGY="optim_grads"
  elif [ "$MEGATRON_FSDP_ZERO_LEVEL" = "3" ]; then
      export DATA_PARALLEL_SHARDING_STRATEGY="optim_grads_params"
  else
    echo "Invalid MEGATRON_FSDP_ZERO_LEVEL: $MEGATRON_FSDP_ZERO_LEVEL"
    exit 1
  fi
else
  export EXP=examples/megatron/configs/MI355X/deepseek_proxy-${PRETRAIN_TYPE}-pretrain.yaml
  OPTIMIZER_FEATURES="--use_precision_aware_optimizer True \
    --main_grads_dtype bf16 \
    --exp_avg_dtype bf16 \
    --exp_avg_sq_dtype bf16"
fi

PRIMUS_TEAM="amd-$(date +%Y%m%d)"
export PRIMUS_TEAM
export PRIMUS_USER=tas
# export PRIMUS_EXP_NAME=deepseek_proxy-type_$PRETRAIN_TYPE-mbs_$MBS-gbs_$GBS-seq_length_$SEQ_LENGTH-cp_$PRIMUS_CP-etp_$PRIMUS_ETP-ep_$PRIMUS_EP-pp_$PRIMUS_PP-turbodeepep_$TURBO_DEEPEEP
export PRIMUS_EXP_NAME=deepseek_proxy-type_$PRETRAIN_TYPE-FSDP_${USE_MEGATRON_FSDP}-zero${MEGATRON_FSDP_ZERO_LEVEL}-mbs_$MBS-gbs_$GBS-seq_length_$SEQ_LENGTH-cp_$PRIMUS_CP-etp_$PRIMUS_ETP-ep_$PRIMUS_EP-pp_$PRIMUS_PP-turbodeepep_$TURBO_DEEPEEP


mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
# ./primus-cli direct --numa \
./primus-cli slurm -N "$NNODES" \
  ${SLURM_TIME:+--time="${SLURM_TIME}"} \
  ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
  ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
  -- --image ${DOCKER_IMAGE} --clean -- --numa \
  -- train pretrain --config "$EXP" \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --seq_length "$SEQ_LENGTH" \
  --max_position_embeddings "$SEQ_LENGTH" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --virtual_pipeline_model_parallel_size "$PRIMUS_VPP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  --export_tensor_parallel_size "$PRIMUS_ETP" \
  --context_parallel_size "$PRIMUS_CP" \
  --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" \
  --recompute_granularity full \
  --recompute_method block \
  --manual_gc True \
  --manual_gc_interval 1 \
  --pp_warmup True  \
  --data_parallel_sharding_strategy "$DATA_PARALLEL_SHARDING_STRATEGY" \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_profiler_activity_cpu "$DISABLE_PROFILER_ACTIVITY_CPU" \
  "$OPTIMIZER_FEATURES" \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
