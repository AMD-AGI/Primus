#!/bin/bash

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key

# export DOCKER_IMAGE="docker.io/tasimage/primus:pr-642"

export NNODES=${PET_NNODES:-1}
export TRAIN_ITERS=10

# export NCCL_DEBUG=INFO
export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
# export GLOO_SOCKET_IFNAME=ens9np0
# export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export CLEAN_DOCKER_CONTAINER=1

export MBS=16
export GBS=512
export PRIMUS_TOTAL_LAYERS=24
export PRIMUS_RECOMPUTE_LAYERS=0
export PRIMUS_EP=8
export PROFILE=False
export PRIMUS_DETERMINISTIC=0
export LEGACY_GG=False
export TURBO_USE_GROUPED_MLP=False
# Enable NUMA binding for better memory locality (increase stability for large models)
# export ENABLE_NUMA_BINDING=1
# export HSA_KERNARG_POOL_SIZE=12582912

if [ "$TURBO_USE_GROUPED_MLP" = "True" ]; then
  export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
#   export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=CK
#   export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPBLASLT
fi

export PRECISION_TYPE=FP8 # BF16
export EXP=examples/megatron/configs/MI355X/lfm2_8B_A1B-${PRECISION_TYPE}-pretrain.yaml
# export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml
export PRIMUS_TEAM=amd
PRIMUS_USER="tas-mi325x-$(date +%Y%m%d)"
export PRIMUS_USER
export PRIMUS_EXP_NAME=lfm2_8B_A1B_${PRECISION_TYPE}_MBS${MBS}_GBS${GBS}_EP${PRIMUS_EP}_legacygg${LEGACY_GG}_turbogg${TURBO_USE_GROUPED_MLP}_${PRIMUS_TURBO_GROUPED_GEMM_BACKEND}
# export PRIMUS_EXP_NAME=debug


mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
./primus-cli direct \
  -- train pretrain --config "$EXP" \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --expert_model_parallel_size $PRIMUS_EP \
  --use_turbo_grouped_mlp $TURBO_USE_GROUPED_MLP \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
#   2>&1 | tee log.txt
