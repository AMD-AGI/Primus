#!/bin/bash

set -x

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
# export GPU_MAX_HW_QUEUES=4
# export CLEAN_DOCKER_CONTAINER=1

export MBS=${MBS:-16}
export GBS=${GBS:-512}
export PRIMUS_TOTAL_LAYERS=24
export PRIMUS_RECOMPUTE_LAYERS=0
export PRIMUS_EP=8
export PROFILE=False
export PRIMUS_DETERMINISTIC=0
export LEGACY_GG=False
export TURBO_USE_GROUPED_MLP=True
export TURBO_USE_PARALLEL_LINEAR=False
# Enable NUMA binding for better memory locality (increase stability for large models)
# export ENABLE_NUMA_BINDING=1
# export HSA_KERNARG_POOL_SIZE=12582912


# ============================================================================
# Grouped GEMM benchmark mode. Run the script once per mode, e.g.
#   MODE=turbo_triton_blockwise bash run.sh
#   MODE=te_delayed_tensorwise  bash run.sh
#
# Supported MODE:
#   turbo_triton_blockwise   primus_turbo gg, TRITON backend, BLOCKWISE  FP8
#   turbo_ck_blockwise       primus_turbo gg, CK     backend, BLOCKWISE  FP8
#   turbo_triton_tensorwise  primus_turbo gg, TRITON backend, TENSORWISE FP8 (curr scaling)
#   turbo_ck_tensorwise      primus_turbo gg, CK     backend, TENSORWISE FP8 (curr scaling)
#   te_delayed_tensorwise    TE  grouped gemm,                DELAYED TENSORWISE FP8
#
# Notes:
# - moe_use_legacy_grouped_gemm is always False here: legacy path routes through
#   Megatron's standalone grouped_gemm package (BF16 only) and bypasses
#   primus_turbo entirely (see DeprecatedGroupedMLP).
# - use_turbo_parallel_linear is kept False so the end-to-end perf delta only
#   reflects the grouped-GEMM path; flip it on if you also want to evaluate the
#   FP8 column/row parallel linear in dense layers.
# - PRIMUS_TURBO_GROUPED_GEMM_BACKEND is consumed inside primus_turbo; it is a
#   no-op for the TE mode.
# ============================================================================
export MODE="${MODE:-turbo_triton_blockwise}"

export PRECISION_TYPE=FP8 # BF16, FP8
export FP8=null # 'e4m3', 'hybrid'
export FP8_RECIPE=null # 'tensorwise', 'delayed', 'mxfp8', 'blockwise', 'custom'
export TE_PRECISION_CONFIG_FILE=null

if [ "$PRECISION_TYPE" = "FP8" ]; then
  export TE_PRECISION_CONFIG_FILE=examples/megatron/configs/MI355X/lfm2_8B_A1B-FP8-te-precision.yaml
  case "$MODE" in
    turbo_triton_blockwise)
      export TURBO_USE_PARALLEL_LINEAR=True
      export TURBO_USE_GROUPED_MLP=True
      export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
      export FP8=e4m3
      export FP8_RECIPE=blockwise
      ;;
    turbo_ck_blockwise)
      export TURBO_USE_PARALLEL_LINEAR=True
      export TURBO_USE_GROUPED_MLP=True
      export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=CK
      export FP8=e4m3
      export FP8_RECIPE=blockwise
      ;;
    turbo_triton_tensorwise)
      export TURBO_USE_PARALLEL_LINEAR=True
      export TURBO_USE_GROUPED_MLP=True
      export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=TRITON
      export FP8=e4m3
      export FP8_RECIPE=tensorwise
      ;;
    turbo_ck_tensorwise)
      export TURBO_USE_PARALLEL_LINEAR=True
      export TURBO_USE_GROUPED_MLP=True
      export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=CK
      export FP8=e4m3
      export FP8_RECIPE=tensorwise
      ;;
    te_delayed_tensorwise)
      export TURBO_USE_PARALLEL_LINEAR=False
      export TURBO_USE_GROUPED_MLP=False
      export FP8=hybrid
      export FP8_RECIPE=delayed
      ;;
    *)
      echo "[ERROR] Unknown MODE='$MODE'. Supported: turbo_triton_blockwise | turbo_ck_blockwise | turbo_triton_tensorwise | turbo_ck_tensorwise | te_delayed_tensorwise" >&2
      exit 2
      ;;
  esac
fi

# export EXP=examples/megatron/configs/MI355X/llama3.1_8B-${PRECISION_TYPE}-pretrain.yaml
export EXP=examples/megatron/configs/MI355X/lfm2_8B_A1B-${PRECISION_TYPE}-pretrain.yaml
# export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml
export PRIMUS_TEAM=amd
PRIMUS_USER="tas-mi325x-$(date +%Y%m%d)"
export PRIMUS_USER
export PRIMUS_EXP_NAME=lfm2_8B_A1B_${PRECISION_TYPE}_MBS${MBS}_GBS${GBS}_EP${PRIMUS_EP}_${MODE}


mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
./primus-cli direct \
  -- train pretrain --config "$EXP" \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --expert_model_parallel_size $PRIMUS_EP \
  --use_turbo_grouped_mlp $TURBO_USE_GROUPED_MLP \
  --use_turbo_parallel_linear $TURBO_USE_PARALLEL_LINEAR \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --fp8 $FP8 \
  --fp8_recipe $FP8_RECIPE \
  --te_precision_config_file $TE_PRECISION_CONFIG_FILE \
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
#   2>&1 | tee log.txt
