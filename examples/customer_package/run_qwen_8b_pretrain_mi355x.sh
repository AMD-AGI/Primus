#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

######################### Training Docker and Variables #########################
# export DOCKER_IMAGE="docker.io/tasimage/primus:pr-316-gfx950"
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-316-gfx950-ainic"
export CLEAN_DOCKER_CONTAINER=1
export SKIP_TRAIN=0

######################### Training Environment Variables #########################
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export GPU_MAX_HW_QUEUES=2
export CPUS_PER_TASK=96

# Set on Primus-Safe Platform
# export MASTER_ADDR=${MASTER_ADDR:-localhost}
# export MASTER_PORT=${MASTER_PORT:-1234}
# export NNODES=${PET_NNODES:-1}
# export NODE_RANK=${PET_NODE_RANK:-0}
# export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# vultr cluster
export NNODES=${NNODES:-1}
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7" # modify based on the GPU NiC settings
export NCCL_SOCKET_IFNAME="enp193s0f1np1"
export GLOO_SOCKET_IFNAME="enp193s0f1np1"
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_TIMEOUT=300

export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
# export USE_ROCM_AITER_ROPE_BACKEND=0
# export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=0


######################### Training Config #########################
export MBS=${MBS:-8}
export GBS=${GBS:-256}
# export GBS=${GBS:-$((64 * NNODES))}
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export TP=${TP:-1}
export ETP=${ETP:-1}
export PP=${PP:-1}
export VPP=${VPP:-1}
export EP=${EP:-1}
export CP=${CP:-1}
export CP_COMM_TYPE=${CP_COMM_TYPE:-"a2a"} # p2p, a2a, allgather or a2a+p2p
export OPTIMIZER=${OPTIMIZER:-adam}
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-0}
export FP8=${FP8:-False} # True for fp8, False for bf16
export PROFILE=${PROFILE:-False}
export DISABLE_CPU_TRACE=${DISABLE_CPU_TRACE:-False}
export PROFILE_STEP_START=${PROFILE_STEP_START:-5}
export PROFILE_STEP_END=${PROFILE_STEP_END:-6}
export TRAIN_ITERS=${TRAIN_ITERS:-10}

# Features legend:
# 0 - Baseline (no extra optimization toggles)
# 1 - Loss fusion helper
# 2 - CPU NUMA binding helper
# 3 - Manual GC helper

if [ -z "${Dense_Features}" ]; then
    Dense_Features=(0 1 2)
else
    # Convert string to array
    # shellcheck disable=SC2128
    read -ra Dense_Features <<< "${Dense_Features}"
fi

FEATURE_ARGS=()

for feature in "${Dense_Features[@]}"; do
    case "$feature" in
    0) ;;
    1)
        FEATURE_ARGS+=("--cross_entropy_fusion_impl" "te")
        FEATURE_ARGS+=("--cross_entropy_loss_fusion" "True")
        ;;
    2)
        # Enable NUMA binding for better memory locality (increase stability for large models)
        export ENABLE_NUMA_BINDING=1
        export HSA_KERNARG_POOL_SIZE=12582912
        ;;
    3)
        FEATURE_ARGS+=("--manual_gc" "True")
        FEATURE_ARGS+=("--manual_gc_interval" "1")
        ;;
    *) ;;
    esac
done

FEATURE_LIST="${Dense_Features[*]}"
FEATURE_TAG=$(printf "%s" "${FEATURE_LIST}" | tr ' ' '-')


VPP_ARGS=()
if [ "$VPP" -gt 1 ]; then
    VPP_ARGS+=("--num_virtual_stages_per_pipeline_rank" "$VPP")
fi

FP8_ARGS=()
if [ "$FP8" = "True" ]; then
    FP8_ARGS+=("--fp8" "hybrid")
fi

RECOMPUTE_ARGS=()
if [ "$RECOMPUTE_LAYERS" -gt 0 ]; then
    RECOMPUTE_ARGS+=("--recompute_granularity" "full")
    RECOMPUTE_ARGS+=("--recompute_method" "block")
    RECOMPUTE_ARGS+=("--recompute_num_layers" "${RECOMPUTE_LAYERS}")
fi

PROFILE_ARGS=()
if [ "$PROFILE" = "True" ]; then
    # --profile-ranks 0 1 2 3 4 5 6 7
    PROFILE_ARGS+=("--profile" "True")
    PROFILE_ARGS+=("--disable_profiler_activity_cpu" "${DISABLE_CPU_TRACE}")
    PROFILE_ARGS+=("--use_pytorch_profiler" "True")
    PROFILE_ARGS+=("--profile_step_start" "${PROFILE_STEP_START}")
    PROFILE_ARGS+=("--profile_step_end" "${PROFILE_STEP_END}")
fi

######################### Training Experiments #########################
PRIMUS_TEAM="date-$(date +%Y%m%d)-qwen3-8B"
export PRIMUS_TEAM
PRIMUS_USER=user-tas
export PRIMUS_USER
# export PRIMUS_EXP_NAME="debug"
export PRIMUS_EXP_NAME="qwen3_8B_MI355X_FP8${FP8}_NNODES${NNODES}_MBS${MBS}_GBS${GBS}_SEQ${SEQ_LENGTH}_REC${RECOMPUTE_LAYERS}_TP${TP}_ETP${ETP}_PP${PP}_VPP${VPP}_EP${EP}_CP${CP}_Profile${PROFILE}-${PROFILE_STEP_START}-${PROFILE_STEP_END}_NoCPUTrace${DISABLE_CPU_TRACE}_Queue${GPU_MAX_HW_QUEUES}_Features${FEATURE_TAG}"

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export DUMP_PP_DIR=$LOG_DIR/pp_dump
export LOG_FILE=$LOG_DIR/training.log
export EXPORT_CONFIG=$LOG_DIR/config.yaml
mkdir -p "$LOG_DIR"
rm -rf "$LOG_FILE"

######################### Training Job #########################
export EXP="examples/megatron/configs/MI355X/qwen3_8B-pretrain.yaml"

echo "--------------------------------" | tee -a "$LOG_FILE"
echo "Begin Training... $(date +%Y%m%d_%H%M%S)" | tee -a "$LOG_FILE"
echo "Training Config: $EXP" | tee -a "$LOG_FILE"
echo "LOG_DIR=${LOG_DIR}" | tee -a "$LOG_FILE"
echo "LOG_FILE=${LOG_FILE}" | tee -a "$LOG_FILE"
echo "FEATURE_ARGS=${FEATURE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "Dense_Features=${FEATURE_LIST}" | tee -a "$LOG_FILE"
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "RECOMPUTE_ARGS=${RECOMPUTE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "PROFILE_ARGS=${PROFILE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

bash ./examples/run_slurm_pretrain.sh \
    --micro_batch_size "$MBS" \
    --global_batch_size "$GBS" \
    --seq_length "$SEQ_LENGTH" \
    --tensor_model_parallel_size "$TP" \
    --expert_tensor_parallel_size "$ETP" \
    --pipeline_model_parallel_size "$PP" \
    --expert_model_parallel_size "$EP" \
    --context_parallel_size "$CP" \
    --cp_comm_type "$CP_COMM_TYPE" \
    --mock_data True \
    --optimizer "$OPTIMIZER" \
    --torch_profiler_use_gzip False \
    "${VPP_ARGS[@]}" \
    "${FEATURE_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
