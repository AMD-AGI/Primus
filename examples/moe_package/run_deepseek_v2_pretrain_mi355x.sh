#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

######################### Training Docker and Variables #########################
# export DOCKER_IMAGE="docker.io/tasimage/primus:pr-316-gfx950-ainic"
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-316-v25.10-ainic"
# export DOCKER_IMAGE="docker.io/rocm/megatron-lm:v25.10"
export CLEAN_DOCKER_CONTAINER=1
export SKIP_TRAIN=0
export CPUS_PER_TASK=96

######################### Training Environment Variables #########################
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}
# export USE_ROCM_AITER_ROPE_BACKEND=0

# Set on Primus-Safe Platform
# export MASTER_ADDR=${MASTER_ADDR:-localhost}
# export MASTER_PORT=${MASTER_PORT:-1234}
# export NNODES=${PET_NNODES:-1}
# export NODE_RANK=${PET_NODE_RANK:-0}
# export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Set on Vultr cluster
export NNODES=4
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7" # modify based on the GPU NiC settings
export NCCL_SOCKET_IFNAME="enp193s0f1np1"
export GLOO_SOCKET_IFNAME="enp193s0f1np1"
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_TIMEOUT=300

######################### Training Config #########################
export MBS=${MBS:-1}
export GBS=${GBS:-256}
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export TP=${TP:-1}
export ETP=${ETP:-1}
export PP=${PP:-4}
export VPP=${VPP:-5}
export EP=${EP:-8}
export CP=${CP:-1}
export CP_COMM_TYPE=${CP_COMM_TYPE:-"a2a"} # p2p, a2a, allgather or a2a+p2p
export ENABLE_MLA=${ENABLE_MLA:-True}
export ENABLE_MTP=${ENABLE_MTP:-False}
export LOAD_BALANCE=${LOAD_BALANCE:-True}
export OPTIMIZER=${OPTIMIZER:-adam}
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-0}
export LEGACY_GG=${LEGACY_GG:-True}
export FP8=${FP8:-False} # True for fp8, False for bf16
export PROFILE=${PROFILE:-False}
export DISABLE_CPU_TRACE=${DISABLE_CPU_TRACE:-False}
export PROFILE_STEP_START=${PROFILE_STEP_START:-5}
export PROFILE_STEP_END=${PROFILE_STEP_END:-6}
export TRAIN_ITERS=${TRAIN_ITERS:-10}

# MoE_Features legend:
# 0 - Baseline (no extra optimization toggles)
# 1 - Turbo attention acceleration
# 2 - Turbo grouped GEMM / MLP fusion
# 3 - Loss fusion helper
# 4 - DeepEP acceleration
# 5 - Sync-free MoE (stage 1/2)
# 6 - CPU NUMA binding helper
# 7 - Manual GC helper
# 8 - Using UCCL-EP
if [ -z "${MoE_Features}" ]; then
    # MoE_Features=(0 7)
    # MoE_Features=(3 7)
    # MoE_Features=(3 4 7)
    # MoE_Features=(3 4 6 7)
    MoE_Features=(3 4 5 6 7 8)
else
    # Convert string to array
    # shellcheck disable=SC2128
    read -ra MoE_Features <<< "$MoE_Features"
fi

FEATURE_ARGS=()
PRIMUS_TURBO_ENABLED="False"
ensure_primus_turbo() {
    if [ "$PRIMUS_TURBO_ENABLED" = "False" ]; then
        FEATURE_ARGS+=("--enable_primus_turbo" "True")
        PRIMUS_TURBO_ENABLED="True"
    fi
}

for feature in "${MoE_Features[@]}"; do
    case "$feature" in
    0) ;;
    1)
        ensure_primus_turbo
        FEATURE_ARGS+=("--use_turbo_attention" "True")
        ;;
    2)
        ensure_primus_turbo
        FEATURE_ARGS+=("--use_turbo_grouped_mlp" "True")
        ;;
    3)
        FEATURE_ARGS+=("--cross_entropy_fusion_impl" "te")
        FEATURE_ARGS+=("--cross_entropy_loss_fusion" "True")
        ;;
    4)
        ensure_primus_turbo
        FEATURE_ARGS+=("--use_turbo_deepep" "True")
        FEATURE_ARGS+=("--turbo_deepep_num_cu" "64")
        FEATURE_ARGS+=("--turbo_deepep_use_comm_stream" "False")
        FEATURE_ARGS+=("--moe_shared_expert_overlap" "False")
        FEATURE_ARGS+=("--moe_router_dtype" "fp32")
        ;;
    5)
        ensure_primus_turbo
        # mi355
        # sync_free moe stage 1 will open router and permutation fusion
        FEATURE_ARGS+=("--turbo_sync_free_moe_stage" "1")

        # mi300/mi325
        # sync_free moe stage 2 will open deepep automatically
        # FEATURE_ARGS+=("--turbo_sync_free_moe_stage" "2")
        # FEATURE_ARGS+=("--moe_shared_expert_overlap" "False")
        # FEATURE_ARGS+=("--moe_use_legacy_grouped_gemm" "True")
        # FEATURE_ARGS+=("--moe_router_dtype" "fp32")
        ;;
    6)
        # Enable NUMA binding for better memory locality (increase stability for large models)
        export ENABLE_NUMA_BINDING=1
        export HSA_KERNARG_POOL_SIZE=12582912
        ;;
    7)
        FEATURE_ARGS+=("--manual_gc" "True")
        FEATURE_ARGS+=("--manual_gc_interval" "1")
        ;;
    8)
        export USING_UEP=1
        ;;
    *) ;;
    esac
done

FEATURE_LIST="${MoE_Features[*]}"
FEATURE_TAG=$(printf "%s" "${FEATURE_LIST}" | tr ' ' '-')

MLA_ARGS=()
if [ "$ENABLE_MLA" = "True" ]; then
    MLA_ARGS+=("--multi_latent_attention" "True")
else
    MLA_ARGS+=("--multi_latent_attention" "False")
fi

MTP_ARGS=()
if [ "$ENABLE_MTP" = "True" ]; then
    MTP_ARGS+=("--mtp_num_layers" "1")
    MTP_ARGS+=("--mtp_loss_scaling_factor" "0.1")
else
    MTP_ARGS+=("--mtp_num_layers" "None")
fi

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
PRIMUS_TEAM="date-$(date +%Y%m%d)-DeepseekV2-Vultr-MI355X"
export PRIMUS_TEAM
PRIMUS_USER=user-tas
export PRIMUS_USER
# export PRIMUS_EXP_NAME="debug"
export PRIMUS_EXP_NAME="DeepSeekV2_MI355X_FP8${FP8}_MBS${MBS}_GBS${GBS}_SEQ${SEQ_LENGTH}_MLA${ENABLE_MLA}_MTP${ENABLE_MTP}_REC${RECOMPUTE_LAYERS}_TP${TP}_ETP${ETP}_PP${PP}_VPP${VPP}_EP${EP}_CP${CP}_Balance${LOAD_BALANCE}_LegacyGG${LEGACY_GG}_Profile${PROFILE}-${PROFILE_STEP_START}-${PROFILE_STEP_END}_NoCPUTrace${DISABLE_CPU_TRACE}_Queue${GPU_MAX_HW_QUEUES}_Features${FEATURE_TAG}"

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export DUMP_PP_DIR=$LOG_DIR/pp_dump
export LOG_FILE=$LOG_DIR/training.log
export EXPORT_CONFIG=$LOG_DIR/config.yaml
mkdir -p "$LOG_DIR"
rm -rf "$LOG_FILE"

######################### Training Job #########################
export EXP="examples/moe_package/configs/MI355X/deepseek_v2-pretrain-baseline.yaml"

echo "--------------------------------" | tee -a "$LOG_FILE"
echo "Begin Training... $(date +%Y%m%d_%H%M%S)" | tee -a "$LOG_FILE"
echo "Training Config: $EXP" | tee -a "$LOG_FILE"
echo "LOG_DIR=${LOG_DIR}" | tee -a "$LOG_FILE"
echo "LOG_FILE=${LOG_FILE}" | tee -a "$LOG_FILE"
echo "FEATURE_ARGS=${FEATURE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "MoE_Features=${FEATURE_LIST}" | tee -a "$LOG_FILE"
echo "MLA_ARGS=${MLA_ARGS[*]}" | tee -a "$LOG_FILE"
echo "MTP_ARGS=${MTP_ARGS[*]}" | tee -a "$LOG_FILE"
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
    --pp_warmup True \
    --moe_router_force_load_balancing "$LOAD_BALANCE" \
    --optimizer "$OPTIMIZER" \
    --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
    --torch_profiler_use_gzip False \
    "${MLA_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    "${VPP_ARGS[@]}" \
    "${FEATURE_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
