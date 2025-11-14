#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

######################### Training Docker and Variables #########################
export DOCKER_IMAGE=${DOCKER_IMAGE:="docker.io/rocm/pytorch-training-private:20250929_gfx950_25dot9_rc4"}
# export DOCKER_IMAGE="docker.io/tasimage/primus:pr-280"
export CLEAN_DOCKER_CONTAINER=1

######################### Training Environment Variables #########################
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}

# Set on Primus-Safe Platform
# export MASTER_ADDR=${MASTER_ADDR:-localhost}
# export MASTER_PORT=${MASTER_PORT:-1234}
# export NNODES=${PET_NNODES:-1}
# export NODE_RANK=${PET_NODE_RANK:-0}
# export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Set on AAC14 cluster
export NNODES=4
export USING_AINIC=1
export NCCL_IB_HCA="rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0,rocep25s0,rocep9s0"
export ANP_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/amd-anp-1.1.0-5"
export RCCL_HOME_DIR="/shared/apps/ubuntu/rocm-7.0.1/rccl-drop-2025-08"
export NCCL_SOCKET_IFNAME="enp193s0f1np1"
export GLOO_SOCKET_IFNAME="enp193s0f1np1"

export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
# export USE_ROCM_AITER_ROPE_BACKEND=0

######################### Training Config #########################
MBS=1
GBS=256
SEQ_LENGTH=4096
TP=1
ETP=1
PP=1
VPP=1
EP=8
CP=1
CP_COMM_TYPE="a2a" # p2p, a2a, allgather or a2a+p2p
ENABLE_MTP=False
LOAD_BALANCE=True
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
LEGACY_GG=True
FP8=False # True for fp8, False for bf16
PROFILE=False
DISABLE_CPU_TRACE=True
PROFILE_STEP_START=5
PROFILE_STEP_END=6
TRAIN_ITERS=10

# MoE_Features legend:
# 0 - Baseline (no extra optimization toggles)
# 1 - Turbo attention + grouped GEMM speedups
# 2 - Sync-free MoE (stage 2)
# 3 - DeepEP acceleration
# 4 - 1F1B MoE overlap
# 5 - Zero-bubble pipeline optimizations
# 6 - Arbitrary pipeline partition (8-way custom layout)
# 7 - CPU NUMA binding helper
# MoE_Features=(0 1 2 3 4 5 6 7)
MoE_Features=(0)

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
		FEATURE_ARGS+=("--use_turbo_grouped_mlp" "True")
		;;
	2)
		ensure_primus_turbo
		FEATURE_ARGS+=("--turbo_sync_free_moe_stage" "2")
		;;
	3)
		ensure_primus_turbo
		FEATURE_ARGS+=("--use_turbo_deepep" "True")
		FEATURE_ARGS+=("--turbo_deepep_num_cu" "32")
		FEATURE_ARGS+=("--turbo_deepep_use_comm_stream" "False")
		;;
	4)
		FEATURE_ARGS+=("--overlap_moe_expert_parallel_comm" "True")
		FEATURE_ARGS+=("--patch_moe_overlap" "True")
		FEATURE_ARGS+=("--delay_wgrad_compute" "False")
		FEATURE_ARGS+=("--moe_shared_expert_overlap" "False")
		;;
	5)
		ensure_primus_turbo
		FEATURE_ARGS+=("--use_turbo_row_parallel_linear" "True")
		FEATURE_ARGS+=("--use_turbo_layer_norm_column_parallel_linear" "True")
		FEATURE_ARGS+=("--use_turbo_column_parallel_linear" "True")
		FEATURE_ARGS+=("--num_virtual_stages_per_pipeline_rank" "1")
		FEATURE_ARGS+=("--patch_zero_bubble" "True")
		;;
	6)
		# TODO: need tuning for the pipeline layout pattern
		# FEATURE_ARGS+=" --pipeline_model_parallel_layout 'Et*3|(tt|)*29,m|L'"
		;;
	7)
		# Enable NUMA binding for better memory locality (increase stability for large models)
		export ENABLE_NUMA_BINDING=1
		# export HSA_KERNARG_POOL_SIZE=12582912
		;;
	*) ;;
	esac
done

FEATURE_LIST="${MoE_Features[*]}"
FEATURE_TAG=$(printf "%s" "${FEATURE_LIST}" | tr ' ' '-')

MTP_ARGS=()
if [ "$ENABLE_MTP" = "True" ]; then
	MTP_ARGS+=("--mtp_num_layers" "1")
	MTP_ARGS+=("--mtp_loss_scaling_factor" "0.1")
fi

VPP_ARGS=()
if [ $VPP -gt 1 ]; then
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
PRIMUS_TEAM="date-$(date +%Y%m%d)"
export PRIMUS_TEAM
PRIMUS_USER=user-tas
export PRIMUS_USER
export PRIMUS_EXP_NAME="DeepSeekV3_MI325X_FP8${FP8}_MBS${MBS}_GBS${GBS}_SEQ${SEQ_LENGTH}_TP${TP}_ETP${ETP}_PP${PP}_VPP${VPP}_EP${EP}_CP${CP}_Balance${LOAD_BALANCE}_LegacyGG${LEGACY_GG}_Profile${PROFILE}(${PROFILE_STEP_START}-${PROFILE_STEP_END})_NoCPUTrace${DISABLE_CPU_TRACE}_Features${FEATURE_TAG}"

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export DUMP_PP_DIR=$LOG_DIR/pp_dump
export LOG_FILE=$LOG_DIR/training.log
export EXPORT_CONFIG=$LOG_DIR/config.yaml
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

######################### Training Job #########################
export EXP="examples/megatron/configs/MI300X/deepseek_v3-pretrain.yaml"

echo "--------------------------------" | tee -a "$LOG_FILE"
echo "Begin Training... $(date +%Y%m%d_%H%M%S)" | tee -a "$LOG_FILE"
echo "Training Config: $EXP" | tee -a "$LOG_FILE"
echo "LOG_DIR=${LOG_DIR}" | tee -a "$LOG_FILE"
echo "LOG_FILE=${LOG_FILE}" | tee -a "$LOG_FILE"
echo "FEATURE_ARGS=${FEATURE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "MoE_Features=${FEATURE_LIST}" | tee -a "$LOG_FILE"
echo "MTP_ARGS=${MTP_ARGS[*]}" | tee -a "$LOG_FILE"
echo "VPP_ARGS=${VPP_ARGS[*]}" | tee -a "$LOG_FILE"
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "RECOMPUTE_ARGS=${RECOMPUTE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "PROFILE_ARGS=${PROFILE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

# --num_layers 4 \
# --moe_layer_freq 1 \
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
	--moe_router_force_load_balancing "$LOAD_BALANCE" \
	--pp_warmup True \
	--manual_gc True \
	--manual_gc_interval 1 \
	--optimizer "$OPTIMIZER" \
	--moe_use_legacy_grouped_gemm "$LEGACY_GG" \
	"${VPP_ARGS[@]}" \
	"${FEATURE_ARGS[@]}" \
	"${RECOMPUTE_ARGS[@]}" \
	"${FP8_ARGS[@]}" \
	"${PROFILE_ARGS[@]}" \
	--train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
