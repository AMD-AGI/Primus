#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -x

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key

export NNODES=${NNODES:-32}

export TRAIN_ITERS=${TRAIN_ITERS:-10}

# Interconnect. The HCA list and the socket interface are site-specific; these
# defaults are for the AINIC nodes this recipe was tuned on. A cluster whose
# front-end NIC is named differently (ens3 on Crusoe) must override both socket
# variables or rendezvous hangs.
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1}"
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens9np0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens9np0}
# base_env.sh defaults this to 3 and runs before the AINIC hook, so the hook's
# own default of 1 never applies. On ionic, GID 3 makes ibv_modify_qp fail with
# "No data available" the first time a QP goes INIT -> RTR.
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}

export MBS=${MBS:-2}
export GBS=${GBS:-$((128 * NNODES))}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-61}
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_PP=${PRIMUS_PP:-16}
export PRIMUS_VPP=${PRIMUS_VPP:-2}
export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-2}

# Mock data and the GC cadence this recipe was tuned with; override to train on
# real data or to collect fewer GC pauses.
export MOCK_DATA=${MOCK_DATA:-True}
export MANUAL_GC=${MANUAL_GC:-True}
export MANUAL_GC_INTERVAL=${MANUAL_GC_INTERVAL:-1}

export PROFILE=${PROFILE:-False}
export TURBO_ATTENTION=${TURBO_ATTENTION:-False}
export TURBO_DEEPEEP=${TURBO_DEEPEEP:-True}
export LEGACY_GG=${LEGACY_GG:-True}
export TURBO_GROUPED_GEMM=${TURBO_GROUPED_GEMM:-False}
export TURBO_RMS_NORM=${TURBO_RMS_NORM:-True}
export APPLY_ROPE_FUSION=True
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
export PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-0}


# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

STAGE=$((PRIMUS_PP * PRIMUS_VPP))
FEATURE_ARGS=()
# The layout value carries no surrounding quotes: overrides are taken literally,
# so a quoted string reaches Megatron's layout parser with the quotes still
# attached and it rejects ' as a layer character.
case $STAGE in
  1)
    # PP=1: there is nothing to lay out, every layer lives on the one stage.
    # This is the shape small-scale bring-up runs need, because EP is bounded by
    # DP = world/(TP*PP) and EP=8 on a single node only works at PP=1.
    # The experiment yaml hardcodes the 32-stage layout, so leaving the flag off
    # is not the same as having no layout: it has to be cleared, or Megatron
    # applies a 61-layer 32-stage plan to a one-stage run.
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "None")
    ;;
  8)
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "Et*7|t*8|t*8|t*8|t*8|t*8|t*7|t*7,L")
    ;;
  16)
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*2,L")
    ;;
  32)
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "Et*1|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L")
    ;;
  *)
    echo "Unsupported STAGE=${STAGE} (PRIMUS_PP=${PRIMUS_PP}, PRIMUS_VPP=${PRIMUS_VPP}). Supported stages: 1, 8, 16, 32." >&2
    exit 1
    ;;
esac

# The layout alone already puts Megatron on the interleaved schedule (it derives
# VPP from stage count / PP), but args.virtual_pipeline_model_parallel_size stays
# None unless it is passed, and validate_args then takes its "not interleaved"
# branch and force-disables overlap_p2p_comm and align_param_gather -- both of
# which trainer_base.yaml asks for. Losing align_param_gather is what lets
# overlap_param_gather hand a chunk's forward parameters whose all-gather has not
# landed: training dies a few steps in with "found NaN in local grad norm for
# bucket #0", on a different rank every run.
[ "$PRIMUS_VPP" -gt 1 ] && FEATURE_ARGS+=(--virtual_pipeline_model_parallel_size "$PRIMUS_VPP")

# Best recompute config for EP8_PP16_VPP2
# 32N
# RECOMP_IDS="0,1,2,4,6,8,10,12,14,16,34,36,38,40,50"
# 64N
# RECOMP_IDS="0,1,2,4,6,8,10,12,14,16,34,36"
# 128N
# RECOMP_IDS="0,1,2,4,6,8,10,12,14"

if [ -n "$RECOMP_IDS" ]; then
  export RECOMP_IDS
  RECOMP_ARGS=(--recompute_layer_ids "$RECOMP_IDS" --recompute_granularity full)
else
  # The experiment yaml ships a recompute_layer_ids list for the 61-layer recipe.
  # It is mutually exclusive with recompute_method and its indices reach 51, so
  # it has to be cleared or validation rejects the run before step 1.
  RECOMP_ARGS=(--recompute_layer_ids None --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" --recompute_granularity full --recompute_method block)
fi

export PRETRAIN_TYPE=${PRETRAIN_TYPE:-BF16}

export EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-pretrain.yaml
PRIMUS_TEAM="amd-$(date +%Y%m%d)"
export PRIMUS_TEAM

PRIMUS_USER="${WORKLOAD_ID:-tas}"
export PRIMUS_USER
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=dsv3-type_$PRETRAIN_TYPE-legacygg_$LEGACY_GG-turbogg_$TURBO_GROUPED_GEMM-turbodeepep_$TURBO_DEEPEEP-turboattn_$TURBO_ATTENTION-autotune_$PRIMUS_TURBO_AUTO_TUNE

if [ -n "$DUMP_PP_DATA" ]; then
  export DUMP_PP_DIR=output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/pp_data
  DUMP_PP_ARGS=(--dump_pp_data True)
else
  DUMP_PP_ARGS=(--dump_pp_data False)
fi

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

# Launcher: `direct` runs here, inside a container that is already up on every
# node; `slurm` allocates the nodes and starts the container itself, which is
# how this is submitted from a login node.
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-direct}
if [ "$PRIMUS_LAUNCHER" = "slurm" ]; then
  : "${DOCKER_IMAGE:?PRIMUS_LAUNCHER=slurm needs DOCKER_IMAGE}"
  export DOCKER_IMAGE
  LAUNCHER_ARGS=(slurm "${SLURM_LAUNCH_CMD:-sbatch}" -N "$NNODES")
  # Spur hands out GPUs only when they are asked for. --exclusive reserves the
  # node but leaves TresPerNode empty, and a node with no GPUs assigned fails to
  # confirm dispatch: the job dies at submit with "dispatch confirmation failed:
  # N of M nodes confirmed", then requeues until it is held. Single-node runs can
  # slip through, which makes this look like a flaky cluster rather than a
  # missing flag.
  LAUNCHER_ARGS+=(--gpus-per-node="${GPUS_PER_NODE:-8}")
  [ -n "${SLURM_TIME:-}" ] && LAUNCHER_ARGS+=(--time="${SLURM_TIME}")
  [ -n "${SLURM_PARTITION:-}" ] && LAUNCHER_ARGS+=(--partition="${SLURM_PARTITION}")
  [ -n "${SLURM_NODELIST:-}" ] && LAUNCHER_ARGS+=(--nodelist="${SLURM_NODELIST}")
  [ -n "${SLURM_EXCLUDE:-}" ] && LAUNCHER_ARGS+=(--exclude="${SLURM_EXCLUDE}")
  [ -n "${SLURM_QOS:-}" ] && LAUNCHER_ARGS+=(--qos="${SLURM_QOS}")
  [ -n "${SLURM_ACCOUNT:-}" ] && LAUNCHER_ARGS+=(--account="${SLURM_ACCOUNT}")
  if [ "${SLURM_LAUNCH_CMD:-sbatch}" = "sbatch" ]; then
    [ "${SLURM_EXCLUSIVE:-1}" != "0" ] && LAUNCHER_ARGS+=(--exclusive)
    # sbatch returns as soon as the job is queued, so the per-node log cannot be
    # tee'd here. %N keeps each node in its own file; a single path would have
    # every node truncate the same file and only the last writer survives.
    export SBATCH_OUTPUT="${SBATCH_OUTPUT:-output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/train_%N.log}"
    export SBATCH_ERROR="${SBATCH_ERROR:-output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/train_%N.err}"
  fi
  # Each patch self-skips (exit 2) when its PRIMUS_* env gate is unset.
  LAUNCHER_ARGS+=(-- --image "${DOCKER_IMAGE}" --clean -- --numa
    --patch runner/helpers/patches/10_fix_libionic_abi4.sh
    --patch runner/helpers/patches/11_fix_lld_stub.sh)
else
  LAUNCHER_ARGS=(direct --numa)
fi

./primus-cli "${LAUNCHER_ARGS[@]}" \
  -- train pretrain --config "$EXP" \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --use_turbo_attention "$TURBO_ATTENTION" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --use_turbo_grouped_gemm "$TURBO_GROUPED_GEMM" \
  --use_turbo_rms_norm "$TURBO_RMS_NORM" \
  --lr 2.2e-4 \
  --min_lr 2.2e-5 \
  --lr_warmup_iters 200 \
  --lr_decay_iters 5000 \
  --lr_decay_style cosine \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --enable_experimental "$APPLY_ROPE_FUSION" \
  --apply_rope_fusion "$APPLY_ROPE_FUSION" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  "${FEATURE_ARGS[@]}" \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  "${RECOMP_ARGS[@]}" \
  "${DUMP_PP_ARGS[@]}" \
  --disable_last_saving True \
  --moe_layer_freq "$PRIMUS_MOE_LAYER_FREQ" \
  --mock_data "$MOCK_DATA" \
  --manual_gc "$MANUAL_GC" \
  --manual_gc_interval "$MANUAL_GC_INTERVAL" \
  --pp_warmup True  \
  --mtp_num_layers 0 \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb True \
  --disable_tensorboard True \
  --turbo_deepep_num_cu 80 \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
