#!/bin/bash

set -x

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"  # make it your own hf token
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"  # make it your own wandb api key

export NNODES=${NNODES:-32}
export SLURM_TIME=48:00:00
export SLURM_PARTITION=amd-aig
export SLURM_NODELIST="uswslocpm2m-106-[030-031,038-039,050,063,069,225,942,1531-1532,1536,1547,1549,1554,1556-1557,1561,1579,1583,1585,1588,1592,1596,1606,1627-1629,1650,1659-1660,1678]"
export SLURM_NODELIST="uswslocpm2m-106-[030-031,038-039,050,063,069,225,942,1531-1532,1536,1547,1549,1554,1556]" # -1557,1561,1579,1583,1585,1588,1592,1596,1606,1627-1629,1650,1659-1660,1678]"
export SLURM_NODELIST="uswslocpm2m-106-[030-031,038-039,050,063,069,225,942,1531-1532,1536,-1557,1561,1579,1583]" #1585,1588,1592,1596,1606,1627-1629,1650,1659-1660,1678]"
export SLURM_NODELIST="uswslocpm2m-106-[030-031,038-039,050,063,069,225,942,1531-1532,1536,1547,1549,1554,1556-1557,1561,1579,1583,1585,1588,1592,1596,1606,1627-1629,1683-1684,1691,1697]"

export TRAIN_ITERS=${TRAIN_ITERS:-10}

# export NCCL_DEBUG=INFO
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA="${NCCL_IB_HCA-ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens9np0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens9np0}"

export MBS=${MBS:-2}
export GBS=$((128 * NNODES))
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-61}
export PRIMUS_MOE_LAYER_FREQ=${PRIMUS_MOE_LAYER_FREQ:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_PP=${PRIMUS_PP:-8}
export PRIMUS_VPP=${PRIMUS_VPP:-2}
export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-2}

export PROFILE=False
export TURBO_ATTENTION=${TURBO_ATTENTION:-False}
export TURBO_DEEPEEP=${TURBO_DEEPEEP:-True}
export LEGACY_GG=${LEGACY_GG:-True}
export TURBO_GROUPED_MLP=${TURBO_GROUPED_MLP:-False}
export TURBO_RMS_NORM=${TURBO_RMS_NORM:-False}
export TURBO_FLOWMOE=${TURBO_FLOWMOE:-False}
export TURBO_FLOWMOE_ENABLE_CHUNKING=${TURBO_FLOWMOE_ENABLE_CHUNKING:-True}
export TURBO_FLOWMOE_CHUNK_TOKENS=${TURBO_FLOWMOE_CHUNK_TOKENS:-7168}
export TURBO_FLOWMOE_MIN_CHUNK_TOKENS=${TURBO_FLOWMOE_MIN_CHUNK_TOKENS:-16384}
export TURBO_FLOWMOE_RECOMPUTE_MODE=${TURBO_FLOWMOE_RECOMPUTE_MODE:-none}
export TURBO_FLOWMOE_PREFETCH_DISPATCH=${TURBO_FLOWMOE_PREFETCH_DISPATCH:-False}
export TURBO_FLOWMOE_PREFETCH_COMBINE=${TURBO_FLOWMOE_PREFETCH_COMBINE:-False}
export TURBO_FLOWMOE_DEBUG=${TURBO_FLOWMOE_DEBUG:-False}
export TURBO_FLOWMOE_LOG_INTERVAL=${TURBO_FLOWMOE_LOG_INTERVAL:-100}
export USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-True}
export MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-bf16}
export EXP_AVG_DTYPE=${EXP_AVG_DTYPE:-bf16}
export EXP_AVG_SQ_DTYPE=${EXP_AVG_SQ_DTYPE:-bf16}
export PP_WARMUP=${PP_WARMUP:-False}
export APPLY_ROPE_FUSION=True
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
export PRIMUS_TURBO_AUTO_TUNE=${PRIMUS_TURBO_AUTO_TUNE:-0}


# Enable NUMA binding for better memory locality (increase stability for large models)
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912

STAGE=$(( PRIMUS_PP *  PRIMUS_VPP))
FEATURE_ARGS=()
case $STAGE in
  1)
    # PP=1 does not need explicit pipeline layout.
    ;;
  8)
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*7|t*8|t*8|t*8|t*8|t*8|t*7|t*7,L'")
    ;;
  16)
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*2,L'")
    ;;
  32)
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*1,L'")
    ;;
  *)
    echo "Unsupported STAGE=${STAGE} (PRIMUS_PP=${PRIMUS_PP}, PRIMUS_VPP=${PRIMUS_VPP}). Supported stages: 8, 16, 32." >&2
    exit 1
    ;;
esac

# 32N best recompute config
# RECOMP_IDS="0,3,4,7,8,11,12,15,19,23,31,32,35,36,39,40,43,44,47,48,51"

MOE_RECOMP_ARGS=()
if [ "$TURBO_FLOWMOE_RECOMPUTE_MODE" = "expert_only" ]; then
  # MemFine-style selective recompute: disable global block recompute and only
  # enable MoE layer recompute in Megatron.
  RECOMP_ARGS=(--recompute_num_layers 0)
  MOE_RECOMP_ARGS=(--moe_layer_recompute True)
elif [ -n "$RECOMP_IDS" ]; then
  export RECOMP_IDS
  RECOMP_ARGS=(--recompute_layer_ids "$RECOMP_IDS" --recompute_granularity full)
else
  RECOMP_ARGS=(--recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" --recompute_granularity full --recompute_method block)
fi

export PRETRAIN_TYPE=${PRETRAIN_TYPE:-BF16}

export EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-pretrain.yaml
PRIMUS_TEAM="amd-$(date +%Y%m%d)"
export PRIMUS_TEAM

PRIMUS_USER="${WORKLOAD_ID}"
export PRIMUS_USER
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document # this is the tokenized data path for the training
export PRIMUS_EXP_NAME=dsv3-pretrain-nnodes_$NNODES-mbs_$MBS-gbs_$GBS-PP_$PRIMUS_PP-EP_$PRIMUS_EP-VPP_$PRIMUS_VPP-turbodeepep_$TURBO_DEEPEEP-legacygg_$LEGACY_GG-turbogg_$TURBO_GROUPED_MLP-turboattn_$TURBO_ATTENTION-ropefusion_$APPLY_ROPE_FUSION-profile_$PROFILE
export PRIMUS_EXP_NAME=debug_dsv3-type_$PRETRAIN_TYPE-legacygg_$LEGACY_GG-turbogg_$TURBO_GROUPED_MLP-turbodeepep_$TURBO_DEEPEEP-turboattn_$TURBO_ATTENTION-autotune_$PRIMUS_TURBO_AUTO_TUNE

if [ -n "$DUMP_PP_DATA" ]; then
  export DUMP_PP_DIR=output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/pp_data
  DUMP_PP_ARGS=(--dump_pp_data True)
else
  DUMP_PP_ARGS=(--dump_pp_data False)
fi

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
# ./primus-cli slurm -N $NNODES \
#   ${SLURM_TIME:+--time="${SLURM_TIME}"} \
#   ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
#   ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
#   -- --image "docker.io/tasimage/primus:pr-609-ainic" --clean -- --numa \

./primus-cli direct --numa \
  -- train pretrain --config "$EXP" \
  --num_layers "$PRIMUS_TOTAL_LAYERS" \
  --train_iters "$TRAIN_ITERS" \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --use_turbo_attention "$TURBO_ATTENTION" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --use_turbo_grouped_mlp "$TURBO_GROUPED_MLP" \
  --use_turbo_rms_norm "$TURBO_RMS_NORM" \
  --use_turbo_flowmoe "$TURBO_FLOWMOE" \
  --turbo_flowmoe_enable_chunking "$TURBO_FLOWMOE_ENABLE_CHUNKING" \
  --turbo_flowmoe_chunk_tokens "$TURBO_FLOWMOE_CHUNK_TOKENS" \
  --turbo_flowmoe_min_chunk_tokens "$TURBO_FLOWMOE_MIN_CHUNK_TOKENS" \
  --turbo_flowmoe_recompute_mode "$TURBO_FLOWMOE_RECOMPUTE_MODE" \
  --turbo_flowmoe_prefetch_dispatch "$TURBO_FLOWMOE_PREFETCH_DISPATCH" \
  --turbo_flowmoe_prefetch_combine "$TURBO_FLOWMOE_PREFETCH_COMBINE" \
  --turbo_flowmoe_debug "$TURBO_FLOWMOE_DEBUG" \
  --turbo_flowmoe_log_interval "$TURBO_FLOWMOE_LOG_INTERVAL" \
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
  "${MOE_RECOMP_ARGS[@]}" \
  "${DUMP_PP_ARGS[@]}" \
  --disable_last_saving True \
  --moe_layer_freq "$PRIMUS_MOE_LAYER_FREQ" \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --pp_warmup "$PP_WARMUP"  \
  --mtp_num_layers 0 \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb True \
  --disable_tensorboard True \
  --turbo_deepep_num_cu 80 \
  --use_precision_aware_optimizer "$USE_PRECISION_AWARE_OPTIMIZER" \
  --main_grads_dtype "$MAIN_GRADS_DTYPE" \
  --exp_avg_dtype "$EXP_AVG_DTYPE" \
  --exp_avg_sq_dtype "$EXP_AVG_SQ_DTYPE" \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK}.txt"
