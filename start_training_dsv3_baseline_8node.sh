#!/bin/bash
#
# DSv3 8-node BASELINE (production config: full recompute, no overlap)
# Mirrors start_training_dsv3.sh but uses primus-cli slurm for scheduling.
#
# Reference: comment in start_training_dsv3_moe_overlap_v2.sh states
#   "~537 TFLOP/s/GPU, ~2152 tokens/GPU/s (iter 5-7)" at 8 nodes, PP=8, EP=8
#
# Compare against: start_training_dsv3_moe_overlap_v3.sh
#   Only change: overlap_moe_expert_parallel_comm=True + selective recompute

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"

export NNODES=8
export SLURM_TIME=4:00:00
export SLURM_PARTITION=amd-aig
export SLURM_NODELIST="uswslocpm2m-106-[2266-2272,2274]"
export TRAIN_ITERS=20

export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0

export MBS=1   # selective recompute needs less memory than full, but requires MBS=1 vs MBS=2
export PRIMUS_SEQ_LENGTH=2048  # reduced from 4096 to give headroom past iteration 1 (98%+ memory at 4096)
export GBS=$((128 * NNODES))   # 1024
export PRIMUS_TOTAL_LAYERS=48  # reduced from 61 to match overlap run for fair comparison
export PRIMUS_MOE_LAYER_FREQ=1
export PRIMUS_EP=8
export PRIMUS_PP=8
export PRIMUS_VPP=2
export PRIMUS_RECOMPUTE_LAYERS=4

export TURBO_ATTENTION=True
export TURBO_DEEPEEP=False
export LEGACY_GG=True           # use legacy grouped GEMM (verified working)
export TURBO_GROUPED_MLP=True
export APPLY_ROPE_FUSION=True
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PRIMUS_TURBO_DEEPEP_TIMEOUT=600
export ENABLE_NUMA_BINDING=1
export HSA_KERNARG_POOL_SIZE=12582912
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TURBO_DEEPEP_NUM_CU=64
# STAGE = PP*VPP = 16 → use 16-stage layout
FEATURE_ARGS=("--turbo_deepep_num_cu" "$TURBO_DEEPEP_NUM_CU")
FEATURE_ARGS+=("--pipeline_model_parallel_layout" "'Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3,L'")

export PRETRAIN_TYPE=FP8
export EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-moe_overlap.yaml
export PRIMUS_TEAM=amd
PRIMUS_USER="tas-$(date +%Y%m%d)"
export PRIMUS_USER
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document
export PRIMUS_EXP_NAME=dsv3-no_overlap_baseline-${PRETRAIN_TYPE}-PP${PRIMUS_PP}-EP${PRIMUS_EP}

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
./primus-cli slurm -N $NNODES --nodelist "$SLURM_NODELIST" \
   -- --image "docker.io/tasimage/primus:pr-609-ainic" --clean \
  -- --numa -- train pretrain --config "$EXP" \
  --num_layers $PRIMUS_TOTAL_LAYERS \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --use_turbo_attention "$TURBO_ATTENTION" \
  --use_turbo_deepep "$TURBO_DEEPEEP" \
  --use_turbo_grouped_mlp "$TURBO_GROUPED_MLP" \
  --lr 2.2e-4 \
  --min_lr 2.2e-5 \
  --lr_warmup_iters 200 \
  --lr_decay_iters 5000 \
  --lr_decay_style cosine \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --enable_experimental $APPLY_ROPE_FUSION \
  --apply_rope_fusion $APPLY_ROPE_FUSION \
  --pipeline_model_parallel_size $PRIMUS_PP \
  --expert_model_parallel_size $PRIMUS_EP \
  --virtual_pipeline_model_parallel_size $PRIMUS_VPP \
  "${FEATURE_ARGS[@]}" \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --recompute_granularity selective \
  --disable_last_saving True \
  --moe_layer_freq $PRIMUS_MOE_LAYER_FREQ \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --pp_warmup True \
  --mtp_num_layers 0 \
  --moe_token_dispatcher_type alltoall \
  --overlap_moe_expert_parallel_comm False \
  --disable_wandb True \
  --disable_tensorboard True \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt"
