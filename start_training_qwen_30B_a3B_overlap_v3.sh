#!/bin/bash
#
# Qwen3-30B-A3B overlap validation (PP=1, GBS=64 = 8 micro-batches of MBS=8)
#
# Small GBS to keep activation memory manageable without full recompute.
# Memory estimate: ~78GB (model+optim) + 8×2.8GB (activations) ≈ 100GB → fits.
#
# Run SIMULTANEOUSLY with start_training_qwen_30B_a3B_overlap_v3_baseline.sh
# to compare throughput.

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"

export NNODES=1
export TRAIN_ITERS=10
export SLURM_TIME=4:00:00
export SLURM_PARTITION=amd-aig
export SLURM_NODELIST="uswslocpm2m-106-2189"

export USING_AINIC=0   # single-node: all-to-all via local RCCL, no AINIC needed
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MBS=1
export GBS=8    # 8 grad-accum steps at MBS=1; activation memory ≈1/8 of MBS=8
export PRIMUS_PP=1
export PRIMUS_EP=8
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export PRETRAIN_TYPE=BF16
export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-${PRETRAIN_TYPE}-moe_overlap.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document
export PRIMUS_EXP_NAME=qwen30b-overlap-MBS${MBS}

mkdir -p output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
bash ./examples/run_slurm_pretrain.sh \
  --train_iters $TRAIN_ITERS \
  --micro_batch_size $MBS \
  --global_batch_size $GBS \
  --use_turbo_deepep $TURBO_DEEPEEP \
  --moe_use_legacy_grouped_gemm $LEGACY_GG \
  --pipeline_model_parallel_size $PRIMUS_PP \
  --expert_model_parallel_size $PRIMUS_EP \
  --cross_entropy_fusion_impl "te" \
  --cross_entropy_loss_fusion True \
  --disable_last_saving True \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --disable_wandb True \
  --disable_tensorboard True \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
