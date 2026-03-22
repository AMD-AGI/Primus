#!/bin/bash
#
# Qwen3-30B-A3B baseline with selective recompute (same config as overlap_v2 EXCEPT no overlap)
# Use this to compare against start_training_qwen_30B_a3B_overlap_v2.sh

export HF_TOKEN="${HF_TOKEN:-'your_hf_token'}"
export WANDB_API_KEY="${WANDB_API_KEY:-'your_wandb_api_key'}"
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"

export NNODES=1
export TRAIN_ITERS=10
export SLURM_TIME=4:00:00
export SLURM_PARTITION=amd-aig
export SLURM_NODELIST="uswslocpm2m-106-2177"

export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4

export MBS=8
export GBS=512
export PRIMUS_PP=1
export PRIMUS_EP=8
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export PRETRAIN_TYPE=BF16
# Use same YAML as overlap but without overlap_moe_expert_parallel_comm
export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-${PRETRAIN_TYPE}-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document
export PRIMUS_EXP_NAME=qwen3_30B_A3B-baseline_selective-${PRETRAIN_TYPE}-PP${PRIMUS_PP}-EP${PRIMUS_EP}

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
  --recompute_granularity selective \
  --disable_last_saving True \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --mtp_num_layers 0 \
  --disable_wandb True \
  --disable_tensorboard True \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
