#!/bin/bash
#
# Validation script: MoE EP A2A Cross-Micro-Batch Overlap for Qwen3-30B-A3B (PP=1)
#
# Design:
#   Enables overlap_moe_expert_parallel_comm to run forward of micro-batch N+1
#   in parallel with backward of micro-batch N at layer granularity:
#
#   comm_stream: combine_bwd(mb_n)  | dispatch_fwd(mb_n+1)->dispatch_bwd(mb_n) | combine_fwd(mb_n+1)
#   comp_stream: attn_fwd(mb_n+1)  | mlp_bwd->dw(mb_n)->mlp_fwd(mb_n+1)      | attn_bwd(mb_n)
#
# Usage:
#   Compare throughput (tokens/sec) vs start_training_qwen_30B_a3B.sh baseline.

export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_api_key"
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-563-ainic"
export NNODES=1
export TRAIN_ITERS=10
export SLURM_TIME=48:00:00
export SLURM_PARTITION=amd-aig-2
export SLURM_NODELIST="uswslocpm2m-106-1962"

export USING_AINIC=1
export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
export GLOO_SOCKET_IFNAME=ens9np0
export NCCL_SOCKET_IFNAME=ens9np0
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export GPU_MAX_HW_QUEUES=4
export CLEAN_DOCKER_CONTAINER=1

export MBS=8
export GBS=$((512 * NNODES))
export PRIMUS_RECOMPUTE_LAYERS=5
export PRIMUS_PP=1
export PRIMUS_EP=8
export PRIMUS_VPP=1
export PROFILE=False
export TURBO_DEEPEEP=True
export LEGACY_GG=True
export PRIMUS_DETERMINISTIC=0

export PRETRAIN_TYPE=BF16
export EXP=examples/megatron/configs/MI355X/qwen3_30B_A3B-${PRETRAIN_TYPE}-pretrain.yaml
export PRIMUS_TEAM=amd
export PRIMUS_USER=tas
export PRIMUS_TOKENIZED_DATA_PATH=/shared_aig/c4/tokenized/c4_en_train_text_document

# ── MoE F/B overlap flags ──────────────────────────────────────────────────
# overlap_moe_expert_parallel_comm: Enable layer-granularity F/B interleaving.
#   Megatron's combined_1f1b_schedule_for_no_pipelining will run F(mb N) and
#   B(mb N-1) in one combined call, overlapping A2A comms with attention/expert
#   compute on separate CUDA streams.
export MOE_OVERLAP=True
# patch_moe_overlap=False: use native Megatron TransformerModelChunkSchedulePlan
#   (requires no WeightGradStore changes; simpler validation path).
export PATCH_MOE_OVERLAP=False
# delay_wgrad_compute=False: ensure weight gradients are computed in-place
#   during the overlap window rather than deferred.
export DELAY_WGRAD=False
# ──────────────────────────────────────────────────────────────────────────

export PRIMUS_EXP_NAME=qwen3_30B_A3B-${PRETRAIN_TYPE}-moe_overlap_${MOE_OVERLAP}-patch_${PATCH_MOE_OVERLAP}-node_${NNODES}-mbs_${MBS}-gbs_${GBS}-EP_${PRIMUS_EP}

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
  --recompute_num_layers $PRIMUS_RECOMPUTE_LAYERS \
  --recompute_granularity full \
  --recompute_method block \
  --disable_last_saving True \
  --mock_data True \
  --manual_gc True \
  --manual_gc_interval 1 \
  --mtp_num_layers 0 \
  --overlap_moe_expert_parallel_comm $MOE_OVERLAP \
  --patch_moe_overlap $PATCH_MOE_OVERLAP \
  --delay_wgrad_compute $DELAY_WGRAD \
  --profile $PROFILE \
  --use_pytorch_profiler $PROFILE \
  --profile_step_end 7 \
  --profile_step_start 6 \
  --disable_wandb True \
  --disable_tensorboard True \
  2>&1 | tee output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log.txt
