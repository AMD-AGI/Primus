#!/bin/bash
set -euo pipefail

CONTAINER=primus_dev1

docker exec "$CONTAINER" bash -c '
set -euo pipefail

cd /io/Primus


export NNODES=1
export MASTER_PORT=29500
export PAGED_STASH_USE_SDMA=0
export PAGED_STASH_SDMA_STREAMS=2
export PAGED_STASH_SDMA_MAX_CHUNK_BYTES=134217728

export PRIMUS_MODEL=gpt_oss_20B
export EXP=examples/megatron/configs/MI300X/gpt_oss_20B-BF16-pretrain.yaml

export GPU_MAX_HW_QUEUES=8
export HSA_NO_SCRATCH_RECLAIM=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.5
export NCCL_GRAPH_REGISTER=0
export NVTE_CK_USES_BWD_V3=1
export APPLY_ROPE_FUSION=True
export LEGACY_GG=True

export USE_PAGED_STASH=1

SYNCFREE_ARGS=()
SYNCFREE_ARGS+=("--use_turbo_deepep=False")
SYNCFREE_ARGS+=("--turbo_deepep_num_cu=64")
SYNCFREE_ARGS+=("--turbo_deepep_use_comm_stream=True")
SYNCFREE_ARGS+=("--turbo_sync_free_moe_stage=0")
SYNCFREE_ARGS+=("--moe_use_fused_router_with_aux_score=True")
SYNCFREE_ARGS+=("--moe_permute_fusion=True")
SYNCFREE_ARGS+=("--use_turbo_grouped_mlp=True")
SYNCFREE_ARGS+=("--moe_paged_stash=True")
SYNCFREE_ARGS+=("--moe_token_dispatcher_type=flex")
SYNCFREE_ARGS+=("--moe_flex_dispatcher_backend=hybridep")
SYNCFREE_ARGS+=("--moe_hybridep_num_sms=80")
SYNCFREE_ARGS+=("--moe_enable_deepep=False")
SYNCFREE_ARGS+=("--moe_expert_rank_capacity_factor=1.5")
SYNCFREE_ARGS+=("--cuda_graph_impl=local")
SYNCFREE_ARGS+=("--cuda_graph_warmup_steps=3")
SYNCFREE_ARGS+=("--moe_pad_experts_for_cuda_graph_inference=True")

bash examples/run_pretrain.sh \
    --profile=True \
    --use_pytorch_profiler=True \
    --profile_step_start 5 \
    --profile_step_end 6 \
    --disable_profiler_activity_cpu False \
    --manual_gc True \
    --manual_gc_interval 1 \
    --enable_experimental "$APPLY_ROPE_FUSION" \
    --apply_rope_fusion "$APPLY_ROPE_FUSION" \
    --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
    --cross_entropy_fusion_impl "te" \
    --cross_entropy_loss_fusion True \
    --use_precision_aware_optimizer True \
    --main_grads_dtype bf16 \
    --exp_avg_dtype bf16 \
    --exp_avg_sq_dtype bf16 \
    --enable_primus_turbo=True \
    --use_turbo_rms_norm=False \
    --use_turbo_attention=False \
    --use_turbo_parallel_linear=False \
    --no_check_for_nan_in_loss_and_grad=True \
    --check_for_nan_in_loss_and_grad=False \
    "${SYNCFREE_ARGS[@]}"
'
