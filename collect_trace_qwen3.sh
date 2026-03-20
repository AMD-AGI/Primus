#!/bin/bash
export HF_TOKEN=your_token
export GPUS_PER_NODE=1
export SKIP_EMERGING_OPTIMIZERS=1
export USE_ROCM_AITER_ROPE_BACKEND=0
export EXP=examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-proxy.yaml
# Qwen3 uses simple top-k routing (no group routing)
NUM_EXPERTS=${1:-16}

case $NUM_EXPERTS in
  16) TOPK=8 ;;  # EP8
   8) TOPK=8 ;;  # EP16
   4) TOPK=4 ;;  # EP32
   2) TOPK=2 ;;  # EP64
   *) echo "Unsupported num_experts=$NUM_EXPERTS. Must be one of: 2, 4, 8, 16"; exit 1 ;;
esac

export HIPBLASLT_LOG_MASK=0XFF

for MBS in 1 2 4; do
  echo "=== Running with num_experts=$NUM_EXPERTS mbs=$MBS ==="
  bash examples/run_pretrain.sh --train_iters=3 \
  --micro_batch_size=$MBS --global_batch_size=$MBS \
  --num_layers=1 --recompute_granularity=full --recompute_method=block --recompute_num_layers=10 \
  --apply_rope_fusion=false  --no_persist_layer_norm=true --disable_compile_dependencies=true \
  --expert_model_parallel_size=1 --multi_latent_attention=false --moe_layer_freq="1" --moe_use_legacy_grouped_gemm=false \
  --num_experts=$NUM_EXPERTS --moe_router_topk=$TOPK \
  > qwen3_bf16_num_experts_${NUM_EXPERTS}_mbs_${MBS}_topk_${TOPK}.txt 2>&1
done

