#!/bin/bash
export HF_TOKEN=your_token
export GPUS_PER_NODE=1
export SKIP_EMERGING_OPTIMIZERS=1
export USE_ROCM_AITER_ROPE_BACKEND=0
export EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml
# Router group params (must match num_experts)
NUM_EXPERTS=${1:-8}

case $NUM_EXPERTS in
  32) NUM_GROUPS=4; GROUP_TOPK=2; TOPK=8 ;;  # EP8
  16) NUM_GROUPS=4; GROUP_TOPK=2; TOPK=8 ;;  # EP16
   8) NUM_GROUPS=2; GROUP_TOPK=2; TOPK=4 ;;  # EP32
   4) NUM_GROUPS=1; GROUP_TOPK=1; TOPK=4 ;;  # EP64
   *) echo "Unsupported num_experts=$NUM_EXPERTS. Must be one of: 4, 8, 16, 32"; exit 1 ;;
esac

export HIPBLASLT_LOG_MASK=0XFF

for MBS in 1 2 4; do
  echo "=== Running with num_experts=$NUM_EXPERTS mbs=$MBS ==="
  bash examples/run_pretrain.sh --train_iters=3 \
  --micro_batch_size=$MBS --global_batch_size=$MBS \
  --num_layers=2 --recompute_granularity=full --recompute_method=block --recompute_num_layers=10 \
  --apply_rope_fusion=false  --no_persist_layer_norm=true --disable_compile_dependencies=true \
  --mtp_num_layers=0 \
  --expert_model_parallel_size=1 --multi_latent_attention=false --moe_layer_freq="'([0]*1+[1]*1)'" --moe_use_legacy_grouped_gemm=false \
  --num_experts=$NUM_EXPERTS --moe_router_topk=$TOPK --moe_router_num_groups=$NUM_GROUPS --moe_router_group_topk=$GROUP_TOPK \
  > ds_v3_bf16_num_experts_${NUM_EXPERTS}_mbs_${MBS}_topk_${TOPK}_num_groups_${NUM_GROUPS}_group_topk_${GROUP_TOPK}.txt 2>&1
done

