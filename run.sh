


# bash  primus-cli slurm  --reservation=mi355-gpu-7_gpu-8_gpu-12_gpu-26_reservation  --exclusive --account=odf --mem=0 \
#  -- train pretrain --config examples/megatron/configs/MI355X/qwen3_30B-FP8-pretrain.yaml \
#     --train_iters=5 \
#     --pp_warmup true

# ---- ROCMoE integration env ------------------------------------------------
# Make `import rocmoe` resolvable (built extension staged here; also installed
# editable via RMoE/install.sh --editable).
export PYTHONPATH=/shared/amdgpu/home/xiaoming_peng_qle/workspace/RMoE/build/m32/python:${PYTHONPATH}
# bufferA is only live within each layer's own forward and its own backward
# (the forward X is side-copied into each layer's per-layer residency buffer),
# so sharing the single transport bufferA across the homogeneous MoE layers is
# correct and saves ~(num_layers-1)x its size. Leave sharing ON (the default).
# Per-expert packing capacity factor (higher = fewer dropped tokens, more mem).
export ROCMOE_OVER_PROVISION=1.5
# Reduce torch allocator fragmentation so it can coexist with ROCMoE hipMalloc.
export PYTORCH_ALLOC_CONF=expandable_segments:True

./primus-cli  direct -- train pretrain --config ./examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml \
                     --num_layers 4 \
                     --pipeline_model_parallel_size 1 \
                     --expert_parallel_size 8 \
                     --moe_layer_freq 1 \
                     --micro_batch_size 1 \
                     --recompute_num_layers 0 \
                     --recompute_method block \
                     --recompute_granularity full \
                     --use_rocmoe true \
                     --use_turbo_deepep false \
                     --train_iters 50
