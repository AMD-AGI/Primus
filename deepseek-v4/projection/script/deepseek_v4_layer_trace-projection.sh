#!/bin/bash
###############################################################################
# DeepSeek-V4 single-layer, single-cr profiling launcher for the perf
# *projection* pipeline (deepseek-v4/projection).
#
# Produces ONE chrome trace for ONE compression-ratio (cr) layer type, captured
# under production-representative conditions so the trace can be turned into a
# clean per-module forward/backward breakdown (see design/01-overview.md):
#
#   * seq_length = 4096            (production per-microbatch token count)
#   * num_layers = 1              (clean per-layer attribution; fits memory @4096)
#   * compress_ratios = [CR]      (one cr per trace; CR in {0,4,128})
#   * optimizer = adam, NON-distributed, fp32 states     (dist-opt ON makes the
#                                                         ROCm Kineto profiler
#                                                         drop dense/HCA compute
#                                                         kernels; NOT muon)
#   * overlap_grad_reduce/param_gather = False           (num_layers=1 breaks
#                                                         Megatron's chained
#                                                         param-sync assert; and
#                                                         with overlap off every
#                                                         microbatch's compute is
#                                                         already clean)
#   * GA = 2  => GBS = 2 * DP * MBS                       (min-grouping removes
#                                                         residual jitter)
#   * recompute = OFF             (capture pure fwd / pure bwd)
#   * profiler ON, with_stack=True, window iter 6 -> 7   (kernel -> nn.module)
#
# Usage (inside the training container, repo root):
#   CR=0   ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
#   CR=4   ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
#   CR=128 ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
#   MODEL=flash CR=4 ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
#
# The trace lands under:
#   output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/tensorboard/*.pt.trace.json
###############################################################################
set -euo pipefail
set -x

export HF_TOKEN="${HF_TOKEN:-}"

# ---------- What to profile -------------------------------------------------
export MODEL=${MODEL:-pro}                 # pro | flash
export CR=${CR:-4}                          # 0 | 4 | 128  (single cr per trace)

# ---------- Model: DeepSeek-V4 (pro or flash) ------------------------------
# Widths (hidden/heads/kv) come from the model yaml via PRIMUS_MODEL; we only
# override the MoE / indexer shape knobs the runner exposes, per variant.
export EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml}
case "$MODEL" in
  pro)
    export PRIMUS_MODEL=${PRIMUS_MODEL:-deepseek_v4_pro}
    export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-384}
    export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
    export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-3072}
    export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-1024}
    ;;
  flash)
    export PRIMUS_MODEL=${PRIMUS_MODEL:-deepseek_v4_flash}
    export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-256}
    export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
    export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-2048}
    export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-512}
    ;;
  *)
    echo "[ERROR] MODEL must be 'pro' or 'flash', got '$MODEL'"; exit 1 ;;
esac
# Megatron aux-loss-free expert bias needs sigmoid; V4 uses sqrtsoftplus -> off.
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=${PRIMUS_MOE_ENABLE_EXPERT_BIAS:-False}

# ---------- Single layer per cr (or 3-layer mix) at production seq ---------
# Pure dense (cr=0) / HCA (cr=128) single layers get CUDA-graph/stream-captured
# (compute hidden from the trace). CR=mix runs a 3-layer [0,4,128] block: the
# dynamic CSA (cr=4) layer keeps the block out of graph capture, so all three
# attention types are visible in one trace and split by kernel name.
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
case "$CR" in
  0|4|128) export PRIMUS_COMPRESS_RATIOS="[$CR]"; export PRIMUS_TOTAL_LAYERS=1 ;;
  mix)     export PRIMUS_COMPRESS_RATIOS="[0,4,128]"; export PRIMUS_TOTAL_LAYERS=3 ;;
  *) echo "[ERROR] CR must be 0, 4, 128 or mix, got '$CR'"; exit 1 ;;
esac

# ---------- Single-node EP=8 (intra-node; DP=8) ----------------------------
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}
export MBS=${MBS:-1}
# DP = world/(TP*PP). On one 8-GPU node with TP=PP=1 => DP=8.
export DP=${DP:-8}
# GA = 2 so the schedule is F1 B1 F2 B2: the clean (overlap-free) forward is F2
# and the clean backward is B1. The parser takes the per-kernel min over the two
# launches, recovering the overlap-free time (design/01-overview.md).
export GBS=${GBS:-$((2 * DP * MBS))}

# ---------- Optimizer: adam + distributed optimizer (zero1) ----------------
# Compute (fwd/bwd) is optimizer-independent; we use adam (a) to model the
# zero1 DP-comm overlap that GA=2 isolates, and (b) to dodge Muon's fp32
# optimizer-state memory blow-up so seq=4096 fits. The optimizer step itself is
# modeled separately in the projection (design/04-projection-math.md Step 4).
export OPTIMIZER=${OPTIMIZER:-adam}
# CRITICAL: distributed optimizer (zero1) MUST be off. With it on, the ROCm
# Kineto profiler silently drops the compute GPU kernels for pure dense (cr=0)
# and HCA (cr=128) layers (only optimizer/comm/elementwise get recorded);
# turning it off makes every cr's kernels visible. dist-opt has no bearing on
# the captured fwd/bwd compute, which is what the projection needs (the
# optimizer step is modeled analytically, design/04 Step 4).
export USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-False}
# Overlap OFF. With num_layers=1, Megatron's chained param-gather sync trips an
# assertion (param_and_grad_buffer.start_param_sync). More importantly, with
# overlap off every microbatch's compute is already overlap-free — exactly the
# clean per-kernel time we want; GA=2 + min-grouping then only removes residual
# jitter. (Production DP comm is assumed hidden in the projection anyway; A2.)
export PRIMUS_OVERLAP_GRAD_REDUCE=${PRIMUS_OVERLAP_GRAD_REDUCE:-False}
export PRIMUS_OVERLAP_PARAM_GATHER=${PRIMUS_OVERLAP_PARAM_GATHER:-False}
# Distributed optimizer (zero1) is the default. The Kineto trace drops the
# compute GPU kernels for pure dense(cr=0)/HCA(cr=128) layers ONLY when the
# distributed optimizer is on; turning it off makes all kernels visible (but
# then precision-aware optimizer must be off too -> fp32 optimizer states).
DISTOPT_ARGS=(--use_distributed_optimizer "$USE_DISTRIBUTED_OPTIMIZER")
if [ "$USE_DISTRIBUTED_OPTIMIZER" = "False" ]; then
  DISTOPT_ARGS+=(--use_precision_aware_optimizer False --main_grads_dtype fp32 --exp_avg_dtype fp32 --exp_avg_sq_dtype fp32)
fi

# ---------- Perf knobs (production V4 Triton attn + Turbo MoE) --------------
export ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-True}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-True}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-True}
export USE_V4_TRITON_ATTENTION=${USE_V4_TRITON_ATTENTION:-True}
export USE_V4_TRITON_CSA_ATTENTION=${USE_V4_TRITON_CSA_ATTENTION:-True}
export USE_V4_TILELANG_ATTENTION=${USE_V4_TILELANG_ATTENTION:-False}
export USE_V4_TILELANG_CSA_ATTENTION=${USE_V4_TILELANG_CSA_ATTENTION:-False}
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-False}
export PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=${PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU:-True}

TURBO_DEEPEP_CLI_ARGS=()
if [ "$USE_TURBO_DEEPEP" = "True" ]; then
  export TURBO_DEEPEP_NUM_CU=${TURBO_DEEPEP_NUM_CU:-80}
  export TURBO_DEEPEP_USE_COMM_STREAM=${TURBO_DEEPEP_USE_COMM_STREAM:-False}
  export MOE_ROUTER_DTYPE=${MOE_ROUTER_DTYPE:-fp32}
  export MOE_SHARED_EXPERT_OVERLAP=${MOE_SHARED_EXPERT_OVERLAP:-False}
  TURBO_DEEPEP_CLI_ARGS=(
    --turbo_deepep_num_cu "$TURBO_DEEPEP_NUM_CU"
    --turbo_deepep_use_comm_stream "$TURBO_DEEPEP_USE_COMM_STREAM"
    --moe_router_dtype "$MOE_ROUTER_DTYPE"
    --moe_shared_expert_overlap "$MOE_SHARED_EXPERT_OVERLAP"
  )
fi

export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
# Wider profiler window (active 3 steps) so at least one full training iteration
# (fwd+bwd+opt) is captured cleanly; min-grouping in the parser then picks the
# clean step. A narrow 1-step window occasionally aligned to the optimizer tail
# only (cr0/cr128), missing the compute kernels.
export TRAIN_ITERS=${TRAIN_ITERS:-12}

# ---------- Profiler: trace with python stacks for kernel->module ----------
export PROFILE=True
# Optional GPU-only trace (drop CPU activity). With CPU activity on, pure dense
# (cr=0) / HCA (cr=128) layers' compute kernels can vanish from the trace
# (profiler/stream-capture interaction); GPU-only capture brings them back, at
# the cost of CPU-side module/stack attribution (fall back to kernel-name rules).
PROFILER_ACTIVITY_ARGS=()
if [ "${DISABLE_PROFILER_CPU:-False}" = "True" ]; then
  PROFILER_ACTIVITY_ARGS=(--disable_profiler_activity_cpu True)
fi
export BACKEND_PATH=${BACKEND_PATH:-"$(pwd)/third_party/Megatron-LM"}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-tas-mi355x-$(date +%Y%m%d)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-projection_${MODEL}_cr${CR}_seq${PRIMUS_SEQ_LENGTH}_ep${PRIMUS_EP}}

if [ ! -d "$BACKEND_PATH" ] || [ -z "$(ls -A "$BACKEND_PATH" 2>/dev/null)" ]; then
  echo "[ERROR] BACKEND_PATH does not exist or is empty: $BACKEND_PATH"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

./primus-cli direct \
  -- train pretrain --config "$EXP" \
  --backend_path "$BACKEND_PATH" \
  --num_layers "$PRIMUS_TOTAL_LAYERS" \
  --train_iters "$TRAIN_ITERS" \
  --lr_warmup_iters 0 \
  --lr_decay_iters "$TRAIN_ITERS" \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --seq_length "$PRIMUS_SEQ_LENGTH" \
  --max_position_embeddings "$PRIMUS_MAX_POSITION_EMBEDDINGS" \
  --rope_type rope \
  --tensor_model_parallel_size "$PRIMUS_TP" \
  --pipeline_model_parallel_size "$PRIMUS_PP" \
  --expert_model_parallel_size "$PRIMUS_EP" \
  --num_experts "$PRIMUS_NUM_EXPERTS" \
  --moe_router_topk "$PRIMUS_MOE_TOPK" \
  --moe_router_enable_expert_bias "$PRIMUS_MOE_ENABLE_EXPERT_BIAS" \
  --moe_ffn_hidden_size "$PRIMUS_MOE_FFN_HIDDEN_SIZE" \
  --index_topk "$PRIMUS_INDEX_TOPK" \
  --v4_grouped_experts_support_clamped_swiglu "$PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU" \
  --compress_ratios "$PRIMUS_COMPRESS_RATIOS" \
  --mtp_num_layers 0 \
  --mock_data True \
  --optimizer "$OPTIMIZER" \
  "${DISTOPT_ARGS[@]}" \
  --enable_primus_turbo "$ENABLE_PRIMUS_TURBO" \
  --use_turbo_attention "$USE_TURBO_ATTENTION" \
  --use_v4_triton_attention "$USE_V4_TRITON_ATTENTION" \
  --use_v4_triton_csa_attention "$USE_V4_TRITON_CSA_ATTENTION" \
  --use_v4_tilelang_attention "$USE_V4_TILELANG_ATTENTION" \
  --use_v4_tilelang_csa_attention "$USE_V4_TILELANG_CSA_ATTENTION" \
  --use_v4_compiled_sinkhorn "$USE_V4_COMPILED_SINKHORN" \
  --use_turbo_deepep "$USE_TURBO_DEEPEP" \
  "${TURBO_DEEPEP_CLI_ARGS[@]}" \
  --use_turbo_grouped_mlp "$TURBO_USE_GROUPED_MLP" \
  --moe_use_legacy_grouped_gemm False \
  --fp8 null \
  --fp8_recipe null \
  --recompute_num_layers 0 \
  --overlap_grad_reduce "$PRIMUS_OVERLAP_GRAD_REDUCE" \
  --overlap_param_gather "$PRIMUS_OVERLAP_PARAM_GATHER" \
  --disable_last_saving True \
  --disable_wandb True \
  --disable_tensorboard False \
  --profile True \
  --use_pytorch_profiler True \
  "${PROFILER_ACTIVITY_ARGS[@]}" \
  --profile_step_start 6 \
  --profile_step_end 9 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK:-0}.txt"
