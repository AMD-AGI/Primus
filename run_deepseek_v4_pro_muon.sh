#!/bin/bash
###############################################################################
# DeepSeek-V4 *Pro* single-node bring-up with the Muon optimizer.
#
# Follows the DeepSeek-V4 paper Pro recipe (§4.2.1 architecture + §4.2.2
# training setup) as closely as a single 8x288GB node allows:
#   1. Model      : deepseek_v4_pro  (61L / d7168 / 384 experts — paper §4.2.1).
#                   Selected via PRIMUS_MODEL, consumed by the
#                   `model: ${PRIMUS_MODEL:...}.yaml` line in the EXP yaml.
#                   Widths come from primus/configs/models/megatron/
#                   deepseek_v4_pro.yaml; we only override the shape knobs the
#                   runner exposes (layers/experts/topk/ffn/index_topk/
#                   compress_ratios) so the runner's smoke defaults don't win.
#   2. Reduced    : 61 layers + seq 4096 do NOT fit one node, so cut depth
#      to fit       (PRIMUS_TOTAL_LAYERS) and seq (PRIMUS_SEQ_LENGTH); full/
#                   uniform recompute on. (CSA-layer gather scales with seq.)
#   3. Optimizer  : Muon (paper §4.2.2: Muon for matrices, AdamW for embedding/
#                   pred-head/RMSNorm — in-tree ChainedOptimizer). Plain `muon`
#                   requires overlap_{grad_reduce,param_gather}=False and
#                   use_distributed_optimizer=False (Megatron asserts).
#   4. Training   : paper §4.2.2 hyperparameters — momentum 0.95, update-RMS
#      setup        0.18 (muon_extra_scale_factor), AdamW eps 1e-20, LR
#                   2.0e-4→2.0e-5, balance-loss 1e-4, and (crucially) a LARGE
#                   batch via gradient accumulation toward the paper's 94.4M
#                   tokens/step. The batch is what amortizes the fixed Muon
#                   Newton-Schulz cost: at GBS=8 (accum 1) NS looks like ~97%
#                   of GEMM (a starved-batch artifact, NOT a Muon bug); at the
#                   paper batch it falls to the reported ~1-3%.
#
# Single-node integration gaps vs the paper (not config — Primus V4 TODO):
#   - MTP depth 1 (MultiTokenPredictionLayer unsupported) -> MTP_NUM_LAYERS=0
#   - expert-bias/noaux_tc (Megatron needs sigmoid; V4 uses sqrtsoftplus) -> off
#   - muon_weight_decay (0.01 vs paper 0.1) / mtp_loss_scaling are yaml-only.
#
# Usage:
#   # paper-faithful single-node run (validated: ~890 TFLOP/s/GPU, Muon ~1% GPU):
#   PRIMUS_TOTAL_LAYERS=2 PRIMUS_COMPRESS_RATIOS="[128,128]" \
#       PRIMUS_SEQ_LENGTH=4096 GBS=256 ./run_deepseek_v4_pro_muon.sh
#   PRIMUS_TOTAL_LAYERS=4 PRIMUS_SEQ_LENGTH=512 GBS=8 \
#       ./run_deepseek_v4_pro_muon.sh                              # cheap validation
#   OPTIMIZER=adam ./run_deepseek_v4_pro_muon.sh                   # A/B vs AdamW
#   PROFILE=True DISABLE_TENSORBOARD=False ...                     # capture 1-step trace
###############################################################################
set -euo pipefail
set -x

export HF_TOKEN="${HF_TOKEN:-}"

export NNODES=${PET_NNODES:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}

# ---------- Model: DeepSeek-V4 Pro -----------------------------------------
export PRIMUS_MODEL=${PRIMUS_MODEL:-deepseek_v4_pro}
export EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml}

# ---------- Pro production widths (paper §4.2.1) ----------------------------
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-384}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
# Megatron only supports aux-loss-free expert bias with the sigmoid score
# function; V4 uses sqrtsoftplus, so disable expert bias (matches the working
# run_deepseek_v4.sh smoke; balancing falls back to seq_aux_loss).
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=${PRIMUS_MOE_ENABLE_EXPERT_BIAS:-False}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-3072}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-1024}

# ---------- Reduced depth + seq to fit single node -------------------------
# MEASURED CEILING (chi2774, 8x288GB, 2026-05-20): with full Pro width
# (384 experts) + Muon, *4 layers @ seq 512 = 268.8 GB/rank (93%)* is about
# the single-node max.  The binding cost is weights + Muon's fp32 optimizer
# states (Muon forces use_precision_aware_optimizer=False, so states cannot
# be bf16) — NOT activations, so lowering seq does not buy more layers.
# 5+ layers OOMs; for more depth use fewer experts or multi-node.
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-512}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}

# Per-layer compression schedule, length == PRIMUS_TOTAL_LAYERS, mirroring the
# Pro pattern: first two HCA(128), then CSA(4)/HCA(128) interleaved, last
# dense+SWA(0).  (Pro full yaml: idx0,1=128; idx>=2 even=4 / odd=128; last=0.)
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-$(python3 - "$PRIMUS_TOTAL_LAYERS" <<'PY'
import sys
n=int(sys.argv[1])
r=[]
for i in range(n):
    if i<2: r.append(128)
    elif i==n-1: r.append(0)
    else: r.append(4 if i%2==0 else 128)
print("["+",".join(map(str,r))+"]")
PY
)}

# ---------- Single-node EP=8 -----------------------------------------------
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}
export MBS=${MBS:-1}
# Paper Pro batch = 94.4M tokens/step (batch-size schedule). On one node we
# approach that regime via GRADIENT ACCUMULATION: grad_accum = GBS/(MBS*DP),
# DP=8 here. Large GBS is what amortizes the fixed Muon Newton-Schulz cost over
# many fwd/bwd microbatches (GBS=8 ⇒ accum=1 ⇒ NS runs every step ⇒ NS looks
# like ~97% of GEMM; that was a starved-batch artifact, NOT a Muon bug). At
# GBS=512 (accum 64) × seq 4096 = ~2.1M tokens/step the optimizer share drops
# toward the paper's reported 1-3%.
export GBS=${GBS:-512}

# ---------- Optimizer: Muon (paper §4.2.2, values for BOTH Flash & Pro) ------
export OPTIMIZER=${OPTIMIZER:-muon}
# The Primus Muon path (primus/backends/megatron/core/optimizer/moun.py) needs
# the emerging_optimizers package, which is NOT bundled in the container. Set
# PRIMUS_INSTALL_EMERGING_OPTIMIZERS so the in-container install hook
# (runner/.../01_install_emerging_optimizers.sh) provisions the pinned commit.
# Gated on a Muon optimizer so an OPTIMIZER=adam A/B run pays nothing.
if [ "$OPTIMIZER" = "muon" ] || [ "$OPTIMIZER" = "dist_muon" ]; then
  export PRIMUS_INSTALL_EMERGING_OPTIMIZERS=${PRIMUS_INSTALL_EMERGING_OPTIMIZERS:-1}
fi
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}            # paper momentum 0.95
# Paper: "rescale the RMS of each update matrix to 0.18 for reutilization of the
# AdamW learning rate."  With scale_mode=spectral the realized update RMS ≈
# extra_scale_factor (orth_grad RMS≈1/√max(m,n), times spectral scale √max(m,n)),
# so 0.18 maps directly here.  Megatron default is 1.0 (≈5.5× too large vs paper).
export MUON_EXTRA_SCALE_FACTOR=${MUON_EXTRA_SCALE_FACTOR:-0.18}
# NOTE: muon_weight_decay is yaml-only (trainer_base.yaml=0.01); paper=0.1.
# No CLI flag, so it stays 0.01 here unless overridden in an EXP yaml.
# Muon hard requirements (Megatron arguments.py:1422):
export USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-False}
export USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-False}

# ---------- Paper §4.2.2 Pro training hyperparameters -----------------------
export LR=${LR:-2.0e-4}                       # Pro peak LR (Flash exp yaml had 1e-5)
export MIN_LR=${MIN_LR:-2.0e-5}               # Pro end LR
export ADAM_EPS=${ADAM_EPS:-1.0e-20}          # paper AdamW eps (NOTE: needs decimal point — Primus parses "1e-20" as a string)
export MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.0001}  # paper balance-loss weight
# Paper MTP depth = 1, but the Primus V4 integration does NOT yet support the
# MTP layer ("Unsupported mtp_model_layer submodules type ... when instantiating
# MultiTokenPredictionLayer"), so default 0 here. Set =1 once V4 MTP lands.
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}

# ---------- Perf knobs (V4 Triton attn + Turbo MoE; same family as proxy) --
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

export PROFILE=${PROFILE:-False}
# Profiler writes the chrome trace via tensorboard_trace_handler(args.tensorboard_dir),
# so tensorboard must be enabled for a trace run.  Default True (smoke); set
# DISABLE_TENSORBOARD=False together with PROFILE=True to capture a trace.
export DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-True}
export BACKEND_PATH=${BACKEND_PATH:-"$(pwd)/third_party/Megatron-LM"}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-tas-mi355x-$(date +%Y%m%d)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_pro_muon_L${PRIMUS_TOTAL_LAYERS}_seq${PRIMUS_SEQ_LENGTH}_ep${PRIMUS_EP}}

if [ ! -d "$BACKEND_PATH" ] || [ -z "$(ls -A "$BACKEND_PATH" 2>/dev/null)" ]; then
  echo "[ERROR] BACKEND_PATH does not exist or is empty: $BACKEND_PATH"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

./primus-cli direct \
  -- train pretrain --config "$EXP" \
  --backend_path "$BACKEND_PATH" \
  --manual_gc True \
  --manual_gc_interval 100 \
  --num_layers "$PRIMUS_TOTAL_LAYERS" \
  --train_iters "$TRAIN_ITERS" \
  --lr_warmup_iters 0 \
  --lr_decay_iters "$TRAIN_ITERS" \
  --micro_batch_size "$MBS" \
  --global_batch_size "$GBS" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --adam_eps "$ADAM_EPS" \
  --moe_aux_loss_coeff "$MOE_AUX_LOSS_COEFF" \
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
  --mtp_num_layers "$MTP_NUM_LAYERS" \
  --mock_data True \
  --optimizer "$OPTIMIZER" \
  --muon_momentum "$MUON_MOMENTUM" \
  --muon_extra_scale_factor "$MUON_EXTRA_SCALE_FACTOR" \
  --use_distributed_optimizer "$USE_DISTRIBUTED_OPTIMIZER" \
  --use_precision_aware_optimizer "$USE_PRECISION_AWARE_OPTIMIZER" \
  --main_grads_dtype fp32 \
  --exp_avg_dtype fp32 \
  --exp_avg_sq_dtype fp32 \
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
  --recompute_num_layers 1 \
  --recompute_granularity full \
  --recompute_method uniform \
  --overlap_grad_reduce False \
  --overlap_param_gather False \
  --disable_last_saving True \
  --disable_wandb True \
  --disable_tensorboard "$DISABLE_TENSORBOARD" \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK:-0}.txt"
