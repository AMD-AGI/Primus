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
#   PRECISION_TYPE=BF16 ./run_deepseek_v4_pro_muon.sh             # A/B vs BF16 (fp8 is default-on)
#   PROFILE=True DISABLE_TENSORBOARD=False ...                     # capture 1-step trace
#
# Precision: FP8 training is ON by default (FP8=e4m3, FP8_RECIPE=tensorwise).
# Paper recipe is ue8m0/mxfp8 but it's not runnable on this gfx950 build (see the
# "FP8 training" block below); tensorwise gives the paper's fp8 layout on the
# weight GEMMs. PRECISION_TYPE=BF16 to A/B back to bf16.
###############################################################################
set -euo pipefail
set -x

export HF_TOKEN="${HF_TOKEN:-}"

export NNODES=${PET_NNODES:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}

# ---------- Model: DeepSeek-V4 Pro -----------------------------------------
export PRIMUS_MODEL=${PRIMUS_MODEL:-deepseek_v4_pro}
export EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml}

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
# Newton-Schulz hardening knobs (matter for mxfp8, where NS can diverge on the
# quant-noised gradient). num_ns_steps = NS iterations (more = better convergence
# on ill-conditioned input); fp32_matmul_prec = precision of the NS matmuls
# ("medium" = tf32-ish, "high" = full fp32 — full precision keeps a near-σ=1
# input from being pushed past the quintic's stable region by rounding error).
export MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS:-5}
export MUON_FP32_MATMUL_PREC=${MUON_FP32_MATMUL_PREC:-medium}
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
# Primus Sync-Free MoE (eliminates the DeepEP host busy-wait on the variable
# per-expert token counts): 0=off, 1=fused router/permute, 2=+no CPU busy-wait
# (turbo deepep + grouped mlp), 3=fully sync-free (+fused act). Stage >=2 needs
# use_turbo_grouped_mlp=True. Auto-enables the required sub-flags. Default 0.
export TURBO_SYNC_FREE_MOE_STAGE=${TURBO_SYNC_FREE_MOE_STAGE:-0}
# Phase 1b: route the dense/attention projections (q_down/kv/o_a etc.) through
# Primus-Turbo linears so they pick up the mxfp8 (CK) path under the fp8 context.
# Default OFF (attention stays bf16, the validated baseline). Set True to enable
# fp8 attention/dense projections. Requires TP=1 and fp8_recipe in {tensorwise,
# blockwise,mxfp8}; the MLA monkey-patch is auto-skipped for V4 (see mla_patches).
export USE_TURBO_PARALLEL_LINEAR=${USE_TURBO_PARALLEL_LINEAR:-False}
# Per-module recipe (paper): routed experts in MXFP4 while the rest of the layer
# stays FP8. Works under the global FP8 recipe (no --fp4/--fp8 conflict): the
# PrimusTurbo grouped MLP routes expert GEMMs through native FP4 (hipBLASLt).
# Default OFF. When on, force the hipBLASLt FP4 backend (no AITER).
export MOE_EXPERTS_FP4=${MOE_EXPERTS_FP4:-False}
if [ "$MOE_EXPERTS_FP4" = "True" ]; then
  export PRIMUS_TURBO_GEMM_BACKEND=${PRIMUS_TURBO_GEMM_BACKEND:-FP4:HIPBLASLT}
fi
# Phase 5 (paper): CSA-indexer QK score in FP4. Rounds q_i and K^{IComp} to
# MXFP4 before the QK product (STE backward); w_i + ReLU/sum tail stay BF16.
# Read directly by the Indexer via PRIMUS_INDEXER_FP4. Default OFF.
export INDEXER_FP4=${INDEXER_FP4:-False}
if [ "$INDEXER_FP4" = "True" ]; then
  export PRIMUS_INDEXER_FP4=1
  # The indexer QK now runs a REAL MXFP4 gemm (pt.ops.gemm_fp4) — force the
  # hipBLASLt FP4 backend (no AITER), same as the MXFP4 expert path.
  export PRIMUS_TURBO_GEMM_BACKEND=${PRIMUS_TURBO_GEMM_BACKEND:-FP4:HIPBLASLT}
fi
# MXFP8 expert-weight caching: expert weights are constant within an optimizer
# step, so re-quantizing them every microbatch + recompute forward (the large
# _mxfp8_quant_weight_fwd kernel) is redundant. When on, PrimusTurboGroupedMLP
# prequantizes once per step and reuses the fp8 buffers (loss-neutral, faster).
# Costs extra bytes/param resident — watch HBM at depth. Only affects the
# mxfp8 (MX_BLOCKWISE) grouped path. Default OFF.
export CACHE_MXFP8_WEIGHT=${CACHE_MXFP8_WEIGHT:-False}
if [ "$CACHE_MXFP8_WEIGHT" = "True" ]; then
  export PRIMUS_TURBO_CACHE_MXFP8_WEIGHT=1
fi
# FP8 attention projections (paper recipe): route q-up / o-proj through the fp8
# turbo linear instead of the bf16 gather/scatter native path. Only valid at
# TP=1 (gather/scatter are no-ops there); for TP>1 the turbo linear rejects
# gather_output/scatter-input and it stays bf16. Default OFF.
export V4_FP8_ATTN_PROJ=${V4_FP8_ATTN_PROJ:-False}
if [ "$V4_FP8_ATTN_PROJ" = "True" ]; then
  export PRIMUS_V4_FP8_ATTN_PROJ=1
fi
export USE_V4_ATTENTION_BACKEND=${USE_V4_ATTENTION_BACKEND:-triton_v2}
export USE_V4_CSA_ATTENTION_BACKEND=${USE_V4_CSA_ATTENTION_BACKEND:-triton_v2}
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-False}
export PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=${PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU:-True}

# ---------- FP8 training (paper §4.x quantization / techblog §9.6) ----------
# Paper recipe: "FP4 + FP8 Mixed" — MoE experts + CSA-Indexer QK in FP4 (MXFP4),
# EVERYTHING ELSE in FP8, all with the **ue8m0** microscaling scale format.
# On this stack the ue8m0 path is `fp8_recipe=mxfp8` → TE MXFP8BlockScaling →
# Primus-Turbo MX_BLOCKWISE granularity with scale_dtype=E8M0 (fp8_utils.py:148),
# i.e. the paper's exact scaling format, native on MI355X/CDNA4.
#
# Integration gap vs the paper (NOT config — Primus V4 TODO, develop techblog
# item 10 "Phase 2 FP4/FP8 Mixed"): the FP4 expert / FP4-Indexer path is not yet
# wired in V4, so experts run at FP8 here (FP8 everywhere) rather than FP4. This
# is the closest supported step toward the paper recipe. FP8 is highly outlier-
# sensitive, which is why the paper pairs it with clamped SwiGLU (swiglu_limit,
# already on above via PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=True).
# Precision toggle — shared interface with run_deepseek_v4.sh / the flash proxy:
#   PRECISION_TYPE=FP8 (default) -> e4m3 + tensorwise; BF16 -> fp8 off.
# FP8 / FP8_RECIPE still override directly (e.g. FP8_RECIPE=blockwise, or FP8=null).
export PRECISION_TYPE=${PRECISION_TYPE:-FP8}
if [ "$PRECISION_TYPE" = "FP8" ]; then
  export FP8=${FP8:-e4m3}                      # forward fp8 format (paper E4M3); "hybrid" = E4M3 fwd / E5M2 bwd
  # Paper recipe is ue8m0 microscaling (mxfp8), but mxfp8 is NOT runnable on this
  # gfx950 build (turbo grouped-GEMM has no MX path; TE ROCm MXFP8 needs K%128==0,
  # V4 has a K=224 proj). `tensorwise` is the working recipe — paper fp8 layout,
  # non-ue8m0 scale. (other: blockwise [TE-ROCm unsupported] / delayed)
  export FP8_RECIPE=${FP8_RECIPE:-tensorwise}
  # mxfp8 (paper ue8m0) + Muon: TRAINS, but shows a transient early-training
  # grad-norm spike. Earlier 8-iter runs only caught the spike and mislabelled it
  # a divergence; a 40-iter run shows it SELF-HEALS and the loss descends cleanly.
  #   What's happening: mxfp8 quant noise ill-conditions the gradient early (random
  #   init), so Muon's Newton-Schulz update norm spikes for ~10-20 iters, then
  #   settles as the model organizes. RAW grads stay ~0.99 throughout; only the
  #   NS-orthogonalized UPDATE norm spikes (Muon-specific: Adam@same-config is flat).
  #   It is NOT a kernel bug (both MX GEMMs ~4% correct on REAL E2E inputs via
  #   capture-replay; error zero-mean).
  #   FIX (verified, L4/64-expert/GBS512/seq128, 40 iters):
  #     no warmup  -> loss 12->0.80, grad-norm peak ~2.4e5 (settles to ~35)
  #     warmup=10  -> loss 12->0.44, grad-norm peak ~2.1e4 (12x lower), no NaN
  #   So LR warmup tames the transient AND improves the loss — and it is what the
  #   paper does. We therefore AUTO-ENABLE warmup for mxfp8 (default 10; override
  #   LR_WARMUP_ITERS). Two more NS-hardening knobs are exposed if needed:
  #   MUON_NUM_NS_STEPS (more iters) and MUON_FP32_MATMUL_PREC=high. tensorwise
  #   stays the conservative default (smooth per-tensor scale, no transient).
  #   NOTE: validated at reduced depth/width; full 61-layer/384-expert is a
  #   separate multi-GPU confirmation.
  if [ "$FP8_RECIPE" = "mxfp8" ]; then
    export LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-10}
    echo "[INFO] FP8_RECIPE=mxfp8 + Muon: expect a transient early grad-norm spike" >&2
    echo "       (self-heals; LR warmup auto-set to ${LR_WARMUP_ITERS} to damp it)." >&2
  fi
else
  export FP8=${FP8:-null}                  # PRECISION_TYPE=BF16 -> disable fp8
  export FP8_RECIPE=${FP8_RECIPE:-null}
fi

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
  --lr_warmup_iters "${LR_WARMUP_ITERS:-0}" \
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
  --moe_router_force_load_balancing "${MOE_FORCE_LOAD_BALANCE:-False}" \
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
  --muon_num_ns_steps "$MUON_NUM_NS_STEPS" \
  --muon_fp32_matmul_prec "$MUON_FP32_MATMUL_PREC" \
  --use_distributed_optimizer "$USE_DISTRIBUTED_OPTIMIZER" \
  --use_precision_aware_optimizer "$USE_PRECISION_AWARE_OPTIMIZER" \
  --main_grads_dtype fp32 \
  --exp_avg_dtype fp32 \
  --exp_avg_sq_dtype fp32 \
  --enable_primus_turbo "$ENABLE_PRIMUS_TURBO" \
  --use_turbo_attention "$USE_TURBO_ATTENTION" \
  --use_v4_attention_backend "$USE_V4_ATTENTION_BACKEND" \
  --use_v4_csa_attention_backend "$USE_V4_CSA_ATTENTION_BACKEND" \
  --use_v4_compiled_sinkhorn "$USE_V4_COMPILED_SINKHORN" \
  --use_turbo_deepep "$USE_TURBO_DEEPEP" \
  --turbo_sync_free_moe_stage "$TURBO_SYNC_FREE_MOE_STAGE" \
  "${TURBO_DEEPEP_CLI_ARGS[@]}" \
  --use_turbo_grouped_mlp "$TURBO_USE_GROUPED_MLP" \
  --use_turbo_parallel_linear "$USE_TURBO_PARALLEL_LINEAR" \
  --moe_experts_fp4 "$MOE_EXPERTS_FP4" \
  --moe_use_legacy_grouped_gemm False \
  --fp8 "$FP8" \
  --fp8_recipe "$FP8_RECIPE" \
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
