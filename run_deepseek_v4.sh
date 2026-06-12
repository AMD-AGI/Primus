#!/bin/bash
set -euo pipefail
set -x

_RUN_START_SEC=$(date +%s)
_RUN_START_TS=$(date '+%Y-%m-%d %H:%M:%S')
_print_run_elapsed() {
  local _end_sec _end_ts _elapsed _exit=$1
  _end_sec=$(date +%s)
  _end_ts=$(date '+%Y-%m-%d %H:%M:%S')
  _elapsed=$((_end_sec - _RUN_START_SEC))
  echo "----------------------------------------"
  echo "run_deepseek_v4.sh wall time"
  echo "  start:   ${_RUN_START_TS}"
  echo "  end:     ${_end_ts}"
  echo "  elapsed: ${_elapsed}s ($((_elapsed / 60))m $((_elapsed % 60))s)"
  echo "  exit:    ${_exit}"
}
trap '_print_run_elapsed $?' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"

export NNODES=${NNODES:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-20}

export DOCKER_IMAGE=${DOCKER_IMAGE:-"tasimage/primus:pr-715-ainic"}
export SLURM_PARTITION=Compute-DCPT
export SLURM_NODELIST="${SLURM_NODELIST:-smci355-ccs-aus-n01-21,smci355-ccs-aus-n01-33,smci355-ccs-aus-n02-25,smci355-ccs-aus-n02-33,smci355-ccs-aus-n03-33,smci355-ccs-aus-n04-21,smci355-ccs-aus-n04-25,smci355-ccs-aus-n04-29,smci355-ccs-aus-n04-33,smci355-ccs-aus-n05-21,smci355-ccs-aus-n05-29,smci355-ccs-aus-n05-33,smci355-ccs-aus-n06-25,smci355-ccs-aus-n06-33,smci355-ccs-aus-n10-29}"
export MASTER_PORT=${MASTER_PORT:-29500}

export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA="ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1"
export GLOO_SOCKET_IFNAME=fenic
export NCCL_SOCKET_IFNAME=fenic
export NCCL_IB_GID_INDEX=1
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}

# Phase-7 fixed knobs for single-node bring-up.
export MBS=${MBS:-1}
export GBS=${GBS:-$((16 * NNODES * MBS))}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}

# Keep this smoke config lightweight for quick bring-up.
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-8}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-128}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-128}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-8}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-2}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-512}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-8}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-"[0,0,4,4,4,4,4,0]"}
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=${PRIMUS_MOE_ENABLE_EXPERT_BIAS:-False}
export PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU=${PRIMUS_V4_GROUPED_EXPERTS_SUPPORT_CLAMPED_SWIGLU:-True}
export PROFILE=${PROFILE:-False}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-False}
export LEGACY_GG=${LEGACY_GG:-False}
# Plan-3 P22 / P23: PrimusTurbo gate (must be on for turbo attention /
# turbo deepep to take effect; enable_primus_turbo gates the
# `before_train` patches that re-bind the spec provider).
export ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-False}
if [ "$USE_TURBO_ATTENTION" = "True" ] || [ "${USE_TURBO_DEEPEP:-False}" = "True" ]; then
  ENABLE_PRIMUS_TURBO=True
fi
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-False}

# Plan-3 P23: Turbo DeepEP-related knobs.  Only emit these CLI flags
# when USE_TURBO_DEEPEP=True so non-deepep runs don't carry unrelated
# overrides.  Best-practice CU count: 64 (or 80) for EP=8, 32 for
# EP>=16 — the EP>=16 cap is asserted by
# `primus/modules/trainer/megatron/utils.py:527`.  DeepEP itself
# requires `moe_router_dtype=fp32` and forbids
# `moe_shared_expert_overlap=True` (both are already V4-Flash YAML
# defaults; we pin them via CLI defensively so a stray YAML override
# or future config edit cannot flip them out from under the Turbo
# path mid-run).
TURBO_DEEPEP_CLI_ARGS=()
if [ "$USE_TURBO_DEEPEP" = "True" ]; then
  if [ "${PRIMUS_EP:-1}" -ge 16 ]; then
    _DEFAULT_TURBO_DEEPEP_NUM_CU=32
  else
    _DEFAULT_TURBO_DEEPEP_NUM_CU=80
  fi
  export TURBO_DEEPEP_NUM_CU=${TURBO_DEEPEP_NUM_CU:-$_DEFAULT_TURBO_DEEPEP_NUM_CU}
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

export PRECISION_TYPE=${PRECISION_TYPE:-BF16}
# Honor an incoming FP8 / FP8_RECIPE env (e.g. FP8_RECIPE=mxfp8); default null
# so non-FP8 runs are unchanged. (Previously these were hard-set to null,
# which silently clobbered a caller-provided recipe.)
export FP8=${FP8:-null}
export FP8_RECIPE=${FP8_RECIPE:-null}

# ---------- Optimizer selection (adam default; muon = DeepSeek-V4 recipe) ----
# OPTIMIZER=adam (default): unchanged behaviour (BF16 precision-aware AdamW
#   from the EXP yaml); overlap_grad_reduce / overlap_param_gather stay ON.
# OPTIMIZER=muon: Primus distributed-Muon path (primus .../optimizer/moun.py).
#   Megatron asserts plain `muon` is incompatible with distributed optimizer +
#   grad/param overlap, so we force them OFF and switch optimizer states to
#   fp32 (Muon does not support the precision-aware optimizer). The
#   Newton-Schulz coefficient set auto-selects 'deepseekv4' (8 aggressive + 2
#   stable) for V4 configs inside get_megatron_muon_optimizer. Requires the
#   emerging_optimizers package -> we set PRIMUS_INSTALL_EMERGING_OPTIMIZERS so
#   the in-container install hook (runner/.../01_install_emerging_optimizers.sh)
#   provisions it.
export OPTIMIZER=${OPTIMIZER:-adam}
export PRIMUS_OVERLAP_GRAD_REDUCE=${PRIMUS_OVERLAP_GRAD_REDUCE:-True}
export PRIMUS_OVERLAP_PARAM_GATHER=${PRIMUS_OVERLAP_PARAM_GATHER:-True}
OPTIMIZER_CLI_ARGS=()
if [ "$OPTIMIZER" = "muon" ] || [ "$OPTIMIZER" = "dist_muon" ]; then
  export PRIMUS_INSTALL_EMERGING_OPTIMIZERS=${PRIMUS_INSTALL_EMERGING_OPTIMIZERS:-1}
  export MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
  export MUON_EXTRA_SCALE_FACTOR=${MUON_EXTRA_SCALE_FACTOR:-0.18}
  # Both plain muon (Megatron asserts) and dist_muon (LayerWiseDistributed-
  # Optimizer docstring: "keep all megatron distributed-optimizer related
  # options OFF"; it manages its own param all-gather, so DDP
  # overlap_param_gather double-drives start_param_sync -> crash) need the
  # DDP grad/param overlap OFF.
  PRIMUS_OVERLAP_GRAD_REDUCE=False
  PRIMUS_OVERLAP_PARAM_GATHER=False
  OPTIMIZER_CLI_ARGS=(
    --optimizer "$OPTIMIZER"
    --muon_momentum "$MUON_MOMENTUM"
    --muon_extra_scale_factor "$MUON_EXTRA_SCALE_FACTOR"
    --use_distributed_optimizer False
    --use_precision_aware_optimizer False
    --main_grads_dtype fp32
    --exp_avg_dtype fp32
    --exp_avg_sq_dtype fp32
  )
fi

# Plan-4 P25 / P26: in-tree Primus Triton kernels for V4 attention.
# Precedence in DeepseekV4Attention.forward:
#   use_turbo_attention > use_v4_tilelang_attention > use_v4_triton_attention > eager   (cr ∈ {0, 128})
#   use_v4_tilelang_csa_attention > use_v4_triton_csa_attention > eager                 (cr == 4)
# These are V4-only; they have no effect on other model types.
export USE_V4_TRITON_ATTENTION=${USE_V4_TRITON_ATTENTION:-True}
export USE_V4_TRITON_CSA_ATTENTION=${USE_V4_TRITON_CSA_ATTENTION:-True}

# Plan-9: FP8 (E4M3) Indexer QK path (CSA selector). Default OFF; flip with
# USE_V4_FP8_INDEXER=True. Passed as a CLI override so it reliably reaches the
# in-container config regardless of env propagation.
export USE_V4_FP8_INDEXER=${USE_V4_FP8_INDEXER:-False}

# Plan-8 tilelang attention kernels (OPTIONAL — default OFF).
# Tilelang is NOT bundled in the default Primus container, so we leave
# both flags off here.  When the container has tilelang installed at
# the plan-8 pin AND the P50..P55 kernels are registered, set these
# to True (e.g. in a sweep / experiment script) to route the dense /
# HCA / CSA paths through tilelang.  Otherwise the dispatcher prints
# a single rank-0 warning and falls back to the Triton path -- training
# continues without error.  Replaces the legacy PRIMUS_V4_TILELANG_ATTN
# env knob (no longer consulted).
export USE_V4_TILELANG_ATTENTION=${USE_V4_TILELANG_ATTENTION:-False}
export USE_V4_TILELANG_CSA_ATTENTION=${USE_V4_TILELANG_CSA_ATTENTION:-False}

# Plan-5 P29 (RESCOPED): wrap sinkhorn_normalize in HyperMixer with a
# cached torch.compile build. Default OFF here; the proxy script
# (run_deepseek_v4_flash_proxy.sh) flips it ON. After G32 + G33b are
# green, the default flips to True for the V4-Flash configs.
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-False}

# Plan-4 P27: TP-side guard for the V4 Triton kernels.
# The dense / HCA / CSA kernels operate on the local head slice (each
# rank only sees H/TP query heads) so TP-sharded execution is correct
# by construction (no in-kernel collective comm needed).  Plan-4 unit
# tests / smoke gates exercise TP=1 only; emit a soft warning when a
# user enables the kernels at TP>1 so any TP-related regression is
# easy to attribute.  TP=1 is the V4-Flash / V4-Pro release default
# (release configs use PP+EP for parallelism, never TP).
if { [ "$USE_V4_TRITON_ATTENTION" = "True" ] || [ "$USE_V4_TRITON_CSA_ATTENTION" = "True" ]; } && [ "${PRIMUS_TP:-1}" -gt 1 ]; then
  echo "[WARN] Plan-4 V4 Triton kernels enabled at PRIMUS_TP=${PRIMUS_TP}>1; this combination is not covered by Plan-4 unit tests / smoke gates (G28..G30 ran TP=1 only). Functionally the kernels operate per-rank on the local H/TP head slice, so this should work, but treat any TP>1 regression as a Plan-4 follow-up."
fi

if [ "$PRECISION_TYPE" = "FP8" ]; then
  export FP8=${FP8:-hybrid}
  export FP8_RECIPE=${FP8_RECIPE:-delayed}
fi

# ---------- MXFP8 + FP8 param-gather (Muon path; Megatron #4987 analogue) ----
# Plan-9: combine the distributed-Muon (LayerWise) path with an MXFP8 forward
# recipe + FP8 parameter all-gather. Enable with FP8_PARAM_GATHER=True (best
# paired with OPTIMIZER=dist_muon + PRECISION_TYPE=FP8 FP8_RECIPE=mxfp8).
# MXFP8 on ROCm/TE requires NVTE_ROCM_ENABLE_MXFP8=1; the mxfp8 param-AG path
# is most memory-efficient with --reuse-grad-buf-for-mxfp8-param-ag. NOTE:
# Megatron auto-disables --fp8-param-gather on TE>=2.0.0 (falls back to a
# bf16/all_gather), so on such containers this exercises the MXFP8 forward +
# dist-Muon path with param-gather requested-but-possibly-downgraded.
export FP8_PARAM_GATHER=${FP8_PARAM_GATHER:-False}
FP8_PARAM_GATHER_CLI_ARGS=()
if [ "$FP8_PARAM_GATHER" = "True" ]; then
  export NVTE_ROCM_ENABLE_MXFP8=${NVTE_ROCM_ENABLE_MXFP8:-1}
  export REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG=${REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG:-True}
  FP8_PARAM_GATHER_CLI_ARGS=(--fp8_param_gather True)
  if [ "$REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG" = "True" ] && [ "$FP8_RECIPE" = "mxfp8" ]; then
    FP8_PARAM_GATHER_CLI_ARGS+=(--reuse_grad_buf_for_mxfp8_param_ag True)
  fi
fi

PP_LAYOUT_ARGS=()
if [ -n "${PRIMUS_PP_LAYOUT:-}" ]; then
  PP_LAYOUT_ARGS=(--pipeline_model_parallel_layout "$PRIMUS_PP_LAYOUT")
fi

PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-0}

export EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml}
export BACKEND_PATH=${BACKEND_PATH:-"$(pwd)/third_party/Megatron-LM"}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-tas-mi355x-$(date +%Y%m%d)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_smoke_${PRECISION_TYPE}_MBS${MBS}_GBS${GBS}_PP${PRIMUS_PP}_EP${PRIMUS_EP}}

if [ ! -d "$BACKEND_PATH" ] || [ -z "$(ls -A "$BACKEND_PATH" 2>/dev/null)" ]; then
  echo "[ERROR] BACKEND_PATH does not exist or is empty: $BACKEND_PATH"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"

export PRIMUS_EXIT_FAST=1

./primus-cli slurm -N "$NNODES" \
  ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
  ${SLURM_NODELIST:+--nodelist="${SLURM_NODELIST}"} \
  -- --image "${DOCKER_IMAGE}" --clean -- --numa \
  -- train pretrain --config "$EXP" \
  --manual_gc True \
  --manual_gc_interval 100 \
  --pp_warmup "${PP_WARMUP:-True}" \
  "${PP_LAYOUT_ARGS[@]}" \
  --moe_router_force_load_balancing True \
  --log_avg_skip_iterations 3 \
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
  --mtp_num_layers "${MTP_NUM_LAYERS:-0}" \
  --mock_data True \
  --enable_primus_turbo "$ENABLE_PRIMUS_TURBO" \
  --use_turbo_attention "$USE_TURBO_ATTENTION" \
  --use_v4_triton_attention "$USE_V4_TRITON_ATTENTION" \
  --use_v4_triton_csa_attention "$USE_V4_TRITON_CSA_ATTENTION" \
  --use_v4_fp8_indexer "$USE_V4_FP8_INDEXER" \
  --use_v4_tilelang_attention "$USE_V4_TILELANG_ATTENTION" \
  --use_v4_tilelang_csa_attention "$USE_V4_TILELANG_CSA_ATTENTION" \
  --use_v4_compiled_sinkhorn "$USE_V4_COMPILED_SINKHORN" \
  --use_turbo_deepep "$USE_TURBO_DEEPEP" \
  "${TURBO_DEEPEP_CLI_ARGS[@]}" \
  --use_turbo_grouped_mlp "$TURBO_USE_GROUPED_MLP" \
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  "${OPTIMIZER_CLI_ARGS[@]}" \
  --fp8 "$FP8" \
  --fp8_recipe "$FP8_RECIPE" \
  "${FP8_PARAM_GATHER_CLI_ARGS[@]}" \
  --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" \
  --recompute_granularity full \
  --recompute_method block \
  --overlap_grad_reduce "$PRIMUS_OVERLAP_GRAD_REDUCE" \
  --overlap_param_gather "$PRIMUS_OVERLAP_PARAM_GATHER" \
  --disable_last_saving True \
  --disable_wandb True \
  --disable_tensorboard True \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK:-0}.txt"
