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
# "fenic" is the cluster NIC; honor inherited values so a non-cluster / direct
# run (e.g. local single-node smoke) can point these at a real interface
# (e.g. lo) — otherwise gloo aborts with "Unable to find address for: fenic".
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-fenic}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic}
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
# "null" is the sentinel for "fp8 off"; honor inherited FP8 / FP8_RECIPE so a
# caller (e.g. run_deepseek_v4_flash_proxy.sh) can set them before exec.
export FP8=${FP8:-null}
export FP8_RECIPE=${FP8_RECIPE:-null}

# Plan-4 P25 / P26: in-tree Primus Triton kernels for V4 attention.
# Precedence in DeepseekV4Attention.forward:
#   use_turbo_attention > use_v4_tilelang_attention > use_v4_triton_attention > eager   (cr ∈ {0, 128})
#   use_v4_tilelang_csa_attention > use_v4_triton_csa_attention > eager                 (cr == 4)
# These are V4-only; they have no effect on other model types.
export USE_V4_TRITON_ATTENTION=${USE_V4_TRITON_ATTENTION:-True}
export USE_V4_TRITON_CSA_ATTENTION=${USE_V4_TRITON_CSA_ATTENTION:-True}

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
  # Default to the paper's ue8m0 microscaling (e4m3 + mxfp8); honor explicit
  # FP8 / FP8_RECIPE overrides. Sentinel-aware because "null" is non-empty, so
  # a plain ${FP8:-...} would keep the off-sentinel instead of defaulting.
  [ "$FP8" = "null" ] && export FP8=e4m3
  [ "$FP8_RECIPE" = "null" ] && export FP8_RECIPE=mxfp8
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

# Launcher: slurm (default, multi-node cluster) or direct (single-node, already
# inside the container — e.g. local smoke on one box). PRIMUS_LAUNCHER=direct
# drops the SLURM/srun + docker-image wrap that 'direct' doesn't use.
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-slurm}
if [ "$PRIMUS_LAUNCHER" = "direct" ]; then
  LAUNCHER_ARGS=(direct)
else
  LAUNCHER_ARGS=(slurm -N "$NNODES")
  [ -n "${SLURM_PARTITION:-}" ] && LAUNCHER_ARGS+=(--partition="${SLURM_PARTITION}")
  [ -n "${SLURM_NODELIST:-}" ] && LAUNCHER_ARGS+=(--nodelist="${SLURM_NODELIST}")
  LAUNCHER_ARGS+=(-- --image "${DOCKER_IMAGE}" --clean -- --numa)
fi

./primus-cli "${LAUNCHER_ARGS[@]}" \
  -- train pretrain --config "$EXP" \
  --manual_gc True \
  --manual_gc_interval 100 \
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
  --mtp_num_layers 0 \
  --mock_data True \
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
  --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
  --fp8 "$FP8" \
  --fp8_recipe "$FP8_RECIPE" \
  --recompute_num_layers "$PRIMUS_RECOMPUTE_LAYERS" \
  --recompute_granularity full \
  --recompute_method block \
  --overlap_grad_reduce True \
  --overlap_param_gather True \
  --disable_last_saving True \
  --disable_wandb True \
  --disable_tensorboard True \
  --profile "$PROFILE" \
  --use_pytorch_profiler "$PROFILE" \
  --profile_step_end 7 \
  --profile_step_start 6 \
  2>&1 | tee "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/log_node_${NODE_RANK:-0}.txt"
