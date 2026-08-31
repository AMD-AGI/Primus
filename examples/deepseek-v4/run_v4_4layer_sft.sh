#!/bin/bash
# Single-node (8x MI308X / gfx942) driver for 4-layer DeepSeek-V4 native SFT.
#
# Why not run_deepseek_v4_flash.sh directly: that script targets multi-node SLURM
# and hardcodes `--mock_data True` plus `--moe_router_force_load_balancing True`
# as literal CLI args. mock_data is fatal under stage:sft (it force-installs
# NullTokenizer), and force-LB freezes the router. Everything else it sets is
# reproduced here or in the experiment yaml.
#
# Usage (run from anywhere; paths resolve relative to the repo):
#   SEQ=4096   SFT_JSONL=<path>/sft_4k.jsonl   bash run_v4_4layer_sft.sh
#   SEQ=131072 SFT_JSONL=<path>/sft_128k.jsonl bash run_v4_4layer_sft.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}" || exit 1

# ---- launcher: single node, already inside the container ----
export PRIMUS_LAUNCHER=direct
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST 2>/dev/null || true
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=${MASTER_PORT:-29517}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export BACKEND_PATH=${BACKEND_PATH:-${REPO}/third_party/Megatron-LM}
export PRIMUS_OUTPUT_ROOT=${PRIMUS_OUTPUT_ROOT:-${REPO}/output}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(whoami)}

# Show OOM tracebacks from every rank, not just rank 0.
export LOCAL_RANKS=${LOCAL_RANKS:-"--local-ranks-filter 0,1,2,3,4,5,6,7"}
# Big variable-size buffers fragment the caching allocator badly at long context.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

# ---- gfx942 / CDNA3 LDS safety ------------------------------------------------
# The V4 sparse-MLA backward Triton kernel is tuned for gfx950 / CDNA4 (160 KB LDS).
# gfx942 / CDNA3 has 64 KB, and the stock pipeline staging asks for 73728 B, so the
# kernel fails to COMPILE (not to run):
#     triton.runtime.errors.OutOfResources: shared memory, Required: 73728, limit: 65536
# num_stages=1 disables Triton's LDS multi-buffering and brings it under the limit.
#
# Measured on MI308X: this is the ONLY switch needed. PRIMUS_HC_TRITON and
# PRIMUS_DSA_DKV_SAFE can both stay at their defaults -- an earlier version of this
# script forced them off after mis-attributing a `Required: 131072` failure (which
# actually came from the triton_v1 ATTENTION backend, not from hyper-connection) to
# the hc kernel. Forcing PRIMUS_HC_TRITON=0 is not needed and costs performance,
# because it drops hyper-connection onto a PyTorch fallback.
_GFX_ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -m1 -oE 'gfx[0-9]+' || true)
if [ "${_GFX_ARCH}" = "gfx942" ]; then
  export PRIMUS_DSA_BWD_NUM_STAGES=${PRIMUS_DSA_BWD_NUM_STAGES:-1}
  echo "[run] gfx942 detected: PRIMUS_DSA_BWD_NUM_STAGES=$PRIMUS_DSA_BWD_NUM_STAGES (64 KB LDS)"
fi

# V4 triton kernel knobs (mirrors run_deepseek_v4_flash.sh).
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-1}
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-1}
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-1}
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-1}
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-1}
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-1}
export PRIMUS_V4_ATTN_BWD_USE_SPLIT=${PRIMUS_V4_ATTN_BWD_USE_SPLIT:-1}
export PRIMUS_V4_CSA_BWD_SEGREDUCE=${PRIMUS_V4_CSA_BWD_SEGREDUCE:-1}

# ---- experiment knobs ----
export PRIMUS_SEQ_LENGTH=${SEQ:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_SEQ_LENGTH}
export SFT_JSONL=${SFT_JSONL:-${REPO}/data/sft/sft_4096.jsonl}
export MBS=${MBS:-1}
export GBS=${GBS:-8}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_ETP=${PRIMUS_ETP:-1}
export PRIMUS_CP=${PRIMUS_CP:-1}
export PRIMUS_SHARD_HEADS=${PRIMUS_SHARD_HEADS:-false}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-256}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-"[0, 0, 128, 128]"}
export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-4}
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
export PRIMUS_HC_MULT=${PRIMUS_HC_MULT:-4}
export PRIMUS_USE_TURBO_DEEPEP=${PRIMUS_USE_TURBO_DEEPEP:-False}
export PRIMUS_USE_V4_ATTENTION_BACKEND=${PRIMUS_USE_V4_ATTENTION_BACKEND:-triton_v2}
export PRIMUS_USE_V4_CSA_ATTENTION_BACKEND=${PRIMUS_USE_V4_CSA_ATTENTION_BACKEND:-triton_v2}

export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-dsv4_4layer_sft_seq${PRIMUS_SEQ_LENGTH}_tp${PRIMUS_TP}_ep${PRIMUS_EP}}
EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash_4layer-BF16-sft.yaml}

LOGDIR="$PRIMUS_OUTPUT_ROOT/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
mkdir -p "$LOGDIR"
echo "[run] EXP=$EXP SEQ=$PRIMUS_SEQ_LENGTH TP=$PRIMUS_TP EP=$PRIMUS_EP GBS=$GBS log=$LOGDIR"

# EXTRA_ARGS is intentionally left unquoted so it splits into separate CLI flags.
# shellcheck disable=SC2086
./primus-cli direct -- train pretrain --config "$EXP" \
  --manual_gc True \
  --manual_gc_interval 100 \
  --pp_warmup False --sequence_parallel False \
  --log_avg_skip_iterations 3 \
  --backend_path "$BACKEND_PATH" \
  ${EXTRA_ARGS:-} \
  2>&1 | tee "$LOGDIR/log_node0.txt"
