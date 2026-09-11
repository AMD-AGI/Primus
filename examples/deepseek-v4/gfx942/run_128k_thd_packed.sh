#!/bin/bash
# One-click 128k DeepSeek-V4 SFT on PACKED sequences (THD) on a single 8x MI308X node.
#
# Same 3-branch model as run_128k_dense_hca_csa.sh (dense+SWA, CSA, HCA) and the same
# CP=8 / TP=1 / EP=8 parallelism -- see that script's header for why CP rather than TP.
# The difference is the data layout: instead of one 128k sequence per sample, many real
# SFT samples are packed into each 128k bin and delimited by cu_seqlens.
#
# WHY THIS IS NOT JUST A DATA CHANGE. Packing is what makes a 128k run train on anything
# real -- alpaca's runtime segments have a median of 84 tokens, so an unpacked 128k row is
# entirely padding. But packing puts sequence boundaries at arbitrary offsets, and V4's
# compressor pools fixed windows of `ratio` rows anchored at each sequence's start. Under
# CP a window can then straddle a shard boundary, with its earlier rows owned by the left
# neighbour. Two things make that work here:
#
#   * a real left-boundary exchange (deepseek_v4_cp.exchange_boundary_hidden) ships the
#     neighbour's trailing rows, so no alignment padding is needed. The alternative --
#     padding every segment up to 128 -- left only 34.2% of tokens supervised against
#     56.1% here, which is the whole reason the exchange was written. Both figures
#     are measured over every pack PackedSFTDataset produces, not estimated;
#   * strict window ownership (a window belongs to the rank holding its LAST row) plus a
#     fixed per-rank capacity, so the pool all-gather stays a uniform collective even
#     though ranks own different numbers of windows.
#
# Attention isolation is STRICT: no query may attend across a packed boundary. That is
# asserted by tests/unit_tests/megatron/transformer/deepseek_v4/test_deepseek_v4_thd_packing.py
# and the CP equivalence of the compressed pool by test_csa_cp_overlap_predecessor.py.
#
# Run from inside a rocm/primus:v26.5-pytorch2.12-te2.15 container:
#
#     bash examples/deepseek-v4/gfx942/run_128k_thd_packed.sh
#
# Required: a local DeepSeek-V4 tokenizer directory (V4_TOKENIZER, or /apps/DeepSeek-V4-Flash).
# Only tokenizer.json / tokenizer_config.json are read; this trains from random init.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
cd "${REPO}"

# ---- prerequisites -----------------------------------------------------------------
if [ -z "${V4_TOKENIZER:-}" ]; then
  for _c in /apps/DeepSeek-V4-Flash /apps/DeepSeek-V4-Flash-FP8; do
    [ -f "${_c}/tokenizer.json" ] && { V4_TOKENIZER="${_c}"; break; }
  done
  V4_TOKENIZER="${V4_TOKENIZER:-/apps/DeepSeek-V4-Flash}"
fi
DATA_DIR="${DATA_DIR:-${REPO}/data/sft}"
BACKEND_PATH="${BACKEND_PATH:-${REPO}/third_party/Megatron-LM}"

if [ ! -f "${V4_TOKENIZER}/tokenizer.json" ]; then
  echo "[error] no tokenizer at ${V4_TOKENIZER}. Set V4_TOKENIZER=<dir> (needs tokenizer.json)." >&2
  exit 1
fi

if [ ! -f "${BACKEND_PATH}/megatron/training/arguments.py" ]; then
  if [ -f /workspace/Primus/third_party/Megatron-LM/megatron/training/arguments.py ]; then
    echo "[setup] copying Megatron-LM from the image"
    mkdir -p "${REPO}/third_party"
    cp -a /workspace/Primus/third_party/Megatron-LM "${REPO}/third_party/"
    rm -f "${BACKEND_PATH}/.git"
  else
    echo "[setup] fetching Megatron-LM submodule"
    git -C "${REPO}" submodule update --init third_party/Megatron-LM
  fi
fi

# Source corpus: NATURAL-LENGTH alpaca rows, one sample per line.
#
# Note this is not the pre-packed {input_ids, labels, cu_seqlens} file that
# prepare_packed_data.py writes -- that tool is an offline measurement aid (it is how the
# supervised-token figures were first estimated) and the training pipeline has no
# loader for its schema. Feeding it here is silently harmless-looking and completely
# wrong: the alpaca formatter finds no instruction/output fields, every sample tokenizes
# to nothing, the loss mask is all zero, and the run completes 10/10 with exit 0 and
# grad norm 0.000. The packing itself is done at runtime by PackedSFTDataset, which is
# what sft_packing_segment_align applies to.
ALPACA_JSONL="${ALPACA_JSONL:-${DATA_DIR}/alpaca_natural.jsonl}"
if [ ! -f "${ALPACA_JSONL}" ]; then
  echo "[setup] fetching natural-length alpaca rows (one-off)"
  python - "$ALPACA_JSONL" <<'PY'
import json, sys
from datasets import load_dataset
out = sys.argv[1]
ds = load_dataset("tatsu-lab/alpaca", split="train")
with open(out, "w", encoding="utf-8") as fh:
    for r in ds:
        fh.write(json.dumps({"instruction": r["instruction"],
                             "input": r["input"],
                             "output": r["output"]}) + "\n")
print(f"[data] {out}: {len(ds)} rows", flush=True)
PY
fi

# ---- launcher ----------------------------------------------------------------------
export PRIMUS_LAUNCHER=direct
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST 2>/dev/null || true
export NNODES=1 NODE_RANK=0 MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29518}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export BACKEND_PATH
export PRIMUS_OUTPUT_ROOT="${PRIMUS_OUTPUT_ROOT:-${REPO}/output}"
export PRIMUS_TEAM="${PRIMUS_TEAM:-amd}"
export PRIMUS_USER="${PRIMUS_USER:-$(whoami)}"
export LOCAL_RANKS="${LOCAL_RANKS:---local-ranks-filter 0,1,2,3,4,5,6,7}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# ---- gfx942 (CDNA3) kernel settings -------------------------------------------------
# See run_128k_dense_hca_csa.sh / README.md for why each of these is needed.
export PRIMUS_DSA_BWD_NUM_STAGES="${PRIMUS_DSA_BWD_NUM_STAGES:-1}"
export PRIMUS_INDEXER_TRITON_FULL="${PRIMUS_INDEXER_TRITON_FULL:-1}"

# ---- memory -------------------------------------------------------------------------
export FUSED_LINEAR_CE="${FUSED_LINEAR_CE:-1}"
export FUSED_CE_CHUNK="${FUSED_CE_CHUNK:-4096}"
export PRIMUS_OVERLAP_PARAM_GATHER="${PRIMUS_OVERLAP_PARAM_GATHER:-false}"

# ---- packing ------------------------------------------------------------------------
export PRIMUS_PACKED=true
# cu_seqlens must actually reach the attention module as packed_seq_params; V4's attention
# is index-driven rather than TE-varlen, so the ROCm TE thd hang does not apply here.
export PRIMUS_PACKED_ATTN=true
# 1 = no alignment padding. The boundary exchange makes straddling windows correct, so
# raising this only throws supervised tokens away (56.1% -> 34.2% at 128), and its one
# real benefit is incidental: aligned segments are all >= 128, so HCA always has a
# window, whereas unaligned it has none on 22 of 39 packs.
export PRIMUS_PACK_ALIGN="${PRIMUS_PACK_ALIGN:-1}"
# The PyTorch window gather is the reference path and is what the tests validate. The
# dsv4_cp fused gather is opt-in and measured at 0.10 ms/step (0.003%), so it is not
# worth the risk here; it also refuses cross-boundary shards, where its window
# enumeration differs from this repo's strict ownership rule.
unset PRIMUS_THD_COMPACT_BACKEND 2>/dev/null || true

# ---- model / parallelism ------------------------------------------------------------
export PRIMUS_SEQ_LENGTH=131072
export PRIMUS_MAX_POSITION_EMBEDDINGS=131072
export SFT_JSONL="${ALPACA_JSONL}"
export PRIMUS_TOTAL_LAYERS=3
export PRIMUS_COMPRESS_RATIOS="[0, 4, 128, 0]"   # dense, CSA, HCA (+MTP slot)
export PRIMUS_RECOMPUTE_LAYERS=3
export PRIMUS_NUM_EXPERTS="${PRIMUS_NUM_EXPERTS:-8}"
export PRIMUS_MOE_TOPK="${PRIMUS_MOE_TOPK:-1}"
export PRIMUS_SHARD_HEADS=false
export PRIMUS_TP=1
export PRIMUS_ETP=1
export PRIMUS_EP=8
export PRIMUS_CP=8
export MBS=1
export GBS=1
export PRIMUS_LR="${PRIMUS_LR:-1.0e-6}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export V4_TOKENIZER
export PRIMUS_EXP_NAME="${PRIMUS_EXP_NAME:-dsv4_3layer_128k_thd_cp8}"

EXP="${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash_4layer-BF16-sft.yaml}"
LOGDIR="${PRIMUS_OUTPUT_ROOT}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOGDIR}"

echo "[run] repo=${REPO}"
echo "[run] PACKED (THD) seq=${PRIMUS_SEQ_LENGTH} layers=${PRIMUS_TOTAL_LAYERS} ratios=${PRIMUS_COMPRESS_RATIOS}"
echo "[run] CP=${PRIMUS_CP} TP=${PRIMUS_TP} EP=${PRIMUS_EP} pack_align=${PRIMUS_PACK_ALIGN} iters=${TRAIN_ITERS}"
echo "[run] data=${SFT_JSONL} (packed at runtime by PackedSFTDataset)"
echo "[run] log=${LOGDIR}/log_node0.txt"

./primus-cli direct -- train pretrain --config "${EXP}" \
  --manual_gc True \
  --manual_gc_interval 100 \
  --pp_warmup False --sequence_parallel False \
  --log_avg_skip_iterations 3 \
  --backend_path "${BACKEND_PATH}" \
  2>&1 | tee "${LOGDIR}/log_node0.txt"
