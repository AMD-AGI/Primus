#!/bin/bash
# One-click 128k DeepSeek-V4 SFT on a single 8x MI308X (gfx942 / CDNA3) node.
#
# Model: 4 decoder layers with compress_ratios [0, 0, 4, 128] -- so the run exercises all
# three V4 attention branches (dense+SWA, CSA, HCA), not just the cheap dense one.
#
# Parallelism is CP=8 / TP=1 / EP=8. That is NOT the obvious choice and it matters a lot:
# the same recipe at TP=8 / CP=1 peaks at 188.63 GB and 9.4 s/step, this one at 42.30 GB
# and 3.83 s/step. The reason is that the dominant tensors do not have a head axis, so TP
# cannot shard them -- the indexer's `scores` is [B, S, P] (heads are already summed out),
# V4's KV is a single MQA latent, and the MoE / residual activations scale with S alone.
# CP shards the sequence and therefore shards all of them.
#
# Everything is resolved relative to this script, so the repo can live anywhere. Run it
# from inside a rocm/primus:v26.5-pytorch2.12-te2.15 container:
#
#     bash examples/deepseek-v4/gfx942/run_128k_dense_hca_csa.sh
#
# Required: a local DeepSeek-V4 tokenizer directory. Point V4_TOKENIZER at it, or drop it
# at /apps/DeepSeek-V4-Flash. Only tokenizer.json / tokenizer_config.json are read -- the
# weights are NOT loaded (no V4 Megatron checkpoint exists; this trains from random init).
#
# Why each gfx942-specific setting is needed is documented in
# examples/deepseek-v4/gfx942/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
cd "${REPO}"

# ---- prerequisites -----------------------------------------------------------------
V4_TOKENIZER="${V4_TOKENIZER:-/apps/DeepSeek-V4-Flash}"
DATA_DIR="${DATA_DIR:-${REPO}/data/sft}"
BACKEND_PATH="${BACKEND_PATH:-${REPO}/third_party/Megatron-LM}"

if [ ! -f "${V4_TOKENIZER}/tokenizer.json" ]; then
  echo "[error] no tokenizer at ${V4_TOKENIZER}. Set V4_TOKENIZER=<dir> (needs tokenizer.json)." >&2
  exit 1
fi

# Megatron-LM: the repo pins it as a submodule, but the container image already ships the
# exact pinned commit, so reuse that instead of a network fetch when it is available.
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

if [ ! -f "${DATA_DIR}/sft_131072.jsonl" ]; then
  echo "[setup] building SFT data (one-off, a few minutes)"
  python "${HERE}/prepare_sft_data.py" --tokenizer "${V4_TOKENIZER}" \
         --out-dir "${DATA_DIR}" --lengths 131072 --rows 16
fi

# ---- launcher ----------------------------------------------------------------------
export PRIMUS_LAUNCHER=direct
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST 2>/dev/null || true
export NNODES=1 NODE_RANK=0 MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29517}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export BACKEND_PATH
export PRIMUS_OUTPUT_ROOT="${PRIMUS_OUTPUT_ROOT:-${REPO}/output}"
export PRIMUS_TEAM="${PRIMUS_TEAM:-amd}"
export PRIMUS_USER="${PRIMUS_USER:-$(whoami)}"
export LOCAL_RANKS="${LOCAL_RANKS:---local-ranks-filter 0,1,2,3,4,5,6,7}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Let an OOM surface as a clean HIP out-of-memory instead of an opaque
# HSA_STATUS_ERROR_EXCEPTION + GPU coredump.
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# ---- gfx942 (CDNA3) kernel settings -------------------------------------------------
# The V4 sparse-MLA backward Triton kernel is tuned for gfx950/CDNA4 (160 KB LDS). gfx942
# has 64 KB and the stock pipeline staging asks for 73728 B, so it fails to COMPILE.
# num_stages=1 turns off Triton's LDS multi-buffering. Measured: this is the only kernel
# switch needed -- PRIMUS_HC_TRITON and PRIMUS_DSA_DKV_SAFE can stay at their defaults.
export PRIMUS_DSA_BWD_NUM_STAGES="${PRIMUS_DSA_BWD_NUM_STAGES:-1}"
# Fused indexer scoring: without it the CSA indexer falls back to an eager einsum that
# materialises [B, S, H, P] -- 512 GiB at 128k.
export PRIMUS_INDEXER_TRITON_FULL="${PRIMUS_INDEXER_TRITON_FULL:-1}"

# ---- memory ---------------------------------------------------------------------------
# Chunked linear + cross-entropy. The LM head's logits are [S, B, vocab]; at 128k with
# vocab 129280 that is 31.6 GiB before the loss even upcasts, and it was measured as the
# single largest allocation in the step -- larger than any attention tensor. Off by
# default upstream; this recipe wants it.
export FUSED_LINEAR_CE="${FUSED_LINEAR_CE:-1}"
export FUSED_CE_CHUNK="${FUSED_CE_CHUNK:-4096}"
# REQUIRED with FUSED_LINEAR_CE: the chunked backward issues one autograd.grad per chunk,
# which changes each parameter's backward-hook firing count and desyncs the distributed
# optimizer's overlapped parameter all-gather. DeepseekV4Model raises if you forget.
export PRIMUS_OVERLAP_PARAM_GATHER="${PRIMUS_OVERLAP_PARAM_GATHER:-false}"

# ---- model / parallelism ------------------------------------------------------------
export PRIMUS_SEQ_LENGTH=131072
export PRIMUS_MAX_POSITION_EMBEDDINGS=131072
export SFT_JSONL="${DATA_DIR}/sft_131072.jsonl"
export PRIMUS_TOTAL_LAYERS=4
export PRIMUS_COMPRESS_RATIOS="[0, 0, 4, 128, 0]"   # dense, dense, CSA, HCA (+MTP slot)
export PRIMUS_RECOMPUTE_LAYERS=4
export PRIMUS_NUM_EXPERTS="${PRIMUS_NUM_EXPERTS:-8}"
export PRIMUS_MOE_TOPK="${PRIMUS_MOE_TOPK:-1}"
# CP=8 shards the sequence; EP=8 shards the experts over the same 8 ranks (the expert side
# decomposes as ETP*EP*PP, which does not include CP, so both can be 8 on 8 GPUs).
# P14 head sharding is off because TP=1 -- see the header for why TP is the wrong lever.
export PRIMUS_SHARD_HEADS=false
export PRIMUS_TP=1
export PRIMUS_ETP=1
export PRIMUS_EP=8
export PRIMUS_CP=8
export MBS=1
export GBS=1
# Random init at 128k diverges at the default 1e-5 (NaN on step 2); 1e-6 is stable.
export PRIMUS_LR="${PRIMUS_LR:-1.0e-6}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export V4_TOKENIZER
export PRIMUS_EXP_NAME="${PRIMUS_EXP_NAME:-dsv4_4layer_128k_cp8}"

EXP="${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash_4layer-BF16-sft.yaml}"
LOGDIR="${PRIMUS_OUTPUT_ROOT}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOGDIR}"

echo "[run] repo=${REPO}"
echo "[run] seq=${PRIMUS_SEQ_LENGTH} layers=${PRIMUS_TOTAL_LAYERS} ratios=${PRIMUS_COMPRESS_RATIOS}"
echo "[run] CP=${PRIMUS_CP} TP=${PRIMUS_TP} EP=${PRIMUS_EP}  iters=${TRAIN_ITERS}"
echo "[run] expect ~42 GB peak, ~3.8 s/step"
echo "[run] log=${LOGDIR}/log_node0.txt"

./primus-cli direct -- train pretrain --config "${EXP}" \
  --manual_gc True \
  --manual_gc_interval 100 \
  --pp_warmup False --sequence_parallel False \
  --log_avg_skip_iterations 3 \
  --backend_path "${BACKEND_PATH}" \
  2>&1 | tee "${LOGDIR}/log_node0.txt"
