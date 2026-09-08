#!/bin/bash
# FULL DeepSeek-V4-Flash SFT at 128k on PACKED sequences (THD), 3 nodes on gfx942.
#
# The complete model -- 43 decoder layers + 1 MTP, 256 experts (top-6) -- at 128k context.
# Delegates to run_full_4k_thd_multinode.sh for the corpus, packing and networking, and
# overrides only what long context needs.
#
# WHY THIS PARALLELISM IS THE ONLY ONE. 24 GPUs and PP=3 is forced by parameter memory, so
# TP x CP x DP = 8 remains. CP=8 is what makes 128k fit at all (it shards the sequence, and
# V4's dominant tensors have no head axis for TP to shard), which leaves DP=1. The expert
# side decomposes independently as ETP=1 x EP=8 x PP=3 = 24.
#
#     TP=1 x PP=3 x CP=8 x DP=1        local rows per rank = 131072/8 = 16384
#
# 16384 stays divisible by the HCA ratio 128, so every compressed branch is exercised.
#
# MEASURED, not estimated. Peak GPU memory against sequence length on this exact recipe:
#
#     seq      peak        note
#      4k    149.84 GB     CP=1; params + optimizer dominate
#     16k    150.41 GB
#     32k    156.03 GB
#     64k    166.06 GB
#    128k    186.32 GB     97.05% of a 192 GB card at offload 0.75
#    128k    163.60 GB     85.21% at offload 0.90  <-- the default here
#
# offload 0.90 is the default because 97% is not a configuration anyone can rely on: the
# reading is device-wide, so another process taking 6 GB is enough to kill the run. Moving
# more optimizer state to the host costs nothing in practice -- there is ~2.9 TB free there.
#
# Usage, per node, in its container:
#   MASTER_ADDR=<node0-ip> NODE_RANK=0 bash examples/deepseek-v4/gfx942/run_full_128k_thd_multinode.sh
#   MASTER_ADDR=<node0-ip> NODE_RANK=1 bash ...
#   MASTER_ADDR=<node0-ip> NODE_RANK=2 bash ...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRIMUS_SEQ_LENGTH="${PRIMUS_SEQ_LENGTH:-131072}"
export PRIMUS_MAX_POSITION_EMBEDDINGS="${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}"

# ---- parallelism -----------------------------------------------------------------------
export PRIMUS_TP="${PRIMUS_TP:-1}"
export PRIMUS_PP="${PRIMUS_PP:-3}"
export PRIMUS_CP="${PRIMUS_CP:-8}"
export PRIMUS_EP="${PRIMUS_EP:-8}"
export PRIMUS_ETP="${PRIMUS_ETP:-1}"
# DP is then 24/(TP*PP*CP) = 1, so GBS is exactly the microbatch count. 3 matches PP, which
# is the smallest value that keeps the pipeline from idling more than it already does.
export MBS="${MBS:-1}"
export GBS="${GBS:-3}"

# ---- memory ------------------------------------------------------------------------------
export PRIMUS_OPTIMIZER_CPU_OFFLOAD="${PRIMUS_OPTIMIZER_CPU_OFFLOAD:-true}"
export PRIMUS_OPTIMIZER_OFFLOAD_FRACTION="${PRIMUS_OPTIMIZER_OFFLOAD_FRACTION:-0.9}"

# Chunked linear + cross-entropy. REQUIRED at this length, not an optimisation: the LM
# head's logits are [S, B, vocab] = 31.6 GB at 128k with vocab 129280, before the loss even
# upcasts. PRIMUS_OVERLAP_PARAM_GATHER must be false alongside it -- the chunked backward
# issues one autograd.grad per chunk, which changes each parameter's hook firing count and
# desyncs the distributed optimizer's overlapped gather. DeepseekV4Model raises if you
# forget.
export FUSED_LINEAR_CE="${FUSED_LINEAR_CE:-1}"
export FUSED_CE_CHUNK="${FUSED_CE_CHUNK:-4096}"
export PRIMUS_OVERLAP_PARAM_GATHER="${PRIMUS_OVERLAP_PARAM_GATHER:-false}"

export PRIMUS_EXP_NAME="${PRIMUS_EXP_NAME:-dsv4_flash_full_128k_thd_${NNODES:-3}node}"

echo "[128k] seq=${PRIMUS_SEQ_LENGTH} TP=${PRIMUS_TP} PP=${PRIMUS_PP} CP=${PRIMUS_CP} EP=${PRIMUS_EP} GBS=${GBS}"
echo "[128k] offload=${PRIMUS_OPTIMIZER_OFFLOAD_FRACTION} fused_ce=${FUSED_LINEAR_CE}/${FUSED_CE_CHUNK}"

exec bash "${HERE}/run_full_4k_thd_multinode.sh"
