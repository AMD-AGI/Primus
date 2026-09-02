#!/bin/bash
# FULL DeepSeek-V4-Flash SFT at 4k on PACKED sequences (THD), multi-node on gfx942.
#
# Same model and parallelism as run_full_4k_multinode.sh -- 43 decoder layers + 1 MTP,
# 256 experts (top-6), TP=1 * PP=3 * CP=1 over 3 nodes, optimizer CPU-offloaded -- with
# sequence packing turned on. That script is exec'd for everything except the data layout,
# so the two recipes cannot drift apart.
#
# Why packing matters at 4k on the full model: alpaca's runtime segments have a median of
# 84 tokens, so an unpacked 4096-token row is mostly padding and the run measures the
# hardware rather than the model. Packing fills the window with real samples and keeps
# attention from crossing sample boundaries (cu_seqlens -> PackedSeqParams -> the V4
# index matrix).
#
# NODE ADDRESSES ARE NOT HARDCODED. On each node, in its container:
#
#   MASTER_ADDR=<node0-ip> NODE_RANK=0 bash examples/deepseek-v4/gfx942/run_full_4k_thd_multinode.sh
#   MASTER_ADDR=<node0-ip> NODE_RANK=1 bash examples/deepseek-v4/gfx942/run_full_4k_thd_multinode.sh
#   MASTER_ADDR=<node0-ip> NODE_RANK=2 bash examples/deepseek-v4/gfx942/run_full_4k_thd_multinode.sh
#
# Everything run_full_4k_multinode.sh documents (NIC selection, NCCL_ALGO=Ring, offload
# fraction) applies unchanged; see README_full_4k_multinode.md.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
cd "${REPO}" || exit 1

DATA_DIR="${DATA_DIR:-${REPO}/data/sft}"

# ---- packing ------------------------------------------------------------------------
export PRIMUS_PACKED="${PRIMUS_PACKED:-true}"
# cu_seqlens must actually reach attention. V4 is index-driven rather than TE-varlen, so
# the ROCm TE thd limitation that makes this default-off elsewhere does not apply.
export PRIMUS_PACKED_ATTN="${PRIMUS_PACKED_ATTN:-true}"
# 1 = no alignment padding; the left-boundary exchange handles straddling windows.
export PRIMUS_PACK_ALIGN="${PRIMUS_PACK_ALIGN:-1}"

# ---- corpus: NATURAL-LENGTH alpaca rows ---------------------------------------------
# Not the pre-packed {input_ids, labels, cu_seqlens} file prepare_packed_data.py writes --
# the training pipeline has no loader for that schema, and feeding it produces a run that
# completes with exit 0 and grad norm 0.000. Packing happens at runtime in
# PackedSFTDataset, which is what PRIMUS_PACK_ALIGN applies to.
ALPACA_JSONL="${ALPACA_JSONL:-${DATA_DIR}/alpaca_natural.jsonl}"
if [ ! -f "${ALPACA_JSONL}" ]; then
  echo "[setup] fetching natural-length alpaca rows (one-off)"
  mkdir -p "${DATA_DIR}"
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
export SFT_JSONL="${ALPACA_JSONL}"

# Pack cache lives on shared /apps, but its FILELOCK must not. With 24 ranks across 3
# hosts, one rank builds under the lock while the other 23 block on it with filelock's
# default timeout=-1; on NFS that also risks ESTALE. Point the lock at node-local disk and
# pre-warm the cache once (single process) before launching, so no rank ever builds under
# a contended lock -- a cold build that outlasts Megatron's 10-minute collective timeout
# takes the whole job down.
export PRIMUS_PACK_LOCK_DIR="${PRIMUS_PACK_LOCK_DIR:-/tmp/primus_pack_locks}"
mkdir -p "${PRIMUS_PACK_LOCK_DIR}"
# Cache on NODE-LOCAL storage, one per node. Sharing it over NFS is worse than it looks:
# the writer stages to a FIXED "<final>.tmp" name and then renames, so two nodes building
# concurrently collide on that path and one dies with ENOENT during os.replace. The lock
# above is node-local and cannot serialise them. Building per node costs a few minutes
# each but is race-free, and the build is deterministic -- no seed, hostname, clock or
# directory order feeds it -- so the nodes end up with identical packs. The launch checks
# that: every node must print the same cache key and the same pack count.
export PRIMUS_PACK_CACHE_DIR="${PRIMUS_PACK_CACHE_DIR:-/root/.cache/primus_packed_sft}"
mkdir -p "${PRIMUS_PACK_CACHE_DIR}"

export PRIMUS_EXP_NAME="${PRIMUS_EXP_NAME:-dsv4_flash_full_4k_thd_${NNODES:-3}node}"

echo "[thd] packing=${PRIMUS_PACKED} packed_attn=${PRIMUS_PACKED_ATTN} align=${PRIMUS_PACK_ALIGN}"
echo "[thd] corpus=${SFT_JSONL} (packed at runtime by PackedSFTDataset)"

exec bash "${HERE}/run_full_4k_multinode.sh"
