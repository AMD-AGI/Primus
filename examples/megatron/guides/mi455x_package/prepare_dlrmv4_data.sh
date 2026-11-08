#!/usr/bin/env bash
# Download and prepare yambda-5b for DLRMv4, CPU only, inside the MI455X image.
#
#   IMAGE=<image> ./prepare_dlrmv4_data.sh <data_dir>
#
#   1. download_dataset.sh: yandex/yambda from HuggingFace + the reference preprocessing
#      (raw/5b, shared_metadata/, processed_5b/)
#   2. the flat-event cache processed_5b/hstu_cache_L4086/ (~227 GB; ~150-190 GB peak host RAM),
#      positions / anchor timestamps, and the offline streaming TS index (window 86400 s)
#   3. verify_dataset.sh: processed md5s and the cache sha256s
# Steps already done are skipped, so the script can be re-run after an interruption.
# Needs ~500 GB free in <data_dir> and network access to huggingface.co.
# READONLY=1 mounts <data_dir> read-only: only checks an already prepared directory.
set -euo pipefail
DATA=${1:?usage: prepare_dlrmv4_data.sh <data_dir>}
IMAGE=${IMAGE:-amdprimus/amdprimus:mi455-training-20261108}
MOUNT_OPT=""
if [ "${READONLY:-0}" = 1 ]; then MOUNT_OPT=":ro"; else mkdir -p "$DATA"; fi

docker run --rm -i \
  -v "$(realpath -m "$DATA")":/data$MOUNT_OPT \
  -e HF_TOKEN="${HF_TOKEN:-}" -e READONLY="${READONLY:-0}" \
  -w /workspace/recommendation \
  "$IMAGE" bash -s <<'EOS'
set -euo pipefail
export MLPERF_LOGGING=0 MLPERF_ALLOW_INDEX_BUILD=1 DLRM_DATA_PATH=/data PYTHONUNBUFFERED=1 \
       PYTHONPATH=/workspace/recommendation
C=/data/processed_5b/hstu_cache_L4086

# Skip the download for a preprocessed tree or for the prepared-dataset layout (cache, no parquet).
if [ ! -f /data/processed_5b/split_meta.json ] && [ ! -f "$C/_READY" ]; then
  ./download_dataset.sh
fi

python3 - <<'PY'
import logging, os
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from generative_recommenders.dlrm_v4.datasets.yambda import DLRMv4YambdaDataset as D, _FlatEventStore
proc = "/data/processed_5b"
cache = f"{proc}/hstu_cache_L4086"
D._ensure_cache_built(cache, proc, 4086)
D._ensure_positions_built(cache, "positions_L4086.npy", 4086)
anchor = f"{cache}/anchor_ts_L4086.npy"
if not os.path.exists(anchor):
    store = _FlatEventStore.load_mmap(cache)
    pos = np.load(f"{cache}/positions_L4086.npy", mmap_mode="r")
    tmp = anchor + ".tmp.npy"
    np.save(tmp, store.flat_timestamps[pos])
    os.replace(tmp, anchor)
logging.info("cache, positions, anchor_ts ready")
PY

T=scripts/ts_index_tool.py
if [ ! -f "$C/ts_indices_L4086_W86400.npy" ] || [ ! -f "$C/ts_offsets_L4086_W86400.npy" ]; then
  python3 $T prebuild --cache-dir "$C" --history-length 4086 --min-history 4086 --window-seconds 86400
fi
# A read-only check cannot record the result in the index manifest; the verify itself must pass.
if ! out=$(python3 $T verify --cache-dir "$C" --history-length 4086 --min-history 4086 --window-seconds 86400 2>&1); then
  echo "$out"
  [ "$READONLY" = 1 ] && grep -q "verify: PASSED" <<< "$out" || exit 1
else
  echo "$out"
fi
./verify_dataset.sh
echo DLRMV4_DATA_READY
EOS
