#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Single-GPU gfx1250 (MI455X) MaxText / JAX repro launcher.
#
#   IMAGE=<jax gfx1250 image> run.sh maxtext <name> [extra docker args...]
#       1-layer Gemma 4 31B-width MaxText run, vocab 131072, lr=0, wd=0, 3 steps,
#       with PRIMUS_MAXTEXT_OPT_DEBUG on. Log: $OUT/<name>.log, arrays: $OUT/<name>/.
#       Extra args are passed to docker run, e.g. -e PRIMUS_LR=3e-5 -e DUMP_HLO=1.
#       CONFIG selects another Primus config (path relative to the Primus root).
#
#   IMAGE=<jax gfx1250 image> run.sh probe <script.py> [extra docker args...]
#       Run one of the standalone probes in this directory. Log: $OUT/<script>.log.
#
# Env: IMAGE (required), OUT (default ./maxtext_gfx1250_debug_out),
#      TIMEOUT_S (default 3600), HIP_DEVICE (default 0).
set -uo pipefail

MODE=${1:?usage: run.sh maxtext|probe <name|script.py> [docker args...]}
TARGET=${2:?usage: run.sh maxtext|probe <name|script.py> [docker args...]}
shift 2

IMAGE=${IMAGE:?set IMAGE to a JAX image with a gfx1250 ROCm plugin}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRIMUS=$(cd "$HERE/../.." && pwd)
OUT=$(mkdir -p "${OUT:-$PWD/maxtext_gfx1250_debug_out}" && cd "${OUT:-$PWD/maxtext_gfx1250_debug_out}" && pwd)
TIMEOUT_S=${TIMEOUT_S:-3600}
# hipBLASLt in pip ROCm wheels keeps its Tensile files one level below where it looks.
TENSILE=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx1250

NAME="gfx1250dbg_${MODE}_$(basename "$TARGET" .py)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# Container group names need not match the host's GIDs for /dev/kfd and /dev/dri.
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

common=(
  --rm -i --name "$NAME"
  --device /dev/kfd --device /dev/dri --group-add "${VIDEO_GID:?}" --group-add "${RENDER_GID:?}"
  --ipc host --shm-size 16G
  -e HIP_VISIBLE_DEVICES="${HIP_DEVICE:-0}"
  -e HIPBLASLT_TENSILE_LIBPATH="$TENSILE"
  -e XLA_GPU_AUTOTUNE_LEVEL=0
  -e PYTHONUNBUFFERED=1
)

case "$MODE" in
  maxtext)
    LOG="$OUT/$TARGET.log"
    mkdir -p "$OUT/$TARGET"
    timeout --signal=KILL "$TIMEOUT_S" docker run "${common[@]}" \
      -v "$PRIMUS":/workspace/primus -v "$OUT/$TARGET":/dump \
      -e MAXTEXT_PATH=/workspace/primus/third_party/maxtext \
      -e NVTE_FUSED_ATTN=0 -e NVTE_FUSED_ATTN_CK=0 -e NVTE_FUSED_ATTN_AOTRITON=0 \
      -e GPUS_PER_NODE=1 -e NNODES=1 \
      -e PRIMUS_STEPS=3 -e PRIMUS_NUM_LAYERS=1 -e PRIMUS_ATTENTION=dot_product \
      -e PRIMUS_VOCAB_SIZE=131072 -e PRIMUS_LR=0 -e PRIMUS_ADAM_WD=0 \
      -e PRIMUS_ABORT_ON_NAN_LOSS=false \
      -e PRIMUS_MAXTEXT_OPT_DEBUG=1 -e PRIMUS_MAXTEXT_OPT_DEBUG_DUMP=/dump \
      "$@" \
      "$IMAGE" bash -lc 'cd /workspace/primus && ./primus-cli direct -- train pretrain --config "$1"' \
      _ "${CONFIG:-examples/maxtext/configs/MI455X/gemma4_31B-bf16-pretrain_1gpu_proxy.yaml}" \
      >"$LOG" 2>&1
    rc=$?
    ;;
  probe)
    LOG="$OUT/$(basename "$TARGET" .py).log"
    timeout --signal=KILL "$TIMEOUT_S" docker run "${common[@]}" \
      -v "$HERE":/probes:ro \
      "$@" \
      "$IMAGE" python3 "/probes/$(basename "$TARGET")" >"$LOG" 2>&1
    rc=$?
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo ">>> $MODE $TARGET exit=$rc log=$LOG"
if [ "$MODE" = maxtext ]; then
  sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -aE "OPTDBG (post|BAD|norms)|completed step" | sed 's/.*\[INFO\] *//' | cut -c1-260
fi
exit "$rc"
