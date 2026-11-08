#!/usr/bin/env bash
# Kimi K3.1 Pro 4-layer BF16 proxy on one node of 4 x MI455X, in the MI455X image.
#
#   IMAGE=<image> ./run_kimi_k3.1_pro.sh <out_dir> [extra train pretrain args...]
#
# Runs examples/megatron/configs/MI455X/kimi_k3.1_pro_4L-BF16-pretrain.yaml from the Primus tree
# baked into the image (/workspace/Primus) with mock data, 12 iterations by default
# (TRAIN_ITERS). Extra arguments are appended to `train pretrain`, e.g. `--use_turbo_deepep False`.
# PRIMUS_SRC=<Primus checkout> mounts that tree over /workspace/Primus instead.
# CONFIG selects another recipe; PRIMUS_ENVS="K=V ..." adds environment variables.
# Writes <out_dir>/train.log.
set -euo pipefail
OUT=${1:?usage: run_kimi_k3.1_pro.sh <out_dir> [extra train pretrain args...]}; shift
IMAGE=${IMAGE:-amdprimus/amdprimus:mi455-training-20261108}
TRAIN_ITERS=${TRAIN_ITERS:-12}
CONFIG=${CONFIG:-examples/megatron/configs/MI455X/kimi_k3.1_pro_4L-BF16-pretrain.yaml}
NAME=${NAME:-kimi_k31_pro_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
SRC_MOUNT=()
[ -n "${PRIMUS_SRC:-}" ] && SRC_MOUNT=(-v "$(cd "$PRIMUS_SRC" && pwd)":/workspace/Primus)
EXTRA_ENVS=()
for kv in ${PRIMUS_ENVS:-}; do EXTRA_ENVS+=(--env "$kv"); done

docker run --rm --name "$NAME" "${SRC_MOUNT[@]}" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --shm-size=64g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -v "$(cd "$OUT" && pwd)":/output \
  -w /workspace/Primus \
  "$IMAGE" \
  ./primus-cli direct \
    --env PRIMUS_GPU_MODEL=MI455X --env NNODES=1 --env GPUS_PER_NODE=4 \
    --env HIP_VISIBLE_DEVICES=0,1,2,3 \
    --env TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
    --env HSA_COREDUMP_PATTERN=/dev/null \
    --env PRIMUS_SKIP_PIP=1 \
    --env PRIMUS_WORKSPACE=/output \
    "${EXTRA_ENVS[@]}" \
    -- train pretrain \
    --config "$CONFIG" \
    --train_iters "$TRAIN_ITERS" "$@" \
  2>&1 | tee "$OUT/train.log"
