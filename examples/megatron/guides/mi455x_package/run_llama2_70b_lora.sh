#!/usr/bin/env bash
# Llama-2-70B LoRA SFT on MI455X, in the MI455X image.
#
#   MODEL=<Llama-2-70b-hf dir> ./run_llama2_70b_lora.sh <out_dir> [extra train posttrain args...]
#
# Default: examples/megatron/configs/MI455X/llama2_70B-BF16-lora-sft.yaml, all 80 layers from the
# local HF snapshot in MODEL, LoRA on qkv/proj/fc1/fc2, Alpaca, 4 GPUs, 20 iterations
# (TRAIN_ITERS). The first run converts MODEL to a Megatron checkpoint under CKPT_DIR (default
# ~/models; ~130 GB) and later runs reuse it.
#
# CONFIG selects another recipe, e.g. the FP8 one or the 4-layer random-init proxy
# llama2_70B-BF16-lora-sft-1gpu-proxy.yaml (no MODEL needed; use NGPUS=1). NGPUS sets the GPU
# count (default 4). tatsu-lab/alpaca, and the proxy's tokenizer, come from HuggingFace into
# HF_CACHE (default ~/.cache/huggingface); set HF_TOKEN if needed.
# Writes <out_dir>/train.log.
set -euo pipefail
OUT=${1:?usage: run_llama2_70b_lora.sh <out_dir> [extra train posttrain args...]}; shift
IMAGE=${IMAGE:-amdprimus/amdprimus:mi455-training-20261108}
TRAIN_ITERS=${TRAIN_ITERS:-}
CONFIG=${CONFIG:-examples/megatron/configs/MI455X/llama2_70B-BF16-lora-sft.yaml}
NGPUS=${NGPUS:-4}
CKPT_DIR=${CKPT_DIR:-$HOME/models}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
NAME=${NAME:-llama2_70b_lora_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT" "$HF_CACHE" "$CKPT_DIR"

MOUNTS=(-v "$(cd "$OUT" && pwd)":/output -v "$(cd "$HF_CACHE" && pwd)":/root/.cache/huggingface
        -v "$(cd "$CKPT_DIR" && pwd)":/data)
ENVS=()
if [ -n "${MODEL:-}" ]; then
  MODEL=$(cd "$MODEL" && pwd)
  MOUNTS+=(-v "$MODEL":"/models/$(basename "$MODEL")":ro)
  ENVS+=(--env PRIMUS_HF_PATH="/models/$(basename "$MODEL")")
elif [[ "$CONFIG" != *proxy* ]]; then
  echo "set MODEL to a local Llama-2-70b-hf snapshot (or CONFIG=...-1gpu-proxy.yaml)" >&2; exit 1
fi
ARGS=()
[ -n "$TRAIN_ITERS" ] && ARGS+=(--train_iters "$TRAIN_ITERS")

docker run --rm --name "$NAME" "${MOUNTS[@]}" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --shm-size=64g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -w /workspace/Primus \
  "$IMAGE" \
  ./primus-cli direct \
    --env PRIMUS_GPU_MODEL=MI455X --env NNODES=1 --env GPUS_PER_NODE="$NGPUS" \
    --env HIP_VISIBLE_DEVICES="$(seq -s, 0 $((NGPUS - 1)))" \
    --env PRIMUS_TURBO_ATTN_BACKEND=triton \
    --env TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
    --env HSA_COREDUMP_PATTERN=/dev/null \
    --env PRIMUS_SKIP_PIP=1 \
    --env PRIMUS_WORKSPACE=/output \
    --env DATA_PATH=/data \
    "${ENVS[@]}" \
    -- train posttrain \
    --config "$CONFIG" "${ARGS[@]}" "$@" \
  2>&1 | tee "$OUT/train.log"
