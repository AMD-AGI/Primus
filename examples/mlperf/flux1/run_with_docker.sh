#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the MLPerf datasets}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

DOCKER_IMAGE=${DOCKER_IMAGE:-zirui3/primus-v26.3-flux:v0.1}
CONFIG=${CONFIG:-examples/mlperf/flux1/flux.1_schnell_t2i-pretrain.yaml}
CONTAINER_NAME=${CONTAINER_NAME:-primus-mlperf-flux1-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

mkdir -p "$OUTPUT_ROOT"

rdma_args=()
if [[ -e /dev/infiniband ]]; then
  rdma_args=(--device=/dev/infiniband)
fi

docker run --rm --init --privileged \
  --name "$CONTAINER_NAME" \
  --device=/dev/kfd --device=/dev/dri "${rdma_args[@]}" --group-add video \
  --ipc=host --network=host --shm-size=20G --ulimit memlock=-1:-1 \
  -v "$REPO_ROOT:/workspace/Primus" \
  -v "$DATA_ROOT:/data" \
  -v "$OUTPUT_ROOT:/output" \
  -w /workspace/Primus \
  -e NNODES="$NNODES" -e NODE_RANK="$NODE_RANK" \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT" \
  -e GPUS_PER_NODE="$GPUS_PER_NODE" \
  -e CONFIG="$CONFIG" \
  -e PRIMUS_WORKSPACE="${PRIMUS_WORKSPACE:-/output/primus_workspace}" \
  -e DATASET_PATH="${DATASET_PATH:-/data/cc12m_preprocessed}" \
  -e EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/data/coco_preprocessed}" \
  -e EMPTY_ENCODINGS_PATH="${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}" \
  -e OUTPUT_DIR="${OUTPUT_DIR:-/output/flux_mlperf}" \
  -e FLUX_FLOAT8_RECIPE="${FLUX_FLOAT8_RECIPE:-tensorwise}" \
  -e ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn_aiter}" \
  -e LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-64}" \
  -e MAX_STEPS="${MAX_STEPS:-30000}" \
  -e GRADIENT_CHECKPOINTING_RATIO="${GRADIENT_CHECKPOINTING_RATIO:-0.25}" \
  -e COMPILE_TRANSFORMER_BLOCKS="${COMPILE_TRANSFORMER_BLOCKS:-true}" \
  -e FSDP2_RESHARD_AFTER_FORWARD="${FSDP2_RESHARD_AFTER_FORWARD:-false}" \
  -e FSDP2_REDUCE_DTYPE="${FSDP2_REDUCE_DTYPE:-fp32}" \
  -e SAVE_STEPS="${SAVE_STEPS:-100}" \
  -e SAVE_STRATEGY="${SAVE_STRATEGY:-dtcp_full}" \
  -e CHECKPOINT_KEEP_LATEST="${CHECKPOINT_KEEP_LATEST:-3}" \
  -e RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}" \
  -e MLPERF_ENABLE="${MLPERF_ENABLE:-true}" \
  -e MLPERF_CLEAR_CACHES="${MLPERF_CLEAR_CACHES:-true}" \
  -e MLLOG_OUTPUT_FILE="${MLLOG_OUTPUT_FILE:-/output/flux_mlperf/mlperf_compliance.log}" \
  -e TARGET_ACCURACY="${TARGET_ACCURACY:-0.586}" \
  -e VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-262144}" \
  -e SEED="${SEED:-10007}" \
  "$DOCKER_IMAGE" bash -c '
    set -euo pipefail
    export MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-$OUTPUT_DIR/mlperf_compliance.log}
    mkdir -p "$OUTPUT_DIR"
    if [[ "${MLPERF_CLEAR_CACHES:-true}" == "true" ]]; then
      sync
      echo 3 > /proc/sys/vm/drop_caches
    fi
    torchrun \
      --nnodes="$NNODES" --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
      --nproc_per_node="$GPUS_PER_NODE" \
      -m primus.cli.main train pretrain \
      --config "$CONFIG"
  '
