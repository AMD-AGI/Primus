#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DOCKER_IMAGE=${DOCKER_IMAGE:-rocm/primus:v26.2}
CONTAINER_NAME=${CONTAINER_NAME:-primus-flux-mlperf}
NOFILE_LIMIT=${NOFILE_LIMIT:-1048576}

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export DATASET_PATH=${DATASET_PATH:-/data/cc12m-preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export OUTPUT_DIR=${OUTPUT_DIR:-/shared_nfs/zirui/runs/flux_rcp_compare/primus}
export SEED=${SEED:-1234}
export SAVE_STEPS=${SAVE_STEPS:-100}
export CHECKPOINT_KEEP_LATEST=${CHECKPOINT_KEEP_LATEST:-3}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}
export ENABLE_WANDB_LOGGER=${ENABLE_WANDB_LOGGER:-false}
export WANDB_PROJECT=${WANDB_PROJECT:-mlperf-flux-rcp}
export WANDB_RUN_ID=${WANDB_RUN_ID:-primus-flux}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-primus-flux}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_OFFLINE=${WANDB_OFFLINE:-false}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-train}

env_names=(
    GPUS_PER_NODE DATASET_PATH EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR
    SAVE_STEPS SAVE_STRATEGY CHECKPOINT_KEEP_LATEST RESUME_FROM_CHECKPOINT
    ENABLE_WANDB_LOGGER WANDB_PROJECT WANDB_RUN_ID WANDB_RUN_NAME WANDB_RESUME
    WANDB_API_KEY WANDB_ENTITY WANDB_GROUP WANDB_TAGS WANDB_OFFLINE WANDB_JOB_TYPE
    LOCAL_BATCH_SIZE MAX_STEPS LR WARMUP_STEPS SEED LOG_FREQ TARGET_ACCURACY
    VAL_CHECK_INTERVAL MLPERF_ENABLE MLPERF_CLEAR_CACHES ATTENTION_BACKEND
    GRADIENT_CHECKPOINTING COMPILE_TRANSFORMER_BLOCKS FSDP2_RESHARD_AFTER_FORWARD
    FLUX_LOW_PRECISION_PROVIDER FLUX_LOW_PRECISION_RECIPE
)
docker_env_args=()
for name in "${env_names[@]}"; do
    if [[ -v "$name" ]]; then
        docker_env_args+=(--env "$name")
    fi
done

docker_tty_args=()
if [[ -t 0 && -t 1 ]]; then
    docker_tty_args=(-it)
fi

docker run "${docker_tty_args[@]}" --rm --init \
    --name "$CONTAINER_NAME" \
    --ulimit "nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host --network=host --privileged \
    --shm-size=20G \
    -v /shared_nfs:/shared_nfs \
    -v /shared_nfs/zirui/models:/models \
    -v /shared_nfs/zirui/data:/data \
    -v "$SCRIPT_DIR:/workspace/code" \
    -w /workspace/code \
    "${docker_env_args[@]}" \
    "$DOCKER_IMAGE" \
    bash local_runs/run_flux_mlperf.sh
