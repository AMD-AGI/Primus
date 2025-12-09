#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -e

# ------------------ Usage Help ------------------

print_usage() {
cat <<EOF
Usage: bash run_local_pretrain.sh

This script launches a Primus pretraining task inside a Docker/Podman container.

Environment Variables:
    DOCKER_IMAGE   Docker image to use [Default: docker.io/rocm/primus:v25.10_gfx942]
    MASTER_ADDR    Master node IP or hostname [Default: localhost]
    MASTER_PORT    Master node port [Default: 1234]
    NNODES         Total number of nodes [Default: 1]
    NODE_RANK      Rank of this node [Default: 0]
    GPUS_PER_NODE  GPUs per node [Default: 8]
    PRIMUS_*       Any environment variable prefixed with PRIMUS_ will be passed into the container.

Example:
    For Megatron: EXP=examples/megatron/exp_pretrain.yaml DATA_PATH=/mnt/data bash run_local_pretrain.sh
    For TorchTitan: EXP=eexamples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml DATA_PATH=/mnt/data bash run_local_pretrain.sh
    For MaxText: BACKEND=MaxText EXP=examples/maxtext/exp_pretrain.yaml bash run_local_pretrain.sh

EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# Path to experiment configuration YAML
EXP=${EXP:-"examples/megatron/exp_pretrain.yaml"}

# Default docker image
if [ "${BACKEND:-}" = "MaxText" ]; then
    DOCKER_IMAGE="docker.io/rocm/jax-training:maxtext-v25.9"
fi
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v25.10_gfx942"}

# Project root
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
echo "PRIMUS_PATH: $PRIMUS_PATH"

# Dataset directory
# DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
DATA_PATH=${DATA_PATH:-"$(pwd)/data"}
echo "DATA_PATH: $DATA_PATH"
mkdir -p "$DATA_PATH"

# ------------------ Cluster Env Defaults ------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ "$NODE_RANK" = "0" ]; then
    echo "========== Cluster info =========="
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "NNODES: $NNODES"
    echo "GPUS_PER_NODE: $GPUS_PER_NODE"
    echo ""
fi

# Pass all PRIMUS_ and NCCL_ environment variables into the container
ENV_ARGS=()

while IFS='=' read -r name _; do
    ENV_ARGS+=("--env" "$name")
done < <(env | grep "^PRIMUS_")
while IFS='=' read -r name _; do
    ENV_ARGS+=("--env" "$name")
done < <(env | grep "^NCCL_")
while IFS='=' read -r name _; do
    ENV_ARGS+=("--env" "$name")
done < <(env | grep "^PRIMUS_TURBO_")
ENV_ARGS+=("--env" "EXP")
ENV_ARGS+=("--env" "BACKEND")
if [ "${BACKEND:-}" = "MaxText" ]; then
    ENV_ARGS+=("--env" "DUMP_HLO")
fi
ENV_ARGS+=("--env" "HF_TOKEN")
ENV_ARGS+=("--env" "WANDB_API_KEY")
ENV_ARGS+=("--env" "ENABLE_NUMA_BINDING")
ENV_ARGS+=("--env" "HSA_KERNARG_POOL_SIZE")
ENV_ARGS+=("--env" "DATABRICKS_TOKEN")
ENV_ARGS+=("--env" "DATABRICKS_HOST")
ENV_ARGS+=("--env" "MLFLOW_TRACKING_URI")
ENV_ARGS+=("--env" "MLFLOW_REGISTRY_URI")
echo "ENV_ARGS: ${ENV_ARGS[*]}"

HOSTNAME=$(hostname)
ARGS=("$@")

VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")
if [[ -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
    VOLUME_ARGS+=(-v "$PATH_TO_BNXT_TAR_PACKAGE":"$PATH_TO_BNXT_TAR_PACKAGE")
fi

# using ainic
if [ "$USING_AINIC" == "1" ]; then
    ENV_ARGS+=("--env" "USING_AINIC")
    ENV_ARGS+=("--env" "RCCL_HOME_DIR")
    ENV_ARGS+=("--env" "ANP_HOME_DIR")
    ENV_ARGS+=("--env" "MPI_HOME_DIR")

    # VOLUME_ARGS+=(-v /mnt/shared:/mnt/shared)
    # VOLUME_ARGS+=(-v /etc/libibverbs.d/:/etc/libibverbs.d:ro)
    # VOLUME_ARGS+=(-v /usr/lib/x86_64-linux-gnu/libibverbs/:/usr/lib/x86_64-linux-gnu/libibverbs/:ro)
fi

export CLEAN_DOCKER_CONTAINER=${CLEAN_DOCKER_CONTAINER:-0}

# ------------------ Optional Container Cleanup ------------------
docker_podman_proxy() {
    if command -v podman &>/dev/null; then
        podman "$@"
    elif command -v docker &>/dev/null; then
        docker "$@"
    else
        echo "Neither Docker nor Podman found!" >&2
        return 1
    fi
}

if [[ "${CLEAN_DOCKER_CONTAINER:-0}" == "1" ]]; then
    echo "Node-${NODE_RANK}: Cleaning up existing containers..."
    CONTAINERS=$(docker_podman_proxy ps -aq)
    if [[ -n "$CONTAINERS" ]]; then
        for cid in $CONTAINERS; do
            docker_podman_proxy rm -f "$cid"
        done
        echo "Node-${NODE_RANK}: Removed containers: $CONTAINERS"
    else
        echo "Node-${NODE_RANK}: No containers to remove."
    fi
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "Node-${NODE_RANK}: Skipping training container launch."
    exit 0
else
    echo "Node-${NODE_RANK}: Launching training container."
fi

# ------------------ Launch Training Container ------------------
docker_podman_proxy run --rm \
    --env MASTER_ADDR \
    --env MASTER_PORT \
    --env NNODES \
    --env NODE_RANK \
    --env GPUS_PER_NODE \
    --env DATA_PATH \
    --env TRAIN_LOG \
    --env HSA_NO_SCRATCH_RECLAIM \
    --env NVTE_CK_USES_BWD_V3 \
    --env GPU_MAX_HW_QUEUES \
    --env GLOO_SOCKET_IFNAME \
    --env REBUILD_BNXT \
    --env PATH_TO_BNXT_TAR_PACKAGE \
    --env MEGATRON_PATH \
    --env TORCHTITAN_PATH \
    --env MAXTEXT_PATH \
    --env BACKEND_PATH \
    --env REBUILD_PRIMUS_TURBO \
    "${ENV_ARGS[@]}" \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged --device=/dev/infiniband \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
        cd $PRIMUS_PATH && \
        bash examples/run_pretrain.sh \"\$@\" 2>&1 && \
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: end, time=$(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
