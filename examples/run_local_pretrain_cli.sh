#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# shellcheck disable=SC2086,SC2048
set -e

# Path to experiment configuration YAML
EXP=${EXP:-"examples/megatron/exp_pretrain.yaml"}

# Default docker image
if [ "${BACKEND:-}" = "MaxText" ]; then
    DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/jax-training:maxtext-v25.9"}
else
    DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v25.10"}
fi

# ------------------ Cluster Env Defaults ------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Project root
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
echo "PRIMUS_PATH: $PRIMUS_PATH"

# Dataset directory
DATA_PATH=${DATA_PATH:-"$(pwd)/data"}
echo "DATA_PATH: $DATA_PATH"
mkdir -p "$DATA_PATH"


bash $PRIMUS_PATH/runner/primus-cli \
container \
    --image $DOCKER_IMAGE \
    --volume $DATA_PATH:$DATA_PATH \
-- \
    --env HSA_NO_SCRATCH_RECLAIM \
    --env NVTE_CK_USES_BWD_V3 \
    --env GPU_MAX_HW_QUEUES \
    --env GLOO_SOCKET_IFNAME \
    -- train pretrain --config $EXP $*