#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v25.10"}
DATA_PATH=${DATA_PATH:-"/data/mlperf_llama31_8b/data"}
HF_TOKEN=${HF_TOKEN:-""}

bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run -d \
    --name dev_primus-$USER \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    --env DATA_PATH="${DATA_PATH}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    -v "${PRIMUS_PATH}:/workspace/Primus" \
    -v "${DATA_PATH}:/data \
    -w "/workspace/Primus" \
    "$DOCKER_IMAGE" sleep infinity
