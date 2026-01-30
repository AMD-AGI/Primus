#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v26.1"}
DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
SANITIZED_USER=$(echo "${USER:-unknown}" | tr -cd '[:alnum:]_-')
if [ -z "$SANITIZED_USER" ]; then
    SANITIZED_USER="unknown"
fi
CONTAINER_NAME=${CONTAINER_NAME:-"dev_primus_${SANITIZED_USER}"}

bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run -d \
    --name "${CONTAINER_NAME}" \
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
    --env DATA_PATH="/data" \
    -v "${PRIMUS_PATH}:/workspace/Primus" \
    -v "${DATA_PATH}:/data" \
    -w "/workspace/Primus" \
    "$DOCKER_IMAGE" sleep infinity
