#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
DOCKER_IMAGE=${DOCKER_IMAGE:-"rocm/megatron-lm-private:c798f55e-b2d8-4e1e-8241-72b49bc39ab0"}
DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}

bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run -d \
    --name dev_primus \
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
    -v "${PRIMUS_PATH}:/workspace/Primus" \
    -v "${DATA_PATH}:${DATA_PATH}" \
    -v /data/mlperf_llama31_8b/data:/data \
    -w "/workspace/Primus" \
    "$DOCKER_IMAGE" sleep infinity
