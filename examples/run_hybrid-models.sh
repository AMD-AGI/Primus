#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -e

DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/pytorch-training:v25.4"}
echo "DOCKER_IMAGE: $DOCKER_IMAGE"
echo "HF_TOKEN: $HF_TOKEN"

ENV_ARGS=()
ENV_ARGS+=("--env" "HF_TOKEN")
ARGS=("$@")

echo "Running Zebra-Llama in ${PWD}"
docker run --rm \
    --env HF_TOKEN \
    --device /dev/dri --device /dev/kfd \
    --device=/dev/infiniband --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME --shm-size 64G --name mla_training \
    $DOCKER_IMAGE /bin/bash -c "\
        echo 'Running Zebra-Llama in ${PWD} with HF_TOKEN: ${HF_TOKEN}' && \
        git clone https://github.com/AMD-AGI/AMD-Hybrid-Models.git && \
        ls && cd AMD-Hybrid-Models/Zebra-Llama && \
        bash install.sh FLASH_ATTN=1 MAMBA=1 && \
        ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/fsdp.yaml train_hybrid/train_distill.py configs/llama3.2_1B/zebra_8MLA8M2_8bt_SFT.yaml 
    " bash "${ARGS[@]}"