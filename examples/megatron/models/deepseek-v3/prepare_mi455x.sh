#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Host-side setup for run_deepseek_v3_proxy_mi455x.sh. Both steps are optional:
#
#   TURBO_DIR=<path>  build that Primus-Turbo checkout in place for DOCKER_IMAGE
#   CONTAINER=<name>  start a container for PRIMUS_LAUNCHER=direct runs
#
#   TURBO_DIR=/path/to/Primus-Turbo CONTAINER=dsv3 bash examples/megatron/models/deepseek-v3/prepare_mi455x.sh
###############################################################################
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "$0")")/../../../.." && pwd)"
DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261015}
TURBO_DIR=${TURBO_DIR:+$(realpath -e "${TURBO_DIR}")}
MOUNTS=(-v "${REPO}:${REPO}")
if [[ -z "${TURBO_DIR}${CONTAINER:-}" ]]; then
    echo "set TURBO_DIR and/or CONTAINER" >&2
    exit 2
fi

if [[ -n "${TURBO_DIR}" ]]; then
    MOUNTS+=(-v "${TURBO_DIR}:${TURBO_DIR}")
    # Build for gfx1250 without probing a GPU, with OpenMPI off the paths so
    # rocSHMEM is not linked in, then install the flydsl version the checkout pins.
    docker run --rm -u "$(stat -c %u:%g "${TURBO_DIR}")" -e HOME=/tmp -v "${TURBO_DIR}:${TURBO_DIR}" -w "${TURBO_DIR}" \
        "${DOCKER_IMAGE}" bash -c '
        set -euo pipefail
        PATH=$(tr : "\n" <<< "${PATH}" | grep -v openmpi | paste -sd:)
        LD_LIBRARY_PATH=$(tr : "\n" <<< "${LD_LIBRARY_PATH:-}" | grep -v openmpi | paste -sd: || true)
        export PATH LD_LIBRARY_PATH
        GPU_ARCHS=gfx1250 python3 setup.py build_ext --inplace
        FLYDSL=$(grep -oE "flydsl==[0-9][0-9.]*" setup.py | head -n1 || true)
        if [[ -n "${FLYDSL}" ]]; then
            pip install -q --no-deps --upgrade --target build/pydeps "${FLYDSL}"
        fi'
fi

if [[ -n "${CONTAINER:-}" ]]; then
    docker run -d --name "${CONTAINER}" --privileged --network host --ipc host \
        --device /dev/kfd --device /dev/dri --group-add video \
        "${MOUNTS[@]}" -w "${REPO}" "${DOCKER_IMAGE}" sleep infinity
fi
