#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: enable build uccl settings.
#
# Trigger:
#   export REBUILD_UCCL=1
#
###############################################################################

set -euo pipefail

if [[ "${REBUILD_UCCL:-0}" != "1" ]]; then
    exit 0
fi

UCCL_DIR="/tmp/uccl"
UCCL_BUILD_DIR="${UCCL_BUILD_DIR:-/tmp/uccl_${HOSTNAME:-$(hostname)}}"
UCCL_REF="${UCCL_REF:-}"
GPU_ARCHS="${GPU_ARCHS:-gfx942;gfx950}"

LOG_INFO_RANK0 "[hook system] REBUILD_UCCL=1 → Building uccl in /tmp "
LOG_INFO_RANK0 "  Build directory : ${UCCL_BUILD_DIR}"
LOG_INFO_RANK0 "  GPU_ARCHS       : ${GPU_ARCHS}"

if [ -d "$UCCL_DIR" ]; then
	LOG_INFO_RANK0 "[hook system] Found existed uccl in /tmp, remove it"
	rm -rf $UCCL_DIR
fi

cd /tmp && git clone https://github.com/uccl-project/uccl.git


pushd $UCCL_DIR

# install dependencies
apt update && apt install -y rdma-core libibverbs-dev libnuma-dev libgoogle-glog-dev

if [[ -n "$UCCL_REF" ]]; then
	LOG_INFO_RANK0 "Checking out UCCL ref: ${UCCL_REF}"
    git fetch --all --tags
    git checkout "${UCCL_REF}"
fi

LOG_INFO_RANK0 "[hook system] Building uccl ep"
cd ep && PYTORCH_ROCM_ARCH="${GPU_ARCHS}" python3 setup.py build && cd ..

LOG_INFO_RANK0 "[hook system] Building uccl ep done"

cp ep/build/**/*.so uccl

pip3 install --no-build-isolation .
LOG_INFO_RANK0 "[hook system] Install uccl done"
# install deep_ep_wrapper
cd $UCCL_DIR/ep/deep_ep_wrapper
pip3 install --no-build-isolation .
LOG_INFO_RANK0 "[hook system] Install deep_ep done"

LOG_INFO_RANK0 "[hook system] Building uccl done."

popd
