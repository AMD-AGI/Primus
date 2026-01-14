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
#   export USING_UCCL=1
#
###############################################################################

set -euo pipefail

if [[ "${USING_UCCL:-0}" != "1" ]]; then
    exit 0
fi

UCCL_DIR="/tmp/uccl"

LOG_INFO_RANK0 "[hook system] USING_UCCL=1 → Building uccl in /tmp "


if [ -d "$UCCL_DIR" ]; then
	LOG_INFO_RANK0 "[hook system] Found existed uccl in /tmp"
else
	cd /tmp && git clone https://github.com/uccl-project/uccl.git
fi

pushd $UCCL_DIR

# install dependencies
apt update && apt install -y rdma-core libibverbs-dev libnuma-dev libgoogle-glog-dev

LOG_INFO_RANK0 "[hook system] Building uccl ep"
cd ep && python3 setup.py build && cd ..

LOG_INFO_RANK0 "[hook system] Building uccl ep done"

cp ep/build/**/*.so uccl

python3 setup.py install
LOG_INFO_RANK0 "[hook system] Install uccl done"
# install deep_ep_wrapper
cd $UCCL_DIR/ep/deep_ep_wrapper
python3 setup.py install
LOG_INFO_RANK0 "[hook system] Install deep_ep done"

LOG_INFO_RANK0 "[hook system] Building uccl done."

popd