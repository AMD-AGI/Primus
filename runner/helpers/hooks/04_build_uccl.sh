#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: enable AINIC environment settings.
#
# Trigger:
#   export USING_AINIC=1
#
# This replaces the old env file:
#   runner/helpers/envs/enable_ainic.sh
#
# Note: hooks must print "env.VAR=VALUE" to persist changes back to the caller.
###############################################################################

set -euo pipefail

if [[ "${USING_UCCL:-0}" != "1" ]]; then
    exit 0
fi

LOG_INFO_RANK0 "[hook system] USING_UCCL=1 → Building uccl in /tmp "

pushd /tmp

if [ -d uccl ]; then
    LOG_INFO_RANK0 "[hook system] Found existed uccl in /tmp, delete it"
	rm -rf uccl
fi

git clone https://github.com/uccl-project/uccl.git
cd uccl
pushd ep

LOG_INFO_RANK0 "[hook system] Building uccl ep"
python3 setup.py build

LOG_INFO_RANK0 "[hook system] Building uccl ep done"
popd
cp ep/build/**/*.so uccl

python3 setup.py install
LOG_INFO_RANK0 "[hook system] Install uccl done"
# install deep_ep_wrapper
pushd ep/deep_ep_wrapper
python3 setup.py install
LOG_INFO_RANK0 "[hook system] Install deep_ep done"
popd

LOG_INFO_RANK0 "[hook system] Cleaning build files"
rm -rf uccl
popd

LOG_INFO_RANK0 "[hook system] Building uccl done."