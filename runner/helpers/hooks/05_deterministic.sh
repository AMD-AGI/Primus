#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: enable deterministic-friendly settings when requested.
#
# Trigger:
#   export PRIMUS_DETERMINISTIC=1
#
# This hook emits env.* lines which will be exported by execute_hooks.sh.
#

set -euo pipefail

if [[ "${PRIMUS_DETERMINISTIC:-0}" != "1" ]]; then
    exit 0
fi

echo "env.NCCL_ALGO=Ring"
echo "env.NVTE_ALLOW_NONDETERMINISTIC_ALGO=0"
echo "env.ROCBLAS_DEFAULT_ATOMICS_MODE=0"
# Disable torch compile to avoid race condition issue in some triton versions.
echo "env.TORCH_COMPILE_DISABLE=1"
