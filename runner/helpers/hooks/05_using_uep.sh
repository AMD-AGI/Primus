#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: select MoE dispatch/combine backend.
#
# Trigger:
#   USING_UEP=1  -> DEEP_EP (requires uccl + deep_ep packages)
#   otherwise    -> TURBO   (MoRI-based, default)
#
# Note: UCCL_*/MORI_* network env defaults are set in
#       runner/helpers/envs/comm_envs.sh (loaded unconditionally by
#       base_env.sh). This hook only handles backend selection and
#       package availability checks.
#
###############################################################################

set -euo pipefail

if [[ "${USING_UEP:-0}" == "1" ]]; then
    LOG_INFO "USING_UEP is enabled, checking required packages..."

    if ! python3 -m pip show uccl &>/dev/null || ! python3 -m pip show deep_ep &>/dev/null; then
        LOG_ERROR "uccl is not installed! Please use pre-installed primus image or set REBUILD_UEP=1."
        exit 1
    fi
    LOG_INFO "uccl package is installed: $(python3 -m pip show uccl | grep Version)"
    LOG_INFO "deep_ep package is installed: $(python3 -m pip show deep_ep | grep Version)"

    echo "env.PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP"
    LOG_INFO "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to DEEP_EP"
else
    echo "env.PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO"
    LOG_INFO "USING_UEP is disabled. PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to TURBO"
fi
