#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: enable using ucep settings.
#
# Trigger:
#   export USING_UEP=1
#
###############################################################################


if [ "$USING_UEP" == "1" ]; then
    LOG_INFO "USING_UEP is enabled, checking required packages..."

    if ! pip show uccl &>/dev/null || ! pip show deep_ep &>/dev/null; then
        LOG_ERROR "uccl is not installed! Please use pre-installed primus image or set REBUILD_UCCL=1."
        exit 1
    fi
    LOG_INFO "uccl package is installed: $(pip show uccl | grep Version)"
    LOG_INFO "deep_ep package is installed: $(pip show deep_ep | grep Version)"

    if [ "$ENABLE_NUMA_BINDING" != "1" ]; then
        LOG_WARN "ENABLE_NUMA_BINDING is not enabled! Please set ENABLE_NUMA_BINDING=1 to avoid dataloader worker exited unexpectedly."
    fi

    export PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP
    LOG_INFO "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to DEEP_EP"
else
    export PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO
    LOG_INFO "USING_UEP is disabled. PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND set to TURBO"
fi
