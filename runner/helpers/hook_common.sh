#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Shared logging for runner hook scripts (prepare_experiment, install hooks, …).
# Source from any hook under runner/helpers/hooks/**:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/hook_common.sh"
# (adjust ../ count: pretrain hooks use ../../../hook_common.sh)
###############################################################################

if [[ -n "${__PRIMUS_COMMON_SOURCED:-}" ]]; then
    return 0
fi

_hook_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${_hook_common_dir}/lib/common.sh"
