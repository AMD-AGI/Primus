#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: optionally rebuild bnxt from a tar package.
#
# Equivalent behavior to the legacy logic in examples/run_pretrain.sh.
#
# Trigger:
#   export REBUILD_BNXT=1
#
# Required:
#   export PATH_TO_BNXT_TAR_PACKAGE=/path/to/libbnxt_re-*.tar.gz
#
# Implementation detail:
#   Calls runner/helpers/rebuild_bnxt.sh (shared implementation).
###############################################################################

set -euo pipefail

REBUILD_BNXT="${REBUILD_BNXT:-0}"
PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-}"

if [[ "${REBUILD_BNXT}" != "1" ]]; then
    exit 0
fi

if [[ -z "${PRIMUS_PATH:-}" ]]; then
    # Best-effort fallback: infer PRIMUS_PATH from this file location
    PRIMUS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export PRIMUS_PATH
fi

if [[ -z "${PATH_TO_BNXT_TAR_PACKAGE}" || ! -f "${PATH_TO_BNXT_TAR_PACKAGE}" ]]; then
    LOG_INFO_RANK0 "[hook system] Skip bnxt rebuild. REBUILD_BNXT=${REBUILD_BNXT}, PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE}"
    exit 0
fi

LOG_INFO_RANK0 "[hook system] REBUILD_BNXT=1 → rebuilding bnxt from ${PATH_TO_BNXT_TAR_PACKAGE}"

PATCH_SCRIPT="${PRIMUS_PATH}/runner/helpers/rebuild_bnxt.sh"
if [[ ! -f "$PATCH_SCRIPT" ]]; then
    LOG_ERROR_RANK0 "[hook system] rebuild_bnxt.sh not found: $PATCH_SCRIPT"
    exit 1
fi

# rebuild_bnxt.sh expects PATH_TO_BNXT_TAR_PACKAGE
export PATH_TO_BNXT_TAR_PACKAGE
bash "$PATCH_SCRIPT"
