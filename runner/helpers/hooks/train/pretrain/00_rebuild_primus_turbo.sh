#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Hook: optionally rebuild Primus-Turbo before running train pretrain.
#
# Trigger:
#   export REBUILD_PRIMUS_TURBO=1
#
# Notes:
# - Hooks run before the main training command, and are the right place for
#   environment preparation steps (vs. patching the training runtime).
# - This hook calls the existing patch script as an implementation detail.
###############################################################################

set -euo pipefail

if [[ "${REBUILD_PRIMUS_TURBO:-0}" != "1" ]]; then
    exit 0
fi

if [[ -z "${PRIMUS_PATH:-}" ]]; then
    # Best-effort fallback: infer PRIMUS_PATH from this file location
    PRIMUS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
    export PRIMUS_PATH
fi

if declare -F LOG_INFO_RANK0 >/dev/null 2>&1; then
    LOG_INFO_RANK0 "[hook train/pretrain] REBUILD_PRIMUS_TURBO=1 → rebuilding Primus-Turbo"
else
    echo "[INFO] [hook train/pretrain] REBUILD_PRIMUS_TURBO=1 → rebuilding Primus-Turbo" >&2
fi

PATCH_SCRIPT="${PRIMUS_PATH}/runner/helpers/rebuild_primus_turbo.sh"
if [[ ! -f "$PATCH_SCRIPT" ]]; then
    echo "[ERROR] [hook train/pretrain] Patch script not found: $PATCH_SCRIPT" >&2
    exit 1
fi

bash "$PATCH_SCRIPT"
