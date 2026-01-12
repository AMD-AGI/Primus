#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: optionally patch Transformer Engine to relax the max supported
# flash-attn version.
#
# Trigger:
#   export PATCH_TE_FLASH_ATTN=1
#
# Implementation:
#   Calls runner/helpers/patch_te_flash_attn_max_version.sh
###############################################################################

set -euo pipefail

if [[ "${PATCH_TE_FLASH_ATTN:-0}" != "1" ]]; then
    exit 0
fi

if [[ -z "${PRIMUS_PATH:-}" ]]; then
    # Best-effort fallback: infer PRIMUS_PATH from this file location
    PRIMUS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export PRIMUS_PATH
fi

if declare -F LOG_INFO_RANK0 >/dev/null 2>&1; then
    LOG_INFO_RANK0 "[hook system] PATCH_TE_FLASH_ATTN=1 → patching Transformer Engine flash-attn max version"
else
    echo "[INFO] [hook system] PATCH_TE_FLASH_ATTN=1 → patching Transformer Engine flash-attn max version" >&2
fi

PATCH_SCRIPT="${PRIMUS_PATH}/runner/helpers/patch_te_flash_attn_max_version.sh"
if [[ ! -f "$PATCH_SCRIPT" ]]; then
    echo "[ERROR] [hook system] Patch script not found: $PATCH_SCRIPT" >&2
    exit 1
fi

bash "$PATCH_SCRIPT"
