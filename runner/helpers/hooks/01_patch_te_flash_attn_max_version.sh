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
#   Inline patch (same as examples/run_pretrain.sh).
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

LOG_INFO_RANK0 "[hook system] PATCH_TE_FLASH_ATTN=1 → patching _flash_attn_max_version in attention.py"

ATTENTION_PY="/opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py"
if [[ ! -f "$ATTENTION_PY" ]]; then
    LOG_ERROR_RANK0 "[hook system] attention.py not found: $ATTENTION_PY"
    exit 1
fi

sed -i 's/_flash_attn_max_version = PkgVersion(\".*\")/_flash_attn_max_version = PkgVersion(\"3.0.0.post1\")/' \
    "$ATTENTION_PY"
