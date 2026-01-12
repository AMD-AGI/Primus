#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Patch Transformer Engine attention.py to relax the max supported flash-attn
# version used by Transformer Engine.
#
# Intended to be run via:
#   primus-cli direct --patch runner/helpers/patch_te_flash_attn_max_version.sh -- ...
#
# Exit codes (for execute_patches.sh):
#   0  - success (patch applied)
#   2  - skipped (already applied or file not found)
#   >2 - failure (stop patch pipeline)
###############################################################################

set -euo pipefail

TARGET_VERSION="${TE_FLASH_ATTN_MAX_VERSION:-3.0.0.post1}"

_find_attention_py() {
    # 1) User override
    if [[ -n "${TE_ATTENTION_PY_PATH:-}" ]]; then
        echo "${TE_ATTENTION_PY_PATH}"
        return 0
    fi

    # 2) Best-effort: locate via python import
    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY' || true
import sys
try:
    import transformer_engine.pytorch.attention as m  # type: ignore
    print(m.__file__)
except Exception:
    sys.exit(1)
PY
        return 0
    fi

    # 3) Fallback to the historical path used in example scripts
    echo "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py"
}

ATTENTION_PY="$(_find_attention_py | tail -n 1)"

if [[ -z "${ATTENTION_PY:-}" || ! -f "${ATTENTION_PY}" ]]; then
    LOG_INFO "Skip TE flash-attn max version patch. attention.py not found: ${ATTENTION_PY:-<empty>}"
    exit 2
fi

LOG_INFO "Patching Transformer Engine attention.py: ${ATTENTION_PY}"
LOG_INFO "Setting _flash_attn_max_version -> PkgVersion(\"${TARGET_VERSION}\")"

if grep -qE "_flash_attn_max_version = PkgVersion\\(\"${TARGET_VERSION//./\\.}\"\\)" "${ATTENTION_PY}"; then
    LOG_INFO "Skip TE flash-attn max version patch. Already set to ${TARGET_VERSION}."
    exit 2
fi

if ! grep -qE "_flash_attn_max_version = PkgVersion\\(\".*\"\\)" "${ATTENTION_PY}"; then
    LOG_ERROR "Failed to patch: expected pattern not found in ${ATTENTION_PY}"
    exit 3
fi

sed -i -E "s/_flash_attn_max_version = PkgVersion\\(\".*\"\\)/_flash_attn_max_version = PkgVersion(\\\"${TARGET_VERSION}\\\")/" "${ATTENTION_PY}"

LOG_SUCCESS "TE flash-attn max version patch applied."
exit 0
