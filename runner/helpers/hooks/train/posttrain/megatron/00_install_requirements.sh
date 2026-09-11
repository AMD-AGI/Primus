#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# ---------------------------------------------------------------------------
# Dependencies for the LEGACY Megatron-Bridge checkpoint-conversion path ONLY
# (AutoBridge.import_ckpt): transformers==4.57.6 + nvidia-modelopt + onnx.
#
# The default native (bridge-free) HF->Megatron converter needs NONE of these,
# so this hook SKIPS the install whenever the native path will run -- otherwise
# it would re-install the very dependencies the native path was built to avoid
# (and clobber the stock transformers). It skips when:
#   * PRIMUS_SKIP_PIP is set (standard pip-skip switch), OR
#   * a Megatron checkpoint is already configured (pretrained_checkpoint -> no
#     conversion at all), OR
#   * the configured HF model resolves to a supported native family and native
#     conversion is not explicitly disabled (native_ckpt_convert: false).
#
# The native-vs-bridge decision reuses 01_convert_checkpoints.py's own helpers
# so it can never drift from the actual conversion behaviour.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

# 1) Standard pip-skip switch.
if [[ -n "${PRIMUS_SKIP_PIP:-}" && "${PRIMUS_SKIP_PIP}" != "0" && "${PRIMUS_SKIP_PIP,,}" != "false" ]]; then
    echo "[00_install_requirements] PRIMUS_SKIP_PIP=${PRIMUS_SKIP_PIP} -> skipping Megatron-Bridge dependency install"
    exit 0
fi

# 2) Find --config among the forwarded args.
CONFIG_FILE=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--config" ]]; then
        CONFIG_FILE="${args[$((i + 1))]:-}"
        break
    fi
done

# 3) Skip when the native (bridge-free) path will handle the conversion.
if [[ -n "${CONFIG_FILE}" ]]; then
    [[ "${CONFIG_FILE}" == /* ]] || CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
    set +e
    DECISION="$(cd "${PRIMUS_ROOT}" && python3 - "${CONFIG_FILE}" 2>/dev/null <<'PY'
import importlib.util
import os
import sys

cfg = sys.argv[1]
hook_path = os.path.join(
    os.getcwd(),
    "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py",
)


def decide():
    spec = importlib.util.spec_from_file_location("_convert_hook", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hf_path, pretrained = mod.get_checkpoint_config(cfg)
    if pretrained:
        return "SKIP"  # checkpoint already provided; no conversion at all
    opts = mod.read_native_opts(cfg)
    fam = mod._detect_native_family(hf_path) if hf_path else None
    if opts["enabled"] is False:
        return "INSTALL"  # bridge explicitly forced -> deps needed
    if opts["enabled"] is True or fam is not None:
        return "SKIP"  # native (bridge-free) path -> deps not needed
    return "INSTALL"  # unsupported family + hf_path -> bridge fallback


try:
    result = decide()
except Exception:
    result = "INSTALL"  # on any uncertainty, keep the legacy behaviour
print("__DECISION__:" + result)
PY
)"
    set -e
    if printf '%s\n' "${DECISION}" | grep -q '__DECISION__:SKIP'; then
        echo "[00_install_requirements] native (bridge-free) conversion will run -> skipping Megatron-Bridge dependency install"
        exit 0
    fi
fi

# 4) Legacy Megatron-Bridge path: install its conversion dependencies.
echo "[+] Installing Megatron-Bridge checkpoint-conversion dependencies..."
DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"
echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

# Minimal bridge conversion set for AutoBridge.import_ckpt(). `nvidia-modelopt`
# is a hard dependency because Megatron-Bridge imports it at module import time
# from its GPT provider / checkpoint-save modules.
pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=2.14.0"
pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" "transformers==4.57.6"
pip install --cache-dir="${PIP_CACHE_DIR}" -U "safetensors>=0.4.0"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt

echo "[OK] Megatron-Bridge dependencies installed"
