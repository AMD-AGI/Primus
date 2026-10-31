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
# so it can never drift from the actual conversion behaviour. When the decision
# fails, the reason is always logged:
#   * a settings error (a ValueError raised by read_hook_settings itself, e.g.
#     native_ckpt_convert: maybe; not one from conversion_plan, such as a
#     corrupt config.json): nothing is installed, because
#     01_convert_checkpoints.py reads the same settings and stops on that error;
#   * any other failure (an import error, a config that cannot be loaded, a
#     python3 without yaml, ...): the dependencies are installed. Installing
#     when unsure costs time; skipping them wrongly makes the Bridge path fail.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

# 1) Standard pip-skip switch.
if [[ -n "${PRIMUS_SKIP_PIP:-}" && "${PRIMUS_SKIP_PIP}" != "0" && "${PRIMUS_SKIP_PIP,,}" != "false" ]]; then
    echo "[00_install_requirements] PRIMUS_SKIP_PIP=${PRIMUS_SKIP_PIP} -> skipping Megatron-Bridge dependency install"
    exit 0
fi

# 2) Find --config among the forwarded args.
#    Both forms (--config <file> and --config=<file>); the last one wins, as in
#    01_convert_checkpoints.py's argparse.
CONFIG_FILE=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--config" ]]; then
        CONFIG_FILE="${args[$((i + 1))]:-}"
    elif [[ "${args[$i]}" == --config=* ]]; then
        CONFIG_FILE="${args[$i]#--config=}"
    fi
done

# 3) Skip unless the Megatron-Bridge path will run (native conversion, an existing checkpoint, or a
#    refused config need no Bridge). The decision uses the
#    same arguments (config and command-line overrides) and the same function
#    (conversion_plan) as 01_convert_checkpoints.py, so both hooks agree.
if [[ -n "${CONFIG_FILE}" ]]; then
    [[ "${CONFIG_FILE}" == /* ]] || CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
    set +e
    DECISION="$(cd "${PRIMUS_ROOT}" && python3 - "${CONFIG_FILE}" "$@" <<'PY'
import importlib.util
import os
import sys

cfg = sys.argv[1]
# the arguments execute_hooks.sh forwards (hook target, --config <file>, overrides); 01 reads
# the overrides from the same list after argparse removes --config
forwarded = sys.argv[2:]
overrides = []
skip = False
for tok in forwarded:
    if skip:
        skip = False
        continue
    if tok == "--config":
        skip = True
        continue
    if tok.startswith("--config="):
        continue
    overrides.append(tok)
hook_path = os.path.join(
    os.getcwd(),
    "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py",
)


class SettingsError(Exception):
    """A ValueError from read_hook_settings only (e.g. native_ckpt_convert: maybe)."""


def decide():
    spec = importlib.util.spec_from_file_location("_convert_hook", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        hf_path, pretrained, opts = mod.read_hook_settings(cfg, overrides)
    except ValueError as e:
        raise SettingsError(f"{type(e).__name__}: {e}") from e
    plan = mod.conversion_plan(hf_path, pretrained, opts)
    # skip / native: no Bridge dependencies; refuse: 01 stops before any conversion
    return "INSTALL" if plan == "bridge" else "SKIP"


try:
    result = decide()
except SettingsError as e:
    # 01_convert_checkpoints.py main() catches the same ValueError from read_hook_settings and
    # exits 1, so nothing is converted.
    print(f"__REASON__:{e}")
    result = "SETTINGS_ERROR"
except Exception as e:  # noqa: BLE001 - any other failure (e.g. a corrupt config.json): install
    print(f"__REASON__:{type(e).__name__}: {e}")
    result = "INSTALL"
print("__DECISION__:" + result)
PY
)"
    STATUS=$?
    set -e
    if printf '%s\n' "${DECISION}" | grep -q '__DECISION__:SKIP'; then
        echo "[00_install_requirements] Megatron-Bridge conversion will not run (native conversion, existing checkpoint, or refused config) -> skipping Megatron-Bridge dependency install"
        exit 0
    fi
    REASON="$(printf '%s\n' "${DECISION}" | sed -n 's/^__REASON__://p')"
    if printf '%s\n' "${DECISION}" | grep -q '__DECISION__:SETTINGS_ERROR'; then
        echo "[00_install_requirements] conversion settings error: ${REASON} -> not installing Megatron-Bridge dependencies; 01_convert_checkpoints.py stops on the same error" >&2
        exit 0
    fi
    if ! printf '%s\n' "${DECISION}" | grep -q '__DECISION__:INSTALL'; then
        REASON="${REASON:-no decision printed}"
    fi
    if [[ -n "${REASON}" ]]; then
        echo "[00_install_requirements] cannot decide the conversion path (exit ${STATUS}): ${REASON} -> installing Megatron-Bridge dependencies" >&2
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
pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=2.14.0,<5.1.0"
pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" "transformers==4.57.6"
pip install --cache-dir="${PIP_CACHE_DIR}" -U "safetensors>=0.4.0"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt

echo "[OK] Megatron-Bridge dependencies installed"
