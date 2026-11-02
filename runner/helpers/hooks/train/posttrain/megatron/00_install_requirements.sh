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
#   * the conversion hook will take the native path, OR
#   * the conversion hook will stop with an error before any converter runs
#     (e.g. a bad native_ckpt_convert value, native_ckpt_convert: true on a family
#     without a native converter, or native_ckpt_convert: false on a family without a
#     Megatron-Bridge model, such as Muse Spark).
# Each case prints its own reason. The Bridge path and a config without hf_path
# install the dependencies (the legacy behaviour), and so does any action this
# script does not know. When the decision itself fails (any exception, SystemExit
# included, or python3 itself), the script prints why and installs; Ctrl-C stops it.
# Note the asymmetry: an "error" action is a config error that plan_conversion()
# recognised, which the conversion hook raises before any converter runs (no install
# needed); an exception means the decision is unknown, so the install is kept.
# --config <path> and --config=<path> are both read; when given more than once, the last
# one counts (as in the conversion hook's argparse). The decision reads the config and a
# local HF config.json; for a Hub id it reads that id's config.json (HF cache, else a
# download) only under an explicit native_ckpt_convert: true.
#
# The decision is 01_convert_checkpoints.py's own plan_conversion(), the function
# main() acts on, so the two can never disagree.
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
    # no break: the last --config wins, as in the conversion hook's argparse
    if [[ "${args[$i]}" == "--config" ]]; then
        CONFIG_FILE="${args[$((i + 1))]:-}"
    elif [[ "${args[$i]}" == --config=* ]]; then
        CONFIG_FILE="${args[$i]#--config=}"
    fi
done

# 3) Skip when the native (bridge-free) path will handle the conversion.
if [[ -n "${CONFIG_FILE}" ]]; then
    [[ "${CONFIG_FILE}" == /* ]] || CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
    set +e
    # stderr is kept in DECISION too: a crash of the interpreter itself is then shown
    DECISION="$(cd "${PRIMUS_ROOT}" && python3 - "${CONFIG_FILE}" 2>&1 <<'PY'
import importlib.util
import os
import sys

cfg = sys.argv[1]
hook_path = os.path.join(
    os.getcwd(),
    "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py",
)

# The plan_conversion() actions that skip the install (the Bridge never runs), with
# why. Every other action -- "bridge", "no_hf_path" and any action added later --
# installs: on uncertainty keep the legacy install.
SKIP_REASONS = {
    "pretrained": "a Megatron checkpoint is configured (no conversion)",
    "native": "native (bridge-free) conversion will run",
    "error": "the conversion hook will stop with an error before any converter runs (see its message)",
}


def decide():
    """``("INSTALL" | "SKIP", reason)``: the hook's own plan_conversion() decision."""
    spec = importlib.util.spec_from_file_location("_convert_hook", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    action = mod.plan_conversion(cfg)[0]
    if action in SKIP_REASONS:
        return "SKIP", SKIP_REASONS[action]
    return "INSTALL", f"conversion action {action}"


try:
    result, reason = decide()
except KeyboardInterrupt:
    sys.exit(130)  # Ctrl-C stops the hook; it never becomes an install decision
except (Exception, SystemExit) as exc:  # noqa: BLE001 - keep the legacy install, say why
    result, reason = "INSTALL", f"could not compute the conversion decision ({type(exc).__name__}: {exc})"
print("__DECISION__:" + result + ":" + reason.replace("\n", " "))
PY
)"
    DECISION_RC=$?
    set -e
    if [[ ${DECISION_RC} -eq 130 ]]; then
        echo "[00_install_requirements] interrupted while computing the conversion decision" >&2
        exit 130
    fi
    DECISION_LINE="$(grep -m1 '^__DECISION__:' <<<"${DECISION}" || true)"
    case "${DECISION_LINE}" in
        __DECISION__:SKIP:*)
            echo "[00_install_requirements] ${DECISION_LINE#__DECISION__:SKIP:} -> skipping Megatron-Bridge dependency install"
            exit 0 ;;
        __DECISION__:INSTALL:*)
            echo "[00_install_requirements] ${DECISION_LINE#__DECISION__:INSTALL:} -> installing the Megatron-Bridge dependencies" ;;
        *)
            echo "[00_install_requirements] could not compute the conversion decision (python3 gave no decision: ${DECISION:0:500}) -> installing the Megatron-Bridge dependencies" ;;
    esac
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
