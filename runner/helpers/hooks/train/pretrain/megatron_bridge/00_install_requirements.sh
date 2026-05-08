#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
PRIMUS_ROOT="${PRIMUS_PATH:-${DEFAULT_PRIMUS_ROOT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --primus_path)
      PRIMUS_ROOT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia_resiliency_ext

pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=2.14.0"

REQ_FILE="${SCRIPT_DIR}/requirements-megatron_bridge.txt"
if [[ -f "${REQ_FILE}" ]] && grep -qE '^[[:space:]]*[^#[:space:]]' "${REQ_FILE}"; then
  echo "[+] Installing Megatron-Bridge dependencies..."
  pip install --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_FILE}"
  echo "[OK] Megatron-Bridge dependencies installed"
fi