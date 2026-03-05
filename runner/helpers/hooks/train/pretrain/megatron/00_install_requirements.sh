#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

if [[ -f "${SCRIPT_DIR}/requirements-megatron.txt" ]]; then
  echo "[+] Installing Megatron dependencies..."
  pip install --cache-dir="${PIP_CACHE_DIR}" -r "${SCRIPT_DIR}/requirements-megatron.txt"
  echo "[OK] Megatron dependencies installed"
fi