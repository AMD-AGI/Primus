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

REQ_FILE="${SCRIPT_DIR}/requirements-maxtext.txt"
if [[ -f "${REQ_FILE}" ]] && grep -qE '^[[:space:]]*[^#[:space:]]' "${REQ_FILE}"; then
  echo "[+] Installing MaxText dependencies..."
  pip install --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_FILE}"
  echo "[OK] MaxText dependencies installed"
fi

REQ_JAX_FILE="${PRIMUS_ROOT}/requirements-jax.txt"
if [[ -f "${REQ_JAX_FILE}" ]]; then
  echo "[+] Installing JAX dependencies..."
  pip install --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_JAX_FILE}"
  echo "[OK] JAX dependencies installed"
else
  echo "[ERROR] Missing required JAX requirements file: ${REQ_JAX_FILE}" >&2
  exit 1
fi
