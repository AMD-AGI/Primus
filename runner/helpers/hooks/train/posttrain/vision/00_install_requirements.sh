#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

if [[ "${PRIMUS_SKIP_PIP:-0}" == "1" ]]; then
  echo "[INFO] PRIMUS_SKIP_PIP=1: skipping vision dependency installation"
  exit 0
fi

DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

REQ_FILE="${SCRIPT_DIR}/requirements-vision.txt"
if [[ -f "${REQ_FILE}" ]] && grep -qE '^[[:space:]]*[^#[:space:]]' "${REQ_FILE}"; then
  echo "[+] Installing vision dependencies..."
  pip install --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_FILE}"
  echo "[OK] vision dependencies installed"
fi

if [[ "${VISION_BUILD_GSPLAT:-0}" == "1" ]]; then
  echo "[+] Building gsplat when it is not already installed..."
  PYTHONPATH="${PRIMUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -c 'from primus.backends.worldmirror.runtime import install_gsplat; from primus.backends.worldmirror.worldmirror_trainer import resolve_worldmirror_repo; install_gsplat(resolve_worldmirror_repo(""))'
  echo "[OK] gsplat is installed"
else
  echo "[INFO] VISION_BUILD_GSPLAT is unset. Stage 2's Gaussian renderer needs gsplat; set VISION_BUILD_GSPLAT=1 to build it."
fi
