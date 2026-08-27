#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Install the MaxDiffusion (JAX) runtime for primus-cli launches.
#
# The MaxText sibling hook pip-installs its requirements inline, but MaxDiffusion
# additionally needs torch from the ROCm wheel index and four source patches, so
# this delegates to examples/maxdiffusion/setup_maxdiffusion_env.sh -- the same
# idempotent script examples/run_pretrain.sh uses. Both launch paths therefore
# share one definition of the environment instead of drifting apart.
#
# Container launches start from a clean image every time, so this runs on every
# launch. Pointing pip at a cache under DATA_PATH (inside the bind-mounted Primus
# checkout) keeps repeat runs off the network.
#
# PRIMUS_SKIP_PIP=1 skips the whole step, for images that already ship the stack.
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
  echo "[INFO] PRIMUS_SKIP_PIP=1: skipping MaxDiffusion env setup (deps from image)"
  exit 0
fi

DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"
export PIP_CACHE_DIR

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

SETUP_SCRIPT="${PRIMUS_ROOT}/examples/maxdiffusion/setup_maxdiffusion_env.sh"
if [[ ! -f "${SETUP_SCRIPT}" ]]; then
  echo "[ERROR] Missing MaxDiffusion setup script: ${SETUP_SCRIPT}" >&2
  exit 1
fi

echo "[+] Setting up MaxDiffusion environment (deps + source patches)..."
PRIMUS_PATH="${PRIMUS_ROOT}" bash "${SETUP_SCRIPT}"
echo "[OK] MaxDiffusion environment ready"
