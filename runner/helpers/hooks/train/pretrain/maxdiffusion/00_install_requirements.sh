#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Install the MaxDiffusion (JAX) Python dependencies for primus-cli launches.
#
# Same shape as the maxtext / megatron / torchtitan siblings: this hook only
# installs packages. The other two concerns are split the same way they are for
# every other backend:
#   - backend path + launcher env (RUN_MODE, JAX coordinator, NVTE_FRAMEWORK)
#     are emitted by this directory's prepare.py
#   - runtime behavior (Shardy partitioner, TensorFlow preload) lives in
#     primus/backends/maxdiffusion/patches/
# so nothing here has to mutate the vendored third_party/maxdiffusion checkout.
#
# requirements-maxdiffusion.txt is deliberately separate from requirements-jax.txt
# (which MaxText installs): it pins transformers 4.x for MaxDiffusion's Flax code,
# and that pin must not be forced onto MaxText runs.
#
# PRIMUS_SKIP_PIP=1 skips this step, for images that already ship the stack.
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

# Load shared logging so output honors PRIMUS_LOG_LEVEL (DEBUG/INFO/WARN/ERROR).
# When invoked through primus-cli the LOG_* functions are already exported, but
# sourcing here keeps the hook usable standalone and under `set -u`.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../../hook_common.sh"

if [[ "${PRIMUS_SKIP_PIP:-0}" == "1" ]]; then
  LOG_INFO "PRIMUS_SKIP_PIP=1: skipping MaxDiffusion dependency install (deps from image)"
  exit 0
fi

DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

# Match pip verbosity to the active log level so WARN/ERROR runs stay quiet
# (suppresses the "Requirement already satisfied" wall) while DEBUG/INFO keep it.
PIP_FLAGS=()
case "${PRIMUS_LOG_LEVEL:-INFO}" in
  WARN|ERROR) PIP_FLAGS+=(-q -q) ;;
esac

LOG_INFO "Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

REQ_FILE="${PRIMUS_ROOT}/requirements-maxdiffusion.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  LOG_ERROR "Missing required MaxDiffusion requirements file: ${REQ_FILE}"
  exit 1
fi

LOG_INFO "Installing MaxDiffusion dependencies..."
pip install "${PIP_FLAGS[@]}" --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_FILE}"
LOG_SUCCESS "MaxDiffusion dependencies installed"
