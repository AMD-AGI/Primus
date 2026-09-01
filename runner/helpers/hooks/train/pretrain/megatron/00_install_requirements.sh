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

# Standard pip-skip switch, matching the posttrain Megatron and MaxDiffusion
# hooks. An image that already ships the stack has nothing to gain here, and on
# a ROCm image it can actively lose: these requirements are the Qwen3.5
# gated-delta-net deps, and causal-conv1d ships no wheel, so pip builds it from
# source and the build environment resolves a fresh `torch` against the image's
# pinned ROCm build. That conflict is unresolvable and aborts the run before
# training starts, for packages the model under test never imports.
if [[ -n "${PRIMUS_SKIP_PIP:-}" && "${PRIMUS_SKIP_PIP}" != "0" && "${PRIMUS_SKIP_PIP,,}" != "false" ]]; then
  echo "[00_install_requirements] PRIMUS_SKIP_PIP=${PRIMUS_SKIP_PIP} -> skipping Megatron pretrain dependency install"
  exit 0
fi

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

REQ_FILE="${SCRIPT_DIR}/requirements-megatron.txt"
if [[ -f "${REQ_FILE}" ]] && grep -qE '^[[:space:]]*[^#[:space:]]' "${REQ_FILE}"; then
  LOG_INFO "Installing Megatron dependencies..."
  pip install "${PIP_FLAGS[@]}" --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_FILE}"
  LOG_SUCCESS "Megatron dependencies installed"
fi
