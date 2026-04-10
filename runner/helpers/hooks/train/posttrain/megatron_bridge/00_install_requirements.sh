#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo root: .../megatron_bridge -> posttrain -> train -> hooks -> helpers -> runner -> Primus
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../" && pwd)"
# Pip cache must live on a path that exists inside THIS environment (e.g. Docker). Schedulers often set
# DATA_PATH to a *host* path (e.g. /home/.../data/mlperf_llama2) for dataset mounts; that path is not
# writable or may not exist in the container, so do not derive PIP_CACHE_DIR from DATA_PATH by default.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PRIMUS_ROOT}/.pip_cache}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia_resiliency_ext

pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=2.14.0"

pip install --cache-dir="${PIP_CACHE_DIR}" -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"

echo "[OK] Megatron-Bridge dependencies installed"
