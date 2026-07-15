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

# Pip cache must live on a path that exists inside THIS environment (e.g. Docker). Schedulers often set
# DATA_PATH to a *host* path (e.g. /home/.../data/mlperf_llama2) for dataset mounts; that path is not
# writable or may not exist in the container, so do not derive PIP_CACHE_DIR from DATA_PATH by default.
# Keep the cache outside the repo so it is never accidentally committed.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/primus-cache/pip}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia_resiliency_ext

pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=2.14.0"

pip install --cache-dir="${PIP_CACHE_DIR}" -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"

# datasets 5.x requires fsspec<=2026.4.0; megatron-bridge deps may upgrade it.
pip install --cache-dir="${PIP_CACHE_DIR}" 'fsspec>=2023.1.0,<=2026.4.0'

echo "[OK] Megatron-Bridge dependencies installed"
