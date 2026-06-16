#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# Install dependencies for Megatron SFT checkpoint conversion
# Uses Megatron-Bridge's AutoBridge for HF → Megatron conversion
echo "[+] Installing Megatron SFT checkpoint conversion dependencies..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

# Set up pip cache directory
DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

echo "[INFO] Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

# Install minimal dependencies for Megatron-Bridge checkpoint conversion
# Note: We only need the conversion utilities, not the full training stack
pip install --cache-dir="${PIP_CACHE_DIR}" -U "datasets>=2.14.0"

# Minimal bridge conversion set for AutoBridge.import_ckpt().
# `nvidia-modelopt` is still a hard dependency today because Megatron-Bridge
# imports it at module import time from GPT provider/checkpoint save modules.
pip install --cache-dir="${PIP_CACHE_DIR}" "onnx==1.20.0rc1"
pip install --cache-dir="${PIP_CACHE_DIR}" "transformers==4.57.6"
pip install --cache-dir="${PIP_CACHE_DIR}" -U "safetensors>=0.4.0"
pip install --cache-dir="${PIP_CACHE_DIR}" -U nvidia-modelopt

echo "[OK] Megatron SFT dependencies installed"
