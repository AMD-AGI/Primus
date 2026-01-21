#!/usr/bin/env bash
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Uninstall nvidia-modelopt if present (incompatible with PyTorch 2.x)
# if pip show nvidia-modelopt &>/dev/null; then
#     echo "[INFO] Uninstalling incompatible nvidia-modelopt..."
#     pip uninstall -y nvidia-modelopt onnx 2>/dev/null || true
# fi

pip install "onnx==1.20.0rc1"
pip install -U nvidia-modelopt
pip install -U nvidia_resiliency_ext

pip install -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"

echo "[OK] Megatron-Bridge dependencies installed"
