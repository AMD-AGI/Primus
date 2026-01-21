#!/usr/bin/env bash
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${SCRIPT_DIR}/requirements-megatron-bridge.txt" ]; then
    pip install -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"
    echo "[OK] Megatron-Bridge dependencies installed"
else
    # Fallback: Install core packages individually
    echo "[WARNING] requirements-megatron-bridge.txt not found, installing manually..."
    pip install transformers==4.57.6
    pip install qwen-vl-utils timm "open-clip-torch>=3.2.0" flash-linear-attention
    pip install megatron-energon bitstring filetype
    pip install nvidia_resiliency_ext "onnx==1.20.0rc1" nvidia-modelopt
    echo "[OK] Packages installed"
fi
