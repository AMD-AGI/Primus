#!/usr/bin/env bash
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge core dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/requirements-megatron-bridge.txt" ]; then
    pip install -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"
else
    # Fallback: Install core packages individually
    pip install qwen-vl-utils timm "open-clip-torch>=3.2.0" flash-linear-attention
    pip install megatron-energon bitstring filetype
fi

# Install optional packages with compatibility notes
echo "[+] Installing optional packages..."
pip install -U nvidia_resiliency_ext

# Note: nvidia-modelopt has compatibility issues with newer PyTorch versions
# Install onnx and nvidia-modelopt only if you need quantization/optimization
if [ "${INSTALL_MODELOPT:-0}" == "1" ]; then
    echo "[+] Installing nvidia-modelopt (compatibility may vary)..."
    pip install "onnx==1.20.0rc1"
    pip install -U nvidia-modelopt
    echo "[OK] nvidia-modelopt installed"
else
    echo "[SKIP] nvidia-modelopt (set INSTALL_MODELOPT=1 to install)"
fi

echo "[OK] Megatron-Bridge dependencies installed"
