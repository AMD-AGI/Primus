#!/usr/bin/env bash
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"

echo "[OK] Megatron-Bridge dependencies installed"
