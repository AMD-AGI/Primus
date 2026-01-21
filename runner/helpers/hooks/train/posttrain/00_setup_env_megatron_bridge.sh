#!/usr/bin/env bash
set -euo pipefail

# Install Megatron-Bridge dependencies
echo "[+] Installing Megatron-Bridge dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install -r "${SCRIPT_DIR}/requirements-megatron-bridge.txt"

echo "[OK] Megatron-Bridge dependencies installed"

# Parse config to get HF model path and prepare checkpoint
echo "[+] Preparing Megatron-Bridge checkpoint..."

# Find --config argument from command line
CONFIG_FILE=""
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i+1))
        CONFIG_FILE="${!j}"
        break
    fi
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "[WARNING] No --config argument found, skipping checkpoint preparation"
    exit 0
fi

# Parse the model config to get hf_path
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PRIMUS_ROOT"

# Extract model config file from main config
MODEL_CONFIG=$(python3 -c "
import sys
sys.path.insert(0, '${PRIMUS_ROOT}')
from primus.core.utils.yaml_utils import parse_yaml
config = parse_yaml('${CONFIG_FILE}')
model_file = config.get('modules', {}).get('post_trainer', {}).get('model', '')
if model_file:
    print(model_file)
else:
    sys.exit(1)
" 2>/dev/null || echo "")

if [[ -z "$MODEL_CONFIG" ]]; then
    echo "[WARNING] Could not extract model config from ${CONFIG_FILE}"
    exit 0
fi

# Resolve model config path
MODEL_CONFIG_PATH="primus/configs/models/megatron_bridge/${MODEL_CONFIG}"
if [[ ! -f "$MODEL_CONFIG_PATH" ]]; then
    echo "[WARNING] Model config not found: ${MODEL_CONFIG_PATH}"
    exit 0
fi

# Extract hf_path from model config
HF_PATH=$(python3 -c "
import sys
sys.path.insert(0, '${PRIMUS_ROOT}')
from primus.core.utils.yaml_utils import parse_yaml
config = parse_yaml('${MODEL_CONFIG_PATH}')
print(config.get('hf_path', ''))
" 2>/dev/null || echo "")

if [[ -z "$HF_PATH" ]]; then
    echo "[WARNING] No hf_path found in ${MODEL_CONFIG_PATH}"
    exit 0
fi

# Set megatron checkpoint path
WORKSPACE="${PRIMUS_WORKSPACE:-./output}"
MEGATRON_PATH="${WORKSPACE}/megatron_checkpoints/$(basename "${HF_PATH}")"

echo "[INFO] HF Model: ${HF_PATH}"
echo "[INFO] Megatron Path: ${MEGATRON_PATH}"

# Check if checkpoint already exists
if [[ -d "$MEGATRON_PATH" ]]; then
    echo "[INFO] Megatron checkpoint already exists at ${MEGATRON_PATH}, skipping conversion"
    exit 0
fi

# Convert checkpoint
echo "[+] Converting HF checkpoint to Megatron format..."
mkdir -p "$(dirname "${MEGATRON_PATH}")"

python3 third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model "${HF_PATH}" \
  --megatron-path "${MEGATRON_PATH}"

echo "[OK] Checkpoint prepared at ${MEGATRON_PATH}"
