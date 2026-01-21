#!/usr/bin/env bash
set -euo pipefail

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

# Parse the complete config with all extends and nested configs
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PRIMUS_ROOT"

# Extract hf_path from fully parsed config
HF_PATH=$(python3 -c "
import sys
sys.path.insert(0, '${PRIMUS_ROOT}')
from primus.core.utils.yaml_utils import parse_yaml_to_namespace
exp = parse_yaml_to_namespace('${CONFIG_FILE}')
# Access post_trainer config
post_trainer = getattr(exp.modules, 'post_trainer', None)
if post_trainer and hasattr(post_trainer, 'hf_path'):
    print(post_trainer.hf_path)
else:
    sys.exit(1)
" 2>/dev/null || echo "")

if [[ -z "$HF_PATH" ]]; then
    echo "[WARNING] No hf_path found in config"
    exit 0
fi

# Set paths
DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
HF_CACHE="${HF_HOME:-${DATA_PATH}/huggingface}/hub"
MEGATRON_PATH="${DATA_PATH}/megatron_checkpoints/$(basename "${HF_PATH}")"

echo "[INFO] HF Model: ${HF_PATH}"
echo "[INFO] HF Cache: ${HF_CACHE}"
echo "[INFO] Megatron Path: ${MEGATRON_PATH}"

# Check if HF checkpoint already downloaded
HF_MODEL_CACHE="${HF_CACHE}/models--$(echo "${HF_PATH}" | tr '/' '--')"
if [[ -d "$HF_MODEL_CACHE" ]]; then
    echo "[INFO] HF checkpoint already cached at ${HF_MODEL_CACHE}"
else
    echo "[INFO] HF checkpoint will be downloaded from ${HF_PATH}"
fi

# Check if Megatron checkpoint already exists
if [[ -d "$MEGATRON_PATH" ]]; then
    echo "[INFO] Megatron checkpoint already exists at ${MEGATRON_PATH}, skipping conversion"
    echo "extra.checkpoint.pretrained_checkpoint=${MEGATRON_PATH}"
    exit 0
fi

# Convert checkpoint
echo "[+] Converting HF checkpoint to Megatron format..."
mkdir -p "$(dirname "${MEGATRON_PATH}")"

python3 third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model "${HF_PATH}" \
  --megatron-path "${MEGATRON_PATH}"

echo "extra.checkpoint.pretrained_checkpoint=${MEGATRON_PATH}"
echo "[OK] Checkpoint prepared at ${MEGATRON_PATH}"
