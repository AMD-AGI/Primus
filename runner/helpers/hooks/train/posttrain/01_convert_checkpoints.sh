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
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$PRIMUS_ROOT"

# Convert CONFIG_FILE to absolute path
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
fi

# Extract hf_path from fully parsed config
HF_PATH=$(python3 -c "
import sys

# Debug output to stderr so it doesn't interfere with stdout capture
print(f'[DEBUG] CONFIG_FILE: ${CONFIG_FILE}', file=sys.stderr)

sys.path.insert(0, '${PRIMUS_ROOT}')
from pathlib import Path
from primus.core.config.primus_config import load_primus_config, get_module_config
from primus.core.utils.yaml_utils import parse_yaml

# Load config using the same method as train_runtime.py
cfg = load_primus_config(Path('${CONFIG_FILE}'), None)
print(f'[DEBUG] cfg type: {type(cfg)}', file=sys.stderr)

# Get post_trainer module config
post_trainer = get_module_config(cfg, 'post_trainer')
print(f'[DEBUG] post_trainer type: {type(post_trainer)}', file=sys.stderr)

if post_trainer is None:
    print('[DEBUG] post_trainer is None', file=sys.stderr)
    sys.exit(1)

if not hasattr(post_trainer, 'params'):
    print('[DEBUG] post_trainer has no params', file=sys.stderr)
    sys.exit(1)

print(f'[DEBUG] post_trainer.params type: {type(post_trainer.params)}', file=sys.stderr)

if not hasattr(post_trainer.params, 'hf_path'):
    print('[DEBUG] post_trainer.params has no hf_path', file=sys.stderr)
    sys.exit(1)

print(post_trainer.params.hf_path)
" || echo "")

echo "[DEBUG] HF_PATH captured: ${HF_PATH}"

if [[ -z "$HF_PATH" ]]; then
    # ANSI color codes
    YELLOW='\033[1;33m'
    RESET='\033[0m'
    echo -e "${YELLOW}[WARNING] No hf_path found in config${RESET}"
    echo -e "${YELLOW}[WARNING] Assuming checkpoint already exists and conversion is not needed${RESET}"
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
    echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"
    exit 0
fi

# Convert checkpoint
echo "[+] Converting HF checkpoint to Megatron format..."
mkdir -p "$(dirname "${MEGATRON_PATH}")"

# Set up Python path for Megatron-Bridge
export PYTHONPATH="${PRIMUS_ROOT}/third_party/Megatron-Bridge/src:${PRIMUS_ROOT}/third_party/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"

python3 third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model "${HF_PATH}" \
  --megatron-path "${MEGATRON_PATH}"

echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"
echo "[OK] Checkpoint prepared at ${MEGATRON_PATH}"
