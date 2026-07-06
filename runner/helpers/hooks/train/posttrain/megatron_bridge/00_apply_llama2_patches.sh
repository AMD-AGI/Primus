#!/bin/bash
###############################################################################
# Apply Llama2-70B LoRA MLPerf patches only for llama2_70b_lora_mlperf_posttrain.yaml.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../../../lib/common.sh" || {
    echo "[ERROR] Failed to load common library" >&2
    exit 1
}

PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"

CONFIG_FILE=""
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--config" ]]; then
        j=$((i+1))
        CONFIG_FILE="${!j}"
        break
    fi
done

if [[ -z "$CONFIG_FILE" ]]; then
    exit 0
fi

if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${PRIMUS_ROOT}/${CONFIG_FILE#./}"
fi

needs_patches="$(python3 -c "
import sys
sys.path.insert(0, '${PRIMUS_ROOT}')
from pathlib import Path
from primus.core.config.primus_config import load_primus_config, get_module_config

cfg_path = Path('${CONFIG_FILE}')
if 'llama2_70b_lora_mlperf_posttrain' not in cfg_path.name:
    sys.exit(0)

cfg = load_primus_config(cfg_path, None)
post = get_module_config(cfg, 'post_trainer')
if post is None:
    sys.exit(0)

model = str(getattr(post, 'model', '') or '')
if model not in ('llama2_70b_lora_mxfp4.yaml', 'llama2_70b_lora_mxfp4'):
    sys.exit(0)

print('1')
" 2>/dev/null || true)"

if [[ "${needs_patches}" != "1" ]]; then
    exit 0
fi

if [[ "${SKIP_PATCHES:-0}" == "1" ]]; then
    LOG_INFO_RANK0 "[llama2-patches] SKIP_PATCHES=1, skipping"
    exit 0
fi

PATCH_SCRIPT="${PRIMUS_ROOT}/examples/megatron_bridge/llama2_70b_lora/apply_patches.sh"
if [[ ! -f "${PATCH_SCRIPT}" ]]; then
    LOG_ERROR_RANK0 "[llama2-patches] Missing ${PATCH_SCRIPT}"
    exit 1
fi

LOG_INFO_RANK0 "[llama2-patches] Llama2-70B LoRA MLPerf config detected — checking Megatron-Bridge / Megatron-LM patches"
PRIMUS_ROOT="${PRIMUS_ROOT}" bash "${PATCH_SCRIPT}"

# primus-cli-direct registers revert on EXIT when this env is set.
echo "env.PRIMUS_LLAMA2_REVERT_PATCHES_ON_EXIT=1"
