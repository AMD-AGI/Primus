#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euo pipefail

# Parse config to get HF model path and prepare checkpoint
LOG_INFO_RANK0 "[+] Preparing Megatron-Bridge checkpoint..."

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
    LOG_ERROR_RANK0 "[WARNING] No --config argument found, skipping checkpoint preparation"
    exit 0
fi

# Parse the complete config with all extends and nested configs
# Repo root: megatron_bridge -> posttrain -> train -> hooks -> helpers -> runner -> Primus (6 levels)
PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../" && pwd)"
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

LOG_DEBUG_RANK0 "HF_PATH captured: ${HF_PATH}"

if [[ -z "$HF_PATH" ]]; then
    LOG_ERROR_RANK0 "[WARNING] No hf_path found in config"
    LOG_ERROR_RANK0 "[WARNING] Assuming checkpoint already exists and conversion is not needed"
    exit 0
fi

# Data root: DATA_PATH is often a *host* path. The path may appear in the container (bind mount
# metadata) but still be unusable. Pick the first root where we can actually mkdir.
resolve_effective_data() {
    local c testdir
    local -a candidates=()
    [[ -n "${MOUNT_DATA_PATH:-}" ]] && candidates+=("${MOUNT_DATA_PATH}")
    candidates+=("/data")
    [[ -n "${DATA_PATH:-}" ]] && candidates+=("${DATA_PATH}")
    candidates+=("${PRIMUS_ROOT}/data")

    for c in "${candidates[@]}"; do
        [[ -z "$c" ]] && continue
        testdir="${c}/.primus_megatron_write_test_$$"
        if mkdir -p "${testdir}" 2>/dev/null; then
            rmdir "${testdir}" 2>/dev/null || rm -rf "${testdir}" 2>/dev/null || true
            printf '%s' "$c"
            return 0
        fi
    done
    mkdir -p "${PRIMUS_ROOT}/data"
    printf '%s' "${PRIMUS_ROOT}/data"
}
EFFECTIVE_DATA="$(resolve_effective_data)"
LOG_INFO_RANK0 "Checkpoint/HF data root: ${EFFECTIVE_DATA}"

# Hugging Face cache dirs: schedulers often export HF_HOME (or HF_HUB_CACHE / TRANSFORMERS_CACHE)
# to host paths that do not exist inside the container. Override when we cannot mkdir there.
HF_FALLBACK="${EFFECTIVE_DATA}/huggingface"
if [[ -n "${HF_HOME:-}" ]]; then
    if mkdir -p "${HF_HOME}/hub/.primus_hf_write_test_$$" 2>/dev/null; then
        rm -rf "${HF_HOME}/hub/.primus_hf_write_test_$$" 2>/dev/null || true
    else
        LOG_INFO_RANK0 "HF_HOME=${HF_HOME} is not usable here; using ${HF_FALLBACK}"
        export HF_HOME="${HF_FALLBACK}"
    fi
else
    export HF_HOME="${HF_FALLBACK}"
fi
mkdir -p "${HF_HOME}/hub"

if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    if mkdir -p "${HF_HUB_CACHE}/.primus_hf_write_test_$$" 2>/dev/null; then
        rm -rf "${HF_HUB_CACHE}/.primus_hf_write_test_$$" 2>/dev/null || true
    else
        LOG_INFO_RANK0 "HF_HUB_CACHE=${HF_HUB_CACHE} is not usable here; using ${HF_HOME}/hub"
        export HF_HUB_CACHE="${HF_HOME}/hub"
    fi
fi

if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
    if mkdir -p "${TRANSFORMERS_CACHE}/.primus_hf_write_test_$$" 2>/dev/null; then
        rm -rf "${TRANSFORMERS_CACHE}/.primus_hf_write_test_$$" 2>/dev/null || true
    else
        LOG_INFO_RANK0 "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} is not usable here; using ${HF_HOME}/hub"
        export TRANSFORMERS_CACHE="${HF_HOME}/hub"
    fi
fi

HF_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# So the training process sees the same paths (execute_hooks exports env.* from hook stdout)
echo "env.HF_HOME=${HF_HOME}"
[[ -n "${HF_HUB_CACHE:-}" ]] && echo "env.HF_HUB_CACHE=${HF_HUB_CACHE}"
[[ -n "${TRANSFORMERS_CACHE:-}" ]] && echo "env.TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"

MEGATRON_PATH="${EFFECTIVE_DATA}/megatron_checkpoints/$(basename "${HF_PATH}")"

LOG_INFO_RANK0 "HF Model: ${HF_PATH}"
LOG_INFO_RANK0 "HF Cache: ${HF_CACHE}"
LOG_INFO_RANK0 "Megatron Path: ${MEGATRON_PATH}"

# Check if HF checkpoint already downloaded
HF_MODEL_CACHE="${HF_CACHE}/models--$(echo "${HF_PATH}" | tr '/' '--')"
if [[ -d "$HF_MODEL_CACHE" ]]; then
    LOG_INFO_RANK0 "HF checkpoint already cached at ${HF_MODEL_CACHE}"
else
    LOG_INFO_RANK0 "HF checkpoint will be downloaded from ${HF_PATH}"
fi

# Check if Megatron checkpoint already exists
if [[ -d "$MEGATRON_PATH" ]]; then
    LOG_INFO_RANK0 "Megatron checkpoint already exists at ${MEGATRON_PATH}, skipping conversion"
    echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"
    exit 0
fi

# Convert checkpoint (only on rank 0, others wait)
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
LOCK_FILE="${MEGATRON_PATH}.converting.lock"
DONE_FILE="${MEGATRON_PATH}.done"

if [[ "$NODE_RANK" == "0" ]]; then
    # Rank 0: perform the conversion
    LOG_INFO_RANK0 "[+] Converting HF checkpoint to Megatron format..."
    mkdir -p "$(dirname "${MEGATRON_PATH}")"

    # Create lock file
    touch "$LOCK_FILE"

    # Set up Python path for Megatron-Bridge
    export PYTHONPATH="${PRIMUS_ROOT}/third_party/Megatron-Bridge/src:${PRIMUS_ROOT}/third_party/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"

    python3 third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
      --hf-model "${HF_PATH}" \
      --megatron-path "${MEGATRON_PATH}"

    # Create done file and remove lock
    touch "$DONE_FILE"
    rm -f "$LOCK_FILE"

    LOG_SUCCESS_RANK0 "Checkpoint prepared at ${MEGATRON_PATH}"
else
    # Other ranks: wait for rank 0 to complete
    LOG_INFO_RANK0 "[RANK ${NODE_RANK}] Waiting for rank 0 to complete checkpoint conversion..."

    # Wait for done file (with timeout)
    timeout=600  # 10 minutes timeout
    elapsed=0
    while [[ ! -f "$DONE_FILE" ]] && [[ $elapsed -lt $timeout ]]; do
        if [[ ! -f "$LOCK_FILE" ]] && [[ ! -f "$DONE_FILE" ]]; then
            # Lock file doesn't exist and done file doesn't exist - rank 0 hasn't started yet
            sleep 2
        else
            # Lock file exists - conversion in progress
            sleep 5
        fi
        elapsed=$((elapsed + 5))
    done

    if [[ ! -f "$DONE_FILE" ]]; then
        echo "[RANK ${NODE_RANK}] Timeout waiting for checkpoint conversion"
        exit 1
    fi

    echo "[OK] [RANK ${NODE_RANK}] Checkpoint ready at ${MEGATRON_PATH}"
fi

echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"
