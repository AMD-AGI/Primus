#!/bin/bash
###############################################################################
# Download SCROLLS gov-report MLPerf dataset and build packed .npy + metadata
# when post_trainer uses dataset_type=mlperf_dataset.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../../../lib/common.sh" || {
    echo "[ERROR] Failed to load common library" >&2
    exit 1
}

PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"
cd "${PRIMUS_ROOT}"

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

read -r DATA_DIR SEQ_LENGTH <<< "$(python3 -c "
import os
import sys
sys.path.insert(0, '${PRIMUS_ROOT}')
from pathlib import Path
from primus.core.config.primus_config import load_primus_config, get_module_config

cfg = load_primus_config(Path('${CONFIG_FILE}'), None)
post = get_module_config(cfg, 'post_trainer')
if post is None:
    sys.exit(0)
params = getattr(post, 'params', None)
if params is None or getattr(params, 'dataset_type', '') != 'mlperf_dataset':
    sys.exit(0)

train_path = getattr(params, 'packed_train_data_path', None) or os.environ.get('PACKED_DATA_DIR', '/data')
train_path = os.path.expandvars(str(train_path))
data_dir = os.path.dirname(train_path) if train_path.endswith('.npy') else train_path
seq_length = int(getattr(params, 'seq_length', 8192) or 8192)
print(data_dir, seq_length)
" 2>/dev/null || true)"

if [[ -z "${DATA_DIR}" ]]; then
    exit 0
fi

SEQ_LENGTH="${SEQ_LENGTH:-8192}"
mkdir -p "${DATA_DIR}"

LOG_INFO_RANK0 "[mlperf-data] Data root: ${DATA_DIR} (seq_length=${SEQ_LENGTH})"
echo "env.PACKED_DATA_DIR=${DATA_DIR}"
echo "env.DATA_PATH=${DATA_DIR}"

if [[ -f "${DATA_DIR}/train.npy" && -f "${DATA_DIR}/validation.npy" && -f "${DATA_DIR}/packed_metadata.jsonl" ]]; then
    LOG_INFO_RANK0 "[mlperf-data] Dataset already present; skipping download"
    exit 0
fi

if [[ -z "${HF_TOKEN:-}" && ( ! -f "${DATA_DIR}/train.npy" || ! -f "${DATA_DIR}/validation.npy" ) ]]; then
    LOG_ERROR_RANK0 "[mlperf-data] HF_TOKEN is required to download regisss/scrolls_gov_report_preprocessed_mlperf_2"
    exit 1
fi

MLPERF_RECIPE_DIR="${PRIMUS_ROOT}/primus/backends/megatron_bridge/recipes/mlperf_llama2_70b"

if [[ ! -f "${DATA_DIR}/train.npy" || ! -f "${DATA_DIR}/validation.npy" ]]; then
    LOG_INFO_RANK0 "[mlperf-data] Downloading and converting MLPerf dataset..."
    python3 "${MLPERF_RECIPE_DIR}/download_dataset.py" --data_dir "${DATA_DIR}"
    python3 "${MLPERF_RECIPE_DIR}/convert_dataset.py" --data_dir "${DATA_DIR}"
fi

if [[ ! -f "${DATA_DIR}/packed_metadata.jsonl" ]]; then
    LOG_INFO_RANK0 "[mlperf-data] Creating packed_metadata.jsonl..."
    python3 "${MLPERF_RECIPE_DIR}/create_metadata.py" "${SEQ_LENGTH}" "${DATA_DIR}/packed_metadata.jsonl"
fi

for f in train.npy validation.npy packed_metadata.jsonl; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        LOG_ERROR_RANK0 "[mlperf-data] Expected file missing: ${DATA_DIR}/${f}"
        exit 1
    fi
done

LOG_SUCCESS_RANK0 "[mlperf-data] MLPerf dataset ready under ${DATA_DIR}"
