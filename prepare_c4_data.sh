#!/bin/bash
###############################################################################
# Prepare C4 English dataset for Megatron training with DeepSeek V3
#
# This script:
#   1. Downloads C4-en data from HuggingFace (configurable amount)
#       GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
#       cd c4
#       git lfs pull --include "en/*"
#   2. Converts to JSONL format
#   3. Tokenizes into Megatron .bin/.idx format using DeepSeekV3Tokenizer
#
# Usage:
#   bash prepare_c4_data.sh [--num_shards N] [--data_dir /path/to/data]
#
# By default downloads 1 shard (~350MB compressed, ~3M documents) for testing.
# Full C4-en has 1024 shards. Adjust --num_shards for more data.
###############################################################################

set -e

# ======================== Configuration ========================
NUM_SHARDS=${NUM_SHARDS:-200}           # Number of C4 shards to download (1-1024)
DATA_DIR=${DATA_DIR:-"/shared/c4"}
PRIMUS_PATH=${PRIMUS_PATH:-"/shared/john/Primus"}
TOKENIZER_TYPE="DeepSeekV3Tokenizer"
TOKENIZER_MODEL="deepseek-ai/DeepSeek-V3"
WORKERS=${WORKERS:-$(nproc)}          # Number of preprocessing workers
HF_TOKEN=${HF_TOKEN:-"your_hf_token"}             # Set your HuggingFace token

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_shards) NUM_SHARDS="$2"; shift 2;;
        --data_dir)   DATA_DIR="$2";   shift 2;;
        --workers)    WORKERS="$2";    shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ======================== Paths ========================
export RAW_DIR="${DATA_DIR}/en"       # Pre-downloaded shards live here
export JSONL_DIR="${DATA_DIR}/jsonl"
export TOKENIZED_DIR="${DATA_DIR}/tokenized"
export TRAIN_OUTPUT_PREFIX="${TOKENIZED_DIR}/c4_en_train"
export NUM_SHARDS

mkdir -p "$RAW_DIR" "$JSONL_DIR" "$TOKENIZED_DIR"

echo "============================================"
echo "C4 English Data Preparation"
echo "============================================"
echo "NUM_SHARDS:     ${NUM_SHARDS} (out of 1024 total)"
echo "DATA_DIR:       ${DATA_DIR}"
echo "PRIMUS_PATH:    ${PRIMUS_PATH}"
echo "TOKENIZER:      ${TOKENIZER_TYPE} / ${TOKENIZER_MODEL}"
echo "WORKERS:        ${WORKERS}"
echo "============================================"

# ======================== Step 1: Merge shards into JSONL ========================
echo ""
echo ">>> Step 1: Merging C4 English shards into JSONL (${NUM_SHARDS} shards)..."
echo "    (Download skipped — using pre-downloaded shards in ${RAW_DIR})"

JSONL_FILE="${JSONL_DIR}/c4_en_train.jsonl"

if [ -f "${JSONL_FILE}" ]; then
    echo "JSONL file already exists: ${JSONL_FILE}"
    echo "Skipping merge. Delete it to re-merge."
else
    # Verify shards exist
    MISSING=0
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        SHARD_NAME=$(printf "c4-train.%05d-of-01024.json.gz" "$i")
        if [ ! -f "${RAW_DIR}/${SHARD_NAME}" ]; then
            echo "  WARNING: Missing shard ${SHARD_NAME}"
            MISSING=$((MISSING + 1))
        fi
    done
    if [ "$MISSING" -gt 0 ]; then
        echo "ERROR: ${MISSING} shard(s) missing in ${RAW_DIR}. Cannot proceed."
        exit 1
    fi

    echo "Decompressing and merging shards into JSONL ..."
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        SHARD_NAME=$(printf "c4-train.%05d-of-01024.json.gz" "$i")
        SHARD_PATH="${RAW_DIR}/${SHARD_NAME}"
        echo "  [${i}/${NUM_SHARDS}] Decompressing ${SHARD_NAME} ..."
        zcat "${SHARD_PATH}" >> "${JSONL_FILE}"
    done

    DOC_COUNT=$(wc -l < "${JSONL_FILE}")
    echo "Done! Total documents: ${DOC_COUNT}"
    echo "Saved to: ${JSONL_FILE}"
fi

echo ">>> Step 1 complete."

# ======================== Step 2: Tokenize ========================
echo ""
echo ">>> Step 2: Tokenizing with ${TOKENIZER_TYPE}..."

JSONL_FILE="${JSONL_DIR}/c4_en_train.jsonl"

if [ -f "${TRAIN_OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${TRAIN_OUTPUT_PREFIX}_text_document.idx" ]; then
    echo "Tokenized files already exist:"
    echo "  ${TRAIN_OUTPUT_PREFIX}_text_document.bin"
    echo "  ${TRAIN_OUTPUT_PREFIX}_text_document.idx"
    echo "Skipping tokenization. Delete them to re-tokenize."
else
    # Need to set up Python path for Megatron imports
    export PYTHONPATH="${PRIMUS_PATH}/third_party/Megatron-LM:${PRIMUS_PATH}:${PYTHONPATH:-}"

    python3 "${PRIMUS_PATH}/examples/megatron/preprocess_data.py" \
        --input "${JSONL_FILE}" \
        --tokenizer-type "${TOKENIZER_TYPE}" \
        --tokenizer-model "${TOKENIZER_MODEL}" \
        --output-prefix "${TRAIN_OUTPUT_PREFIX}" \
        --workers "${WORKERS}" \
        --append-eod \
        --partitions 1

    echo ">>> Step 2 complete."
fi

# ======================== Summary ========================
echo ""
echo "============================================"
echo "Data preparation complete!"
echo "============================================"
echo ""
echo "Tokenized data files:"
ls -lh "${TOKENIZED_DIR}/"
echo ""
echo "To use this data for training, set in run_dsv3.sh:"
echo ""
echo "  1. Change:  --mock_data True  →  --mock_data False"
echo "  2. Add env:  export PRIMUS_TOKENIZED_DATA_PATH=${TRAIN_OUTPUT_PREFIX}_text_document"
echo ""
echo "Or pass directly via environment variable before running:"
echo "  export PRIMUS_TOKENIZED_DATA_PATH=${TRAIN_OUTPUT_PREFIX}_text_document"
echo ""
echo "============================================"
