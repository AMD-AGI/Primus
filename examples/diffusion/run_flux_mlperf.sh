#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export DATASET_PATH=${DATASET_PATH:-/data/cc12m_preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export OUTPUT_DIR=${OUTPUT_DIR:-/output/flux_mlperf}
export MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-$OUTPUT_DIR/mlperf_compliance.log}

export FLUX_FLOAT8_RECIPE=${FLUX_FLOAT8_RECIPE:-tensorwise}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash_attn_aiter}
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-64}
export MAX_STEPS=${MAX_STEPS:-30000}
export GRADIENT_CHECKPOINTING_RATIO=${GRADIENT_CHECKPOINTING_RATIO:-0.25}
export COMPILE_TRANSFORMER_BLOCKS=${COMPILE_TRANSFORMER_BLOCKS:-true}
export FSDP2_RESHARD_AFTER_FORWARD=${FSDP2_RESHARD_AFTER_FORWARD:-false}
export FSDP2_REDUCE_DTYPE=${FSDP2_REDUCE_DTYPE:-fp32}
export MLPERF_ENABLE=${MLPERF_ENABLE:-true}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}
export SEED=${SEED:-10007}

mkdir -p "$OUTPUT_DIR"
if [[ "${MLPERF_CLEAR_CACHES:-true}" == "true" ]]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches
fi

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-8}" \
  -m primus.cli.main train pretrain \
  --config examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml
