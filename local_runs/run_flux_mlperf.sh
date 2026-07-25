#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ "$NNODES" == "1" ]]; then
    export MASTER_PORT=$((20000 + RANDOM % 40000))
  else
    export MASTER_PORT=29500
  fi
else
  export MASTER_PORT
fi
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export CONFIG=${CONFIG:-examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml}
export DATASET_PATH=${DATASET_PATH:-/data/cc12m-preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export PROMPT_DROPOUT_PROB=${PROMPT_DROPOUT_PROB:-0.1}

export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash_attn_aiter}
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-64}
export MAX_STEPS=${MAX_STEPS:-30000}
export LR=${LR:-2e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
export FSDP2_RESHARD_AFTER_FORWARD=${FSDP2_RESHARD_AFTER_FORWARD:-true}
export SAVE_STRATEGY=${SAVE_STRATEGY:-none}
export LOG_FREQ=${LOG_FREQ:-10}
export MLPERF_ENABLE=${MLPERF_ENABLE:-true}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}
export RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
export BENCH_SKIP_STEPS=${BENCH_SKIP_STEPS:-1}

LOG_FILE=${LOG_FILE:-local_runs/flux_mlperf_${RUN_TAG}.log}
SUMMARY_FILE=${SUMMARY_FILE:-local_runs/flux_mlperf_${RUN_TAG}_summary.txt}
OUTPUT_DIR=${OUTPUT_DIR:-local_runs/flux_mlperf_${RUN_TAG}_output}
MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-local_runs/flux_mlperf_${RUN_TAG}_mllog.txt}
RANK_LOG_DIR=${RANK_LOG_DIR:-local_runs/flux_mlperf_${RUN_TAG}_ranklogs}
export OUTPUT_DIR
export MLLOG_OUTPUT_FILE

require_file() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo "[run_flux_mlperf] missing required file: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "[run_flux_mlperf] missing required directory: $path" >&2
    exit 2
  fi
}

preflight() {
  require_file "$CONFIG"
  require_dir "$DATASET_PATH"
  require_file "$DATASET_PATH/state.json"
  require_file "$DATASET_PATH/dataset_info.json"
  if [[ "${MLPERF_ENABLE,,}" == "true" || "$MLPERF_ENABLE" == "1" ]]; then
    require_dir "$EVAL_DATASET_PATH"
    require_file "$EVAL_DATASET_PATH/state.json"
    require_file "$EVAL_DATASET_PATH/dataset_info.json"
  fi
  require_dir "$EMPTY_ENCODINGS_PATH"
  require_file "$EMPTY_ENCODINGS_PATH/t5_empty.npy"
  require_file "$EMPTY_ENCODINGS_PATH/clip_empty.npy"
}

summarize_metrics() {
  python3 - "$LOG_FILE" "$SUMMARY_FILE" "$BENCH_SKIP_STEPS" <<'PY'
from __future__ import annotations

import os
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
skip = int(sys.argv[3])

ansi_re = re.compile(r"\x1b\[[0-9;]*m")
metric_re = re.compile(
    r"step=(?P<step>\d+).*?"
    r"mem=(?P<alloc>[0-9.]+)/(?P<reserved>[0-9.]+)GB "
    r"peak_mem=(?P<peak>[0-9.]+)GB.*?"
    r"step_time=(?P<step_time>[0-9.]+)s "
    r"throughput=(?P<tps>[0-9.]+)samples/gpu/s"
)

records_by_step: dict[int, dict[str, float]] = {}
if log_path.exists():
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        line = ansi_re.sub("", raw_line)
        match = metric_re.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        records_by_step[step] = {
            "alloc_gb": float(match.group("alloc")),
            "reserved_gb": float(match.group("reserved")),
            "peak_gb": float(match.group("peak")),
            "step_time_s": float(match.group("step_time")),
            "tps_samples_per_gpu_s": float(match.group("tps")),
        }

records = [(step, records_by_step[step]) for step in sorted(records_by_step)]
used = records[skip:] if skip > 0 else records

def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")

def maximum(values: list[float]) -> float:
    return max(values) if values else float("nan")

step_times = [record["step_time_s"] for _, record in used]
tps_values = [record["tps_samples_per_gpu_s"] for _, record in used]
step_rates = [1.0 / value for value in step_times if value > 0]
alloc_values = [record["alloc_gb"] for _, record in used]
reserved_values = [record["reserved_gb"] for _, record in used]
peak_values = [record["peak_gb"] for _, record in used]

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as handle:
    handle.write("FLUX MLPerf benchmark summary\n")
    handle.write(f"log_file: {log_path}\n")
    handle.write(f"steps_logged: {len(records)}\n")
    handle.write(f"bench_skip_logged_steps: {skip}\n")
    handle.write(f"steps_used: {len(used)}\n")
    handle.write(f"nnodes: {os.environ.get('NNODES', '')}\n")
    handle.write(f"gpus_per_node: {os.environ.get('GPUS_PER_NODE', '')}\n")
    handle.write(f"local_batch_size: {os.environ.get('LOCAL_BATCH_SIZE', '')}\n")
    handle.write(f"max_steps: {os.environ.get('MAX_STEPS', '')}\n")
    handle.write(f"lr: {os.environ.get('LR', '')}\n")
    handle.write(f"warmup_steps: {os.environ.get('WARMUP_STEPS', '')}\n")
    handle.write(f"dataset_path: {os.environ.get('DATASET_PATH', '')}\n")
    handle.write(f"eval_dataset_path: {os.environ.get('EVAL_DATASET_PATH', '')}\n")
    handle.write(f"empty_encodings_path: {os.environ.get('EMPTY_ENCODINGS_PATH', '')}\n")
    handle.write(f"attention_backend: {os.environ.get('ATTENTION_BACKEND', '')}\n")
    handle.write(f"mlperf_enable: {os.environ.get('MLPERF_ENABLE', '')}\n")
    handle.write(f"target_accuracy: {os.environ.get('TARGET_ACCURACY', '')}\n")
    handle.write(f"val_check_interval: {os.environ.get('VAL_CHECK_INTERVAL', '')}\n")
    handle.write(f"mean_gpu_allocated_gb: {mean(alloc_values):.4f}\n")
    handle.write(f"mean_gpu_reserved_gb: {mean(reserved_values):.4f}\n")
    handle.write(f"max_gpu_peak_mem_gb: {maximum(peak_values):.4f}\n")
    handle.write(f"mean_step_time_s: {mean(step_times):.4f}\n")
    handle.write(f"mean_step_per_s: {mean(step_rates):.4f}\n")
    handle.write(f"mean_tps_samples_per_gpu_s: {mean(tps_values):.4f}\n")

if not records:
    print(f"[run_flux_mlperf] no metric lines found in {log_path}", file=sys.stderr)
    sys.exit(1)

print(summary_path.read_text(encoding="utf-8"), end="")
PY
}

preflight
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$SUMMARY_FILE")" "$OUTPUT_DIR" "$RANK_LOG_DIR"

echo "[run_flux_mlperf] config=$CONFIG"
echo "[run_flux_mlperf] dataset_path=$DATASET_PATH"
echo "[run_flux_mlperf] eval_dataset_path=$EVAL_DATASET_PATH"
echo "[run_flux_mlperf] empty_encodings_path=$EMPTY_ENCODINGS_PATH"
echo "[run_flux_mlperf] output_dir=$OUTPUT_DIR"
echo "[run_flux_mlperf] log_file=$LOG_FILE"
echo "[run_flux_mlperf] summary_file=$SUMMARY_FILE"
echo "[run_flux_mlperf] mllog_output_file=$MLLOG_OUTPUT_FILE"
echo "[run_flux_mlperf] rank_log_dir=$RANK_LOG_DIR"
echo "[run_flux_mlperf] nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node=$GPUS_PER_NODE"
echo "[run_flux_mlperf] steps=$MAX_STEPS local_batch_size=$LOCAL_BATCH_SIZE lr=$LR warmup_steps=$WARMUP_STEPS"
echo "[run_flux_mlperf] mlperf_enable=$MLPERF_ENABLE target_accuracy=$TARGET_ACCURACY val_check_interval=$VAL_CHECK_INTERVAL"

set +e
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --log-dir="$RANK_LOG_DIR" --tee=3 \
  -m primus.cli.main train pretrain \
  --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
train_status=${PIPESTATUS[0]}
set -e

summary_status=0
summarize_metrics || summary_status=$?

if [[ "$train_status" -ne 0 ]]; then
  echo "[run_flux_mlperf] training failed with exit code $train_status" >&2
  exit "$train_status"
fi
if [[ "$summary_status" -ne 0 ]]; then
  echo "[run_flux_mlperf] metric summary failed with exit code $summary_status" >&2
  exit "$summary_status"
fi
