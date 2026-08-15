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
export DATASET_PATH=${DATASET_PATH:-/data/cc12m_preprocessed}
export EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/data/coco_preprocessed}
export EMPTY_ENCODINGS_PATH=${EMPTY_ENCODINGS_PATH:-/data/empty_encodings}
export PROMPT_DROPOUT_PROB=${PROMPT_DROPOUT_PROB:-0.1}

export FLUX_FLOAT8_RECIPE=${FLUX_FLOAT8_RECIPE:-}
if [[ -z "${ATTENTION_BACKEND:-}" ]]; then
  if [[ "$FLUX_FLOAT8_RECIPE" == "tensorwise" ]]; then
    export ATTENTION_BACKEND=flash_attn_aiter
  else
    export ATTENTION_BACKEND=sdpa
  fi
else
  export ATTENTION_BACKEND
fi
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-64}
export MAX_STEPS=${MAX_STEPS:-30000}
export LR=${LR:-2e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
export GRADIENT_CHECKPOINTING_RATIO=${GRADIENT_CHECKPOINTING_RATIO:-0.25}
export COMPILE_TRANSFORMER_BLOCKS=${COMPILE_TRANSFORMER_BLOCKS:-true}
export COMPILE_STRATEGY=${COMPILE_STRATEGY:-per_block}
export COMPILE_BACKEND=${COMPILE_BACKEND:-inductor}
export COMPILE_FULLGRAPH=${COMPILE_FULLGRAPH:-true}
export COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-false}
export COMPILE_OUTPUT_HEAD=${COMPILE_OUTPUT_HEAD:-false}
export FSDP2_RESHARD_AFTER_FORWARD=${FSDP2_RESHARD_AFTER_FORWARD:-false}
export TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE:-}
export FSDP2_REDUCE_DTYPE=${FSDP2_REDUCE_DTYPE:-fp32}
export PROFILE=${PROFILE:-false}
export PROFILE_RANK=${PROFILE_RANK:-0}
export PROFILE_WAIT_STEPS=${PROFILE_WAIT_STEPS:-10}
export PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-2}
export PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS:-10}
export PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-}
export PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-false}
# performance_only isolates steady training; nemo_mlperf prewarms and times
# train/eval blocks with MLPerf semantics.
export FLUX_PERFORMANCE_MODE=${FLUX_PERFORMANCE_MODE:-nemo_mlperf}
case "$FLUX_PERFORMANCE_MODE" in
  performance_only)
    export MLPERF_ENABLE=false
    export MLPERF_WARMUP_TRAIN_STEPS=0
    export MLPERF_WARMUP_VALIDATION_STEPS=0
    : "${BENCH_SKIP_STEPS:=10}"
    ;;
  nemo_mlperf)
    export MLPERF_ENABLE=true
    export MLPERF_WARMUP_TRAIN_STEPS=${MLPERF_WARMUP_TRAIN_STEPS:-2}
    export MLPERF_WARMUP_VALIDATION_STEPS=${MLPERF_WARMUP_VALIDATION_STEPS:-2}
    : "${BENCH_SKIP_STEPS:=1}"
    ;;
  *)
    echo "[run_flux_mlperf] unsupported FLUX_PERFORMANCE_MODE=$FLUX_PERFORMANCE_MODE" >&2
    exit 2
    ;;
esac
export BENCH_SKIP_STEPS
export DISABLE_CHECKPOINT=true
export SAVE_STEPS=0
export SAVE_STRATEGY=none
export CHECKPOINT_KEEP_LATEST=0
export RESUME_FROM_CHECKPOINT=
export LOG_FREQ=${LOG_FREQ:-10}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}
export SEED=${SEED:-10007}
export MLPERF_CLEAR_CACHES=${MLPERF_CLEAR_CACHES:-true}
export RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
export TORCHRUN_TEE=${TORCHRUN_TEE:-true}

LOG_FILE=${LOG_FILE:-local_runs/flux_mlperf_${RUN_TAG}.log}
SUMMARY_FILE=${SUMMARY_FILE:-local_runs/flux_mlperf_${RUN_TAG}_summary.txt}
OUTPUT_DIR=${OUTPUT_DIR:-local_runs/flux_mlperf_output}
FINAL_MLLOG_OUTPUT_FILE=${MLLOG_OUTPUT_FILE:-local_runs/flux_mlperf_${RUN_TAG}_mllog.txt}
if [[ "$FLUX_PERFORMANCE_MODE" == "nemo_mlperf" ]]; then
  export MLLOG_OUTPUT_FILE="/tmp/primus-${RUN_TAG}-mllog.txt"
else
  export MLLOG_OUTPUT_FILE="$FINAL_MLLOG_OUTPUT_FILE"
fi
RANK_LOG_DIR=${RANK_LOG_DIR:-local_runs/flux_mlperf_${RUN_TAG}_ranklogs}
export OUTPUT_DIR
export ENABLE_WANDB_LOGGER=${ENABLE_WANDB_LOGGER:-false}
export WANDB_PROJECT=${WANDB_PROJECT:-mlperf-flux1}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-flux-mlperf}
export WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-$OUTPUT_DIR/wandb}
export WANDB_OFFLINE=${WANDB_OFFLINE:-false}
export WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-train}

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

clear_os_caches() {
  if [[ "${MLPERF_ENABLE,,}" != "true" && "$MLPERF_ENABLE" != "1" ]]; then
    return
  fi
  if [[ "${MLPERF_CLEAR_CACHES,,}" != "true" && "$MLPERF_CLEAR_CACHES" != "1" ]]; then
    echo "[run_flux_mlperf] WARNING: OS cache clear disabled; run is not submission-compliant" >&2
    return
  fi
  sync
  if [[ "$(id -u)" == "0" ]]; then
    echo 3 > /proc/sys/vm/drop_caches
  else
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  fi
  echo "[run_flux_mlperf] cleared OS caches on node_rank=$NODE_RANK"
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
    handle.write(f"performance_mode: {os.environ.get('FLUX_PERFORMANCE_MODE', '')}\n")
    handle.write(f"mlperf_warmup_train_steps: {os.environ.get('MLPERF_WARMUP_TRAIN_STEPS', '')}\n")
    handle.write(f"mlperf_warmup_validation_steps: {os.environ.get('MLPERF_WARMUP_VALIDATION_STEPS', '')}\n")
    handle.write(f"save_steps: {os.environ.get('SAVE_STEPS', '')}\n")
    handle.write(f"save_strategy: {os.environ.get('SAVE_STRATEGY', '')}\n")
    handle.write(f"disable_checkpoint: {os.environ.get('DISABLE_CHECKPOINT', '')}\n")
    handle.write(f"resume_from_checkpoint: {os.environ.get('RESUME_FROM_CHECKPOINT', '')}\n")
    handle.write(f"seed: {os.environ.get('SEED', '')}\n")
    handle.write(f"dataset_path: {os.environ.get('DATASET_PATH', '')}\n")
    handle.write(f"eval_dataset_path: {os.environ.get('EVAL_DATASET_PATH', '')}\n")
    handle.write(f"empty_encodings_path: {os.environ.get('EMPTY_ENCODINGS_PATH', '')}\n")
    handle.write(f"attention_backend: {os.environ.get('ATTENTION_BACKEND', '')}\n")
    handle.write(f"float8_recipe: {os.environ.get('FLUX_FLOAT8_RECIPE', '')}\n")
    handle.write(f"float8_gemm_backend: {os.environ.get('FLUX_FP8_GEMM_BACKEND', '')}\n")
    handle.write(f"gradient_checkpointing_ratio: {os.environ.get('GRADIENT_CHECKPOINTING_RATIO', '')}\n")
    handle.write(f"compile_transformer_blocks: {os.environ.get('COMPILE_TRANSFORMER_BLOCKS', '')}\n")
    handle.write(f"compile_strategy: {os.environ.get('COMPILE_STRATEGY', '')}\n")
    handle.write(f"compile_backend: {os.environ.get('COMPILE_BACKEND', '')}\n")
    handle.write(f"compile_fullgraph: {os.environ.get('COMPILE_FULLGRAPH', '')}\n")
    handle.write(f"compile_dynamic: {os.environ.get('COMPILE_DYNAMIC', '')}\n")
    handle.write(f"compile_output_head: {os.environ.get('COMPILE_OUTPUT_HEAD', '')}\n")
    handle.write(f"fsdp2_reshard_after_forward: {os.environ.get('FSDP2_RESHARD_AFTER_FORWARD', '')}\n")
    handle.write(f"torch_compile_mode: {os.environ.get('TORCH_COMPILE_MODE', '')}\n")
    handle.write(f"fsdp2_reduce_dtype: {os.environ.get('FSDP2_REDUCE_DTYPE', '')}\n")
    handle.write(f"profile: {os.environ.get('PROFILE', '')}\n")
    handle.write(f"profile_output_dir: {os.environ.get('PROFILE_OUTPUT_DIR', '')}\n")
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

terminate_training() {
  trap - INT TERM
  local children=()
  mapfile -t children < <(jobs -pr)
  for child in "${children[@]}"; do
    kill -INT "$child" 2>/dev/null || true
  done
  for child in "${children[@]}"; do
    wait "$child" 2>/dev/null || true
  done
  exit 130
}

preflight
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$SUMMARY_FILE")" "$OUTPUT_DIR" "$RANK_LOG_DIR"
clear_os_caches

echo "[run_flux_mlperf] config=$CONFIG"
echo "[run_flux_mlperf] dataset_path=$DATASET_PATH"
echo "[run_flux_mlperf] eval_dataset_path=$EVAL_DATASET_PATH"
echo "[run_flux_mlperf] empty_encodings_path=$EMPTY_ENCODINGS_PATH"
echo "[run_flux_mlperf] output_dir=$OUTPUT_DIR"
echo "[run_flux_mlperf] log_file=$LOG_FILE"
echo "[run_flux_mlperf] summary_file=$SUMMARY_FILE"
echo "[run_flux_mlperf] performance_mode=$FLUX_PERFORMANCE_MODE"
echo "[run_flux_mlperf] mllog_output_file=$FINAL_MLLOG_OUTPUT_FILE local=$MLLOG_OUTPUT_FILE"
echo "[run_flux_mlperf] mlperf_warmup=train:$MLPERF_WARMUP_TRAIN_STEPS validation:$MLPERF_WARMUP_VALIDATION_STEPS"
echo "[run_flux_mlperf] rank_log_dir=$RANK_LOG_DIR"
echo "[run_flux_mlperf] nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node=$GPUS_PER_NODE"
echo "[run_flux_mlperf] steps=$MAX_STEPS local_batch_size=$LOCAL_BATCH_SIZE lr=$LR warmup_steps=$WARMUP_STEPS"
echo "[run_flux_mlperf] checkpoint_disabled=$DISABLE_CHECKPOINT save_steps=$SAVE_STEPS strategy=$SAVE_STRATEGY keep_latest=$CHECKPOINT_KEEP_LATEST resume=${RESUME_FROM_CHECKPOINT:-none}"
echo "[run_flux_mlperf] wandb=$ENABLE_WANDB_LOGGER project=$WANDB_PROJECT run_name=$WANDB_RUN_NAME"
echo "[run_flux_mlperf] seed=$SEED mlperf_enable=$MLPERF_ENABLE target_accuracy=$TARGET_ACCURACY val_check_interval=$VAL_CHECK_INTERVAL"
echo "[run_flux_mlperf] checkpoint_ratio=$GRADIENT_CHECKPOINTING_RATIO compile=$COMPILE_TRANSFORMER_BLOCKS strategy=$COMPILE_STRATEGY compile_output_head=$COMPILE_OUTPUT_HEAD reshard_after_forward=$FSDP2_RESHARD_AFTER_FORWARD"
echo "[run_flux_mlperf] compile_backend=$COMPILE_BACKEND fullgraph=$COMPILE_FULLGRAPH dynamic=$COMPILE_DYNAMIC mode=$TORCH_COMPILE_MODE fsdp2_reduce_dtype=$FSDP2_REDUCE_DTYPE"
echo "[run_flux_mlperf] float8_recipe=${FLUX_FLOAT8_RECIPE:-bf16} fp8_gemm_backend=${FLUX_FP8_GEMM_BACKEND:-default}"

trap terminate_training INT TERM
torchrun_log_args=()
if [[ "${TORCHRUN_TEE,,}" == "true" || "$TORCHRUN_TEE" == "1" ]]; then
  torchrun_log_args=(--log-dir="$RANK_LOG_DIR" --tee=3)
fi
set +e
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  "${torchrun_log_args[@]}" \
  -m primus.cli.main train pretrain \
  --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
train_status=${PIPESTATUS[0]}
set -e
trap - INT TERM

if [[ "$MLLOG_OUTPUT_FILE" != "$FINAL_MLLOG_OUTPUT_FILE" && -f "$MLLOG_OUTPUT_FILE" ]]; then
  mkdir -p "$(dirname "$FINAL_MLLOG_OUTPUT_FILE")"
  cp "$MLLOG_OUTPUT_FILE" "$FINAL_MLLOG_OUTPUT_FILE"
fi

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
