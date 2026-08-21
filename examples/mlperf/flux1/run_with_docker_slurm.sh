#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${SLURM_JOB_ID:?Run this script inside a Slurm allocation}"
: "${SLURM_NNODES:?Missing SLURM_NNODES from the allocation}"
: "${SLURM_JOB_NODELIST:?Missing SLURM_JOB_NODELIST from the allocation}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the MLPerf datasets}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

command -v srun >/dev/null || { echo "srun is required" >&2; exit 1; }

export NNODES=${NNODES:-$SLURM_NNODES}
if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ "$NNODES" == "1" ]]; then
    MASTER_ADDR=${SLURMD_NODENAME:-$SLURM_JOB_NODELIST}
  elif hostnames=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null); then
    MASTER_ADDR=${hostnames%%$'\n'*}
  elif [[ -n "${SPUR_PEER_NODES:-}" ]]; then
    MASTER_ADDR=${SPUR_PEER_NODES%%,*}
    MASTER_ADDR=${MASTER_ADDR%%:*}
  else
    echo "Set MASTER_ADDR: this scheduler cannot expand SLURM_JOB_NODELIST" >&2
    exit 1
  fi
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-29500}

srun_args=(--nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1)
if command -v spur >/dev/null; then
  srun_args+=(--jobid="$SLURM_JOB_ID" --overlap)
fi

forward_vars=(
  DATA_ROOT OUTPUT_ROOT DOCKER_IMAGE CONFIG PRIMUS_WORKSPACE GPUS_PER_NODE DP_REPLICATE
  DATASET_PATH EVAL_DATASET_PATH EMPTY_ENCODINGS_PATH OUTPUT_DIR
  FLUX_FLOAT8_RECIPE FLUX_FP8_GEMM_BACKEND ATTENTION_BACKEND
  TORCHINDUCTOR_BENCHMARK_FUSION PRIMUS_FLUX_AITER_ATOMIC_FP32 PRIMUS_FLUX_REUSE_FP8_INPUT
  LOCAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS LR WARMUP_STEPS
  GRADIENT_CHECKPOINTING_RATIO COMPILE_TRANSFORMER_BLOCKS
  COMPILE_STRATEGY COMPILE_FULLGRAPH COMPILE_DYNAMIC
  FSDP2_RESHARD_AFTER_FORWARD FSDP2_REDUCE_DTYPE FLUX_PERFORMANCE_MODE
  SAVE_STEPS SAVE_STRATEGY CHECKPOINT_KEEP_LATEST RESUME_FROM_CHECKPOINT
  MLPERF_ENABLE MLPERF_CLEAR_CACHES MLLOG_OUTPUT_FILE TARGET_ACCURACY
  VAL_CHECK_INTERVAL SEED
)
env_args=(NNODES="$NNODES" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT")
for name in "${forward_vars[@]}"; do
  [[ -v "$name" ]] && env_args+=("$name=${!name}")
done

srun "${srun_args[@]}" env "${env_args[@]}" bash "$SCRIPT_DIR/run_with_docker.sh"
