#!/usr/bin/env bash
# MLPerf DLRMv4 HSTU ranker on yambda-5b, one node of 4 x MI455X, in the MI455X image.
#
#   IMAGE=<image> DATA=<dataset dir> ./run_dlrmv4.sh <out_dir> [extra docker -e/-v args...]
#
# DATA is the prepared yambda-5b directory (processed_5b/ with the hstu_cache_L4086 flat-event
# cache; see README.md). Trains STEPS (default 200) steps from START_TS=0 at a local batch of 1024
# (global 4096), eval off, with the MI455X restricted-feature gin overlay. The run stops itself at
# STEPS with exit code 42 (DIE_AT_STEP). Writes train.log, mlperf.log and environment.txt to
# <out_dir>; tools/mi455/run_kit/analyze.py <out_dir> prints the median global samples/s.
# Extra arguments go to `docker run` before the image and override the defaults below.
set -uo pipefail
OUT=${1:?usage: run_dlrmv4.sh <out_dir> [extra docker args...]}; shift
IMAGE=${IMAGE:-amdprimus/amdprimus:mi455-training-20261108}
DATA=${DATA:?set DATA to the prepared yambda-5b dataset directory}
STEPS=${STEPS:-200}
OVERLAY=${OVERLAY:-/workspace/recommendation/generative_recommenders/dlrm_v4/train/gin/mi455/a37_1_smoke_minimal.gin}
NAME=${NAME:-dlrmv4_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

docker run --rm --name "$NAME" \
  --network=bridge --ipc=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v "$DATA":/data/mlperf_dlrm_v4:ro \
  -v "$OUT":/output \
  -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_NET=Socket -e NCCL_PROTO=Simple \
  -e NCCL_ALGO=Ring -e NCCL_SOCKET_IFNAME=lo -e RCCL_MSCCL_ENABLE=0 -e RCCL_MSCCLPP_ENABLE=0 \
  -e GPU_MAX_HW_QUEUES=2 -e HSA_ENABLE_SDMA=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTORCH_ALLOC_CONF= -e PYTORCH_CUDA_ALLOC_CONF= \
  -e AMDGCN_USE_BUFFER_OPS=0 -e TRITON_FULL_AUTOTUNE=0 \
  -e HSA_COREDUMP_PATTERN=/dev/null \
  -e HSTU_HAMMER_KERNEL=TRITON -e HSTU_BWD_MAX_VGPR=0 -e HSTU_BWD_BLOCK_N=128 \
  -e GR_TRITON_PLAIN_PTRS=1 -e FP8_WARMUP=0 -e CKPT_TIME_INTERVAL_S=0 -e FLYDSL_GPU_ARCH=gfx1250 \
  -e EMBEDDING_ROW_SCALE=1.0 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/opt/flydsl-0.2.4 \
  -e DLRM_DATA_PATH=/data/mlperf_dlrm_v4 \
  -e GPUS_PER_NODE=4 -e NNODES=1 -e NODE_RANK=0 -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 \
  -e SEED=1 -e START_TS=0 -e BATCH_SIZE=1024 -e DIE_AT_STEP="$STEPS" \
  -e EVAL_EVERY_N_WINDOWS=0 -e EVAL_EVERY_DATA_PCT=0 \
  -e METRIC_LOG_FREQ=1 -e OUTPUT_TRACE=0 \
  -e RUN_NAME="$NAME" \
  -e MLPERF_LOGGING=1 -e MLPERF_LOG_PATH=/output/mlperf.log -e MLPERF_SUBMISSION_PLATFORM=MI455X \
  -e DLRM_GIN_OVERLAY="$OVERLAY" \
  "$@" \
  -w /workspace/recommendation \
  "$IMAGE" \
  bash -c 'env | sort > /output/environment.txt; exec python -m generative_recommenders.dlrm_v4.train.train_ranker --dataset yambda-5b --mode streaming-train-eval' \
  > "$OUT/train.log" 2>&1
RC=$?
echo "exit code $RC (42 = stopped at DIE_AT_STEP=$STEPS)" | tee "$OUT/exit-status.txt"
if grep -q "Step $STEPS " "$OUT/train.log" || grep -q "Step $((STEPS-1)) " "$OUT/train.log"; then
  echo "reached step $STEPS"
else
  echo "run did not reach step $STEPS; see $OUT/train.log" >&2; exit 1
fi
