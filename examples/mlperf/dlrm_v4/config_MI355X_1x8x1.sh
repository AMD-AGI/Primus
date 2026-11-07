#!/bin/bash
# MLPerf DLRMv4 (dlrmv4) configuration for MI355X, 1 node x 8 GPUs.

export DGXSYSTEM=MI355X_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

export PRIMUS_PATH=/workspace/Primus
export PRIMUS_MLPERF=1
export EXP=${PRIMUS_PATH}/examples/mlperf/dlrm_v4/configs/MI355X/dlrm_v4-yambda5b-mlperf.yaml
export DLRM_DATA_PATH=/data

# MLPerf RCP shape for one node: global batch 8192 (1024 per GPU), LR 1e-6 (the
# trainer's gin values, spelled out for the submission). 2 nodes: 2e-6, 4 nodes:
# 4e-6. The first eval follows the MLPerf rule for the global batch
# (SKIP_EVAL_EPOCH_PCT unset); evals every 0.1% of the data to AUC 0.75.
export BATCH_SIZE=1024
export DENSE_LR=1e-6
export SPARSE_LR=1e-6
export EVAL_EVERY_DATA_PCT=0.001
export AUC_THRESHOLD=0.75

# Data state at the start of the run: evict and verify the dataset before INIT,
# then stream it back into the page cache once run_start is logged.
export CLEAR_CACHES=1
export WARM_CACHE_AFTER_RUN_START=1

export ENABLE_MLLOG=1
export MLPERF_LOGGING=1
export MLPERF_LOG_PATH=/results/mlperf_logging.out
export MLLOG_SUBMISSION_PLATFORM=MI355X
MLLOG_CONFIG_FILENAME=$(basename "${BASH_SOURCE[0]}")
export MLLOG_CONFIG_FILENAME

# Recipe environment. The image's HIP device-queue settings (GPU_USE_DEVICE_QUEUE=1,
# DEBUG_CLR_AQL_DEV_QUEUE=1) are kept: +0.23% 200-step throughput over 0 on v26.9
# (MI355X, 2 runs each; the ROCm 10.2 port measured 0 vs 1 as neutral and ran 0).
export HSTU_HAMMER_KERNEL=TRITON
export FLYDSL_GPU_ARCH=gfx950
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1
# numpy's transparent huge pages stall every rank on a fragmented host (T29).
export NUMPY_MADVISE_HUGEPAGE=0
