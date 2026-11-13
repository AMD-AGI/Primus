#!/usr/bin/env bash

export TORCH_DIST_USE_NCCL2=${TORCH_DIST_USE_NCCL2:-1}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-2}
export FLUX_FSDP_RS_STREAM=${FLUX_FSDP_RS_STREAM:-normal}
export MLLOG_CONFIG_FILENAME=${MLLOG_CONFIG_FILENAME:-config_2n_gbs512_a6w4_v26_9.sh}
# shellcheck source=examples/mlperf/flux1/config_2n_gbs512_a6w4.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config_2n_gbs512_a6w4.sh"
