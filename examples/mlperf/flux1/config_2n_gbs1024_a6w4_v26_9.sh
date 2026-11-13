#!/usr/bin/env bash

export MLLOG_CONFIG_FILENAME=${MLLOG_CONFIG_FILENAME:-config_2n_gbs1024_a6w4_v26_9.sh}
export WARMUP_STEPS=${WARMUP_STEPS:-1600}
# shellcheck source=examples/mlperf/flux1/config_2n_gbs512_a6w4_v26_9.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config_2n_gbs512_a6w4_v26_9.sh"
export GRADIENT_ACCUMULATION_STEPS=2
export GLOBAL_BATCH_SIZE=1024
