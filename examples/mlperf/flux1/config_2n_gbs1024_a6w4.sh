#!/usr/bin/env bash

# Two-node adaptation of the four-node schedule; not an AMD submission profile.
export MLLOG_CONFIG_FILENAME=${MLLOG_CONFIG_FILENAME:-config_2n_gbs1024_a6w4.sh}
# shellcheck source=examples/mlperf/flux1/config_4n_gbs1024.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config_4n_gbs1024.sh"

export NNODES=2
export GRADIENT_ACCUMULATION_STEPS=2
export DGXSYSTEM=2n_gbs1024_a6w4
