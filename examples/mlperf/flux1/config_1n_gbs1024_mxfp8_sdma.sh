#!/usr/bin/env bash

export FLUX_TRAINING_RECIPE=mxfp8
export FSDP_ALL_GATHER_BACKEND=rccl_sdma
source "$(dirname -- "${BASH_SOURCE[0]}")/config_1n_gbs1024.sh"
