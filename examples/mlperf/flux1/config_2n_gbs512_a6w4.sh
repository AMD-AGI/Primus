#!/usr/bin/env bash

# Scale the single-node microbatch recipe to two nodes without adding fusion/switch changes.
export MLLOG_CONFIG_FILENAME=${MLLOG_CONFIG_FILENAME:-config_2n_gbs512_a6w4.sh}
if [[ "${FLUX_CLUSTER:-}" == "crusoe" ]]; then
    export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-0}
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
    export NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ:-1}
fi
# shellcheck source=examples/mlperf/flux1/config_1n_gbs256_a6w4.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config_1n_gbs256_a6w4.sh"
export NNODES=2
export GLOBAL_BATCH_SIZE=512
