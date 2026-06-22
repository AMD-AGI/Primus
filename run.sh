#!/bin/bash

export GLOO_SOCKET_IFNAME=ens1f0
export NCCL_SOCKET_IFNAME=ens1f0
export GPUS_PER_NODE=1

export LD_PRELOAD=/root/rocm-libraries/projects/hipblaslt/hipblaslt-install/lib/libhipblaslt.so.1${LD_PRELOAD:+:$LD_PRELOAD}
export PRIMUS_TURBO_GEMM_BACKEND=hipblaslt
export PRIMUS_TURBO_GROUPED_GEMM_BACKEND=hipblaslt

./primus-cli --debug direct train pretrain --config qwen3_235B_A22B-FP8.turbo.yaml
# ./primus-cli --debug direct train pretrain --config qwen3_235B_A22B-FP8.te.yaml
