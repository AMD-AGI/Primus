#!/bin/bash
# Run the DSv4 Flash proxy (1 GPU, EP=1, 32 experts) with opt3_best.env.
# usage: TRITON_MAIN=<triton repo>/python bash run_opt3_1gpu.sh [eager|graph] [iters]
# Run inside the Primus container with Primus at /workspace/Primus and Primus-Turbo at /workspace/Primus-Turbo.
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-graph}; ITERS=${2:-20}
source "$HERE/opt3_best.env"
: "${TRITON_MAIN:?set TRITON_MAIN to a Triton main checkout python dir with gluon cdna5 (dsv4up needs it)}"
export PYTHONPATH=$HERE/fla_stub:$TRITON_MAIN:/workspace/Primus-Turbo${PYTHONPATH:+:$PYTHONPATH}
EXTRA="--turbo_mega_moe_precision,fp8_block"
[ "$MODE" = graph ] && EXTRA="--cuda_graph_impl,full_iteration,--check_for_nan_in_loss_and_grad,False,$EXTRA"
export PRIMUS_EXTRA_TRAIN_ARGS=$EXTRA
export PRIMUS_LAUNCHER=direct GPUS_PER_NODE=1 NNODES=1 PRIMUS_TP=1 PRIMUS_PP=1 PRIMUS_EP=1 MBS=1 GBS=1
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-32} USE_TURBO_DEEPEP=False ENABLE_PRIMUS_TURBO=True
export PRECISION_TYPE=BF16 FP8_RECIPE=tensorwise TRAIN_ITERS=$ITERS PRIMUS_GPU_MODEL=MI455X HSA_COREDUMP_PATTERN=/dev/null PYTORCH_ROCM_ARCH=gfx1250
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261109}
cd /workspace/Primus
exec bash examples/megatron/models/deepseek-v4/run_deepseek_v4_flash_proxy.sh
