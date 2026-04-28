#!/bin/bash

set -e

mkdir -p /results

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29502}
export EXP=${EXP:-/workspace/code/conf/llama3.1_8B-pretrain.yaml}
export DATA_PATH=${DATA_PATH:-/data}

echo "============================================"
echo "MLPerf LLama3.1 8B Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "Train iters: ${PRIMUS_TRAIN_ITERS}"
echo "Eval interval: ${PRIMUS_EVAL_INTERVAL}"
echo "Enable MLPerf logging: ${ENABLE_MLPERF}"
echo "MLLOG_TRAIN_LOSS_LOG_FREQ: ${MLLOG_TRAIN_LOSS_LOG_FREQ}"
echo "MLLOG_TARGET_EVAL_LOSS: ${MLLOG_TARGET_EVAL_LOSS}"
echo "MLLOG_SUBMISSION_BENCHMARK: ${MLLOG_SUBMISSION_BENCHMARK}"
echo "MLLOG_SUBMISSION_DIVISION: ${MLLOG_SUBMISSION_DIVISION}"
echo "MLLOG_SUBMISSION_ORG: ${MLLOG_SUBMISSION_ORG}"
echo "MLLOG_SUBMISSION_PLATFORM: ${MLLOG_SUBMISSION_PLATFORM}"
echo "============================================"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py

ret_code=$?

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

result=$(( end - start ))
result_name="LLAMA3.1_8B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0

