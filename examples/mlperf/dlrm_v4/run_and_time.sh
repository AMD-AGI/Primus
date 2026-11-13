#!/bin/bash
# MLPerf DLRMv4 timed run inside the container (config_<system>.sh sourced or
# passed in the environment). Prints the MLPerf RESULT line.
set -e

cd "${PRIMUS_PATH}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-/results/train.dlrm_v4.log}"

# Under multi-node SLURM, take the node layout from the scheduler.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
    export NNODES NODE_RANK
fi

echo "============================================"
echo "MLPerf DLRMv4 (HSTU, yambda-5b) Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DLRM_DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}  Nodes: ${NNODES}  Rank: ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Seed:   ${SEED:-random}"
echo "============================================"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

set +e
"${PRIMUS_PATH}/primus-cli" direct -- \
    train pretrain \
    --config "${EXP}" \
    2>&1 | tee "${TRAIN_LOG_FILE}"
ret_code=${PIPESTATUS[0]}
set -e

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

result=$(( end - start ))
echo "RESULT,DLRM_V4,${SEED:-},$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit "$ret_code"
fi
exit 0
