#!/bin/bash

set -e

# Create results directory
mkdir -p /results

cd "${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b"

# Under multi-node SLURM (run_with_docker_slurm.sh), inherit rendezvous + node
# sizing from SLURM env so we can scale to N nodes without editing the config
# file. Single-node SLURM jobs (NNODES=1) fall through to the config defaults
# so torchrun doesn't try to do c10d rdzv against MASTER_ADDR=localhost.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

echo "============================================"
echo "MLPerf GPT-OSS-20B Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "Rank:   ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Launch through Primus CLI and keep the real exit code even though output is
# piped through tee.
set +e
"${PRIMUS_PATH}/primus-cli" direct -- \
    train pretrain \
    --config "${EXP}" \
    2>&1 | tee train.mlperfpretrain.exp.log
ret_code=${PIPESTATUS[0]}
set -e

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="GPT_OSS_20B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit "$ret_code"
fi

exit 0
