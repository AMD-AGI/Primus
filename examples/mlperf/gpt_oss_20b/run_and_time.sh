#!/bin/bash

set -e

# Create results directory
mkdir -p /results

cd "${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-train.mlperfpretrain.exp.log}"

# Under a multi-node scheduler wrapper, inherit rendezvous + node sizing from
# SLURM so the same benchmark config scales without edits. Single-node jobs
# fall through to the config defaults.
if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -gt 1 ]]; then
    NNODES="${SLURM_NNODES}"
    NODE_RANK="${SLURM_NODEID:-0}"
fi

# TE 2.15 lazily compiles CK attention blobs. Populate both GPT-OSS windows
# once before torchrun so eight ranks do not race while writing the cache.
if [ "${MLPERF_RUNTIME_SERIES:-v26.3}" = "v26.5" ] \
    && [ "${MLPERF_SKIP_ATTENTION_PREWARM:-0}" != "1" ]; then
    python3 /opt/mlperf-gpt-oss-20b/prewarm_attention.py
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
    2>&1 | tee "${TRAIN_LOG_FILE}"
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
