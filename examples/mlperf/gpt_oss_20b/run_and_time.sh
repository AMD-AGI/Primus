#!/bin/bash

set -e

mkdir -p /results

cd "${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-train.mlperfpretrain.exp.log}"

# Under multi-node SLURM (run_with_docker.sh / a scheduler wrapper), inherit
# rendezvous + node sizing from SLURM env so we can scale without editing the
# config file. Single-node jobs fall through to the config defaults.
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

if [[ -n "${PRIMUS_TOKENIZER_MODEL:-}" ]]; then
    echo "Tokenizer: ${PRIMUS_TOKENIZER_MODEL} (environment override)"
elif [[ -d /model && -n "$(ls -A /model 2>/dev/null)" ]]; then
    export PRIMUS_TOKENIZER_MODEL=/model
    echo "Tokenizer: /model (local mount)"
else
    unset PRIMUS_TOKENIZER_MODEL
    echo "Tokenizer: meta-llama/Llama-3.1-8B (HuggingFace default)"
fi

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
result_name="GPT_OSS_20B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit "$ret_code"
fi

exit 0
