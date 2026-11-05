#!/bin/bash
# GLM-5.3-Flash on MI355X with the Primus `direct` launcher (run inside the
# rocm/primus container on every node).
#
#   Full model, 45 layers (bf16 ~600 GB): >= 4 nodes for training, e.g.
#     NNODES=4 NODE_RANK=<r> MASTER_ADDR=<node0> PRIMUS_PP=4 PRIMUS_EP=8 \
#       GLM5_HF_LOAD=/path/GLM-5.3-Flash bash run_glm5_3_flash.sh
#   Smoke test, first 8 layers on one node:
#     PRIMUS_MODEL=glm5_3_flash_8L bash run_glm5_3_flash.sh
#   Forward-only logprob eval (1 node, EP8, no optimizer state):
#     GLM5_HF_LOAD=... GLM5_LOGPROB_EVAL_INPUT=tokens.jsonl \
#       GLM5_LOGPROB_EVAL_OUTPUT=logprobs.jsonl bash run_glm5_3_flash.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO}"

export PRIMUS_LAUNCHER=direct
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST 2>/dev/null || true
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29537}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export BACKEND_PATH=${BACKEND_PATH:-${REPO}/third_party/Megatron-LM}
export PRIMUS_OUTPUT_ROOT=${PRIMUS_OUTPUT_ROOT:-${REPO}/output}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(whoami)}
if [ "${NNODES}" = 1 ]; then
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
fi
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}

export PRIMUS_MODEL=${PRIMUS_MODEL:-glm5_3_flash}
export GLM5_TOKENIZER=${GLM5_TOKENIZER:-${GLM5_HF_LOAD:-zai-org/GLM-5.3-Flash}}
export PRIMUS_SEQ_LENGTH=${SEQ:-${PRIMUS_SEQ_LENGTH:-4096}}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-8}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-${PRIMUS_MODEL}_n${NNODES}_tp${PRIMUS_TP}_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}}
EXP=${EXP:-examples/megatron/configs/MI355X/glm5_3_flash-BF16-pretrain.yaml}

LOGDIR="$PRIMUS_OUTPUT_ROOT/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
mkdir -p "$LOGDIR"
echo "[glm5.3] model=$PRIMUS_MODEL nodes=$NNODES tp=$PRIMUS_TP pp=$PRIMUS_PP ep=$PRIMUS_EP seq=$PRIMUS_SEQ_LENGTH log=$LOGDIR"

# shellcheck disable=SC2086
./primus-cli direct -- train pretrain --config "$EXP" \
  --backend_path "$BACKEND_PATH" \
  ${EXTRA_ARGS:-} \
  2>&1 | tee "$LOGDIR/log_node${NODE_RANK}.txt"
