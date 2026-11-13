#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Enhanced-ILD loss of fixed weights (lr 0) on packed text, on one GPU, against the same frozen
# teacher as the packed ILD recipes. Score a run's checkpoints, its upcycled init, and the
# reference (convert_hylo_reference_to_megatron.py) on the same test split, then compare the logs
# with summarize_hylo_ild_eval.py.
#
# Usage: tools/hybrid/run_hylo_ild_eval.sh <m2|mla> <name> <checkpoint dir> <packed data prefix>
#
# Environment:
#   HYLO_LLAMA32_1B_MEGATRON_CKPT, HYLO_TOKENIZER_MODEL  the teacher and tokenizer, as for training
#   HYLO_EVAL_ITERS  steps of two windows each; half the test split's "windows" count in the
#                    stats file prepare_hylo_packed_data.py writes (default 348)
#   GPU              device index (default 0)
#   MASTER_PORT      (default 29650)
#
# A checkpoint directory holds latest_checkpointed_iteration.txt; to score one iteration of a run,
# point at a directory whose tracker names that iteration.
set -euo pipefail

KIND=${1:?m2 or mla}
NAME=${2:?name}
CKPT=${3:?checkpoint directory}
DATA=${4:?packed data prefix}
case "$KIND" in
  m2 | mla) ;;
  *)
    echo "unknown kind '$KIND' (m2 or mla)" >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")/../.."
ITERS=${HYLO_EVAL_ITERS:-348}
echo "[$(date -u +%T)] hylo-ild-eval-$NAME: $CKPT on $DATA, GPU ${GPU:-0}, $ITERS iterations"
PRIMUS_EXP_NAME="hylo-ild-eval-$NAME" HYLO_EVAL_CKPT="$CKPT" HYLO_EVAL_DATA="$DATA" HYLO_EVAL_ITERS="$ITERS" \
  GPUS_PER_NODE=1 HIP_VISIBLE_DEVICES="${GPU:-0}" MASTER_PORT="${MASTER_PORT:-29650}" \
  ./primus-cli direct -- train pretrain --config "examples/megatron/configs/MI300X/hylo_${KIND}_1B-ild-eval.yaml"
