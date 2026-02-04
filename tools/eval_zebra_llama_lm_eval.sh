#!/usr/bin/env bash
###############################################################################
# Evaluate Zebra-Llama (HF checkpoint) with lm-eval-harness
#
# Default checkpoint:
#   output/zebra_llama_1B_hf_iter_0150000
#
# Usage:
#   cd /vfs/silo/mingyyan/home_backup/Primus
#   ./tools/eval_zebra_llama_lm_eval.sh \
#     --checkpoint output/zebra_llama_1B_hf_iter_0150000 \
#     --tasks arc_easy,hellaswag,winogrande \
#     --batch-size 8 \
#     --dtype bfloat16 \
#     --output eval_results/zebra_llama_1B_iter_0150000
###############################################################################

set -euo pipefail

CHECKPOINT="output/zebra_llama_1B_hf_iter_0150000"
TASKS="arc_easy,hellaswag,winogrande"
BATCH_SIZE="8"
DTYPE="bfloat16"
OUTPUT="eval_results/zebra_llama_1B_iter_0150000"
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

cd /vfs/silo/mingyyan/home_backup/Primus

export TOKENIZERS_PARALLELISM=false

MODEL_ARGS="pretrained=${CHECKPOINT},trust_remote_code=True,dtype=${DTYPE}"

echo "========================================================================"
echo "Zebra-Llama lm-eval"
echo "========================================================================"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Tasks:       ${TASKS}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Dtype:       ${DTYPE}"
echo "Output:      ${OUTPUT}"
if [[ -n "${LIMIT}" ]]; then
  echo "Limit:       ${LIMIT}"
fi
echo "========================================================================"

mkdir -p "${OUTPUT}"

CMD=(python3 -m lm_eval
  --model hf
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo ""
echo "[Run] ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "========================================================================"
echo "Done. Results in: ${OUTPUT}"
echo "========================================================================"

