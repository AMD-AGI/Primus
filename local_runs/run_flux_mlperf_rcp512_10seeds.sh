#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

RESULT_DIR=${RESULT_DIR:-local_runs/flux_mlperf_rcp512_results}
mkdir -p "$RESULT_DIR"

# Fixed, unique seeds make the campaign reproducible while satisfying the
# MLPerf requirement that repeated runs not reuse one seed.
SEEDS=(10007 11239 12763 13999 15173 16661 17881 19013 20507 21911)

for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  tag="rcp512_run${index}_seed${seed}_$(date +%Y%m%d_%H%M%S)"
  echo "[run_flux_mlperf_rcp512_10seeds] run=$index seed=$seed"
  SEED="$seed" \
  RUN_TAG="$tag" \
  MLLOG_OUTPUT_FILE="$RESULT_DIR/result_${index}.txt" \
  LOG_FILE="$RESULT_DIR/run_${index}.log" \
  SUMMARY_FILE="$RESULT_DIR/run_${index}_summary.txt" \
  OUTPUT_DIR="$RESULT_DIR/run_${index}_output" \
  RANK_LOG_DIR="$RESULT_DIR/run_${index}_ranklogs" \
    bash "$SCRIPT_DIR/run_flux_mlperf.sh"
  python3 -m mlperf_logging.compliance_checker \
    --usage training \
    --ruleset 6.0.0 \
    --werror \
    --log_output "$RESULT_DIR/run_${index}_compliance.log" \
    "$RESULT_DIR/result_${index}.txt"
done

python3 -m mlperf_logging.rcp_checker \
  --rcp_usage training \
  --rcp_version 6.0.0 \
  --verbose \
  --log_output "$RESULT_DIR/rcp_checker.log" \
  "$RESULT_DIR"
