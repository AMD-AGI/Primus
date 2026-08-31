#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Run the full flux1 submission campaign: ten timed runs, each with its own
# seed, then check the collected logs as a set.
#
# Ten is not a convention -- mlperf_logging/rcp_checker/rcp_checker.py maps
# flux1 to 10 runs, and the RCP comparison is made over that many results. A
# run that ends without converging still produces a result file and still
# counts as one of the ten; dropping it would bias the set.

set -euo pipefail

: "${PRIMUS_PATH:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${RESULTS_DIR:=/results}"
: "${NUM_RUNS:=10}"
: "${SEED_BASE:=42}"
: "${MLPERF_RULESET:=6.0.0}"
export PRIMUS_PATH RESULTS_DIR

mkdir -p "${RESULTS_DIR}"

failed_runs=()
for (( index = 0; index < NUM_RUNS; index++ )); do
    echo
    echo "########## flux1 run ${index} of ${NUM_RUNS} ##########"
    # Each run is checked on its own inside run_and_time.sh; a failure here is
    # recorded and the campaign continues, because nine good runs plus a
    # diagnosis beats stopping on the first bad one after hours of compute.
    if ! RUN_INDEX="${index}" \
         PRIMUS_SEED="$(( SEED_BASE + index ))" \
         bash "${PRIMUS_PATH}/examples/mlperf/flux1/megatron/run_and_time.sh"; then
        failed_runs+=("${index}")
    fi
done

echo
echo "########## campaign summary ##########"
if (( ${#failed_runs[@]} > 0 )); then
    echo "Runs that exited non-zero: ${failed_runs[*]}"
else
    echo "All ${NUM_RUNS} runs completed."
fi

echo
echo "Comparing the collected results against the reference convergence points:"
python3 -m mlperf_logging.rcp_checker \
    --rcp_usage training \
    --rcp_version "${MLPERF_RULESET}" \
    --log_output "${RESULTS_DIR}/rcp_checker.out" \
    --verbose \
    "${RESULTS_DIR}" \
    || echo "[MLPerf] WARNING: RCP comparison failed; see the output above"
