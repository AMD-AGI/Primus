#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# One timed MLPerf Training run of FLUX.1-Schnell on the Megatron backend.
#
# The logging patch fails closed on every value it would otherwise have to
# guess -- submission identity, precision disclosures, whether caches were
# dropped, where the log goes, which recipe produced it. This script is where
# those answers come from, which is why it is part of the submission rather
# than a convenience wrapper.
#
# One invocation produces one result file. A submission needs ten of them,
# each with its own seed; see run_campaign.sh.

set -euo pipefail

: "${PRIMUS_PATH:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
export PRIMUS_PATH

# --- What is being submitted ------------------------------------------------
: "${EXP:=${PRIMUS_PATH}/examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_fp8_mlperf.yaml}"
: "${MLLOG_SUBMISSION_ORG:=AMD}"
: "${MLLOG_SUBMISSION_DIVISION:=closed}"
: "${MLLOG_SUBMISSION_PLATFORM:=MI355X}"
: "${MLLOG_SUBMISSION_STATUS:=onprem}"
export EXP MLLOG_SUBMISSION_ORG MLLOG_SUBMISSION_DIVISION
export MLLOG_SUBMISSION_PLATFORM MLLOG_SUBMISSION_STATUS

# --- Numerics disclosure ----------------------------------------------------
# The compliance checker accepts a fixed vocabulary here (see
# mlperf_logging/compliance_checker/training_6.0.0/common.yaml). mxfp6 is not
# in it yet, so an MXFP6 run needs the format approved upstream before its log
# can pass; describing the run as anything else would be a false disclosure.
: "${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR:=fp8}"
: "${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN:=bfloat16}"
: "${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM:=bfloat16}"
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM

# --- This run ---------------------------------------------------------------
: "${RESULTS_DIR:=/results}"
: "${RUN_INDEX:=0}"
: "${PRIMUS_SEED:=$((42 + RUN_INDEX))}"
: "${MLLOG_OUTPUT_FILE:=${RESULTS_DIR}/result_${RUN_INDEX}.txt}"
export RESULTS_DIR RUN_INDEX PRIMUS_SEED MLLOG_OUTPUT_FILE

mkdir -p "${RESULTS_DIR}"

# --- Cold start -------------------------------------------------------------
# cache_clear is a claim about the machine, so drop the caches here and report
# what actually happened rather than what was requested. Dropping caches needs
# privileges the container may not have; a run that could not do it says so.
: "${MLPERF_CLEAR_CACHES:=true}"
if [[ "${MLPERF_CLEAR_CACHES}" == "true" ]]; then
    if sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
        echo "[MLPerf] Dropped page cache"
    else
        echo "[MLPerf] WARNING: could not drop page cache; reporting cache_clear=false"
        MLPERF_CLEAR_CACHES=false
    fi
fi
export MLPERF_CLEAR_CACHES

echo "============================================"
echo "MLPerf FLUX.1-Schnell Training (Megatron)"
echo "============================================"
echo "Recipe:    ${EXP}"
echo "Run index: ${RUN_INDEX}"
echo "Seed:      ${PRIMUS_SEED}"
echo "Result:    ${MLLOG_OUTPUT_FILE}"
echo "Division:  ${MLLOG_SUBMISSION_DIVISION} (${MLLOG_SUBMISSION_ORG} / ${MLLOG_SUBMISSION_PLATFORM})"
echo "Precision: linear=${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR}" \
     "attn=${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN}" \
     "comm=${MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM}"
echo "============================================"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${start_fmt}"

set +e
"${PRIMUS_PATH}/primus-cli" direct -- \
    train pretrain \
    --config "${EXP}" \
    2>&1 | tee "${RESULTS_DIR}/train_flux1_${RUN_INDEX}.log"
ret_code=${PIPESTATUS[0]}
set -e

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${end_fmt}"

result=$(( end - start ))
echo "RESULT,FLUX1,${PRIMUS_SEED},${result},${MLLOG_SUBMISSION_ORG},${start_fmt}"

if [[ ${ret_code} != 0 ]]; then
    echo "Training failed with exit code: ${ret_code}"
    exit "${ret_code}"
fi

# --- Check the artifact before it is treated as a result --------------------
# Finding out at submission time that a week of runs produced unparseable logs
# is the failure this guards against.
: "${MLPERF_RULESET:=6.0.0}"
: "${CHECK_COMPLIANCE:=1}"
if [[ "${CHECK_COMPLIANCE}" == "1" ]]; then
    python3 -m mlperf_logging.compliance_checker \
        --usage training \
        --ruleset "${MLPERF_RULESET}" \
        --log_output "${RESULTS_DIR}/compliance_${RUN_INDEX}.out" \
        "${MLLOG_OUTPUT_FILE}" \
        || echo "[MLPerf] WARNING: compliance check failed for run ${RUN_INDEX}"
fi

exit 0
