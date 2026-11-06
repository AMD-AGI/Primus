#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Re-record the DLRMv4 trainer's per-stack tables for a new image (run inside the
# image, on one MI355X node). The trainer checks each table's stack against the
# running one and ignores tables recorded on another stack, so after an image
# change (torch, hipBLASLt, rocBLAS, FBGEMM builds) one-time costs move into the
# timed region until the tables are re-recorded:
#
#   blas LABEL
#       BLAS first-use replay table (T26): 8 scan parts on 8 GPUs, merged, verified
#       (rescan after replaying the table) and filtered to the rows a replay fixes.
#       Writes generative_recommenders/dlrm_v4/train/blas_replay_table_LABEL.jsonl.
#   shim OUT_DIR
#       Builds the code-object load monitor (coload_shim.c) into OUT_DIR. Run a
#       full training run with LD_PRELOAD=OUT_DIR/coload_shim.so
#       COLOAD_LOG_DIR=<dir> to collect the load logs for `hip`.
#   hip LABEL LOAD_DIR MLPERF_LOG [TRACE_GLOB]
#       HIP preload table (T12) from that run's load logs (and optional Kineto
#       traces): writes generative_recommenders/dlrm_v4/train/hip_preload_table_LABEL.jsonl.
#
# Then add the new file names to blas_warmup._BLAS_REPLAY_TABLES /
# hip_preload.STACK_TABLES. Tables are written to $TABLE_DIR (default: the trainer's
# train/ folder), work files to $WORK (default /tmp/dlrm_tables).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${HERE}/../../.." && pwd)"
SRC="${PRIMUS_ROOT}/primus/backends/recommendation/dlrm_v4/src"
TRAIN_DIR="${SRC}/generative_recommenders/dlrm_v4/train"
WORK="${WORK:-/tmp/dlrm_tables}"
TABLE_DIR="${TABLE_DIR:-$TRAIN_DIR}"
NGPUS="${NGPUS:-8}"
mkdir -p "$WORK" "$TABLE_DIR"

# gen_blas_replay_table.py scan|verify, one part per GPU in parallel; fails if any part fails.
run_parts() {  # <scan|verify> [table]
    local mode=$1 table=${2:-} pids=() g rc=0 args
    for ((g = 0; g < NGPUS; g++)); do
        if [[ "$mode" == scan ]]; then
            args=(scan "$g" "$NGPUS" "$WORK/scan_${g}.jsonl")
        else
            args=(verify "$g" "$NGPUS" "$table" "$WORK/verify_${g}.jsonl")
        fi
        HIP_VISIBLE_DEVICES=$g PYTHONPATH="$SRC" python3 "$HERE/gen_blas_replay_table.py" "${args[@]}" \
            > "$WORK/${mode}_${g}.log" 2>&1 &
        pids+=($!)
    done
    for g in "${!pids[@]}"; do
        if ! wait "${pids[$g]}"; then
            echo "[tables] $mode part $g failed:" >&2
            tail -20 "$WORK/${mode}_${g}.log" >&2
            rc=1
        fi
    done
    return $rc
}

case "${1:-}" in
    blas)
        label=${2:?label, e.g. rocm10.2.0a20261020}
        table="$TABLE_DIR/blas_replay_table_${label}.jsonl"
        echo "[tables] BLAS scan on $NGPUS GPUs..."
        run_parts scan
        python3 "$HERE/gen_blas_replay_table.py" merge "$table" "$WORK"/scan_*.jsonl
        echo "[tables] verify: replay the table, rescan with a shifted interleave"
        run_parts verify "$table"
        grep -h '^REPLAY' "$WORK"/verify_*.log | head -1
        # Keep the rows a replay fixes (library loads); drop costs that recur at every new M.
        mv "$table" "$WORK/blas_unfiltered.jsonl"
        python3 "$HERE/gen_blas_replay_table.py" filter "$WORK/blas_unfiltered.jsonl" "$table" "$WORK"/verify_*.jsonl
        ;;
    shim)
        out=${2:?output dir}
        mkdir -p "$out"
        gcc -O2 -fPIC -shared -Wall -o "$out/coload_shim.so" "$HERE/coload_shim.c" -ldl -lpthread
        echo "[tables] built $out/coload_shim.so"
        ;;
    hip)
        label=${2:?label} loads=${3:?load log dir} mllog=${4:?mlperf.log of that run} traces=${5:-}
        site=$(python3 -c 'import os, torch; print(os.path.dirname(os.path.dirname(torch.__file__)))')
        libs=(--lib "$site/torch/lib/libtorch_hip.so")
        for so in "$site"/fbgemm_gpu/fbgemm_gpu*.so; do libs+=(--lib "$so"); done
        args=("${libs[@]}" --loads "1n=$loads/coload.*.log" --run-start "1n=$mllog"
              --stack "$label" --out "$TABLE_DIR/hip_preload_table_${label}.jsonl"
              --summary "$WORK/hip_preload_summary.json")
        [[ -n "$traces" ]] && args+=(--trace "1n=$traces")
        PYTHONPATH="$SRC" python3 "$HERE/gen_hip_preload_table.py" "${args[@]}"
        ;;
    *)
        sed -n '9,29p' "${BASH_SOURCE[0]}"
        exit 2
        ;;
esac
