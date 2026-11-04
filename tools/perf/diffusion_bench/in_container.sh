#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Steps executed inside the benchmark container (invoked by run_benchmarks.sh via `docker exec`;
# this folder is mounted read-only at /diffusion_bench).
#
# Usage: in_container.sh <step> [args...]
#   setup                                   patch Makefile, record software versions
#   download all|<model>...                 download HF assets (all -> `make download_assets`)
#   diffusion <model> <batch_size> [hydra overrides...]
#   dlrm <TF32|FP32> [batch_size]
#   cleanup                                 kill leftover training processes
#   finalize <uid> <gid>                    hand results / HF cache ownership back to the host user
#
# Exit codes: 0 success, 2 out-of-memory, 3 benchmark reported failure, 4 not run (dry run),
#             other = error.
set -uo pipefail

DIFF_DIR=/workspace/AMDiffusionBenchmark
DLRM_DIR=/workspace/DLRMBenchmark
RESULTS=/results
HF_CACHE=/root/.cache/huggingface

log() { echo "[in_container $(date '+%F %T')] $*"; }

retry() {
    local n=0 max=3 shown="$*"
    # No caller passes the token as an argument; this keeps it so if one does.
    [[ -n "${HF_TOKEN:-}" ]] && shown=${shown//"$HF_TOKEN"/<redacted>}
    until "$@"; do
        n=$((n + 1))
        if (( n >= max )); then
            log "FAILED after ${max} attempts: $shown"
            return 1
        fi
        log "Retry ${n}/${max}: $shown"
        sleep 10
    done
}

step_setup() {
    local tag=${1:-main}
    cd "$DIFF_DIR" || exit 1
    # The token login becomes a no-op: hf reads HF_TOKEN from the environment,
    # and on a command line the token would show in the host's `ps`.
    # shellcheck disable=SC2016  # $(HF_TOKEN) is the Makefile's literal text
    sed -i 's/huggingface-cli login/hf auth login/g; s/huggingface-cli download/hf download/g;
            s/hf auth login --token \$(HF_TOKEN)/true/g' Makefile
    log "Makefile patched to use 'hf download', with HF_TOKEN from the environment"

    mkdir -p "$RESULTS/env"
    {
        echo "date: $(date -Is)"
        echo "hostname: $(hostname)"
        echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null || hipconfig --version 2>/dev/null || echo unknown)"
        echo "python: $(python --version 2>&1)"
        python -c 'import torch; print("torch:", torch.__version__, "| hip:", torch.version.hip, "| gpus:", torch.cuda.device_count(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")' 2>&1
        for repo in "$DIFF_DIR" "$DLRM_DIR"; do
            echo "$(basename "$repo"): $(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
        done
    } | tee "$RESULTS/env/versions_${tag}.txt"
    pip freeze > "$RESULTS/env/pip_freeze_${tag}.txt" 2>/dev/null || true
    git -C "$DIFF_DIR" status --short > "$RESULTS/env/AMDiffusionBenchmark_git_status.txt" 2>/dev/null || true
}

yaml_value() {
    # yaml_value <file> <key>  -> top-level scalar value
    sed -n "s/^$2:[[:space:]]*//p" "$1" | head -1 | tr -d "\"'"
}

step_download() {
    cd "$DIFF_DIR" || exit 1
    # hf uses HF_TOKEN from the environment; no login, which would put the
    # token on a command line.
    [[ -n "${HF_TOKEN:-}" ]] && log "Using HF_TOKEN from the environment"

    if [[ "${1:-all}" == "all" ]]; then
        # The Makefile includes .env.example (HF_TOKEN=), which would mask the exported token;
        # -e lets the environment win.
        retry make -e download_assets </dev/null
        return
    fi

    local model cfg repo dataset
    declare -A seen=()
    for model in "$@"; do
        cfg="config/train_args/${model}.yaml"
        [[ -f "$cfg" ]] || { log "Unknown model config: $cfg"; return 1; }
        repo=$(yaml_value "$cfg" model_path)
        dataset=$(yaml_value "$cfg" train_data_path)
        if [[ -n "$dataset" && -z "${seen[$dataset]:-}" ]]; then
            seen[$dataset]=1
            log "Downloading dataset $dataset"
            retry hf download --repo-type=dataset "$dataset" >/dev/null || return 1
        fi
        if [[ -n "$repo" && -z "${seen[$repo]:-}" ]]; then
            seen[$repo]=1
            log "Downloading model $repo"
            retry hf download "$repo" >/dev/null || return 1
        fi
    done
    log "Download completed."
}

step_diffusion() {
    local model=$1 bs=$2
    shift 2
    cd "$DIFF_DIR" || exit 1

    local out="$RESULTS/diffusion/${model}_bs${bs}"
    rm -rf "$out"
    mkdir -p "$out"

    local cmd=(python launcher.py "train_args=${model}" "train_args.train_batch_size=${bs}"
               "train_args.logging_dir=${out}" "$@")
    log "Running: ${cmd[*]}"
    "${cmd[@]}"
    local rc=$?

    # Checkpoints are written into the container after every run (tens of GB); not needed.
    rm -rf "$DIFF_DIR/outputs/checkpoint"

    # launcher.py exits 0 even when the training run fails; the real status is in runs_summary.csv.
    local status
    status=$(python - "$out/runs_summary.csv" <<'EOF'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    print(rows[-1].get("status", "unknown") if rows else "missing")
except FileNotFoundError:
    print("missing")
EOF
)
    log "launcher exit code=${rc}, run status=${status}"
    case "$status" in
        success) return 0 ;;
        oom|oom-skipped) return 2 ;;
        not_run) return 4 ;;
        *) return 3 ;;
    esac
}

newest_train_log() {
    # launch_training_single_node.sh names these log_<time>.txt, so ls -t is safe here.
    # shellcheck disable=SC2012
    ls -1t training_logs/log_*.txt 2>/dev/null | head -1
}

step_dlrm() {
    local precision=$1 bs=${2:-}
    cd "$DLRM_DIR" || exit 1

    if [[ -n "$bs" ]]; then
        sed -i "s/^export BATCH_SIZE=.*/export BATCH_SIZE=${bs}/" training_config.sh
    fi
    log "DLRM config:"
    cat training_config.sh; echo

    rm -rf training_logs results.csv
    mkdir -p training_logs

    # The launch script redirects all training output into training_logs/log_<time>.txt;
    # stream that file so progress shows up in our log too.
    ./launch_training_single_node.sh -p "$precision" &
    local pid=$!
    local train_log=""
    while kill -0 "$pid" 2>/dev/null && [[ -z "$train_log" ]]; do
        sleep 2
        train_log=$(newest_train_log)
    done
    if [[ -n "$train_log" ]]; then
        tail -n +1 -F --pid="$pid" "$train_log" 2>/dev/null
    fi
    wait "$pid"
    local rc=$?
    train_log=$(newest_train_log)

    local out="$RESULTS/dlrm/${precision}"
    rm -rf "$out"
    mkdir -p "$out"
    cp training_config.sh "$out/"
    [[ -n "$train_log" ]] && cp "$train_log" "$out/train.log"
    [[ -f results.csv ]] && cp results.csv "$out/results.csv"
    log "launch script exit code=${rc}; artifacts saved to ${out}"

    if [[ -z "$train_log" ]]; then
        return 3
    fi
    if grep -qiE "out of memory|OutOfMemoryError" "$train_log"; then
        return 2
    fi
    if grep -qE "Traceback|ChildFailedError" "$train_log"; then
        return 3
    fi
    return "$rc"
}

step_cleanup() {
    pkill -9 -f 'launcher.py|accelerate launch|train.py|torchrun|dlrm_main.py|launch_training_single_node' 2>/dev/null || true
    sleep 5
}

step_finalize() {
    local uid=$1 gid=$2
    chown -R "${uid}:${gid}" "$RESULTS" 2>/dev/null || true
    # Only files created by this container (root-owned); the cache may be shared with other users.
    find "$HF_CACHE" -user 0 -print0 2>/dev/null | xargs -0 -r chown -h "${uid}:${gid}" 2>/dev/null || true
}

step=${1:-}
shift || true
case "$step" in
    setup)     step_setup "$@" ;;
    download)  step_download "$@" ;;
    diffusion) step_diffusion "$@" ;;
    dlrm)      step_dlrm "$@" ;;
    cleanup)   step_cleanup ;;
    finalize)  step_finalize "$@" ;;
    *) echo "Unknown step: '$step'" >&2; exit 64 ;;
esac
