#!/usr/bin/env bash
###############################################################################
# End-to-End FSDP2 Benchmark: LLaMA 3.1 8B (full 32 layers)
# Compares standard PyTorch FSDP2 vs veScale RaggedShard FSDP
#
# Usage (inside container):
#   cd /home/xiaompen/Primus-dev
#   bash tests/fsdp_bench/run_e2e_benchmark.sh
###############################################################################
set -euo pipefail

PRIMUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOKENIZER="$PRIMUS_ROOT/data/huggingface/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
CONFIG="$PRIMUS_ROOT/examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml"
LOG_DIR="$PRIMUS_ROOT/output/e2e_bench"
WARMUP_ITERS=3
TRAIN_ITERS=15   # 3 warmup + 12 measured

mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[e2e-bench]${NC} $*"; }
ok()   { echo -e "${GREEN}[e2e-bench]${NC} $*"; }
sep()  { echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── run training and capture log ──────────────────────────────────────────────
run_training() {
    local tag="$1"; local log="$2"; shift 2
    local extra_args=("$@")
    log "Starting [$tag] → $log"
    cd "$PRIMUS_ROOT"
    bash runner/primus-cli-direct.sh -- train pretrain \
        --config "$CONFIG" \
        --num_layers 32 \
        --train_iters "$TRAIN_ITERS" \
        --micro_batch_size 2 \
        --global_batch_size 16 \
        --overlap_grad_reduce false \
        --overlap_param_gather false \
        --tokenizer_model "$TOKENIZER" \
        "${extra_args[@]}" \
        > "$log" 2>&1
}

# ── parse metrics from log ────────────────────────────────────────────────────
parse_avg_elapsed() {
    # Extract the "average" part from "elapsed time per iteration (ms): X/Y"
    # Y is the rolling avg. Skip first WARMUP_ITERS lines.
    grep 'elapsed time per iteration' "$1" | sed 's/\x1B\[[0-9;]*m//g' | \
        grep -oP 'elapsed time per iteration \(ms\): [\d.]+/\K[\d.]+' | \
        tail -n +$((WARMUP_ITERS + 1)) | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n}'
}

parse_avg_tflops() {
    grep 'throughput per GPU' "$1" | sed 's/\x1B\[[0-9;]*m//g' | \
        grep -oP 'throughput per GPU \(TFLOP/s/GPU\): [\d.]+/\K[\d.]+' | \
        tail -n +$((WARMUP_ITERS + 1)) | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n}'
}

parse_avg_tokens() {
    grep 'tokens per GPU' "$1" | sed 's/\x1B\[[0-9;]*m//g' | \
        grep -oP 'tokens per GPU \(tokens/s/GPU\): [\d.]+/\K[\d.]+' | \
        tail -n +$((WARMUP_ITERS + 1)) | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n}'
}

parse_peak_mem() {
    # hip mem usage in GB
    grep 'hip mem usage' "$1" | sed 's/\x1B\[[0-9;]*m//g' | tail -1 | \
        grep -oP 'hip mem usage/free/total/usage_ratio: \K[\d.]+(?=GB)'
}

# ── RUN 1: Baseline FSDP2 ─────────────────────────────────────────────────────
sep
log "${BOLD}RUN 1: Standard PyTorch FSDP2 (baseline)${NC}"
sep
LOG_BASELINE="$LOG_DIR/baseline_fsdp2.log"
if PRIMUS_VESCALE_RAGGED_SHARD_FSDP=0 run_training "baseline-fsdp2" "$LOG_BASELINE" \
    --use_torch_fsdp2 true; then
    ok "Baseline FSDP2 run finished."
else
    echo "WARNING: Baseline exited non-zero (parsing metrics anyway)."
fi

# ── RUN 2: veScale RaggedShard FSDP ───────────────────────────────────────────
sep
log "${BOLD}RUN 2: veScale RaggedShard FSDP${NC}"
sep
LOG_VESCALE="$LOG_DIR/vescale_ragged_fsdp.log"
if PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1 run_training "vescale-ragged-fsdp" "$LOG_VESCALE" \
    --use_torch_fsdp2 true; then
    ok "veScale RaggedShard run finished."
else
    echo "WARNING: veScale exited non-zero (parsing metrics anyway)."
fi

# ── RESULTS ───────────────────────────────────────────────────────────────────
sep
echo -e "${BOLD}  End-to-End Results — LLaMA 3.1 8B (32 layers, 8x MI355X)${NC}"
echo -e "  FSDP2 on/off: on  |  Overlap: off  |  DP=8 TP=1 PP=1  |  BS=16"
sep

B_ELAPSED=$(parse_avg_elapsed "$LOG_BASELINE")
V_ELAPSED=$(parse_avg_elapsed "$LOG_VESCALE")
B_TFLOPS=$(parse_avg_tflops "$LOG_BASELINE")
V_TFLOPS=$(parse_avg_tflops "$LOG_VESCALE")
B_TOKENS=$(parse_avg_tokens "$LOG_BASELINE")
V_TOKENS=$(parse_avg_tokens "$LOG_VESCALE")
B_MEM=$(parse_peak_mem "$LOG_BASELINE")
V_MEM=$(parse_peak_mem "$LOG_VESCALE")

speedup() { awk "BEGIN{if($2>0)printf \"%.3f\", $1/$2; else print \"N/A\"}"; }

SPEEDUP_T=$(speedup "$B_ELAPSED" "$V_ELAPSED")
SPEEDUP_F=$(speedup "$V_TFLOPS" "$B_TFLOPS")

printf "\n"
printf "  %-38s  %12s  %14s\n" "Metric" "Baseline" "veScale RaggedShard"
printf "  %-38s  %12s  %14s\n" "------" "--------" "-------------------"
printf "  %-38s  %10sms  %12sms\n" "Iter time avg (ms, excl. warmup)" "$B_ELAPSED" "$V_ELAPSED"
printf "  %-38s  %10s    %12s\n"   "Throughput (TFLOP/s/GPU)"         "$B_TFLOPS"  "$V_TFLOPS"
printf "  %-38s  %10s    %12s\n"   "Tokens/s/GPU"                     "$B_TOKENS"  "$V_TOKENS"
printf "  %-38s  %9sGB  %11sGB\n"  "Peak GPU mem (rank-7)"            "$B_MEM"     "$V_MEM"
printf "  %-38s  %26s\n"           "Speedup (iter time)"              "${SPEEDUP_T}x"
printf "  %-38s  %26s\n"           "Throughput gain"                  "${SPEEDUP_F}x"
printf "\n"
sep
log "Logs: $LOG_BASELINE"
log "      $LOG_VESCALE"
sep
