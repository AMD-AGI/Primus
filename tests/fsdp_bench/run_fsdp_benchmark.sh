#!/usr/bin/env bash
###############################################################################
# FSDP2 Benchmark: LLaMA 3.1 8B
# Compares standard PyTorch FSDP2 vs veScale RaggedShard FSDP
#
# Hardware: 8x AMD MI355X @ 288 GB VRAM each
# Config:   DP=8, TP=1, PP=1 (pure data-parallel, maximizes FSDP communication)
#
# Usage (inside container):
#   cd /home/xiaompen/Primus-dev
#   bash tests/fsdp_bench/run_fsdp_benchmark.sh
#
# Environment:
#   GPUS_PER_NODE=8     (default)
#   PRIMUS_WARMUP=3     warm-up iterations to skip (default: 3)
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(realpath "$SCRIPT_DIR/../..")"
EXP_CONFIG="$SCRIPT_DIR/exp_fsdp_bench.yaml"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
WARMUP="${PRIMUS_WARMUP:-3}"
LOG_DIR="$PRIMUS_ROOT/output/fsdp_bench"

mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[bench]${NC} $*"; }
warn() { echo -e "${YELLOW}[bench]${NC} $*"; }
ok()   { echo -e "${GREEN}[bench]${NC} $*"; }
sep()  { echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── verify veScale is importable ──────────────────────────────────────────────
check_vescale() {
    python3 -c "from vescale.dtensor.placement_types import RaggedShard; print('veScale OK')" \
        2>/dev/null && return 0
    warn "veScale not found — installing from third_party/veScale/ ..."
    pip install --ignore-requires-python -q \
        -e "$PRIMUS_ROOT/third_party/veScale/" && ok "veScale installed"
}
check_vescale

# ── launch helper ─────────────────────────────────────────────────────────────
# Directly uses torchrun (same as run_pretrain.sh) but from the dev path.
run_training() {
    local tag="$1"
    local log_file="$2"
    shift 2
    # extra env vars forwarded as "KEY=VAL ..." pairs
    local extra_env=("$@")

    local site_packages
    site_packages=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    export PYTHONPATH="${PRIMUS_ROOT}:${site_packages}:${PYTHONPATH:-}"

    local cmd=(
        torchrun
        --standalone
        --nnodes 1
        --nproc_per_node "$GPUS_PER_NODE"
        "$PRIMUS_ROOT/primus/cli/main.py"
        train pretrain
        --config "$EXP_CONFIG"
    )

    log "Starting [$tag] → $log_file"
    log "Command: ${cmd[*]}"

    # Build environment
    local env_export=""
    for kv in "${extra_env[@]}"; do
        env_export="$kv $env_export"
    done

    eval "env $env_export ${cmd[*]}" 2>&1 | tee "$log_file"
    return "${PIPESTATUS[0]}"
}

# ── metric parser ─────────────────────────────────────────────────────────────
# Parse Megatron log lines of the form:
#   iteration   10/   15 | ... elapsed time per iteration (ms): 1234.5 | ...
#   throughput per GPU (TFLOP/s): 45.6
parse_metrics() {
    local log_file="$1"
    local skip="$2"   # number of initial iterations to skip (warmup)

    # elapsed time per iteration (ms) — skip first $skip measurement lines
    local times
    times=$(grep -oP 'elapsed time per iteration \(ms\): \K[\d.]+' "$log_file" \
            | tail -n +$((skip + 1)))

    if [ -z "$times" ]; then
        echo "N/A N/A N/A N/A"
        return
    fi

    local count mean min max
    count=$(echo "$times" | wc -l)
    mean=$(echo "$times" | awk '{s+=$1}END{printf "%.1f", s/NR}')
    min=$(echo  "$times" | awk 'NR==1||$1<m{m=$1}END{printf "%.1f", m}')
    max=$(echo  "$times" | awk 'NR==1||$1>m{m=$1}END{printf "%.1f", m}')

    echo "$count $mean $min $max"
}

parse_tflops() {
    local log_file="$1"
    local skip="$2"
    grep -oP 'throughput per GPU \(TFLOP/s\): \K[\d.]+' "$log_file" \
        | tail -n +$((skip + 1)) \
        | awk '{s+=$1}END{if(NR>0)printf "%.2f", s/NR; else print "N/A"}'
}

parse_peak_memory() {
    # Look for the memory report that Megatron emits at the end:
    # "[Rank 0] after training is done: max allocated: X MB"
    local log_file="$1"
    grep -oP '(?<=max allocated: )[\d.]+(?= MB)' "$log_file" \
        | awk '{if($1>max||NR==1)max=$1}END{printf "%.0f", max}' \
        || echo "N/A"
}

# ─────────────────────────────────────────────────────────────────────────────
# RUN 1: Baseline — standard PyTorch FSDP2
# ─────────────────────────────────────────────────────────────────────────────
sep
log "${BOLD}RUN 1: Standard PyTorch FSDP2 (baseline)${NC}"
sep

LOG_BASELINE="$LOG_DIR/baseline_fsdp2.log"
if run_training "baseline-fsdp2" "$LOG_BASELINE" \
    "PRIMUS_VESCALE_RAGGED_SHARD_FSDP=0"; then
    ok "Baseline run finished."
else
    warn "Baseline run exited with non-zero (may still have valid metrics)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# RUN 2: veScale RaggedShard FSDP
# ─────────────────────────────────────────────────────────────────────────────
sep
log "${BOLD}RUN 2: veScale RaggedShard FSDP${NC}"
sep

LOG_VESCALE="$LOG_DIR/vescale_ragged_fsdp.log"
if run_training "vescale-ragged-fsdp" "$LOG_VESCALE" \
    "PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1"; then
    ok "veScale RaggedShard run finished."
else
    warn "veScale RaggedShard run exited with non-zero (may still have valid metrics)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
sep
echo -e "${BOLD}  FSDP2 Performance Comparison — LLaMA 3.1 8B  (8x MI355X)${NC}"
sep

read -r _cnt_b mean_b min_b max_b <<< "$(parse_metrics "$LOG_BASELINE"  "$WARMUP")"
read -r _cnt_v mean_v min_v max_v <<< "$(parse_metrics "$LOG_VESCALE"   "$WARMUP")"

tflops_b=$(parse_tflops    "$LOG_BASELINE" "$WARMUP")
tflops_v=$(parse_tflops    "$LOG_VESCALE"  "$WARMUP")

mem_b=$(parse_peak_memory  "$LOG_BASELINE")
mem_v=$(parse_peak_memory  "$LOG_VESCALE")

# Compute speedup
speedup="N/A"
if [[ "$mean_b" != "N/A" && "$mean_v" != "N/A" ]]; then
    speedup=$(awk "BEGIN{printf \"%.3f\", $mean_b / $mean_v}")
fi

printf "\n"
printf "  %-36s  %10s  %10s\n" "Metric" "Baseline" "veScale RaggedShard"
printf "  %-36s  %10s  %10s\n" "------" "--------" "-------------------"
printf "  %-36s  %9sms  %9sms\n"  "Iter time (mean, excl. warmup)"   "$mean_b"   "$mean_v"
printf "  %-36s  %9sms  %9sms\n"  "Iter time (min)"                  "$min_b"    "$min_v"
printf "  %-36s  %9sms  %9sms\n"  "Iter time (max)"                  "$max_b"    "$max_v"
printf "  %-36s  %9s    %9s\n"    "Throughput (TFLOP/s/GPU)"         "$tflops_b" "$tflops_v"
printf "  %-36s  %9s MB %9s MB\n" "Peak GPU memory (rank-0)"         "$mem_b"    "$mem_v"
printf "  %-36s  %20s\n"          "Speedup (baseline / veScale)"     "${speedup}x"
printf "\n"

sep
log "Raw logs:"
log "  Baseline : $LOG_BASELINE"
log "  veScale  : $LOG_VESCALE"
sep

# Emit speedup verdict
if [[ "$speedup" != "N/A" ]]; then
    result=$(awk "BEGIN{v=$speedup; if(v>1.02)print \"faster\"; else if(v<0.98)print \"slower\"; else print \"neutral\"}")
    case "$result" in
        faster)  ok  "veScale RaggedShard is ${BOLD}${speedup}x faster${NC} than baseline FSDP2." ;;
        slower)  warn "veScale RaggedShard is ${speedup}x (slower than baseline — investigate)." ;;
        neutral) log  "Performance is comparable (speedup=${speedup}x, within ±2%)." ;;
    esac
fi
