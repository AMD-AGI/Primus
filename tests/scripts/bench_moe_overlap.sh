#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Benchmark: MoE Expert↔Comm Overlap vs Baseline
#
# Uses DeepSeek-V2-Lite (16B, 64 experts, EP=8) on 8x MI300X.
# Compares:
#   1) Baseline: turbo_moe_expert_comm_overlap = false
#   2) Overlap:  turbo_moe_expert_comm_overlap = true (num_chunks=2)
#
# Usage:
#   bash tests/scripts/bench_moe_overlap.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PRIMUS_ROOT"

CONFIG="examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml"
ITERS=20
LOG_DIR="logs/bench_moe_overlap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " MoE Expert↔Comm Overlap Benchmark"
echo " Model: DeepSeek-V2-Lite (16B, 64 experts)"
echo " Config: $CONFIG"
echo " Iterations: $ITERS"
echo " Log dir: $LOG_DIR"
echo "=============================================="

# --- Run 1: Baseline (no overlap) ---
echo ""
echo "[1/2] Running BASELINE (no overlap)..."
MASTER_PORT=29501 \
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29501 \
    -m primus.cli.main train pretrain \
    --config "$CONFIG" \
    train_iters=$ITERS \
    turbo_moe_expert_comm_overlap=false \
    log_interval=1 \
    2>&1 | tee "${LOG_DIR}/baseline.log"

echo "[1/2] Baseline complete."

# --- Run 2: Overlap (num_chunks=2) ---
echo ""
echo "[2/2] Running OVERLAP (num_chunks=2)..."
MASTER_PORT=29502 \
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29502 \
    -m primus.cli.main train pretrain \
    --config "$CONFIG" \
    train_iters=$ITERS \
    turbo_moe_expert_comm_overlap=true \
    turbo_moe_overlap_num_chunks=2 \
    log_interval=1 \
    2>&1 | tee "${LOG_DIR}/overlap.log"

echo "[2/2] Overlap complete."

# --- Compare results ---
echo ""
echo "=============================================="
echo " Results Summary"
echo "=============================================="

extract_throughput() {
    local log_file="$1"
    local label="$2"
    # Look for "elapsed time per iteration" lines, skip first few warmup iters
    local avg_ms
    avg_ms=$(grep -oP 'elapsed time per iteration \(ms\): \K[0-9.]+' "$log_file" \
        | tail -n +4 \
        | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    echo "$label: avg ${avg_ms} ms/iter"
}

extract_throughput "${LOG_DIR}/baseline.log" "Baseline"
extract_throughput "${LOG_DIR}/overlap.log"  "Overlap "

echo ""
echo "Full logs in: $LOG_DIR"
echo "=============================================="
