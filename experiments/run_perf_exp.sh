#!/bin/bash
###############################################################################
# Run ONE 50-iter perf experiment on the Pure-GDN 1B model.
#
# Usage:
#   bash experiments/run_perf_exp.sh <exp_id> <yaml_path> [ENV_VAR=val ...]
#
# Examples:
#   # Re-verify the EXP7 winning config (50 iters, ~4 min)
#   bash experiments/run_perf_exp.sh exp7_repro \
#       examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_exp7-fsdp-overlap.yaml
#
#   # Test a new env-var toggle without editing the YAML
#   bash experiments/run_perf_exp.sh my_test \
#       examples/megatron/configs/MI300X/zebra_llama_1B_gdn_pure_exp7-fsdp-overlap.yaml \
#       PRIMUS_FUSED_CE_CHUNKS=16 MY_OTHER_FLAG=1
#
# Produces:
#   primus_perf_<exp_id>.log              (full Megatron log)
#   primus_perf_<exp_id>.launcher.out     (launcher stdout/stderr)
#   primus_perf_<exp_id>.summary.txt      (one-line steady-state iter time + loss)
###############################################################################
set -euo pipefail

EXP_ID=${1:?missing exp_id}
EXP_YAML=${2:?missing yaml path}
shift 2

LOG="primus_perf_${EXP_ID}.log"
SUMMARY="primus_perf_${EXP_ID}.summary.txt"
LAUNCHER_OUT="primus_perf_${EXP_ID}.launcher.out"

export EXP="${EXP_YAML}"
export LOG="${LOG}"

# Any additional env-var=value pairs passed as args are exported here
for kv in "$@"; do
    export "${kv?}"
done

# Auto-detect FSDP YAML and apply NCCL channel clamp.
# At 99% VRAM, Megatron-FSDP's extra comm groups need their workspace
# (NCCL_BUFFSIZE × #channels) kept small. See README_PERF_GDN_1B.md.
if grep -qE "^[[:space:]]+use_megatron_fsdp:[[:space:]]+true" "${EXP_YAML}"; then
    export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-1}
    export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
    export NCCL_NCHANNELS_PER_PEER=${NCCL_NCHANNELS_PER_PEER:-1}
    echo "[exp ${EXP_ID}] FSDP detected → applied NCCL channel clamp"
fi

echo "════════════════════════════════════════════════════════════════════════"
echo "  PERF EXPERIMENT: ${EXP_ID}"
echo "    yaml: ${EXP_YAML}"
echo "    log:  ${LOG}"
echo "    extra env: $* "
echo "════════════════════════════════════════════════════════════════════════"

# Quick sanity check
[ -f "${EXP_YAML}" ] || { echo "FAIL: yaml missing: ${EXP_YAML}"; exit 1; }

# Run synchronously under the production launcher so we keep the same
# ROCm allocator knobs + NCCL_BUFFSIZE + FLA_* env vars.
# Skip preflight checks (we already validated them on prod).
export SKIP_DISK_CHECK=1 SKIP_GPU_CHECK=1 SKIP_TOKENIZER_CHECK=1 SKIP_PATCH_CHECK=1
bash launch_gdn_pure_1B_100B.sh > "${LAUNCHER_OUT}" 2>&1 &
LAUNCHER_PID=$!
echo "[exp ${EXP_ID}] launcher pid=${LAUNCHER_PID}"
wait "${LAUNCHER_PID}"
EXIT_CODE=$?
echo "[exp ${EXP_ID}] launcher exit=${EXIT_CODE}"

# Extract a summary
python3 - <<PY > "${SUMMARY}"
import re, statistics, sys
log = "${LOG}"
try:
    txt = open(log, errors="replace").read()
except FileNotFoundError:
    print(f"FAIL: no log file {log}")
    sys.exit(0)

iter_re = re.compile(
    r"iteration\s+(\d+)/\s*\d+\s+\|\s+consumed samples:\s+\d+\s+\|\s+elapsed time per iteration \(ms\):\s+([\d.]+)(?:/([\d.]+))?\s+\|\s+throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)(?:/([\d.]+))?\s+\|.*?lm loss:\s+([\d.E+-]+)"
)
seen = {}
for m in iter_re.finditer(txt):
    it = int(m.group(1))
    seen[it] = (float(m.group(2)), float(m.group(4)), float(m.group(6)))   # cur_ms, cur_tflops, loss

print("exp_id=${EXP_ID}")
print("exit_code=${EXIT_CODE}")
print("yaml=${EXP_YAML}")
print(f"iters_completed={max(seen) if seen else 0}")
print()
if not seen:
    print("FAIL: no iters logged")
    sys.exit(0)

# Steady-state windows: iters 21..50 (skip warmup).
# Flush vs normal is determined by elapsed time, not iter index, since the
# logged ms is a window-average that may straddle a flush boundary.
all_ms = []
loss_first = None
loss_last  = None
for it in sorted(seen):
    ms, tflops, loss = seen[it]
    if it == min(seen): loss_first = loss
    loss_last = loss
    if it < 21: continue
    all_ms.append((it, ms))

if all_ms:
    # Split at the median: anything significantly above median is a flush-touching window
    sorted_ms = sorted(m for _, m in all_ms)
    median = sorted_ms[len(sorted_ms)//2]
    threshold = median * 1.10  # >10% slower than median = a flush window
    normal = [m for _, m in all_ms if m < threshold]
    flush  = [m for _, m in all_ms if m >= threshold]
    print(f"  classifier: median={median:.1f} threshold={threshold:.1f}")
    print(f"  all 30+ iters logged: {[(it, round(m,0)) for it,m in all_ms]}")
    print()
    if normal:
        nmean = statistics.mean(normal)
        print(f"  normal iters  n={len(normal)} mean={nmean:.1f} ms  min={min(normal):.1f}  max={max(normal):.1f}")
    if flush:
        fmean = statistics.mean(flush)
        print(f"  flush iters   n={len(flush)} mean={fmean:.1f} ms  min={min(flush):.1f}  max={max(flush):.1f}")
    print()
    if normal:
        print(f"steady_state_ms={statistics.mean(normal):.1f}")
        print(f"steady_state_vs_fla_pct={(statistics.mean(normal)/2310 - 1)*100:+.1f}%   (FLA=2310 ms)")
print(f"loss_first={loss_first}")
print(f"loss_last={loss_last}")
PY

echo
echo "─── SUMMARY for ${EXP_ID} ─────────────────────────────────────────────"
cat "${SUMMARY}"
echo "───────────────────────────────────────────────────────────────────────"
