#!/usr/bin/env bash
# Is the bf16 MegaMoE backward corruption a bad autotune config?
#
# The bf16 dispatch grouped GEMM autotunes num_dispatch_cu over (16, 32, 64) -- how many blocks of
# the persistent kernel take the comm role instead of a GEMM tile. The autotuner picks by measured
# latency and caches the winner per container in ~/.flydsl/autotune, so *which* candidate runs is a
# property of the machine. That matches the symptom better than anything else looked at so far:
#
#   - iteration-1 grad norm lands on 1,881,300 / 1,881,455 / 1,881,473 across runs -- within 0.01%,
#     which is a deterministic wrong answer, not a race
#   - the same turbo commit is wrong on one node and right on another
#   - mxfp8 is unaffected: separate kernel, separate config space
#   - adding the L2-invalidate acquire the mxfp8 path has changed nothing (1,881,473 after)
#
# Each case pins one candidate via MEGA_BF16_DISPATCH_CU and clears the autotune cache first --
# without the clear a cached key short-circuits the configs list and the pin is ignored. The
# unpinned case is the control: it reports what the autotuner picks on this machine unaided.
#
# 5 iterations: the signal is a factor of a million on iteration 1. A case that comes back clean
# still needs 50 iterations before it is called good.
set -uo pipefail

NODE=${NODE:-smci355-ccs-aus-n02-21}
CONTAINER=${CONTAINER:-xiaoming-dev}
ITERS=${TRAIN_ITERS:-5}
REPO=${REPO:-/perf_apps/xiaoming/Primus}
TURBO=${TURBO:-/perf_apps/xiaoming/MegaMoE}
OUT=${OUT:-$REPO/ab_2x2/dispatch_cu_probe/$(date +%m%d-%H%M%S)}
CASES=${CASES:-"16 32 64 auto"}

busy=$(ssh -o BatchMode=yes "$NODE" \
    "docker exec $CONTAINER bash -c 'ps -eo stat,args | grep \"[c]li/main.py\" | grep -cv \"^Z\"'" \
    2>/dev/null | tr -d '\r')
[ "${busy:-0}" -gt 0 ] && { echo "refusing: $NODE has $busy live training procs" >&2; exit 3; }

mkdir -p "$OUT"
{
    echo "node    : $NODE ($CONTAINER)"
    echo "driver  : $(ssh -o BatchMode=yes "$NODE" 'rocm-smi --showdriverversion 2>/dev/null | grep -oP "Driver version: \K\S+"' 2>&1)"
    echo "turbo   : $(git -C "$TURBO" rev-parse --short=8 HEAD) $(git -C "$TURBO" log -1 --pretty=%s | cut -c1-46)"
    echo "primus  : $(git -C "$REPO" rev-parse --short HEAD)"
    echo "arm     : bf16 MegaMoE (fused), $ITERS iterations per case"
    echo "cases   : $CASES (num_dispatch_cu; 'auto' = let the autotuner choose)"
    echo "started : $(date -Is)"
} | tee "$OUT/launch.txt"

for cu in $CASES; do
    log="$OUT/cu$cu.log"
    pin=""
    [ "$cu" != auto ] && pin="-e MEGA_BF16_DISPATCH_CU=$cu"
    echo ""
    echo "########## num_dispatch_cu=$cu  $(date -Is)"
    # The cache is cleared per case, not once: a pinned run writes its own winner back, and the
    # next case would then read that key instead of tuning or honouring its own pin.
    ssh -o BatchMode=yes "$NODE" "docker exec $CONTAINER rm -rf /root/.flydsl/autotune" >/dev/null 2>&1
    # shellcheck disable=SC2086
    ssh -o BatchMode=yes "$NODE" "docker exec \
        -e PRECISION=bf16 \
        -e USE_MEGA_MOE=True \
        -e TRAIN_ITERS=$ITERS \
        -e LOG=$log \
        -e ALL_RANKS=1 \
        $pin \
        $CONTAINER bash $REPO/run.sh" >"$log.outer" 2>&1
    echo "########## num_dispatch_cu=$cu done rc=$? $(date -Is)"
    # What the autotuner actually settled on, so the 'auto' case is interpretable and a pinned
    # case can be shown to have taken effect.
    ssh -o BatchMode=yes "$NODE" \
        "docker exec $CONTAINER cat /root/.flydsl/autotune/_compiled_dispatch_grouped_gemm.json" \
        >"$OUT/cu$cu.chosen.json" 2>/dev/null
done

echo ""
python3 - "$OUT" <<'PY'
import json
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?lm loss:\s*([\d.E+naN-]+).*?grad norm:\s*([\d.naN-]+)")
out = pathlib.Path(sys.argv[1])


def num(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


print("## bf16 MegaMoE: is the corruption a num_dispatch_cu config?\n")
print((out / "launch.txt").read_text().rstrip())
print("\nReference: iteration-1 grad norm 1.449, loss 12.01474. Corrupted runs report ~1.88e6.\n")
print("| num_dispatch_cu | it1 loss | it1 grad norm | chosen by autotuner | verdict |")
print("|---|---|---|---|---|")
for log in sorted(out.glob("cu*.log"), key=lambda p: (p.stem != "cuauto", p.stem)):
    case = log.stem[2:]
    text = ANSI.sub("", log.read_text(errors="replace"))
    rows = {}
    for i, loss, gn in ITER.findall(text):
        i, loss, gn = int(i), num(loss), num(gn)
        if i not in rows or not (gn <= rows[i][1]):
            rows[i] = (loss, gn)
    chosen = "-"
    cj = out / f"cu{case}.chosen.json"
    if cj.exists() and cj.stat().st_size:
        try:
            vals = {c.get("num_dispatch_cu") for c in json.loads(cj.read_text()).values()}
            chosen = ",".join(str(v) for v in sorted(v for v in vals if v is not None))
        except Exception:  # noqa: BLE001
            chosen = "unparsed"
    if not rows:
        print(f"| {case} | — | — | {chosen} | **no iteration logged** |")
        continue
    loss, gn = rows[min(rows)]
    print(f"| {case} | {loss:.5f} | {gn:,.3f} | {chosen} | {'clean' if gn <= 100 else '**CORRUPTED**'} |")
PY
echo "=== out: $OUT"
