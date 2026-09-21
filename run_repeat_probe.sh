#!/usr/bin/env bash
# Is the bf16 MegaMoE gradient corruption intermittent, on one build?
#
# Every verdict in this investigation so far has been a single run, and those verdicts stopped
# adding up: on n02-21, turbo e5487abc came back clean four times in the morning and the same
# source (734eb17f plus a no-op config wrapper) came back corrupted four times in the afternoon.
# Same node, same container, equivalent code. Either something changed between them that is not in
# the source, or the failure is intermittent and every n=1 verdict here -- including the ones that
# exonerated turbo commits, mxfp8 and the stock MoE -- measured a coin flip.
#
# This runs one build repeatedly and reports the rate. No rebuild and no checkout between reps:
# the build, the container and the machine are held fixed so the only thing varying is the run.
#
# 5 iterations per rep: iteration-1 grad norm is 1.449 when clean and ~1.88e6 when not, so a rep
# is decided on its first logged iteration and the rest is wall time.
set -uo pipefail

NODE=${NODE:-smci355-ccs-aus-n02-21}
CONTAINER=${CONTAINER:-xiaoming-dev}
PRECISION=${PRECISION:-bf16}
ITERS=${TRAIN_ITERS:-5}
REPS=${REPS:-3}
MEGA=${USE_MEGA_MOE:-True}
REPO=${REPO:-/perf_apps/xiaoming/Primus}
TURBO=${TURBO:-/perf_apps/xiaoming/MegaMoE}
OUT=${OUT:-$REPO/ab_2x2/repeat_probe/$(date +%m%d-%H%M%S)}

busy=$(ssh -o BatchMode=yes "$NODE" \
    "docker exec $CONTAINER bash -c 'ps -eo stat,args | grep \"[c]li/main.py\" | grep -cv \"^Z\"'" \
    2>/dev/null | tr -d '\r')
[ "${busy:-0}" -gt 0 ] && { echo "refusing: $NODE has $busy live training procs" >&2; exit 3; }

mkdir -p "$OUT"
{
    echo "node    : $NODE ($CONTAINER)"
    echo "driver  : $(ssh -o BatchMode=yes "$NODE" 'rocm-smi --showdriverversion 2>/dev/null | grep -oP "Driver version: \K\S+"' 2>&1)"
    echo "turbo   : $(git -C "$TURBO" rev-parse --short=8 HEAD) $(git -C "$TURBO" log -1 --pretty=%s | cut -c1-44)"
    echo "primus  : $(git -C "$REPO" rev-parse --short HEAD)"
    echo "arm     : $PRECISION, use_turbo_mega_moe=$MEGA, $ITERS iterations x $REPS reps"
    echo "note    : no rebuild and no checkout between reps"
    echo "started : $(date -Is)"
} | tee "$OUT/launch.txt"

for rep in $(seq 1 "$REPS"); do
    log="$OUT/rep$rep.log"
    echo ""
    echo "########## rep$rep  $(date -Is)"
    ssh -o BatchMode=yes "$NODE" "docker exec \
        -e PRECISION=$PRECISION \
        -e USE_MEGA_MOE=$MEGA \
        -e TRAIN_ITERS=$ITERS \
        -e LOG=$log \
        -e ALL_RANKS=1 \
        $CONTAINER bash $REPO/run.sh" >"$log.outer" 2>&1
    echo "########## rep$rep done rc=$? $(date -Is)"
done

echo ""
python3 - "$OUT" <<'PY'
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


print("## Same build, repeated: is the corruption intermittent?\n")
print((out / "launch.txt").read_text().rstrip())
print("\nClean is iteration-1 grad norm ~1.4; corrupted is ~1.88e6.\n")
print("| rep | it1 loss | it1 grad norm | verdict |")
print("|---|---|---|---|")
clean = bad = 0
for log in sorted(out.glob("rep*.log")):
    text = ANSI.sub("", log.read_text(errors="replace"))
    rows = {}
    for i, loss, gn in ITER.findall(text):
        i, loss, gn = int(i), num(loss), num(gn)
        if i not in rows or not (gn <= rows[i][1]):
            rows[i] = (loss, gn)
    if not rows:
        print(f"| {log.stem} | — | — | **no iteration logged** |")
        continue
    loss, gn = rows[min(rows)]
    ok = gn <= 100
    clean, bad = clean + ok, bad + (not ok)
    print(f"| {log.stem} | {loss:.5f} | {gn:,.3f} | {'clean' if ok else '**CORRUPTED**'} |")
print(f"\n{clean} clean / {bad} corrupted out of {clean + bad}.")
if clean and bad:
    print("\nIntermittent on one build: every single-run verdict in this investigation is a sample, "
          "not a measurement.")
PY
echo "=== out: $OUT"
