#!/usr/bin/env bash
# Where does the bf16 MegaMoE backward corruption live, on a node that reproduces it?
#
# The symptom: iteration-1 grad norm is ~1.88e6 against a reference of 1.449, while iteration-1
# loss matches the reference to six digits. Forward correct, backward wrong. It reproduces on
# n06-25 (driver 6.16.13) on every turbo commit tried, from 2026-08-18 to main, and does not
# reproduce on n01-29 (driver 6.14.14) on any of them -- so it is not a turbo regression and
# bisecting turbo cannot find it.
#
# 5 iterations per case: the signal is a factor of a million on the *first* iteration, so a longer
# run only adds wall time. A case that comes back clean here still needs 50 iterations before it is
# believed, because the reference trajectory is what proves training actually works.
#
# No rebuild between cases -- every case runs the same turbo build, which is the point: the only
# thing that varies is which code path the MoE takes.
#
# Note on flags that do NOT belong here: --use_turbo_deepep and --turbo_sync_free_moe_stage change
# the *baseline* MoE only. MegaMoE replaces the whole layer including dispatch and combine and
# ignores both, so toggling them against a MegaMoE arm compares a configuration with itself.
set -uo pipefail

NODE=${NODE:-smci355-ccs-aus-n06-25}
CONTAINER=${CONTAINER:-xiaoming-dev}
PRECISION=${PRECISION:-bf16}
ITERS=${TRAIN_ITERS:-5}
REPO=${REPO:-/perf_apps/xiaoming/Primus}
TURBO=${TURBO:-/perf_apps/xiaoming/MegaMoE}
OUT=${OUT:-$REPO/ab_2x2/mega_grad_probe/$(date +%m%d-%H%M%S)}

# "<name> <USE_MEGA_MOE> <env assignments or -> [extra args]"
#
# baseline first: it is the control. If the stock MoE is also corrupted then nothing below the
# MegaMoE layer can be trusted and the remaining cases say nothing about MegaMoE.
CASES=(
    "baseline          False -"
    "mega              True  -"
    "mega_curstream    True  PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1"
)
[ -n "${ONLY:-}" ] && IFS=';' read -r -a CASES <<<"$ONLY"

busy=$(ssh -o BatchMode=yes "$NODE" \
    "docker exec $CONTAINER bash -c 'ps -eo stat,args | grep \"[c]li/main.py\" | grep -cv \"^Z\"'" \
    2>/dev/null | tr -d '\r')
[ "${busy:-0}" -gt 0 ] && { echo "refusing: $NODE has $busy live training procs" >&2; exit 3; }
vram=$(ssh -o BatchMode=yes "$NODE" \
    "rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card/{print \$3}' | sort -rn | head -1" \
    2>/dev/null | tr -d '\r')
if [ -n "$vram" ] && [ "$vram" -gt $((4 * 1024 * 1024 * 1024)) ]; then
    echo "refusing: $NODE has $((vram / 1024 / 1024 / 1024)) GiB VRAM held on some GPU" >&2
    exit 3
fi

mkdir -p "$OUT"
{
    echo "node    : $NODE ($CONTAINER)"
    echo "driver  : $(ssh -o BatchMode=yes "$NODE" 'rocm-smi --showdriverversion 2>/dev/null | grep -oP "Driver version: \K\S+"' 2>&1)"
    echo "image   : $(ssh -o BatchMode=yes "$NODE" "docker inspect -f '{{.Config.Image}}' $CONTAINER" 2>&1)"
    echo "turbo   : $(git -C "$TURBO" rev-parse --short=8 HEAD) $(git -C "$TURBO" log -1 --pretty=%s | cut -c1-50)"
    echo "primus  : $(git -C "$REPO" rev-parse --short HEAD)"
    echo "arm     : $PRECISION, $ITERS iterations per case"
    echo "started : $(date -Is)"
} | tee "$OUT/launch.txt"

for spec in "${CASES[@]}"; do
    read -r name mega envs extra <<<"$spec"
    [ "$envs" = "-" ] && envs=""
    log="$OUT/$name.log"
    echo ""
    echo "########## $name (mega=$mega) $envs $extra  $(date -Is)"
    # shellcheck disable=SC2086
    ssh -o BatchMode=yes "$NODE" "docker exec \
        -e PRECISION=$PRECISION \
        -e USE_MEGA_MOE=$mega \
        -e TRAIN_ITERS=$ITERS \
        -e LOG=$log \
        -e ALL_RANKS=1 \
        -e EXTRA_ARGS='${extra:-}' \
        $(for e in $envs; do printf -- '-e %s ' "$e"; done) \
        $CONTAINER bash $REPO/run.sh" >"$log.outer" 2>&1
    echo "########## $name done rc=$? $(date -Is)"
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


print("## bf16 MegaMoE backward corruption probe\n")
print((out / "launch.txt").read_text().rstrip())
print("\nReference: iteration-1 grad norm 1.449, iteration-1 loss 12.01474.\n")
print("| case | it1 loss | it1 grad norm | verdict |")
print("|---|---|---|---|")
for log in sorted(out.glob("*.log")):
    text = ANSI.sub("", log.read_text(errors="replace"))
    rows = {}
    for i, loss, gn in ITER.findall(text):
        i, loss, gn = int(i), num(loss), num(gn)
        # all ranks log; keep the worst grad norm, the corruption does not hit every rank
        if i not in rows or not (gn <= rows[i][1]):
            rows[i] = (loss, gn)
    if not rows:
        why = "trace error" if "unexpected keyword" in text else "no iteration logged"
        print(f"| {log.stem} | — | — | **{why}** |")
        continue
    loss, gn = rows[min(rows)]
    ok = gn <= 100
    print(f"| {log.stem} | {loss:.5f} | {gn:,.3f} | {'clean' if ok else '**CORRUPTED**'} |")
PY
echo "=== out: $OUT"
