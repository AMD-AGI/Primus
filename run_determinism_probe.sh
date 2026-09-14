#!/usr/bin/env bash
# Is the stock-MoE loss jitter nondeterminism, or code drift between runs?
#
# The three bf16.baseline runs so far each sat on a different Primus/MegaMoE commit, so their spread
# (loss@50 of 5.68 / 5.21 / 5.99) cannot tell the two apart. Two reps of the same binary can: if they
# diverge, the path is nondeterministic, and no amount of commit archaeology was ever going to
# explain it.
#
# 10 iterations is enough. The divergence shows at iteration 3 and reaches 0.1 by iteration 5, so a
# 50-iteration run would only add wall time to a question already answered.
#
# The cases walk down the stock MoE stack, which is the only thing that differs from the MegaMoE arm
# (deterministic across all three runs). Sync-free stage 1 force-enables use_turbo_deepep, so
# turning DeepEP off means turning sync-free off first -- hence the middle case, which separates the
# two instead of changing both at once.
set -uo pipefail

NODE=${NODE:-smci355-ccs-aus-n04-33}
CONTAINER=${CONTAINER:-xiaoming-dev}
ITERS=${TRAIN_ITERS:-10}
REPS=${REPS:-2}
PRECISION=${PRECISION:-bf16}
REPO=${REPO:-/home/xiaompen/Primus}
MEGAMOE=${MEGAMOE:-/home/xiaompen/MegaMoE}
OUT=${OUT:-$REPO/ab_2x2/$(date +%m%d-%H%M%S)-determinism}

# "<name> [extra args]"
CASES=(
    "asis"
    "nosyncfree --turbo_sync_free_moe_stage 0"
    "nodeepep --turbo_sync_free_moe_stage 0 --use_turbo_deepep False"
)

mkdir -p "$OUT"
{
    echo "node      : $NODE ($CONTAINER)"
    echo "precision : $PRECISION, baseline (stock) MoE"
    echo "iters/rep : $ITERS, reps: $REPS"
    echo "primus git: $(git -C $REPO rev-parse --short HEAD) (working tree, wgrad fix applied)"
    echo "megamoe   : $(git -C "$MEGAMOE" rev-parse --short HEAD)"
    echo "started   : $(date -Is)"
} | tee "$OUT/launch.txt"

for spec in "${CASES[@]}"; do
    read -r name extra <<<"$spec"
    for rep in $(seq 1 "$REPS"); do
        log="$OUT/$name.rep$rep.log"
        echo ""
        echo "########## $name rep$rep  $extra  $(date -Is)"
        ssh -o BatchMode=yes "$NODE" "docker exec \
            -e PRECISION=$PRECISION \
            -e USE_MEGA_MOE=False \
            -e TRAIN_ITERS=$ITERS \
            -e LOG=$log \
            -e EXTRA_ARGS='$extra' \
            $CONTAINER bash $REPO/run.sh" >"$log.outer" 2>&1
        echo "########## $name rep$rep done rc=$? $(date -Is)"
    done
done

echo ""
python3 - "$OUT" <<'PY'
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOSS = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?lm loss:\s*([\d.E+-]+)")
out = pathlib.Path(sys.argv[1])


def traj(p):
    return {int(i): float(v) for i, v in LOSS.findall(ANSI.sub("", p.read_text(errors="replace")))}


print("## Same binary, two reps: does the stock MoE reproduce?\n")
for name in ("asis", "nosyncfree", "nodeepep"):
    reps = [traj(p) for p in sorted(out.glob(f"{name}.rep*.log")) if p.stat().st_size]
    reps = [r for r in reps if r]
    if len(reps) < 2:
        print(f"### {name}: {len(reps)} usable rep(s), cannot compare\n")
        continue
    common = sorted(set(reps[0]) & set(reps[1]))
    diffs = [abs(reps[0][i] - reps[1][i]) for i in common]
    first = next((i for i, d in zip(common, diffs) if d > 0), None)
    print(f"### {name}")
    print("| iteration | " + " | ".join(str(i) for i in common) + " |")
    print("|---" * (len(common) + 1) + "|")
    print("| rep1 | " + " | ".join(f"{reps[0][i]:.5f}" for i in common) + " |")
    print("| rep2 | " + " | ".join(f"{reps[1][i]:.5f}" for i in common) + " |")
    print("| |diff| | " + " | ".join(f"{d:.2e}" for d in diffs) + " |")
    verdict = "REPRODUCIBLE (bitwise on every iteration)" if first is None else f"NONDETERMINISTIC from iteration {first}"
    print(f"\n{verdict}, max |diff| {max(diffs):.2e}\n")
PY
echo "=== out: $OUT"
