#!/usr/bin/env bash
# Bisect a MegaMoE (Primus-Turbo) regression over turbo commits, one commit per invocation.
#
# The arm is the *fused MegaMoE* one, not a baseline: the failure being chased is inside the fused
# dispatch GEMM, which the stock MoE path never enters.
#
# bf16 by default rather than mxfp8, because #483 (220ead50, 2026-08-28) dropped the rd_cm/st_cm
# parameters of `_emit_lds_repack` while the mxfp8 dispatch kernel still passes them. Every commit
# from there to main therefore fails at *trace* time with `TypeError: unexpected keyword argument
# 'rd_cm'` before it can reach the failure this script is looking for, so an mxfp8 bisect would
# just re-find #483. bf16 does not touch that helper and runs to the real symptom.
#
# The verdict is the *training result*, not whether the run survives: three separate readings, any
# one of which condemns a commit.
#
#   1. iteration-1 grad norm -- reference 1.449, a broken stack reports ~1.8e6. A factor of a
#      million on the very first iteration, while iteration-1 loss still matches the reference to
#      six digits, which is what says the forward pass is fine and the backward is not.
#   2. loss@50 -- reference 4.533. This is the reading that cannot be faked: a commit whose
#      gradients are subtly wrong can pass (1) and still fail to train.
#   3. any Inf or NaN.
#
# 50 iterations, not a handful, because the corruption is intermittent -- #482's own row-level probe
# found 7 of 8 runs carrying 1-6 corrupted rows out of 8192. A short run can therefore come back
# clean from a bad commit, and a false GOOD sends the bisect down the wrong half. The Inf that
# eventually kills a run lands on a different iteration every time (3, 9 and 21 have all been
# seen), so it is reported but never relied on as the signal.
#
# ALL_RANKS=1: the Inf guard fires on whichever rank sees it first, and with the default
# rank-0-only filter its traceback never reaches the log -- all that survives is torchrun's
# "exitcode 1 (local_rank: N)", which is how this symptom first got mistaken for a hang.
#
# Rebuilds incrementally (the build tree is kept) because a bisect hop is a few commits and ninja
# only recompiles what changed. If a verdict looks wrong, re-run that commit with CLEAN=1: a
# cross-checkout incremental build can link an object whose source file no longer exists.
#
# /perf_apps is shared across nodes, so both the checkout and the rebuild are visible everywhere at
# once. Never run this while any node has training holding the old .so mmapped -- and never edit the
# turbo tree while a run is in flight: the install is editable and FlyDSL re-reads source at trace
# time, so a mid-run edit kills the run.
set -uo pipefail

COMMIT=${1:?usage: run_bisect_turbo.sh <turbo-commit>}
NODE=${NODE:-smci355-ccs-aus-n06-25}
CONTAINER=${CONTAINER:-xiaoming-dev}
PRECISION=${PRECISION:-bf16}
ITERS=${TRAIN_ITERS:-50}
# Reference grad norm is 1.449 and a broken stack reports ~1.8e6, so anything inside three orders
# of magnitude of either end is unambiguous; a verdict is only withheld if a run lands between.
GOOD_MAX=${GOOD_MAX:-100}
# Reference loss@50 is 4.533, and the three reference runs of 2026-08-24/25 agreed to 0.0004. The
# tolerance is wide anyway because DeepEP's dispatcher is nondeterministic run to run (the
# 2026-08-25 probe measured a 0.8 band at 50 iterations with it on), and a commit that corrupts
# gradients misses by whole units, not by tenths.
REF_LOSS=${REF_LOSS:-4.533}
LOSS_TOL=${LOSS_TOL:-1.0}
CLEAN=${CLEAN:-0}
REPO=${REPO:-/perf_apps/xiaoming/Primus}
TURBO=${TURBO:-/perf_apps/xiaoming/MegaMoE}

sha=$(git -C "$TURBO" rev-parse --short=8 "$COMMIT" 2>/dev/null) || {
    echo "error: '$COMMIT' is not a commit in $TURBO" >&2
    exit 2
}
OUT=${OUT:-$REPO/ab_2x2/turbo_bisect/$sha}
mkdir -p "$OUT"

busy=$(ssh -o BatchMode=yes "$NODE" \
    "docker exec $CONTAINER bash -c 'ps -eo stat,args | grep \"[c]li/main.py\" | grep -cv \"^Z\"'" \
    2>/dev/null | tr -d '\r')
# Zombies are excluded deliberately: killing a torchrun parent leaves its ranks unreaped by the
# container's init, and hundreds accumulate over a bisect. They hold no memory and would otherwise
# make every step refuse to start.
[ "${busy:-0}" -gt 0 ] && {
    echo "refusing: $NODE still has $busy live training procs" >&2
    exit 3
}

# A killed run can keep its VRAM for a while after its processes are gone, and the next step then
# dies with "0 bytes is free" while PyTorch reports only its own ~67 GB -- which reads like a code
# problem and is not one. Idle is ~300 MB per GPU, a loaded one is tens of GB, so 4 GB separates
# them without being tight.
vram=$(ssh -o BatchMode=yes "$NODE" \
    "rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card/{print \$3}' | sort -rn | head -1" \
    2>/dev/null | tr -d '\r')
if [ -n "$vram" ] && [ "$vram" -gt $((4 * 1024 * 1024 * 1024)) ]; then
    echo "refusing: $NODE still has $((vram / 1024 / 1024 / 1024)) GiB VRAM held on some GPU" >&2
    exit 3
fi

subject=$(git -C "$TURBO" log -1 --pretty=%s "$COMMIT")

# Checkout runs inside the container, as root, for the same reason the build does: the container
# has already created parts of .git/modules as root (submodule git dirs), and a submodule this
# commit does not carry has to be deinited, which needs write access to them. From the host that
# fails with `index.lock: Permission denied` partway through, leaving a half-switched tree.
# safe.directory: root in the container does not own the bind-mounted repo either.
# SKIP_BUILD=1 runs against whatever is already installed. The checkout+install below is there for
# bisecting, where every step is a different commit; running a second arm on a build that is
# already current re-links the .so under LTO for several minutes and changes nothing.
if [ "${SKIP_BUILD:-0}" = 1 ]; then
    installed=$(ssh -o BatchMode=yes "$NODE" \
        "docker exec $CONTAINER python3 -c 'import primus_turbo;print(primus_turbo.__version__)'" \
        2>/dev/null | tr -d '\r')
    echo "########## skipping build; installed turbo $installed, tree at $sha  $(date -Is)"
    build_rc=0
else
    echo "########## checkout + build $sha  $(date -Is)"
ssh -o BatchMode=yes "$NODE" "docker exec -e GPU_ARCHS=gfx950 $CONTAINER bash -lc '
set -e
git config --global --add safe.directory \"*\"
cd $TURBO
git checkout -q --recurse-submodules $COMMIT
# --recurse-submodules deinits a submodule the target commit does not carry, but moving back to a
# commit that does carry it leaves it empty rather than re-cloning. Walking across 3rdparty/hipkittens
# being added that way costs a build that fails on a missing kittens.cuh, ten minutes in.
git submodule update --init --recursive 2>&1 | tail -2
echo \"turbo  : \$(git rev-parse --short=8 HEAD) \$(git log -1 --format=%ad --date=short)\"
[ ${CLEAN} = 1 ] && rm -rf build
start=\$(date +%s)
pip install -e . --no-build-isolation --no-deps 2>&1 | tail -6
echo \"build seconds: \$(( \$(date +%s) - start ))\"
'" >"$OUT/build.log" 2>&1
    build_rc=$?
    tail -3 "$OUT/build.log"
fi
[ $build_rc -ne 0 ] && {
    echo "=== $sha  $subject"
    echo "VERDICT: SKIP (build failed, see $OUT/build.log)"
    exit 0
}

{
    echo "turbo   : $sha  $subject"
    echo "date    : $(git -C "$TURBO" log -1 --date=short --pretty=%ad)"
    echo "primus  : $(git -C "$REPO" rev-parse --short HEAD)"
    echo "node    : $NODE  container: $CONTAINER"
    echo "arm     : $PRECISION MegaMoE (fused), $ITERS iterations"
    echo "clean   : $CLEAN"
    echo "verdict : it1 grad norm <= $GOOD_MAX (ref 1.449), loss@$ITERS within $LOSS_TOL of $REF_LOSS, no Inf/NaN"
} | tee "$OUT/launch.txt"

log="$OUT/$PRECISION.mega.log"
echo "########## run $sha  $(date -Is)"
ssh -o BatchMode=yes "$NODE" "docker exec \
    -e PRECISION=$PRECISION \
    -e USE_MEGA_MOE=True \
    -e TRAIN_ITERS=$ITERS \
    -e LOG=$log \
    -e ALL_RANKS=1 \
    $CONTAINER bash $REPO/run.sh" >"$log.outer" 2>&1
echo "########## run $sha done rc=$? $(date -Is)"

python3 - "$log" "$sha" "$subject" "$GOOD_MAX" "$ITERS" "$REF_LOSS" "$LOSS_TOL" <<'PY'
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# loss and grad norm come off the same line, so one regex keeps them paired per iteration.
ITER = re.compile(r"iteration\s+(\d+)/\s*\d+ .*?lm loss:\s*([\d.E+naN-]+).*?grad norm:\s*([\d.naN-]+)")
log, sha, subject = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
good_max, want, ref_loss, loss_tol = (float(sys.argv[4]), int(sys.argv[5]),
                                      float(sys.argv[6]), float(sys.argv[7]))


def num(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


text = ANSI.sub("", log.read_text(errors="replace")) if log.exists() else ""
# All ranks log, so the same iteration appears up to eight times; keep the worst grad norm seen,
# since the corruption does not hit every rank.
rows = {}
for i, loss, gn in ITER.findall(text):
    i, loss, gn = int(i), num(loss), num(gn)
    if i not in rows or not (gn <= rows[i][1]):
        rows[i] = (loss, gn)

trace = "unexpected keyword argument" in text
inf = "Unexpected result inf" in text or "Unexpected result nan" in text
nan_seen = any(v != v for r in rows.values() for v in r)
print(f"\n=== {sha}  {subject}")
marks = [i for i in (1, 2, 10, 20, 30, 40, want) if i in rows]
if rows and max(rows) not in marks:
    marks.append(max(rows))
for i in marks:
    loss, gn = rows[i]
    print(f"  it{i}: loss={loss:.5f}  grad_norm={gn:,.3f}")
if trace:
    print("  TRACE ERROR: unexpected keyword argument (the #483 rd_cm break)")
if inf:
    print("  INF/NAN GUARD fired (downstream of the corrupted gradients)")

reasons = []
if rows:
    gn1 = rows[min(rows)][1]
    last = max(rows)
    final = rows[last][0]
    if not (gn1 <= good_max):
        reasons.append(f"it1 grad norm {gn1:,.0f} vs reference 1.449")
    if nan_seen or inf:
        reasons.append("Inf/NaN in the trajectory")
    if last < want:
        reasons.append(f"stopped at iteration {last}/{want}")
    elif not (abs(final - ref_loss) <= loss_tol):
        reasons.append(f"loss@{last} {final:.4f} vs reference {ref_loss} (tol {loss_tol})")

if trace:
    verdict = "SKIP (trace-time break, never reaches a gradient)"
elif not rows:
    verdict = "SKIP (no iteration logged -- inspect the log)"
elif reasons:
    verdict = "BAD -- " + "; ".join(reasons)
else:
    verdict = (f"GOOD (it1 grad norm {rows[min(rows)][1]:.3f}, "
               f"loss@{max(rows)} {rows[max(rows)][0]:.4f})")
print(f"VERDICT: {verdict}")
PY
echo "=== out: $OUT"
