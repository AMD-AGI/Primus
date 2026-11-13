#!/bin/bash
# Official score for the committed tree. Agents must not edit this file.
# Higher is better: SCORE = 1000 / steady-state ms/iter.
# A failed, incomplete, NaN, or GPU-collided run scores 0.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/eat/env.sh"

score0() {
    echo "$*" >&2
    echo "SCORE: 0"
    exit 0
}

GPU="${EAT_GPUS%% *}"
GPU="${GPU:-0}"
ITERS=6
NAME="eat_m31flash_$(date +%Y%m%d_%H%M%S)_$$"
PORT="$((20000 + ($$ % 20000)))"

export EXP="examples/megatron/configs/MI455X/minimax_m3_1_flash-BF16-pretrain.yaml"
export GPUS_PER_NODE=1
export HIP_VISIBLE_DEVICES="$GPU"
export PRIMUS_TOTAL_LAYERS=4
export MBS=1
export GBS=1
export TRAIN_ITERS="$ITERS"
export PRIMUS_SEQ_LENGTH=4096
export PRIMUS_MAX_POSITION_EMBEDDINGS=4096
export MOCK_DATA=True
export PROFILE=False
export PRIMUS_MSA_BACKEND=flydsl
export PRIMUS_LAUNCHER=container
export MASTER_PORT="$PORT"
export PRIMUS_TEAM=eat
export PRIMUS_USER=eval
export PRIMUS_EXP_NAME="$NAME"
export PRIMUS_WORKSPACE="$ROOT/output"

LOG="$ROOT/output/${PRIMUS_TEAM}/${PRIMUS_USER}/${NAME}/train.log"
mkdir -p "$(dirname "$LOG")"
CAPTURE="$(mktemp)"
trap 'rm -f "$CAPTURE"' EXIT

echo "eval name=$NAME gpu=$GPU port=$PORT log=$LOG"
set +e
gpulock run -g "$GPU" --max-wait-foreign 3h -n "eat m3.1-flash eval $NAME" -- \
    timeout --signal=TERM --kill-after=60 45m \
    bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh \
    >"$CAPTURE" 2>&1
RC=$?
set -e
cat "$CAPTURE"

python3 - "$LOG" "$CAPTURE" "$ITERS" "$RC" <<'PY'
import re, sys
log_path, cap_path, iters, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cap = open(cap_path, errors="replace").read()
try:
    text = open(log_path, errors="replace").read()
except OSError:
    text = ""
text = text or cap

def fail(why):
    print(why, file=sys.stderr)
    print("SCORE: 0")
    raise SystemExit(0)

if rc != 0:
    fail(f"training exit {rc}")
if re.search(r"INTRUDERS|is on a GPU this run locked", cap):
    fail("gpu collision; result discarded")
rows = re.findall(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)(?:/([\d.]+))?",
    text,
)
if not rows:
    fail("no elapsed-time lines")
step, total, inst, avg = rows[-1]
if int(step) < iters or int(total) != iters:
    fail(f"incomplete run, last iteration {step}/{total}, want {iters}")
losses = re.findall(r"lm loss:\s*([^\s|]+)", text)
if not losses:
    fail("no lm loss")
try:
    loss = float(losses[-1])
except ValueError:
    fail(f"non-numeric lm loss {losses[-1]}")
if loss != loss or abs(loss) == float("inf"):
    fail("non-finite lm loss")
if re.search(r"found NaN|lm loss:\s*nan", text, re.I):
    fail("NaN in log")
# Bit-exact tree (iter 1) prints this at iteration 6. The Adam remainder
# path drifts by ~8e-4 and is rejected. Kernel noise inside 1e-4 still passes.
ref_loss = 12.28070
if abs(loss - ref_loss) > 1e-4:
    fail(f"lm loss {loss:.5f} differs from reference {ref_loss} by more than 1e-4")
ms = float(avg or inst)
if not (20.0 <= ms <= 20000.0):
    fail(f"steady ms/iter {ms} outside sanity range")
print(f"steady_ms={ms:.3f} lm_loss={loss:.5f} iters={step}/{total}")
print(f"SCORE: {1000.0 / ms:.6f}")
PY
