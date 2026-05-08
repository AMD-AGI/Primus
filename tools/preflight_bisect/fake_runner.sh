#!/bin/bash
###############################################################################
# fake_runner.sh — drop-in runner replacement for bisect.py testing.
#
# Simulates a bad node without running a real preflight, so that
# bisect.py's identification logic can be validated on a live cluster.
#
# Environment variables (pass via bisect.py --preflight-env):
#   BAD_NODE      hostname of the node that should appear faulty (required)
#   HANG_SECONDS  if >0, the bad node sleeps this long before exiting 1,
#                 which lets you exercise the --trial-timeout-sec hang path
#                 (default: 0 → instant failure)
#   GOOD_SLEEP    seconds a healthy node sleeps to simulate a real run
#                 (default: 2)
#
# Usage with bisect.py:
#   python tools/preflight_bisect/bisect.py \
#       --nodelist "chi2867,chi2879" \
#       --partition mi355x \
#       --output-dir "output/bisect-$(date +%Y%m%d-%H%M%S)" \
#       --trial-timeout-sec 600 \
#       --slurm-time 00:15:00 \
#       --runner tools/preflight_bisect/fake_runner.sh \
#       --preflight-env BAD_NODE=chi2879
###############################################################################
set -euo pipefail

BAD_NODE="${BAD_NODE:-}"
HANG_SECONDS="${HANG_SECONDS:-0}"
GOOD_SLEEP="${GOOD_SLEEP:-2}"

THIS_HOST="$(hostname -s)"

if [[ -z "$BAD_NODE" ]]; then
    echo "[fake_runner] ERROR: BAD_NODE is not set. Pass --preflight-env BAD_NODE=<hostname> to bisect.py." >&2
    exit 2
fi

if [[ "$THIS_HOST" == "$BAD_NODE" ]]; then
    echo "[fake_runner] $THIS_HOST == BAD_NODE ($BAD_NODE): simulating failure"
    if (( HANG_SECONDS > 0 )); then
        echo "[fake_runner] sleeping ${HANG_SECONDS}s to simulate a hang"
        sleep "$HANG_SECONDS"
    fi
    exit 1
fi

echo "[fake_runner] $THIS_HOST != BAD_NODE ($BAD_NODE): simulating healthy run (sleep ${GOOD_SLEEP}s)"
sleep "$GOOD_SLEEP"
exit 0
