#!/usr/bin/env bash
# Per-iteration metrics from a Primus Megatron training log (last rank only) and the harmonic mean
# of tokens/s/GPU over iterations >= first (default 4, the window tools/perf/extract_results.py uses).
# Usage: iter_metrics.sh <log> [first_iter=4]
set -euo pipefail

LOG=${1:?training log}
FIRST=${2:-4}

sed 's/\x1b\[[0-9;]*m//g' "$LOG" | awk -v first="$FIRST" '
  /iteration +[0-9]+\/ *[0-9]+ \|/ {
    if (!match($0, /rank-[0-9]+\/[0-9]+/)) next
    split(substr($0, RSTART + 5, RLENGTH - 5), rw, "/")
    if (rw[1] + 0 != rw[2] - 1) next
    match($0, /iteration +[0-9]+/); it = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", it)
    ms = tok = loss = mem = ""
    if (match($0, /elapsed time per iteration \(ms\): [0-9.]+/)) { ms = substr($0, RSTART, RLENGTH); sub(/.*: /, "", ms) }
    if (match($0, /tokens\/s\/GPU inst\/harmonic mean: [0-9.]+/)) { tok = substr($0, RSTART, RLENGTH); sub(/.*: /, "", tok) }
    if (match($0, /lm loss: [0-9.E+-]+/)) { loss = substr($0, RSTART, RLENGTH); sub(/.*: /, "", loss) }
    if (match($0, /hip mem usage\/free\/total\/usage_ratio: [0-9.]+GB\/[0-9.]+GB\/[0-9.]+GB\/[0-9.]+%/)) {
      mem = substr($0, RSTART, RLENGTH); sub(/.*: /, "", mem); split(mem, m, "/"); mem = m[1] " (" m[4] ")"
    }
    if (match($0, /number of nan iterations: +[0-9]+/)) { v = substr($0, RSTART, RLENGTH); sub(/.*: +/, "", v); nans += v }
    if (match($0, /number of skipped iterations: +[0-9]+/)) { v = substr($0, RSTART, RLENGTH); sub(/.*: +/, "", v); skips += v }
    printf "it=%-4s ms=%-10s tok=%-10s loss=%-14s mem=%s\n", it, ms, tok, loss, mem
    if (it + 0 >= first && tok + 0 > 0) { n++; s += 1 / tok }
    last_loss = loss
  }
  END {
    if (n) printf "harmonic mean tokens/s/GPU (iterations >= %d): %.1f over %d iterations\n", first, n / s, n
    else print "no tokens/s/GPU values found"
    printf "last loss: %s | nan iterations: %d | skipped iterations: %d\n", last_loss, nans, skips
  }'
