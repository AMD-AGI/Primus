#!/bin/bash
# =============================================================================
# compare_node_perf.sh — summarize and compare trainer throughput across runs.
#
# Parses the perf lines emitted by MetricsLogger:
#   train - Step N perf: local_sps=.. global_sps=.. step_ms=.. elapsed_sec=.. \
#           total_samples=.. [tflops_algo/gpu=.. mfu=..% ...]
#
# For each log it drops the first $WARMUP perf windows (Triton autotune + first-
# step compile dominate those and are not steady state) and reports the MEDIAN
# of the rest. Then, treating the FIRST log as the baseline, it prints the
# scaling of global_sps and the change in per-rank local_sps / step_ms — i.e.
# how close 2 nodes get to the ideal 2x (the gap = cross-node allreduce cost).
#
# batch_size is per-rank, so this is a weak-scaling comparison: ideal is
# global_sps_2node / global_sps_1node == world_size_2node / world_size_1node.
#
# The perf lines exist only when the run logged training metrics
# (METRIC_LOG_FREQ>0). The gin default is METRIC_LOG_FREQ=0 (OFF, the MLPerf
# submission setting; yambda_5b.gin mlf/env_int), so launch perf runs with
# METRIC_LOG_FREQ=20 (as in the launch_slurm.sh perf-pair example) or 50, the
# same value in every arm. A log with no perf lines is reported as an error and
# the script exits 1.
#
# Usage:  bash scripts/compare_node_perf.sh <log1> [log2 ...]
#         WARMUP=3 bash scripts/compare_node_perf.sh perf_1node.log perf_2node.log
# =============================================================================
set -uo pipefail
WARMUP=${WARMUP:-3}

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <log1> [log2 ...]" >&2
  exit 2
fi

# Emits one summary row per log via awk: label nwin world_size med_local med_global med_stepms med_mfu
summarize() {
  local log="$1"
  awk -v warmup="$WARMUP" '
    function median(arr, n,   i, tmp, j, key) {
      # copy to a 1..n indexed array and sort
      delete s; i=0
      for (key in arr) { i++; s[i]=arr[key] }
      n=i
      for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (s[j]<s[i]) { tmp=s[i]; s[i]=s[j]; s[j]=tmp }
      if (n==0) return 0
      if (n%2) return s[(n+1)/2]
      return (s[n/2]+s[n/2+1])/2.0
    }
    /perf: local_sps=/ {
      n++
      if (n<=warmup) next
      # field-split on key=value tokens
      for (k=1;k<=NF;k++) {
        if ($k ~ /^local_sps=/)  { split($k,a,"="); L[n]=a[2] }
        if ($k ~ /^global_sps=/) { split($k,a,"="); G[n]=a[2] }
        if ($k ~ /^step_ms=/)    { split($k,a,"="); S[n]=a[2] }
        if ($k ~ /^mfu=/)        { split($k,a,"="); m=a[2]; sub(/%/,"",m); M[n]=m }
      }
      kept++
    }
    END {
      # world_size inferred from the launcher topology line if present
      ws=0
      printf "row\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.1f\n", kept, ws, median(L), median(G), median(S), median(M)
    }
  ' "$log"
}

# world_size is logged by launch_smoke_8gpu.sh as "world_size=NN"; grab it.
ws_of() { grep -oE "world_size=[0-9]+" "$1" 2>/dev/null | head -1 | cut -d= -f2; }

printf "\n%-34s %5s %3s %12s %12s %9s %6s\n" "LOG" "win" "WS" "local_sps" "global_sps" "step_ms" "mfu%"
printf -- "-----------------------------------------------------------------------------------------------\n"
declare -a BASE
i=0
rc=0
# A log is usable only if it has perf lines (METRIC_LOG_FREQ>0 runs).
has_perf() { grep -q "perf: local_sps=" "$1" 2>/dev/null; }
for log in "$@"; do
  if [ ! -s "$log" ]; then printf "%-34s  (missing/empty)\n" "$(basename "$log")"; rc=1; continue; fi
  if ! has_perf "$log"; then
    printf "%-34s  (no perf lines)\n" "$(basename "$log")"
    echo "ERROR: $log has no 'train - Step N perf:' lines; the run likely had training metrics OFF" \
         "(METRIC_LOG_FREQ=0, the gin default). Relaunch with METRIC_LOG_FREQ=20 or 50." >&2
    rc=1; continue
  fi
  row=$(summarize "$log"); ws=$(ws_of "$log"); ws=${ws:-?}
  kept=$(echo "$row" | cut -f2); ml=$(echo "$row" | cut -f4); mg=$(echo "$row" | cut -f5); ms=$(echo "$row" | cut -f6); mm=$(echo "$row" | cut -f7)
  printf "%-34s %5s %3s %12s %12s %9s %6s\n" "$(basename "$log")" "$kept" "$ws" "$ml" "$mg" "$ms" "$mm"
  if [ "$i" -eq 0 ]; then BASE=("$ws" "$ml" "$mg" "$ms"); fi
  i=$((i+1))
done

# Scaling vs first log (baseline); only when every log was usable
if [ "$#" -ge 2 ] && [ "$rc" -eq 0 ] && [ -n "${BASE[2]:-}" ]; then
  printf -- "-----------------------------------------------------------------------------------------------\n"
  printf "Scaling vs baseline (%s, WS=%s, global_sps=%s):\n" "$(basename "$1")" "${BASE[0]}" "${BASE[2]}"
  i=0
  for log in "$@"; do
    [ "$i" -eq 0 ] && { i=1; continue; }
    [ -s "$log" ] || continue
    row=$(summarize "$log"); ws=$(ws_of "$log"); ws=${ws:-0}
    mg=$(echo "$row" | cut -f5); ml=$(echo "$row" | cut -f4); ms=$(echo "$row" | cut -f6)
    awk -v base_ws="${BASE[0]}" -v base_g="${BASE[2]}" -v base_l="${BASE[1]}" -v base_s="${BASE[3]}" \
        -v ws="$ws" -v g="$mg" -v l="$ml" -v s="$ms" -v name="$(basename "$log")" 'BEGIN{
      ideal = (base_ws>0)? ws/base_ws : 0
      gain  = (base_g>0)?  g/base_g  : 0
      eff   = (ideal>0)?   100*gain/ideal : 0
      ldeg  = (base_l>0)?  100*(l/base_l-1) : 0
      sgrow = (base_s>0)?  100*(s/base_s-1) : 0
      printf "  %-30s  global_sps %.2fx (ideal %.2fx, %.0f%% scaling eff) | local_sps %+.1f%% | step_ms %+.1f%%\n", \
             name, gain, ideal, eff, ldeg, sgrow
    }'
  done
fi
echo
exit "$rc"
