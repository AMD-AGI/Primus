#!/usr/bin/env bash
# GPU guard for the DSv3 optimisation work (agent bc-50a3eeec).
# - Every CHECK_INTERVAL (600 s) looks at GPUs 0-3: utilisation and the KFD processes holding VRAM.
# - A process is "ours" when it or one of its ancestors has "dsv3opt" in its command line.
# - free_since = last time anyone else was seen on the GPUs (utilisation > 0 with none of our
#   processes running, or any foreign VRAM holder). Experiments may start once free_since is
#   >= 30 min old (see can_start.sh).
# - While our run is active (marker file RUNNING), checks every RUN_INTERVAL (60 s) for foreign
#   holders; if one appears, kills our processes and records it.
set -u
D=/home/boyan/project/deepep/dsv3opt
STATE=$D/gpu_state
LOG=$D/gpu_guard.log
CONTAINER=${CONTAINER:-dev_mi455_deepep_yangbo}
CHECK_INTERVAL=${CHECK_INTERVAL:-600}
RUN_INTERVAL=${RUN_INTERVAL:-60}
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

gpu_ids() {
  for n in $(ls -d /sys/class/kfd/kfd/topology/nodes/* | sort -t/ -k8 -n); do
    id=$(cat "$n/gpu_id"); [ "$id" != 0 ] && echo "$id"
  done
}

is_ours() {  # host pid
  local p=$1 i
  for i in $(seq 30); do
    [ -r /proc/$p/cmdline ] || return 1
    tr '\0' ' ' < /proc/$p/cmdline | grep -q dsv3opt && return 0
    p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null) || return 1
    [ -z "$p" ] || [ "$p" -le 1 ] && return 1
  done
  return 1
}

holders() {  # prints "pid ours|foreign" for every process with VRAM on any GPU
  local ids; ids=$(gpu_ids)
  for pdir in /sys/class/kfd/kfd/proc/*; do
    [ -d "$pdir" ] || continue
    local pid; pid=$(basename "$pdir"); local v=0
    for id in $ids; do
      local x; x=$(cat "$pdir/vram_$id" 2>/dev/null || echo 0); v=$((v + x))
    done
    [ "$v" -gt 0 ] || continue
    if is_ours "$pid"; then echo "$pid ours"; else echo "$pid foreign"; fi
  done
}

util_max() {
  local m=0 i u
  for i in 1 2 3; do
    for u in $(timeout 30 rocm-smi --showuse 2>/dev/null | sed -n 's/.*GPU use (%): *\([0-9]*\).*/\1/p'); do
      [ "$u" -gt "$m" ] && m=$u
    done
    sleep 2
  done
  echo "$m"
}

kill_ours() {
  log "KILL: foreign GPU user seen while our run was active; killing dsv3opt processes"
  # every run starts in its own session (setsid) whose leader has dsv3opt in its command line;
  # the training workers below it do not, so kill whole sessions
  timeout 60 docker exec "$CONTAINER" bash -c \
    'sids=$( (cat /tmp/*dsv3opt*.pid 2>/dev/null; for p in $(pgrep -f "[d]sv3opt"); do ps -o sid= -p "$p"; done) | tr -d " " | sort -u)
     echo "sessions: $sids"
     for s in $sids; do pkill -TERM -s "$s"; done; sleep 15
     for s in $sids; do pkill -KILL -s "$s"; done' >> "$LOG" 2>&1
  touch "$D/KILLED"; rm -f "$D/RUNNING"
}

free_since=$( [ -f "$STATE" ] && sed -n 's/^free_since=//p' "$STATE" || true )
[ -n "${free_since:-}" ] || free_since=$(date +%s)
log "guard start (free_since=$free_since)"
last_full=0
while true; do
  now=$(date +%s)
  running=0; [ -f "$D/RUNNING" ] && running=1
  if [ $running = 1 ] || [ $((now - last_full)) -ge "$CHECK_INTERVAL" ]; then
    h=$(holders); nf=$(echo "$h" | grep -c foreign || true); no=$(echo "$h" | grep -c ours || true)
    if [ $((now - last_full)) -ge "$CHECK_INTERVAL" ]; then
      u=$(util_max); last_full=$now
      busy=0
      [ "$nf" -gt 0 ] && busy=1
      [ "$no" -eq 0 ] && [ $running = 0 ] && [ "$u" -gt 0 ] && busy=1
      [ $busy = 1 ] && free_since=$now
      log "check: util_max=$u ours=$no foreign=$nf running=$running free_for=$(( (now-free_since)/60 ))min"
    elif [ "$nf" -gt 0 ]; then
      free_since=$now
    fi
    if [ $running = 1 ] && [ "$nf" -gt 0 ]; then
      log "foreign holders: $(echo "$h" | grep foreign | tr '\n' ' ')"
      kill_ours; free_since=$now
    elif [ $running = 1 ] && [ "$((10#$(TZ=Asia/Shanghai date +%H)))" -ge "${DAY_FROM:-8}" ]; then
      log "daytime in UTC+8 (user: GPUs at night only)"
      kill_ours
    fi
    printf 'free_since=%s\nlast_check=%s\nforeign=%s\nours=%s\n' "$free_since" "$now" "$nf" "$no" > "$STATE"
  fi
  sleep "$RUN_INTERVAL"
done
