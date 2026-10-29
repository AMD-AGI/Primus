#!/bin/bash
# DSv3 optimisation runs of bc-50a3eeec (copy of e2e/run_v2_e2e.sh): Primus worktree Primus-dsv3opt,
# Turbo worktree Primus-Turbo-dsv3opt. Every command line carries "dsv3opt" so gpu_guard.sh can tell
# our processes apart; RUNNING marks a batch in progress, KILLED (written by the guard) stops it. Cases run one after another, each in its own process group.
#   MODEL     gptoss20b | dsv3   (GPT-OSS 20B full model, or the MI455X DeepSeek-V3 4-layer proxy)
#   CASES     "name|ENV=VAL ENV=VAL|megatron args;..."  ENV may set TURBO_DIR (empty: installed Turbo)
#             and PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND
#   ROUTINGS  "even uniform"  (--moe_router_force_load_balancing_type, appended to every case)
#   GPUS      physical GPUs (default 0,1,2,3); EP size = number of GPUs. With 2 GPUs: GPT-OSS MBS 4 /
#             GBS 32, DSv3 64 experts / GBS 32 (Xiaoming's EP2 proxy), unless MBS / GBS are set
#   REPS ITERS RUN_TIMEOUT STALL_TIMEOUT TAG
set -u
BASE=/home/boyan/project/deepep
PRIMUS=${PRIMUS:-$BASE/Primus-dsv3opt}
CONTAINER=${CONTAINER_NAME:-dev_mi455_deepep_yangbo}
MODEL=${MODEL:-dsv3}
V2=$BASE/Primus-Turbo-dsv3opt
OPT=$BASE/dsv3opt
DEFAULT_CASES="off_inst||;cpp160|PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=cpp|--use_turbo_deepep True --turbo_deepep_num_cu 160"
DEFAULT_CASES+=";off_v2|TURBO_DIR=$V2|"
DEFAULT_CASES+=";flydsl160|TURBO_DIR=$V2 PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl|--use_turbo_deepep True --turbo_deepep_num_cu 160"
for cu in 32 64 160; do
  DEFAULT_CASES+=";v2_$cu|TURBO_DIR=$V2 PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND=flydsl_v2|--use_turbo_deepep True --turbo_deepep_num_cu $cu"
done
CASES=${CASES:-$DEFAULT_CASES}
ROUTINGS=${ROUTINGS:-"even uniform"}
REPS=${REPS:-2}
ITERS=${ITERS:-20}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
STALL_TIMEOUT=${STALL_TIMEOUT:-600}
TAG=dsv3opt_${TAG:-$MODEL}
GPUS=${GPUS:-0,1,2,3}
NGPU=$(echo "$GPUS" | tr ',' ' ' | wc -w)
LOCAL_GPUS=$(seq -s, 0 $((NGPU - 1)))
MODEL_ENV=(-e PRIMUS_EP="$NGPU")
if [ "$NGPU" -eq 2 ]; then
  if [ "$MODEL" = gptoss20b ]; then MODEL_ENV+=(-e MBS="${MBS:-4}" -e GBS="${GBS:-32}")
  else MODEL_ENV+=(-e PRIMUS_NUM_EXPERTS=64 -e GBS="${GBS:-32}"); fi
else
  [ -n "${MBS:-}" ] && MODEL_ENV+=(-e MBS="$MBS")
  [ -n "${GBS:-}" ] && MODEL_ENV+=(-e GBS="$GBS")
fi

case "$MODEL" in
  gptoss20b) LAUNCHER=examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh ;;
  dsv3) LAUNCHER=examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

D="$BASE/ut_logs/e2e_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$D"
ln -sfn "$D" "$OPT/latest"
rm -f "$OPT/KILLED"; touch "$OPT/RUNNING"
trap 'rm -f "$OPT/RUNNING"' EXIT
echo $$ > "$D/run.pid"
S="$D/STATUS"
# CACHE: share one directory between a smoke and its full run, so the full run starts with warm FlyDSL
# kernels (ranks that JIT-compile for > 5 s apart trip the gfx1250 watchdog)
CACHE=${CACHE:-/tmp/v2e2e_cache_$(date +%s)}
log() { echo "$(date -u +%T) $*" >> "$S"; }
gitv() { git -c safe.directory="$1" -C "$1" log -1 --format='%h %s' 2>/dev/null; }

log "gpus=$GPUS model_env=[${MODEL_ENV[*]}]"
log "start model=$MODEL routings=[$ROUTINGS] reps=$REPS iters=$ITERS"
log "cases=[$CASES]"
log "primus $(gitv "$PRIMUS")"
log "turbo v2 worktree $(gitv "$V2")"
log "turbo installed: $(docker exec "$CONTAINER" python3 -c 'import primus_turbo,os;print(primus_turbo.__version__,os.path.dirname(primus_turbo.__file__))' 2>&1 | tail -1)"
log "triton/inductor cache $CACHE"

# PIDs that may stay on a GPU (stuck idle processes, agreed with the user); the old list died with the
# 2026-10-28 05:27 UTC reboot
IGNORE_PIDS=${IGNORE_PIDS:-}
gpus_free() {
  local g pids p
  for p in $(bash "$BASE/e2e/gpu_pids.sh" "$GPUS"); do
    [[ " $IGNORE_PIDS " == *" $p "* ]] || return 1
  done
}

clean_shm() {
  # runs that crash in teardown (HIP VMM bug) leave their nccl-* segments behind; only ours are running
  docker exec "$CONTAINER" bash -c 'pgrep -f "pretrain|torchrun" >/dev/null || find /dev/shm -maxdepth 1 -name "nccl-*" -delete' 2>/dev/null
}

run_case() {
  local name=$1 envs=$2 extra=$3 routing=$4 rep=$5
  local exp="${TAG}_${name}_${routing}_r${rep}"
  local out="$D/$exp.out" pidf="/tmp/$exp.$$.pid" rcf="$D/$exp.rc"
  local waited=0
  until gpus_free; do
    if [ "$waited" -ge "${WAIT_FREE:-7200}" ]; then
      log "ABORT: GPUs $GPUS not free for ${waited}s before $exp"
      echo "    foreign GPU PIDs: $(bash "$BASE/e2e/gpu_pids.sh" "$GPUS" | tr "\n" " ")" >> "$S"
      return 2
    fi
    [ "$waited" -eq 0 ] && { log "$exp waiting for GPUs $GPUS"; echo "    foreign GPU PIDs: $(bash "$BASE/e2e/gpu_pids.sh" "$GPUS" | tr "\n" " ")" >> "$S"; }
    sleep 60; waited=$((waited + 60))
  done
  [ -f "$OPT/KILLED" ] && { log "ABORT: guard killed our run (another GPU user)"; return 2; }
  clean_shm
  extra="$extra --moe_router_force_load_balancing_type $routing"
  local env_args=() kv
  for kv in $envs; do env_args+=(-e "$kv"); done
  log "$exp RUNNING env=[$envs] extra=[$extra]"
  local t0=$(date +%s) cmd
  if [ "$MODEL" = gptoss20b ]; then
    cmd="MODEL=20B EXTRA_ARGS='$extra' bash $LAUNCHER"
  else
    cmd="bash $LAUNCHER $extra"
  fi
  docker exec -i -w "$PRIMUS" \
    -e PRIMUS_LAUNCHER=direct -e PRIMUS_TEAM=amd -e PRIMUS_USER=boyan \
    -e PRIMUS_EXP_NAME="$exp" -e TRAIN_ITERS="$ITERS" -e GPUS_PER_NODE="$NGPU" "${MODEL_ENV[@]}" \
    -e HIP_VISIBLE_DEVICES="$LOCAL_GPUS" -e ROCR_VISIBLE_DEVICES="$GPUS" \
    -e TRITON_CACHE_DIR="$CACHE/triton" -e TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    -e FLYDSL_RUNTIME_CACHE_DIR="$CACHE/flydsl" "${env_args[@]}" \
    "$CONTAINER" bash -c "setsid bash -c \"$cmd\" > '$out' 2>&1 < /dev/null & echo \$! > '$pidf'; wait \$!; echo \$? > '$rcf'" &
  local client=$! last_size=-1 last_change=$t0 verdict=""
  while kill -0 "$client" 2>/dev/null; do
    sleep 15
    local now=$(date +%s) size=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$now; fi
    if [ $((now - t0)) -ge "$RUN_TIMEOUT" ]; then verdict="TIMEOUT"; fi
    if [ $((now - last_change)) -ge "$STALL_TIMEOUT" ]; then verdict="STALL"; fi
    if [ -n "$verdict" ]; then
      local pg
      pg=$(docker exec "$CONTAINER" cat "$pidf" 2>/dev/null)
      log "$exp $verdict after $((now - t0))s (log idle $((now - last_change))s): killing process group $pg"
      tail -5 "$out" | sed 's/^/    /' >> "$S"
      [ -n "$pg" ] && docker exec "$CONTAINER" kill -TERM -- "-$pg" 2>/dev/null
      sleep 20
      [ -n "$pg" ] && docker exec "$CONTAINER" kill -KILL -- "-$pg" 2>/dev/null
      break
    fi
  done
  wait "$client" 2>/dev/null
  local rc
  rc=$(cat "$rcf" 2>/dev/null || echo "?")
  log "$exp DONE rc=$rc ${verdict:-ok} $(( $(date +%s) - t0 ))s"
  [ -f "$OPT/KILLED" ] && log "$exp was killed by the guard"
  bash "$BASE/e2e/iter_metrics.sh" "$out" 4 > "$D/$exp.metrics" 2>&1
  tail -2 "$D/$exp.metrics" | sed "s/^/    /" >> "$S"
  grep -m3 -iE "Traceback|Error|out of memory|NaN" "$out" | cut -c1-200 | sed 's/^/    !! /' >> "$S"
  docker exec "$CONTAINER" rm -f "$pidf"
  sleep 10
  return 0
}

IFS=';' read -r -a case_list <<< "$CASES"
for rep in $(seq "$REPS"); do
  for routing in $ROUTINGS; do
    for spec in "${case_list[@]}"; do
      IFS='|' read -r cname cenv cextra <<< "$spec"
      run_case "$cname" "$cenv" "$cextra" "$routing" "$rep" || { log "stopping"; exit 1; }
    done
  done
done
clean_shm
log "ALL DONE"
