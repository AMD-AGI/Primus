#!/usr/bin/env bash
# The flydsl_v2 combine fix (41655cf2): unit tests, then the sync-free cases end to end.
set -u
D=/home/boyan/project/deepep/dsv3opt
V2=/home/boyan/project/deepep/Primus-Turbo-dsv3opt
CONTAINER=${CONTAINER_NAME:-dev_mi455_deepep_yangbo}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/batches.log"; }
log "batch b7 queued (unit tests + sync-free e2e)"
until bash "$D/can_start.sh" >/dev/null; do sleep 60; done
log "batch b7 start ($(bash "$D/can_start.sh"))"
rm -f "$D/KILLED"; touch "$D/RUNNING"
out="$D/unit_b7.out" pidf="/tmp/dsv3opt_unit.$$.pid"
docker exec -i -e HIP_VISIBLE_DEVICES=0,1,2,3 -e ROCR_VISIBLE_DEVICES=0,1,2,3 -e V2="$V2" "$CONTAINER" \
  bash -c "setsid bash $D/unit_tests.sh > '$out' 2>&1 < /dev/null & echo \$! > '$pidf'; wait \$!; echo rc=\$? >> '$out'"
docker exec "$CONTAINER" rm -f "$pidf"
rm -f "$D/RUNNING"
log "batch b7 unit tests: $(tail -1 "$out")"
[ -f "$D/KILLED" ] && { log "batch b7 stopped by the guard"; exit 2; }

B=PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND
DE="--use_turbo_deepep True --turbo_deepep_num_cu 160"
SF="--turbo_sync_free_moe_stage 3 --turbo_sync_free_permute_factor 1.05"
HS="--get_batch_skip_tp1_broadcast True --check_for_nan_in_loss_and_grad False --check_for_nan_in_grad_norm True"
C="v2s3|TURBO_DIR=$V2 $B=flydsl_v2|$DE --turbo_sync_free_moe_stage 3"
C+=";v2sf|TURBO_DIR=$V2 $B=flydsl_v2|$DE $SF"
C+=";v2sfh|TURBO_DIR=$V2 $B=flydsl_v2|$DE $SF $HS"
export CASES=$C ROUTINGS=even REPS=1 ITERS=20 MODEL=dsv3 CACHE=/tmp/dsv3opt_cache TAG=b7
log "batch b7 e2e start"
bash "$D/run_e2e.sh"
log "batch b7 done rc=$? ($(readlink -f "$D/latest"))"
