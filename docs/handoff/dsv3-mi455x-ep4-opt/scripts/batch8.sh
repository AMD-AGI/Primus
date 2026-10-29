#!/usr/bin/env bash
# After b7: gradient_accumulation_fusion on top of flydsl_v2 (needs the gfx1250 FlyDSL beta=1 wgrad, b1abeff3).
set -u
D=/home/boyan/project/deepep/dsv3opt
V2=/home/boyan/project/deepep/Primus-Turbo-dsv3opt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/batches.log"; }
log "batch b8 queued (gradient_accumulation_fusion e2e, after b7)"
until grep -qE "batch b7 (done|stopped)" "$D/batches.log"; do sleep 60; done
grep -q "batch b7 stopped" "$D/batches.log" && { log "batch b8 skipped: b7 was stopped"; exit 2; }
if ! grep -q "ALL OK" "$D/unit_b7.out"; then
  log "batch b8 skipped: beta=1 bench in unit_b7.out is not ALL OK"; exit 3
fi
until bash "$D/can_start.sh" >/dev/null; do sleep 60; done
log "batch b8 start ($(bash "$D/can_start.sh"))"

B=PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND
DE="--use_turbo_deepep True --turbo_deepep_num_cu 160"
SF="--turbo_sync_free_moe_stage 3 --turbo_sync_free_permute_factor 1.05"
HS="--get_batch_skip_tp1_broadcast True --check_for_nan_in_loss_and_grad False --check_for_nan_in_grad_norm True"
G="--gradient_accumulation_fusion True"
C="v2|TURBO_DIR=$V2 $B=flydsl_v2|$DE"
C+=";v2g|TURBO_DIR=$V2 $B=flydsl_v2|$DE $G"
C+=";v2sfhg|TURBO_DIR=$V2 $B=flydsl_v2|$DE $SF $HS $G"
P="--profile True --use_pytorch_profiler True --profile_step_start 6 --profile_step_end 8 --profile_ranks [0,1,2,3]"
C+=";v2sfhgP|TURBO_DIR=$V2 $B=flydsl_v2|$DE $SF $HS $G $P"
export CASES=$C ROUTINGS=even REPS=1 ITERS=20 MODEL=dsv3 CACHE=/tmp/dsv3opt_cache TAG=b8
log "batch b8 e2e start"
bash "$D/run_e2e.sh"
log "batch b8 done rc=$? ($(readlink -f "$D/latest"))"
