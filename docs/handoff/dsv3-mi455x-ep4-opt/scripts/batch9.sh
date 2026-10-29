#!/usr/bin/env bash
# After b8: shared-expert overlap under DeepEP (Primus fc305f29) and DeepEP on its own comm stream.
set -u
D=/home/boyan/project/deepep/dsv3opt
V2=/home/boyan/project/deepep/Primus-Turbo-dsv3opt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/batches.log"; }
log "batch b9 queued (shared expert overlap / deepep comm stream, after b8)"
until grep -qE "batch b8 (done|skipped|stopped)" "$D/batches.log"; do sleep 60; done
until bash "$D/can_start.sh" >/dev/null; do sleep 60; done
log "batch b9 start ($(bash "$D/can_start.sh"))"

B=PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND
DE="--use_turbo_deepep True --turbo_deepep_num_cu 160"
SF="--turbo_sync_free_moe_stage 3 --turbo_sync_free_permute_factor 1.05"
HS="--get_batch_skip_tp1_broadcast True --check_for_nan_in_loss_and_grad False --check_for_nan_in_grad_norm True"
G="--gradient_accumulation_fusion True"
O="--moe_shared_expert_overlap True"
S="--turbo_deepep_use_comm_stream True"
E="TURBO_DIR=$V2 $B=flydsl_v2"
C="v2sfh|$E|$DE $SF $HS"
C+=";v2sfho|$E|$DE $SF $HS $O"
C+=";v2sfhg|$E|$DE $SF $HS $G"
C+=";v2sfhgo|$E|$DE $SF $HS $G $O"
C+=";v2sfhgs|$E|$DE $SF $HS $G $S"
C+=";v2sfhgos|$E|$DE $SF $HS $G $O $S"
M="--create_attention_mask_in_dataloader False"
C+=";v2sfhgm|$E|$DE $SF $HS $G $M"
C+=";v2sfhgom|$E|$DE $SF $HS $G $O $M"
export CASES=$C ROUTINGS=even REPS=1 ITERS=20 MODEL=dsv3 CACHE=/tmp/dsv3opt_cache TAG=b9
log "batch b9 e2e start"
bash "$D/run_e2e.sh"
log "batch b9 done rc=$? ($(readlink -f "$D/latest"))"
