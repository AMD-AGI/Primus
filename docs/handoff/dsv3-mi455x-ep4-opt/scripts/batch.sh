#!/usr/bin/env bash
# One batch of DSv3 proxy EP4 runs (bc-50a3eeec), started only once the GPUs have been free of
# other users for 30 min (can_start.sh), then dsv3opt/run_e2e.sh.
#   batch.sh <tag> "<cases>"   cases as in run_e2e.sh; ROUTINGS / REPS / ITERS / PROFILE pass through
set -u
D=/home/boyan/project/deepep/dsv3opt
TAG=${1:?tag}
export CASES=${2:?cases}
export ROUTINGS=${ROUTINGS:-even} REPS=${REPS:-1} ITERS=${ITERS:-20} MODEL=dsv3
export CACHE=${CACHE:-/tmp/dsv3opt_cache}
export TAG
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/batches.log"; }
log "batch $TAG queued: cases=[$CASES] routings=[$ROUTINGS] reps=$REPS iters=$ITERS"
until bash "$D/can_start.sh" >/dev/null; do sleep 60; done
log "batch $TAG start ($(bash "$D/can_start.sh"))"
bash "$D/run_e2e.sh"
log "batch $TAG done rc=$? ($(readlink -f "$D/latest"))"
