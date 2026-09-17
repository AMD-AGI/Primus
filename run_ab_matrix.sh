#!/usr/bin/env bash
# 2x2 A/B: fused MegaMoE vs the stock MoE, once at bf16 and once at mxfp8. Runs from the host
# against one container, one arm at a time -- run.sh starts with `pkill -9 python`, so two arms must
# never overlap.
#
# Precision is a property of the config rather than of a flag, which is why run.sh takes PRECISION
# and picks the yaml itself: turbo_mega_moe_precision covers only the expert GEMMs, so an arm that
# set it without switching the yaml would run its attention and dense MLP at the other precision.
#
#   PRECISION=bf16   BF16 yaml, MegaMoE bf16 experts vs stock bf16
#   PRECISION=mxfp8  FP8 yaml (fp8 e4m3, mxfp8 recipe), MegaMoE mxfp8 experts vs stock turbo mxfp8
#
# Arms are interleaved (both mega arms, then both baselines) so drift over the ~40 minute window
# cannot land entirely on one arm of a pair.
set -uo pipefail

# LOCAL=1: driver and container on the same host — docker exec only, no ssh.
# Default LOCAL=0 keeps the old remote-driver layout (ssh to NODE, then docker exec).
LOCAL=${LOCAL:-0}
NODE=${NODE:-smci355-ccs-aus-n04-21}
CONTAINER=${CONTAINER:-xiaoming-dev}
LAYERS=${NUM_LAYERS:-4}
ITERS=${TRAIN_ITERS:-50}
REPO=${REPO:-/perf_apps/xiaoming/Primus}
MEGAMOE=${MEGAMOE:-/perf_apps/xiaoming/MegaMoE}
# Appended to every arm, so a flag that changes the question rather than one arm stays symmetric.
# For an accuracy comparison that means COMMON_EXTRA="--turbo_sync_free_moe_stage 0
# --use_turbo_deepep False": the turbo DeepEP dispatcher is nondeterministic run to run (0.1 of loss
# within 10 iterations, 0.8 by 50), which is larger than the gap between the arms, so with it on no
# single-run loss difference means anything. MegaMoE replaces the whole MoE layer including dispatch
# and ignores both flags, but they are passed to it anyway to keep the command lines identical.
COMMON_EXTRA=${COMMON_EXTRA:-}
# Under the repo rather than the old .ab/ convention in the MegaMoE tree: a hidden directory in a
# different repository is not somewhere anyone looks for a log.
OUT=${OUT:-$REPO/ab_2x2/$(date +%m%d-%H%M%S)}

# "<PRECISION> <USE_MEGA_MOE> [extra args]"
# mxfp8.baseline leads because it is the arm under suspicion, so a bad stack shows up in the first
# 20 minutes instead of the last. No pair runs back to back either way, which is what keeps drift
# over the window from landing entirely on one side of a comparison.
ARMS=(
    "mxfp8 False"
    "bf16  True"
    "mxfp8 True"
    "bf16  False"
)
[ -n "${ONLY:-}" ] && ARMS=("$ONLY")

run_in_container() {
    docker exec \
        -e PRECISION="$1" \
        -e USE_MEGA_MOE="$2" \
        -e NUM_LAYERS="$LAYERS" \
        -e TRAIN_ITERS="$ITERS" \
        -e LOG="$3" \
        -e EXTRA_ARGS="$4" \
        "$CONTAINER" bash "$REPO/run.sh"
}

mkdir -p "$OUT"
# The yamls are snapshotted, not just named: an edit between arms would silently make them
# incomparable, and the diff is the only record that the FP8 config is not the committed one.
cp "$REPO"/examples/megatron/configs/MI355X/deepseek_v3-{BF16,FP8}-pretrain.yaml "$OUT/"
git -C "$REPO" diff -- examples/megatron/configs/MI355X/ >"$OUT/configs.diff"
if [ "$LOCAL" = 1 ]; then
    host_label="$(hostname -s) (local, $CONTAINER)"
    image="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>&1)"
    turbo="$(docker exec "$CONTAINER" python -c \
        'import primus_turbo,os;print(os.path.dirname(primus_turbo.__file__))' 2>&1)"
else
    host_label="$NODE ($CONTAINER)"
    image="$(ssh -o BatchMode=yes "$NODE" "docker inspect -f '{{.Config.Image}}' $CONTAINER" 2>&1)"
    turbo="$(ssh -o BatchMode=yes "$NODE" "docker exec $CONTAINER python -c \
        'import primus_turbo,os;print(os.path.dirname(primus_turbo.__file__))'" 2>&1)"
fi
{
    echo "node       : $host_label"
    echo "local      : $LOCAL"
    # The image is now a variable of the experiment, not a constant: it decides whether turbo comes
    # from the release build in site-packages or an editable install of a working tree.
    echo "image      : $image"
    echo "layers     : $LAYERS"
    echo "iterations : $ITERS"
    echo "turbo      : $turbo"
    echo "megamoe git: $(git -C "$MEGAMOE" rev-parse --short HEAD 2>&1)"
    echo "primus  git: $(git -C "$REPO" rev-parse --short HEAD 2>&1) (+$(wc -l <"$OUT/configs.diff") lines of config diff)"
    echo "started    : $(date -Is)"
} | tee "$OUT/launch.txt"

for spec in "${ARMS[@]}"; do
    read -r precision mega extra <<<"$spec"
    extra="$extra $COMMON_EXTRA"
    arm=mega
    [ "$mega" = True ] || arm=baseline
    name="$precision.$arm"
    log="$OUT/$name.log"
    echo ""
    echo "########## $name $extra -> $log  $(date -Is)"
    start=$SECONDS
    if [ "$LOCAL" = 1 ]; then
        run_in_container "$precision" "$mega" "$log" "$extra" >"$OUT/$name.outer.log" 2>&1
    else
        ssh -o BatchMode=yes "$NODE" "docker exec \
            -e PRECISION=$precision \
            -e USE_MEGA_MOE=$mega \
            -e NUM_LAYERS=$LAYERS \
            -e TRAIN_ITERS=$ITERS \
            -e LOG=$log \
            -e EXTRA_ARGS='$extra' \
            $CONTAINER bash $REPO/run.sh" >"$OUT/$name.outer.log" 2>&1
    fi
    echo "########## $name done rc=$? in $((SECONDS - start))s  $(date -Is)"
done

echo ""
echo "=== out: $OUT"
python3 "$(dirname "$0")/tools/parse_ab_matrix.py" "$OUT" | tee "$OUT/summary.md"
