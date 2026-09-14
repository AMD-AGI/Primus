



set -o pipefail

# PRECISION is the only precision knob: it picks the config, because a config is what decides the
# precision of everything outside the experts. turbo_mega_moe_precision covers the expert GEMMs
# alone, so setting it without switching the yaml would leave attention and the dense MLP on the
# other precision and the arm would be neither bf16 nor mxfp8.
PRECISION=${PRECISION:-mxfp8}
case "$PRECISION" in
    bf16) EXP=examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml ;;
    # Carries `fp8: e4m3` + `fp8_recipe: mxfp8`, so the baseline arm's GEMMs are block-scaled mxfp8
    # rather than trainer_base's `delayed` default, whose amax history would still be warming up
    # after 50 iterations and would make the baseline look worse than it is.
    mxfp8) EXP=examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml ;;
    *)
        echo "PRECISION must be bf16 or mxfp8, got '$PRECISION'" >&2
        exit 2
        ;;
esac
export EXP
# Build Primus-Turbo from source before training (hook)
# export REBUILD_PRIMUS_TURBO=1
# export PRIMUS_TURBO_REF=9b5d3092efcbc087657b233d8e9ae662cee6ec6b
# export GPU_ARCHS=gfx950

# /workspace/Primus is the stale copy baked into the image (no MegaMoE); use the bind-mounted repo
cd /perf_apps/xiaoming/Primus

# Use the MegaMoE working tree's python (flydsl / mega_moe) on top of the image's
# turbo .so -- see runner/helpers/patches/12_overlay_megamoe_python.sh
export PRIMUS_MEGAMOE_SRC=/perf_apps/xiaoming/MegaMoE

pkill -9 python
pkill -9 python
pkill -9 python3
pkill -9 python3

# The fused MegaMoE on/off switch, so an A/B does not need this file edited between arms. Normalized
# because Megatron accepts `true` while the log name below compares against `True`: an unnormalized
# lowercase value would label a MegaMoE run "baseline".
USE_MEGA_MOE=${USE_MEGA_MOE:-True}
case "${USE_MEGA_MOE,,}" in
    true | 1) USE_MEGA_MOE=True ;;
    false | 0) USE_MEGA_MOE=False ;;
    *)
        echo "USE_MEGA_MOE must be True or False, got '$USE_MEGA_MOE'" >&2
        exit 2
        ;;
esac
# The experts follow PRECISION, which is what makes each arm a same-precision comparison. Overriding
# it deliberately crosses the two (bf16 model, mxfp8 experts) -- the reference for how much the
# quantization alone costs, since then nothing else in the model changes.
MEGA_PRECISION=${MEGA_PRECISION:-$PRECISION}
# Depth is overridable too: a 4-layer step is ~half the wall time when the question is a loss
# trajectory rather than throughput.
NUM_LAYERS=${NUM_LAYERS:-4}
# Short iteration counts are for smoke-testing an arm before spending the full run on it.
TRAIN_ITERS=${TRAIN_ITERS:-50}
GBS=${GBS:-512}
# Escape hatch for a one-off arm that needs a flag this file does not carry, e.g. giving the bf16
# baseline the turbo grouped GEMM: EXTRA_ARGS="--use_turbo_grouped_gemm True".
EXTRA_ARGS=${EXTRA_ARGS:-}
USE_TURBO_DEEPEEP=${USE_TURBO_DEEPEEP:-True}
# Both the precision and the arm go in the log name: the four arms of a 2x2 would otherwise
# overwrite each other, and /perf_apps is shared across nodes so nothing here is private to a
# machine. A driver that wants its own layout passes LOG.
ARM=mega
[ "$USE_MEGA_MOE" = "True" ] || ARM=baseline
if [ -z "${LOG:-}" ]; then
    LOG=log.$PRECISION.$ARM
    # A crossed run (bf16 model, mxfp8 experts) is a third thing and must not land on either of the
    # two same-precision names.
    [ "$ARM" = mega ] && [ "$MEGA_PRECISION" != "$PRECISION" ] && LOG="$LOG.$MEGA_PRECISION-experts"
fi

# Parallelism (EP-only) + fused MegaMoE
bash ./primus-cli  direct --numa \
  --patch runner/helpers/patches/11_fix_lld_stub.sh \
-- train pretrain --config $EXP \
  --num_layers $NUM_LAYERS \
  --micro_batch_size 2 \
  --global_batch_size $GBS \
  --train_iters $TRAIN_ITERS \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --expert_model_parallel_size 8 \
  --moe_layer_freq 1 \
  --moe_shared_expert_intermediate_size None \
  --pipeline_model_parallel_layout null \
  --recompute_granularity null \
  --recompute_num_layers 0 \
  --recompute_layer_ids null \
  --moe_router_force_load_balancing_type uniform \
  --enable_primus_turbo True \
  --use_turbo_deepep $USE_TURBO_DEEPEEP \
  --use_turbo_mega_moe $USE_MEGA_MOE \
  --turbo_mega_moe_precision $MEGA_PRECISION \
  --mock_data True $EXTRA_ARGS 2>&1 | tee "$LOG"
  
  # --patch runner/helpers/patches/12_overlay_megamoe_python.sh \