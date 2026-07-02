#!/bin/bash
###############################################################################
# Primus PROJECTION (memory / performance) for DeepSeek-V4 on one gfx1250 GPU.
#
# Sibling of run_deepseek_v4_pro_muon_1gpu.sh. Same validated gfx1250 docker
# recipe and the same required workarounds, but instead of pretraining it runs
# the Primus projection tool (docs/projection.md): benchmark a couple of layers
# on this single GPU and analytically project memory + training performance to
# a multi-node target cluster.
#
# Usage:
#   ./run_dsv4_projection_1gpu.sh                       # performance, pro, ->8 nodes
#   MODE=memory ./run_dsv4_projection_1gpu.sh           # memory projection only
#   PRIMUS_MODEL=deepseek_v4_flash ./run_dsv4_projection_1gpu.sh   # flash model
#   TARGET_NODES=16 ./run_dsv4_projection_1gpu.sh       # project to 16 nodes
#   PROFILING_MODE=simulate GPU_ARCH=mi355x MODE=performance ./run_dsv4_projection_1gpu.sh  # CPU-only
###############################################################################
set -eo pipefail

SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
export DOCKER_IMAGE=${DOCKER_IMAGE:-registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/dist/feba897")}

# ---------- What to project -------------------------------------------------
export MODE=${MODE:-performance}                       # memory | performance
export PRIMUS_MODEL=${PRIMUS_MODEL:-deepseek_v4_pro}   # deepseek_v4_pro | deepseek_v4_flash
export EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml}
export BENCHMARK_GPUS=${BENCHMARK_GPUS:-1}             # benchmark on this many GPUs (1 here)
# This box has ONE physical GPU. We invoke `primus-cli direct --single`
# (ONE python3 process, NOT torchrun) —
# the same trick the validated run_dsv3_projection script uses. In --single mode
# torchrun is not used, so GPUS_PER_NODE no longer drives --nproc_per_node; it
# serves ONLY as the TARGET node size (8 GPUs/node), while the projection spawns
# its own nproc=1 benchmark subprocess pinned to the single physical GPU. This
# gives correct intra-/inter-node comm modeling for the real 8-GPU-node cluster.
export GPUS_PER_NODE=8                                  # TARGET node size (8 nodes x 8 = 64 GPUs)
export TARGET_NODES=${TARGET_NODES:-8}                  # production TP1*PP8*EP8 = 64 GPUs = 8 nodes
export PROFILING_MODE=${PROFILING_MODE:-benchmark}     # benchmark | simulate | both
export GPU_ARCH=${GPU_ARCH:-}                          # e.g. mi355x for --profiling-mode simulate

# ---------- Required gfx1250 workarounds (see the pretrain launcher) ---------
export HSA_NO_SCRATCH_RECLAIM=1
# gfx1250 MES async-queue hang workaround — matched EXACTLY to the training
# launcher (run_deepseek_v4_pro_muon_1gpu.sh): AMD_SERIALIZE_COPY=3 alone, which
# was bisected sufficient for the V4-Pro proxy (KERNEL serialize + LAUNCH_BLOCKING
# found unnecessary, default off). If the projection still hangs with this, the
# cause is the full-model build (all 61 layers on one rank), not these knobs.
export AMD_SERIALIZE_COPY=${AMD_SERIALIZE_COPY:-3}
export AMD_SERIALIZE_KERNEL=${AMD_SERIALIZE_KERNEL:-0}
export HIP_LAUNCH_BLOCKING=${HIP_LAUNCH_BLOCKING:-0}
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}           # flaky SDMA completion-signal workaround
# Turbo-free TE-native FP8 (tensorwise / Float8CurrentScaling), matching the
# training launcher. Without this the model uses TE DelayedScaling, which asserts
# against the V4 attention's save_original_input. fp8_utils.py reads this env.
export PRIMUS_FP8_DISABLE_TURBO=${PRIMUS_FP8_DISABLE_TURBO:-1}
export NVTE_ROCM_ENABLE_MXFP8=${NVTE_ROCM_ENABLE_MXFP8:-1}
# RCCL all_reduce(AVG) hangs even at world_size=1 -> sitecustomize rewrites AVG->SUM/ws.
# Also put the vendored Emerging-Optimizers on PYTHONPATH so the muon optimizer
# (emerging_optimizers.*) imports — required for OPTIMIZER=muon.
export PYTHONPATH_IN="$SCRIPT_DIR/rccl_avg_workaround:$SCRIPT_DIR/third_party/Emerging-Optimizers"

LOG=${LOG:-dsv4-projection-${MODE}.log}

# Config overrides appended as trailing CLI key/value pairs. V4 uses its OWN
# attention (multi_latent_attention=false + yarn via dual_rope), but Megatron's
# stock validate_args rejects rope_type=yarn unless MLA is on. The projection
# benchmarks a stock-Megatron layer (not the V4 custom attention), so force
# rope_type=rope at the Megatron-arg level — V4 applies yarn internally and
# rope-vs-yarn is a cheap elementwise op (negligible for timing).
# (moe_router_score_function: V4 uses sqrtsoftplus, but Megatron's stock
# validate requires sigmoid for expert-bias aux-loss-free routing; the score
# function is a pointwise on router logits and does not change GEMM timing.)
# (moe_token_dispatcher_type: V4 uses the turbo "flex" dispatcher, which asserts
# TPxEP>1; the single-GPU benchmark runs at EP=1. Force "alltoall" — dispatcher
# type only affects MoE *communication* (modeled analytically), not expert-GEMM
# compute, so benchmarked layer time is unchanged.)
# (enable_primus_turbo / use_turbo_deepep: a Primus patch [moe_dispatcher_patches]
# force-replaces the dispatcher with the turbo DeepEP "flex" one when BOTH are
# true, and flex asserts TPxEP>1 (can't benchmark on 1 GPU). gfx1250 runs
# turbo-free anyway (training disables it), so force them off -> standard
# alltoall dispatcher, single-GPU-benchmarkable.)
# (gradient_accumulation_fusion: needs APEX fused_weight_gradient_mlp_cuda, not
# in this container; off here exactly as in the training launcher.)
# (optimizer: use the yaml default adam — the projection only benchmarks layer
# fwd/bwd, so the optimizer choice doesn't affect timing, and adam avoids muon's
# "Emerging Optimizers" package dependency that isn't in this container. The
# trainer.py adam/muon get_*_optimizer call sites were patched to match the
# bundled Megatron signature [dropped the removed no_wd_decay_cond/scale_lr_cond/
# lr_mult positionals that collided with use_gloo_process_groups].)
# (tokenizer NullTokenizer: the benchmark builds a mock dataset; Megatron's
# MockGPTDataset JSON-serializes the tokenizer via .unique_identifiers, which the
# DeepSeekV4 HuggingFace tokenizer lacks -> crash. NullTokenizer has it, needs no
# HF download, and preserves vocab_size (129280) so embedding/LM-head GEMM dims
# are unchanged. Layer compute is tokenizer-independent.)
# (use_v4_triton_attention/csa: enable the fused flash-style V4 attention kernels
# instead of eager attention so the benchmarked attention time is representative
# of the real training config — eager materializes [B,H,S,S] and hugely inflates
# attention at seq 4096. Verified working on gfx1250.)
# V4-specific flags mirrored from the training launcher (now that the V4 builder
# is used, the real DeepseekV4MoE/HybridLayer is built and needs these): clamped
# SwiGLU support on the grouped backend, turbo off, legacy/permute-fusion off.
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:---rope_type rope --moe_router_score_function sigmoid --moe_token_dispatcher_type alltoall --enable_primus_turbo false --use_turbo_deepep false --use_turbo_grouped_mlp false --use_turbo_parallel_linear false --use_v4_compiled_sinkhorn false --moe_use_legacy_grouped_gemm false --moe_permute_fusion false --gradient_accumulation_fusion false --mtp_num_layers 0 --tokenizer_type NullTokenizer --use_v4_triton_attention true --use_v4_triton_csa_attention true}

# ---------- Build the projection CLI args -----------------------------------
PROJ_ARGS="projection $MODE --config $EXP"
if [ "$MODE" = "performance" ]; then
    PROJ_ARGS="$PROJ_ARGS --benchmark-gpus $BENCHMARK_GPUS --target-nodes $TARGET_NODES --profiling-mode $PROFILING_MODE"
    [ -n "$GPU_ARCH" ] && PROJ_ARGS="$PROJ_ARGS --gpu-arch $GPU_ARCH"
fi
PROJ_ARGS="$PROJ_ARGS $EXTRA_OVERRIDES"

# TE wheel install prefix (same as pretrain launcher)
TE_INSTALL="true"
if ls "${TE_WHEEL_DIR}"/transformer_engine-*.whl >/dev/null 2>&1; then
    TE_INSTALL="pip install --quiet --force-reinstall --no-deps ${TE_WHEEL_DIR}/transformer_engine-*.whl && \
        pip install --quiet einops nvdlfw-inspect onnxscript onnx pydantic importlib-metadata packaging transformers pybind11"
fi
# simulate mode (CPU-only, no model instantiation) needs the Origami GEMM model.
ORIGAMI_INSTALL="true"
if [ "$PROFILING_MODE" = "simulate" ] || [ "$PROFILING_MODE" = "both" ]; then
    ORIGAMI_INSTALL="pip install --quiet 'git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python' || echo '[warn] origami install failed'"
fi

VOLUME_ARGS=(-v "$SCRIPT_DIR":"$SCRIPT_DIR")
[[ -d "$TE_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TE_WHEEL_DIR":"$TE_WHEEL_DIR")

echo "[projection] mode=$MODE model=$PRIMUS_MODEL target_nodes=$TARGET_NODES profiling_mode=$PROFILING_MODE config=$EXP"

# Same docker invocation as run_deepseek_v4_pro_muon_1gpu.sh (validated gfx1250 recipe).
docker run --rm \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged \
    --name primus-v4-projection \
    -e NNODES=1 -e GPUS_PER_NODE="$GPUS_PER_NODE" \
    -e MASTER_ADDR=localhost -e MASTER_PORT=1234 \
    -e GLOO_SOCKET_IFNAME=lo -e NCCL_SOCKET_IFNAME=lo \
    -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=1 \
    -e HSA_NO_SCRATCH_RECLAIM="$HSA_NO_SCRATCH_RECLAIM" \
    -e AMD_SERIALIZE_COPY="$AMD_SERIALIZE_COPY" -e AMD_SERIALIZE_KERNEL="$AMD_SERIALIZE_KERNEL" \
    -e HIP_LAUNCH_BLOCKING="$HIP_LAUNCH_BLOCKING" -e HSA_ENABLE_SDMA="$HSA_ENABLE_SDMA" \
    -e PRIMUS_FP8_DISABLE_TURBO="$PRIMUS_FP8_DISABLE_TURBO" -e NVTE_ROCM_ENABLE_MXFP8="$NVTE_ROCM_ENABLE_MXFP8" \
    -e PRIMUS_PROJ_MAX_LAYERS="${PRIMUS_PROJ_MAX_LAYERS:-1}" -e PRIMUS_PROJ_COMPRESS_RATIOS="${PRIMUS_PROJ_COMPRESS_RATIOS:-}" \
    -e PRIMUS_MODEL="$PRIMUS_MODEL" -e PYTHONUNBUFFERED=1 \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        set -e && cd $SCRIPT_DIR && \
        export PYTHONPATH=$PYTHONPATH_IN:\${PYTHONPATH:-} && \
        ${TE_INSTALL} && \
        ${ORIGAMI_INSTALL} && \
        pip install --quiet -r requirements.txt && \
        echo '==================== DSV4 PROJECTION ($MODE) ====================' && \
        bash runner/primus-cli direct --single -- $PROJ_ARGS" \
    2>&1 | tee "$LOG"
