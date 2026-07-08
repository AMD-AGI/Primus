#!/bin/bash
###############################################################################
# DeepSeek-V4 *Pro* + Muon single-GPU bring-up on mi455 / gfx1250 (1 GPU).
#
# Single-GPU sibling of run_deepseek_v4_flash_proxy_1gpu.sh, for the Pro model.
# The upstream run_deepseek_v4_pro_muon.sh uses `primus-cli direct`, which
# assumes it is ALREADY inside the 8x288GB MI355X container at EP=8. This host
# is one gfx1250 box with no SLURM and one GPU, so this script instead wraps
# the SAME examples/run_pretrain.sh entrypoint in a local `docker run` (the
# validated gfx1250 docker + TransformerEngine recipe from
# ../../mi450/Primus/run_dsv3_proxy_4L.sh), selects the Pro model via
# PRIMUS_MODEL, and scales it down to a MINIMUM single-GPU proxy:
#
#   - model           deepseek_v4_pro      hidden 7168 / 128 heads / head_dim
#                                          512 / o_groups 16 (full Pro widths
#                                          from deepseek_v4_pro.yaml)
#   - parallel        TP=1 PP=1 EP=1       (single GPU; no SLURM, no DeepEP)
#   - num_layers      4                    (vs production 61; MINIMUM slice
#                                          that still exercises every V4
#                                          attention layer kind)
#   - compress_ratios [128,128,4,0]        Pro pattern (first two HCA, then
#                                          CSA, last dense+SWA) -> covers
#                                          HCA cr=128 / CSA cr=4 / dense cr=0
#   - num_experts     48  topk 1           (production 384/topk6 div by 8 =
#                                          the per-rank shape of the EP=8
#                                          production run; topk ceil(6/8)=1.
#                                          ~48 experts x 66M x 4L + Muon fp32
#                                          states + seq-4096 activations -> 273
#                                          GB peak (measured), fits the 432 GiB
#                                          card; full 384 will NOT fit.)
#   - moe_ffn_hidden  3072                 (full Pro MoE width; from yaml)
#   - seq_length      4096                 (raised from 512: at GBS=8 the fixed
#                                          Muon Newton-Schulz cost otherwise
#                                          dominates GPU time; 4096
#                                          tokens/microbatch amortizes it to a
#                                          representative profile. CSA gather
#                                          scales w/ seq. Set 512 for a fast smoke.)
#   - index_topk      64                   (CSA top-K; <= cr=4 pool seq/4, i.e.
#                                          4096/4=1024 at the default seq)
#   - precision       FP8 e4m3 + tensorwise (paper fp8 LAYOUT, per-tensor scale;
#                                          mxfp8/ue8m0 is GUARDED off — diverges
#                                          on this build. CK-free TE path.
#                                          PRECISION_TYPE=BF16 for a BF16 A/B.)
#
# Optimizer: Muon (paper §4.2.2), same recipe as the upstream pro_muon runner:
#   momentum 0.95, update-RMS scale 0.18, AdamW eps 1e-20, LR 2.0e-4->2.0e-5,
#   balance-loss 1e-4. Muon hard-requires use_distributed_optimizer=False +
#   use_precision_aware_optimizer=False (so optimizer states stay fp32 — this
#   is the binding memory cost, NOT activations). Set OPTIMIZER=adam to A/B.
#   NOTE: at the tiny single-GPU GBS the Newton-Schulz cost looks huge as a %
#   of GEMM (starved-batch artifact, not a Muon bug); raise GBS to amortize.
#
# Correctness-first defaults (this is a "does V4-Pro train at all on 1 gfx1250
# GPU" bring-up, not a perf push). Eager attention; Turbo/DeepEP/tilelang/
# compiled-Sinkhorn/plan-6 Triton fusions all OFF; stock hipBLASLt; profiler
# OFF. Every knob is ${VAR:-DEFAULT}-guarded for command-line A/B.
#
# REQUIRED gfx1250 fix kept ON (single-GPU-safe): RCCL all_reduce(op=AVG) hangs
# even at world_size=1 on this build and Megatron's MoE aux-loss reduce uses
# AVG; sitecustomize on PYTHONPATH rewrites AVG -> SUM/world_size. See
# rccl_avg_workaround/sitecustomize.py.
#
# Usage:
#   ./run_deepseek_v4_pro_muon_1gpu.sh                                  # 10-iter smoke
#   OPTIMIZER=adam ./run_deepseek_v4_pro_muon_1gpu.sh                   # A/B vs AdamW
#   PRIMUS_TOTAL_LAYERS=6 ./run_deepseek_v4_pro_muon_1gpu.sh            # deeper slice (watch HBM)
#   PRIMUS_NUM_EXPERTS=8 PRIMUS_MOE_TOPK=2 ./run_deepseek_v4_pro_muon_1gpu.sh  # tiny MoE
#   HIP_VISIBLE_DEVICES=3 ./run_deepseek_v4_pro_muon_1gpu.sh            # pin a card
###############################################################################
set -eo pipefail

export DOCKER_IMAGE=${DOCKER_IMAGE:-registry-sc-harbor.amd.com/framework/therock-npi@sha256:feba897e2a32a2465b8b296ed2662b2ad6136b5f1cf6f6c2716a3674aafc30f3}
SCRIPT_DIR=$(realpath -m "$(dirname "$0")")
export TE_DIR=${TE_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/TransformerEngine")}
export TE_WHEEL_DIR=${TE_WHEEL_DIR:-$(realpath -m "$SCRIPT_DIR/../../mi450/dist/feba897")}

# ---------- Attention backend env (TE side) --------------------------------
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=0
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_USE_CK_GEMM=0
export NVTE_FLASH_ATTN=0

# ---------- hipBLASLt: STOCK by default; opt-in TUNED (PRIMUS_TUNED_HIPBLASLT=1) -
# Stock is the safe default: the older feba897 tuned bundle DEADLOCKED on a
# backward-FP8 GSU split-K kernel on this host (GPU wedge -> node reboot,
# 2026-06-10). PRIMUS_TUNED_HIPBLASLT=1 opts into a freshly built GridBased
# gfx1250 tuned library (qwen3/dsv3 tuned; swept clean on dsv4 fwd shapes) via
# LD_PRELOAD + HIPBLASLT_TENSILE_LIBPATH. This is a GUARDED EXPERIMENT: run with
# a watchdog and expect a possible node reboot if the backward path still wedges.
# NOTE: never export an EMPTY HIPBLASLT_TENSILE_LIBPATH into the container — a
# missing path breaks even stock hipBLASLt ("Cannot read TensileLibrary...").
export PRIMUS_TUNED_HIPBLASLT=${PRIMUS_TUNED_HIPBLASLT:-0}
export HBL_TUNED_RELEASE=${HBL_TUNED_RELEASE:-/home/yanyuqin/hipblaslt/rocm-libraries/projects/hipblaslt/build/release}
if [ "$PRIMUS_TUNED_HIPBLASLT" = "1" ]; then
    if [ ! -f "$HBL_TUNED_RELEASE/library/libhipblaslt.so.1" ]; then
        echo "[hipblaslt] ERROR: tuned lib not found at $HBL_TUNED_RELEASE/library/libhipblaslt.so.1" >&2
        exit 1
    fi
    # LD_PRELOAD / LD_LIBRARY_PATH are injected INSIDE the container (below) so the
    # image's own rocm/torch lib paths are preserved (prepend, not override).
    export HIPBLASLT_TENSILE_LIBPATH="$HBL_TUNED_RELEASE/Tensile/library/gfx1250"
    echo "[hipblaslt] TUNED (opt-in): libpath=$HIPBLASLT_TENSILE_LIBPATH, LD_PRELOAD=libhipblaslt.so.1 (GUARDED: watch for wedge)"
else
    unset HIPBLASLT_DIR HIPBLASLT_LD_PRELOAD HIPBLASLT_TENSILE_LIBPATH
    echo "[hipblaslt] STOCK (container built-in gfx1250 catalog)"
fi

# ---------- REQUIRED gfx1250 RCCL AVG->SUM workaround -----------------------
export PYTHONPATH="$SCRIPT_DIR/rccl_avg_workaround:${PYTHONPATH:-}"
# Real primus_turbo imports flydsl at import time; put FLYDSL_PKG_DIR on PYTHONPATH.
export FLYDSL_PKG_DIR=${FLYDSL_PKG_DIR:-}
if [ -n "$FLYDSL_PKG_DIR" ] && [ -d "$FLYDSL_PKG_DIR/flydsl" ]; then
    export PYTHONPATH="$FLYDSL_PKG_DIR:$PYTHONPATH"
fi

# SDMA OFF on this host (run 3 debugging, 2026-06-10): an SDMA H2D copy
# intermittently never signals completion — py-spy --native showed the trainer
# pinned in rocr BusyWaitSignal under a trivial `torch.tensor(n, device=dev)`
# in Megatron get_batch, GPU 0%, dmesg clean. Killing the stuck proc then
# leaves MES unrecoverable (recovery disabled) -> node reboot. Blit-kernel
# copies are slower but don't use the flaky SDMA queues. This may ALSO be the
# true cause of the run-1 "permute autotune wedge" (same stuck-queue
# signature; permute fusion possibly innocent).
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}

# ---------- Distributed / NCCL: single GPU, loopback only -------------------
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export RCCL_DISABLE_AMDSMI=1
export NCCL_AMDSMI_DISABLE=1
export USING_AINIC=0

export GPUS_PER_NODE=1
export NNODES=1
export PYTHONUNBUFFERED=1

# REQUIRED gfx1250 workaround for V4-Pro (default ON). The Pro model build
# (hidden_size 7168, NOT a multiple of 4096) leaves a memory layout that wedges
# the process's first high-priority MES queue creation at iter-1 get_batch
# -> deadlock -> node reboot (debugged 2026-06-11; root cause = MES queue
# creation vs non-4096-aligned allocation layout). AMD_SERIALIZE_COPY=3 alone
# prevents it (kernels stay async; small iter-time cost on the eager
# proxy). Bisected: KERNEL serialize + LAUNCH_BLOCKING are NOT needed, so they
# default off. Set AMD_SERIALIZE_COPY=0 only to re-demonstrate the hang.
export AMD_SERIALIZE_COPY=${AMD_SERIALIZE_COPY:-3}
export AMD_SERIALIZE_KERNEL=${AMD_SERIALIZE_KERNEL:-0}
export HIP_LAUNCH_BLOCKING=${HIP_LAUNCH_BLOCKING:-0}

# ---------- Model: DeepSeek-V4 Pro (selected via the EXP yaml model: line) --
export PRIMUS_MODEL=${PRIMUS_MODEL:-deepseek_v4_pro}

# ---------- Pro MINIMUM single-GPU proxy shape ------------------------------
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-1}
# 3 layers (down from 4): the 4-layer model sits near the 432GB gfx1250's
# capacity and OOMs at iter 2 (Muon keeps fp32 optimizer states). Dropping
# the last layer frees the headroom for a warm step.
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-3}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-48}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-1}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-3072}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-64}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
export MBS=${MBS:-1}
export GBS=${GBS:-8}
export TRAIN_ITERS=${TRAIN_ITERS:-10}

# Per-layer compression schedule, length == PRIMUS_TOTAL_LAYERS, mirroring the
# Pro pattern (first two HCA(128), then CSA(4)/HCA(128) interleaved, last
# dense+SWA(0)). Pure-bash generator (no host python needed).
gen_pro_compress_ratios() {
    local n=$1 i r=()
    for ((i = 0; i < n; i++)); do
        if   (( i < 2 ));     then r+=(128)
        elif (( i == n-1 ));  then r+=(0)
        elif (( i % 2 == 0)); then r+=(4)
        else                       r+=(128)
        fi
    done
    local IFS=,
    echo "[${r[*]}]"
}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-$(gen_pro_compress_ratios "$PRIMUS_TOTAL_LAYERS")}

# ---------- Optimizer: Muon (paper §4.2.2) ---------------------------------
export OPTIMIZER=${OPTIMIZER:-muon}
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
export MUON_EXTRA_SCALE_FACTOR=${MUON_EXTRA_SCALE_FACTOR:-0.18}
export USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-False}
export USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-False}
export LR=${LR:-2.0e-4}
export MIN_LR=${MIN_LR:-2.0e-5}
export ADAM_EPS=${ADAM_EPS:-1.0e-20}   # needs a decimal point — Primus parses "1e-20" as a string
export MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.0001}
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}   # V4 MTP layer not yet supported in-tree
# Pro uses sqrtsoftplus; Megatron only supports aux-loss-free expert bias with
# sigmoid, so disable expert bias (balancing falls back to seq_aux_loss).
export PRIMUS_MOE_ENABLE_EXPERT_BIAS=${PRIMUS_MOE_ENABLE_EXPERT_BIAS:-False}

# Muon needs fp32 optimizer states (precision-aware off forces this anyway).
OPT_DTYPE_ARGS="--main_grads_dtype fp32 --exp_avg_dtype fp32 --exp_avg_sq_dtype fp32"

# ---------- Perf knobs: V4 attention backends ON; turbo paths OFF ----------
# V4 attention backend (replaces the unfused/eager path). Covers the dense +
# HCA layers (compress_ratio in {0, 128}) via USE_V4_ATTENTION_BACKEND and the
# CSA layers (compress_ratio == 4) via USE_V4_CSA_ATTENTION_BACKEND.
# Validated on gfx1250 after the WMMA tile-floor fix (06ae5214).
export USE_V4_ATTENTION_BACKEND=${USE_V4_ATTENTION_BACKEND:-triton_v2}
export USE_V4_CSA_ATTENTION_BACKEND=${USE_V4_CSA_ATTENTION_BACKEND:-triton_v2}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-False}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-False}
# Projections: the FP8 yaml sets use_turbo_parallel_linear=true (PrimusTurboLinear);
# turbo-free here -> TELinear. Override off so no turbo GEMM is invoked.
export USE_TURBO_PARALLEL_LINEAR=${USE_TURBO_PARALLEL_LINEAR:-False}
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-False}
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-0}
# RoPE Triton: default ON. Trace (2026-06-25, L3) attributed 960 kernels / 513 tiny
# (<5us) / 38.6 ms to the eager rotary-embedding path — a launch-bound fusion target.
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-1}
# Sinkhorn Triton fused FWD/BWD: default ON. The eager Sinkhorn-Knopp loop
# (n_iters=20) launches ~18,600 tiny sum/add/div kernels per step (5,616 on the
# fwd side alone); the Triton path emits exactly 1 fwd + 1 bwd kernel per call.
# Measured 2026-06-25 (0612, L3, FP8): total GPU events 80,962 -> 62,340, sinkhorn
# GPU kernels 5,616 -> 48, warm step ~2,890 -> ~2,797 ms (+3.2%), 0 NaN / loss
# bit-identical. Falls back to eager when the shape/device is unsupported. Set =0 to A/B.
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-1}
# HyperConnection mHC Triton: default ON. The mHC HyperMixer glue (pre/post/comb
# projections + scales), separate from the already-fused HC-expand and sinkhorn.
# Trace (2026-06-25, L3): 1,320 kernels / 872 tiny (<5us) / 52 ms — top remaining
# launch-bound target after sinkhorn.
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-1}
# CSA indexer Triton: kept OFF — inert at L3 (compress_ratios [128,128,0] has NO CSA
# layer, so the indexer never runs). Enable only with a CSA layer (>=4 layers).
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-0}
export PRIMUS_INDEXER_TRITON_FULL=${PRIMUS_INDEXER_TRITON_FULL:-0}
# V4 MoE router Triton: default ON. Trace (2026-06-25, L3): 432 kernels / 208 tiny /
# 5.3 ms — marginal, but launch-bound and correctness-neutral.
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-1}

export ENABLE_PRIMUS_TURBO=False
if [ "$USE_TURBO_ATTENTION" = "True" ] || [ "$USE_TURBO_DEEPEP" = "True" ] || [ "$TURBO_USE_GROUPED_MLP" = "True" ]; then
    ENABLE_PRIMUS_TURBO=True
fi

# MoE permute fusion OFF for Pro on gfx1250: the Triton permute_with_mask_map
# BACKWARD autotune wedges the GPU stream at Pro shapes (48 experts / hidden
# 7168) — cuda.synchronize inside triton do_bench never returns (debugged
# 2026-06-10 via py-spy; flash shapes 32 experts / hidden 4096 autotune fine).
# Eager permute is the safe path; flip =True to retry after a triton fix.
export MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-False}

export PROFILE=${PROFILE:-False}
# PyTorch profiler writes the chrome trace via tensorboard_trace_handler(
# args.tensorboard_dir), so the tensorboard dir MUST be enabled to get a trace.
# Default tensorboard off, but auto-enable it whenever PROFILE=True so a
# profiled run actually produces a trace. profile window = steps [START,END);
# need TRAIN_ITERS > PROFILE_STEP_END.
export DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-True}
if [ "$PROFILE" = "True" ]; then export DISABLE_TENSORBOARD=False; fi
export PROFILE_STEP_START=${PROFILE_STEP_START:-6}
export PROFILE_STEP_END=${PROFILE_STEP_END:-7}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-gfx1250-1gpu}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_pro_muon_1gpu_L${PRIMUS_TOTAL_LAYERS}_E${PRIMUS_NUM_EXPERTS}_seq${PRIMUS_SEQ_LENGTH}}

PRIMUS_PATH="$SCRIPT_DIR"
DATA_PATH="${PRIMUS_PATH}/data"
mkdir -p "$DATA_PATH"

EXP=${EXP:-examples/megatron/configs/MI355X/deepseek_v4_flash-FP8-pretrain.yaml}
LOG=${LOG:-deepseek-v4-pro-muon-1gpu.log}

# ---------- FP8 training (matches upstream run_deepseek_v4_pro_fp8_paper.sh) --
# PRECISION_TYPE=FP8 (default) -> FP8=e4m3, FP8_RECIPE=tensorwise. This is the
# paper's fp8 LAYOUT (all weight GEMMs in fp8: MoE expert GEMMs via TEGroupedMLP,
# attention QKV/O + dense proj via TELinear; attention core QK^T/softmax*V stays
# BF16; mHC/Sinkhorn fp32; embedding/head/RMSNorm/router BF16; optimizer fp32) —
# but with a per-TENSOR scale instead of the paper's ue8m0 microscale.
#
# CK-free by construction: this 1gpu launcher already has TURBO_USE_GROUPED_MLP=
# False + USE_TURBO_DEEPEP=False, so experts route to TEGroupedMLP (hipBLASLt),
# not PrimusTurboGroupedMLP (ck_grouped_gemm). NVTE_ROCM_ENABLE_MXFP8=1 is set
# by examples/run_pretrain.sh.
#
# WHY NOT mxfp8 (paper ue8m0): two blockers on this build, both upstream-
# root-caused. (1) TE-ROCm MXFP8 asserts GEMM K % 128 == 0 (rocm_gemm.hip:1529)
# and V4 has non-128 K dims (e.g. K=224, K=32) -> errors out. (2) Even if it ran,
# mxfp8's e8m0 per-block quant noise AMPLIFIES MULTIPLICATIVELY through backward
# depth -> divergence. tensorwise's smooth per-tensor fp32 scale is stable
# (upstream: loss 12 -> 0.82 at full depth). So mxfp8 is GUARDED below.
#
# The earlier FP8 no-op (decoder skipped the fp8 context) is fixed upstream
# (commit b662c40b) and lives in the mounted repo, so FP8 now actually engages.
# A/B back to BF16 with PRECISION_TYPE=BF16 (or FP8=null).
export NVTE_ROCM_ENABLE_MXFP8=${NVTE_ROCM_ENABLE_MXFP8:-1}
# TURBO-FREE FP8: primus_turbo is only an import-shim in this gfx1250 container,
# so the turbo FP8 path (PrimusTurboQuantConfig / primus_turbo_fp8_autocast)
# can't run. Force the TE-native fp8_autocast branch (fp8_utils.py honors this).
export PRIMUS_FP8_DISABLE_TURBO=${PRIMUS_FP8_DISABLE_TURBO:-1}
export PRECISION_TYPE=${PRECISION_TYPE:-FP8}
if [ "$PRECISION_TYPE" = "FP8" ]; then
    export FP8=${FP8:-e4m3}
    export FP8_RECIPE=${FP8_RECIPE:-tensorwise}
    # GUARD: mxfp8 (paper ue8m0) diverges for V4 on this build. Refuse it unless
    # explicitly forced, matching upstream run_deepseek_v4_pro_muon.sh.
    if [ "$FP8_RECIPE" = "mxfp8" ] && [ "${MXFP8_I_KNOW_ITS_BROKEN:-0}" != "1" ]; then
        echo "[FATAL] FP8_RECIPE=mxfp8 diverges for V4 on this build (TE K%128 assert +" >&2
        echo "        e8m0 depth-amplified instability). Use FP8_RECIPE=tensorwise," >&2
        echo "        or set MXFP8_I_KNOW_ITS_BROKEN=1 to force it anyway." >&2
        exit 1
    fi
else
    export FP8=${FP8:-null}
    export FP8_RECIPE=${FP8_RECIPE:-null}
fi

if [ "$TURBO_USE_GROUPED_MLP" = "True" ]; then
  export PRIMUS_BIAS_SWIGLU_FUSION=True
fi

if [ ! -d "$PRIMUS_PATH/third_party/Megatron-LM" ] || \
   [ -z "$(ls -A "$PRIMUS_PATH/third_party/Megatron-LM" 2>/dev/null)" ]; then
    echo "[ERROR] third_party/Megatron-LM missing/empty -> run: git submodule update --init --recursive" >&2
    exit 1
fi

echo "[pro] model=$PRIMUS_MODEL layers=$PRIMUS_TOTAL_LAYERS experts=$PRIMUS_NUM_EXPERTS seq=$PRIMUS_SEQ_LENGTH optimizer=$OPTIMIZER compress_ratios=$PRIMUS_COMPRESS_RATIOS"

# V4-Pro single-GPU overrides (trailing args -> run_pretrain.sh -> primus cli
# train pretrain --config $EXP ...). Mirrors run_deepseek_v4_pro_muon.sh's CLI
# set, minus the DeepEP wiring, scaled to one GPU / minimum layers.
# overlap_grad_reduce/param_gather stay OFF: upstream enabled them for multi-
# node DP scaling (needs the distributed optimizer + the indexer-param freeze),
# but at single-GPU DP=1 they are no-ops, and Muon requires them off anyway.
PROXY_OVERRIDES="\
    --backend_path $PRIMUS_PATH/third_party/Megatron-LM \
    --train_iters $TRAIN_ITERS \
    --lr_warmup_iters 0 \
    --lr_decay_iters $TRAIN_ITERS \
    --num_layers $PRIMUS_TOTAL_LAYERS \
    --compress_ratios $PRIMUS_COMPRESS_RATIOS \
    --micro_batch_size $MBS \
    --global_batch_size $GBS \
    --lr $LR \
    --min_lr $MIN_LR \
    --adam_eps $ADAM_EPS \
    --moe_aux_loss_coeff $MOE_AUX_LOSS_COEFF \
    --seq_length $PRIMUS_SEQ_LENGTH \
    --max_position_embeddings $PRIMUS_MAX_POSITION_EMBEDDINGS \
    --rope_type rope \
    --tensor_model_parallel_size $PRIMUS_TP \
    --pipeline_model_parallel_size $PRIMUS_PP \
    --expert_model_parallel_size $PRIMUS_EP \
    --num_experts $PRIMUS_NUM_EXPERTS \
    --moe_router_topk $PRIMUS_MOE_TOPK \
    --moe_router_enable_expert_bias $PRIMUS_MOE_ENABLE_EXPERT_BIAS \
    --moe_ffn_hidden_size $PRIMUS_MOE_FFN_HIDDEN_SIZE \
    --index_topk $PRIMUS_INDEX_TOPK \
    --v4_grouped_experts_support_clamped_swiglu True \
    --mtp_num_layers $MTP_NUM_LAYERS \
    --mock_data True \
    --moe_router_force_load_balancing True \
    --log_avg_skip_iterations 3 \
    --optimizer $OPTIMIZER \
    --muon_momentum $MUON_MOMENTUM \
    --muon_extra_scale_factor $MUON_EXTRA_SCALE_FACTOR \
    --use_distributed_optimizer $USE_DISTRIBUTED_OPTIMIZER \
    --use_precision_aware_optimizer $USE_PRECISION_AWARE_OPTIMIZER \
    $OPT_DTYPE_ARGS \
    --enable_primus_turbo $ENABLE_PRIMUS_TURBO \
    --use_turbo_attention $USE_TURBO_ATTENTION \
    --use_turbo_deepep $USE_TURBO_DEEPEP \
    --use_turbo_grouped_mlp $TURBO_USE_GROUPED_MLP \
    --use_turbo_parallel_linear $USE_TURBO_PARALLEL_LINEAR \
    --use_v4_attention_backend $USE_V4_ATTENTION_BACKEND \
    --use_v4_csa_attention_backend $USE_V4_CSA_ATTENTION_BACKEND \
    --use_v4_compiled_sinkhorn $USE_V4_COMPILED_SINKHORN \
    --moe_use_legacy_grouped_gemm False \
    --moe_permute_fusion $MOE_PERMUTE_FUSION \
    --fp8 $FP8 \
    --fp8_recipe $FP8_RECIPE \
    --recompute_num_layers 0 \
    --recompute_granularity full \
    --recompute_method block \
    --gradient_accumulation_fusion False \
    --overlap_grad_reduce False \
    --overlap_param_gather False \
    --disable_last_saving True \
    --disable_wandb True \
    --disable_tensorboard $DISABLE_TENSORBOARD \
    --profile $PROFILE \
    --use_pytorch_profiler $PROFILE \
    --profile_step_start $PROFILE_STEP_START \
    --profile_step_end $PROFILE_STEP_END \
    --bias_swiglu_fusion "$PRIMUS_BIAS_SWIGLU_FUSION" \
    --torch_profiler_use_gzip True"

ENV_ARGS=()
for v in DOCKER_IMAGE NVTE_FUSED_ATTN NVTE_FUSED_ATTN_CK NVTE_FUSED_ATTN_AOTRITON \
         PRIMUS_TURBO_GEMM_BACKEND PRIMUS_TURBO_GROUPED_GEMM_BACKEND TURBO_WHEEL_DIR FLYDSL_PKG_DIR \
         NVTE_FLASH_ATTN NVTE_USE_CK_GEMM NVTE_ROCM_ENABLE_MXFP8 PRIMUS_FP8_DISABLE_TURBO PYTHONPATH HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM \
         TORCH_COMPILE_DISABLE TORCHINDUCTOR_COMPILE_THREADS TRITON_CACHE_DIR \
         HSA_SIGNAL_ABORT_TIMEOUT HSA_ENABLE_INTERRUPT \
         HIP_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY \
         AMD_LOG_LEVEL AMD_LOG_MASK MASTER_PORT TORCH_NCCL_HIGH_PRIORITY \
         NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_IB_HCA NCCL_SOCKET_IFNAME \
         GLOO_SOCKET_IFNAME RCCL_DISABLE_AMDSMI NCCL_AMDSMI_DISABLE USING_AINIC \
         GPUS_PER_NODE NNODES PYTHONUNBUFFERED TE_DIR TE_WHEEL_DIR PRIMUS_MODEL \
         PRIMUS_SEQ_LENGTH PRIMUS_MAX_POSITION_EMBEDDINGS \
         PRIMUS_TEAM PRIMUS_USER PRIMUS_EXP_NAME \
         PRIMUS_STACK_GROUPED_WEIGHT_TRITON PRIMUS_ROPE_TRITON \
         PRIMUS_SINKHORN_TRITON PRIMUS_HC_TRITON PRIMUS_INDEXER_TRITON \
         PRIMUS_INDEXER_TRITON_FULL PRIMUS_V4_ROUTER_TRITON \
         PRIMUS_TURBO_FUSE_GROUPED_WGRAD PRIMUS_TURBO_FUSE_WGRAD_DEBUG \
         PRIMUS_MUON_BATCHED_NS PRIMUS_COMPRESS_ROPE_CACHE PRIMUS_COMPRESS_POOL_TRITON; do
    ENV_ARGS+=("--env" "$v")
done
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && ENV_ARGS+=("--env" "HIP_VISIBLE_DEVICES")
# EXTRA_CLI: extra trailing --flag value overrides appended after PROXY_OVERRIDES
# (argparse last-wins), for dimension bisects etc.
[[ -n "${EXTRA_CLI:-}" ]] && ENV_ARGS+=("--env" "EXTRA_CLI")

# Persistent Triton compile cache. The --rm container makes TRITON_CACHE_DIR
# ephemeral, so every run recompiles ALL kernels from scratch (the slow CPU-bound
# LLVM step that dominates iteration 1, esp. under this node's MCE storm). Mount a
# host dir so compiled kernels (hsaco) are reused across runs -> iter-1 of every
# later run with the same shapes skips the cold compile. Triton keys cache entries
# by kernel-source + arch + constexpr hash, so a wheel/arch/shape change auto-
# invalidates (safe to keep warm). Disable with PRIMUS_TRITON_CACHE_DIR="".
export PRIMUS_TRITON_CACHE_DIR=${PRIMUS_TRITON_CACHE_DIR:-$PRIMUS_PATH/.triton_cache_shared}
if [ -n "$PRIMUS_TRITON_CACHE_DIR" ]; then
    mkdir -p "$PRIMUS_TRITON_CACHE_DIR"
    export TRITON_CACHE_DIR="$PRIMUS_TRITON_CACHE_DIR"
    echo "[triton] persistent compile cache: $TRITON_CACHE_DIR ($(find "$TRITON_CACHE_DIR" -maxdepth 1 -type d 2>/dev/null | wc -l) entries)"
fi

VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")
[[ -d "$TE_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TE_WHEEL_DIR":"$TE_WHEEL_DIR")
[[ -d "$TE_DIR" ]] && VOLUME_ARGS+=(-v "$TE_DIR":"$TE_DIR")
[[ -n "${TURBO_WHEEL_DIR:-}" && -d "$TURBO_WHEEL_DIR" ]] && VOLUME_ARGS+=(-v "$TURBO_WHEEL_DIR":"$TURBO_WHEEL_DIR")
[[ -n "${FLYDSL_PKG_DIR:-}" && -d "$FLYDSL_PKG_DIR/flydsl" ]] && VOLUME_ARGS+=(-v "$FLYDSL_PKG_DIR":"$FLYDSL_PKG_DIR")
[[ -n "${TRITON_CACHE_DIR:-}" ]] && VOLUME_ARGS+=(-v "$TRITON_CACHE_DIR":"$TRITON_CACHE_DIR")
# Opt-in tuned hipBLASLt: mount the built library at the same path and pass the
# loader env into the container (only when enabled, to keep stock runs untouched).
if [ "$PRIMUS_TUNED_HIPBLASLT" = "1" ]; then
    VOLUME_ARGS+=(-v "$HBL_TUNED_RELEASE":"$HBL_TUNED_RELEASE")
    ENV_ARGS+=("--env" "PRIMUS_TUNED_HIPBLASLT" "--env" "HIPBLASLT_TENSILE_LIBPATH" \
               "--env" "HBL_TUNED_RELEASE")
fi
# Container-side loader injection for the tuned lib (prepends to the image's paths).
HBL_PRELOAD_PREFIX=""
if [ "$PRIMUS_TUNED_HIPBLASLT" = "1" ]; then
    HBL_PRELOAD_PREFIX="export LD_LIBRARY_PATH=\"\$HBL_TUNED_RELEASE/library:\${LD_LIBRARY_PATH:-}\" && export LD_PRELOAD=\"\$HBL_TUNED_RELEASE/library/libhipblaslt.so.1\${LD_PRELOAD:+:\$LD_PRELOAD}\" && echo \"[hipblaslt] container LD_PRELOAD=\$LD_PRELOAD\" && "
fi

TE_INSTALL_PREFIX="\
    if ls ${TE_WHEEL_DIR}/transformer_engine-*.whl >/dev/null 2>&1; then \
        echo '[TE] installing prebuilt wheel from ${TE_WHEEL_DIR}' && \
        pip install --quiet --force-reinstall --no-deps ${TE_WHEEL_DIR}/transformer_engine-*.whl && \
        pip install --quiet einops nvdlfw-inspect onnxscript onnx pydantic importlib-metadata packaging transformers pybind11; \
    else \
        echo '[TE] WARNING: no TE wheel found at ${TE_WHEEL_DIR}; run will likely fail'; \
    fi && \
    echo '[deps] installing Primus requirements' && \
    pip install --quiet -r requirements.txt && \
    if [ -n \"${TURBO_WHEEL_DIR:-}\" ] && ls ${TURBO_WHEEL_DIR}/primus_turbo-*.whl >/dev/null 2>&1; then \
        echo '[turbo] installing real primus_turbo wheel from ${TURBO_WHEEL_DIR}' && \
        pip install --quiet --force-reinstall --no-deps ${TURBO_WHEEL_DIR}/primus_turbo-*.whl && \
        python -c 'import primus_turbo, primus_turbo.pytorch as _; print(\"[turbo] primus_turbo\", primus_turbo.__version__, \"imported OK\")'; \
    fi && "

docker run --rm \
    "${ENV_ARGS[@]}" \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged \
    --name primus-v4-pro-muon-1gpu \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        set -e && cd $PRIMUS_PATH && \
        ${HBL_PRELOAD_PREFIX}\
        ${TE_INSTALL_PREFIX}\
        echo '==================== V4-PRO + MUON 1-GPU PROXY (gfx1250, BF16, eager, no profiler) ====================' && \
        EXP=$EXP PRIMUS_MODEL=$PRIMUS_MODEL GPUS_PER_NODE=1 NNODES=1 bash examples/run_pretrain.sh \
            ${PROXY_OVERRIDES} ${EXTRA_CLI:-}" \
    2>&1 | tee "$LOG"
