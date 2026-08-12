#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Kimi K3 -- "CURVE" shape BF16 pretraining, SINGLE NODE (1 x 8 MI355X).
#
# WHAT "CURVE" IS, AND WHY IT EXISTS
#   curve is the SINGLE-NODE, reproducible PROXY for the 8-layer official-width
#   model (see run_kimi_k3_8L_official_pretrain_mi355x.sh, ~215 B params, >= 4
#   nodes). The official shape only shards its 896-expert optimizer state across
#   expert-DP = DP/EP, so it cannot run on one node; curve scales the model DOWN
#   just far enough that the WHOLE model + optimizer fits on ONE 8-GPU node, while
#   PRESERVING every architectural mechanism that is actually under test.
#
#   SCALED DOWN together (so the shape stays balanced, not just shallower):
#       hidden        7168 -> 2048
#       routed experts 896 -> 32          (EP=8 -> exactly 4 local experts / GPU)
#       top-k           16 -> 4           (activation fraction 4/32 = 12.5%, dense
#                                          enough to train every expert in ~500 steps)
#       attn heads      96 -> 16
#       MoE FFN       3072 -> 1024
#       seq          4096 -> 2048
#       (num_layers stays 24: 1 dense + 23 MoE, KDA/full interleave "([1]*3+[0])*6")
#
#   PRESERVED at the released values (this is the point of curve):
#       * MLA + KDA interleave (3:1 body, same polarity as the release).
#       * Stable-Latent-MoE bottleneck ratio EXACTLY 0.5 (routed_expert_hidden_size
#         1024 = hidden/2, mirroring 3584/7168), with the in-bottleneck RMSNorm.
#       * situ (Moonshot's soft-clamped SwiGLU, gate beta 4.0 / up beta 25.0).
#       * sigmoid router + noaux_tc expert bias (moe_router_enable_expert_bias).
#       * RELEASED MLA head geometry: qk_head_dim 128 + qk_pos_emb_head_dim 64
#         -> q_head_dim 192, so softmax_scale stays 192**-0.5 (NoPE, not scaled).
#
# PARAMETER COUNT (derived from kimi_k3_curve.yaml, not measured)
#   Each routed expert is a SwiGLU MLP in the 1024-dim latent: 2*1024*1024 +
#   1024*1024 ~= 3.15 M.
#     routed experts   32 * 23 MoE layers * 3.15 M            ~= 2.32 B  (dominant)
#     untied embed+head 2 * 151680 * 2048                     ~= 0.62 B
#     KDA (18 layers)  18 * (q,k,v,gate,out = 5 * 2048*2048)  ~= 0.38 B
#     shared experts   23 * (2*2048*2048 + 2048*2048)         ~= 0.29 B
#     dense layer 0    2*2048*9728 + 9728*2048                ~= 0.06 B
#     latent proj      23 * (2048*1024 + 1024*2048)           ~= 0.10 B
#     MLA (6 layers) + router + norms                         ~= 0.06 B
#     -------------------------------------------------------------------
#     TOTAL                                                   ~= 3.85 B
#   The model + optimizer fit on 8 x 288 GB. The perf default still turns ON the
#   distributed + precision-aware (bf16-state) optimizer (PERF_OPT=1): at EP=8 only the
#   1.5 B non-expert params shard, but the bf16 grad/moment states cut memory and it is
#   part of the measured winner. Activation recompute keeps the micro-batch in memory:
#   MBS=16 with full/block/6 + flydsl peaks at ~210 GB of 288 (measured); MBS=8 with
#   full/block/12 peaked at 139.6 GB; MBS=16 without recompute OOMs (276 GB).
#
# PERFORMANCE -- MEASURED, 1 node x 8 MI355X, bf16, mock data, docker.io/rocm/primus:v26.4
#   Measured 2026-08-12 with the DEFAULTS below: attn_res_backend=flydsl, MBS=16,
#   GBS=128, seq 2048, recompute full/block/6, PERF_OPT=1 (distributed +
#   precision-aware optimizer, bf16 grads/moments), Turbo grouped_gemm + rms_norm +
#   permute, NUMA binding ON, kda_backend=fla. The perf run additionally exported the
#   perf reference's allocator env (matches perf/run_perf.sh and the 211 reference):
#       HSA_NO_SCRATCH_RECLAIM=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
#       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_ITERS=0
#
#   >>> THE SINGLE BIGGEST LEVER IS NUMA BINDING, AND IT WAS SILENTLY OFF. <<<
#   `primus-cli direct` has its OWN --numa/--no-numa flag (default AUTO = OFF) and does
#   NOT read ENABLE_NUMA_BINDING -- that env is only consulted by run_pretrain.sh's
#   torchrun wrapper (the multi-node path). So on this single-node launch path NUMA
#   binding never actually happened, even with MoE feature 3 on. Forwarding it as
#   `primus-cli direct --numa` (added below, gated on ENABLE_NUMA_BINDING) took the
#   SAME recipe from ~94 -> ~194 TFLOP/s/GPU. This shape is overhead-bound, so per-rank
#   numactl cpubind/membind is worth ~2x here (perf/FINDINGS calls it the biggest win).
#
#   Measured throughput (single 60-iter run, EVAL disabled, 0 NaN / 0 skipped for all
#   60 iters; MBS=16 FITS at ~210 GB peak of 288 = ~73%):
#       iters 40-60 (harmonic mean)     ~189-194 TFLOP/s/GPU   (~1585 ms/iter)
#       iters 50-60 (instantaneous)     ~193-195 TFLOP/s/GPU
#   Throughput was STILL RISING at iter 60 (this shape needs ~150 iters to reach
#   steady state), so this 60-iter window UNDER-reports the sustained number. The
#   internal 450-iter perf reference for the identical knobs reaches 211 TFLOP/s/GPU
#   steady-state at iter 400 (1460 ms/iter, 0 NaN). A clean self-measured 450-iter
#   headline is not quoted here: every idle amd-spur node reachable in this session was
#   sharing GPUs with non-SLURM docker containers, which hung/crashed the long runs.
#   For a steady-state number run TRAIN_ITERS>=450. Always quote which segment you use.
#
#   flydsl FIXES the MBS=16 instability. On eager, MBS=16 NaNs/OOMs on iteration 1 at
#   attention_residual.py's 1024 MiB fp32 up-cast of the concatenated residual
#   candidates. The fused flydsl mixer (one kernel per direction) removes that up-cast,
#   so MBS=16 trains clean AND fits. flydsl needs gfx950/CDNA4 + the `flydsl` pip
#   package (present in rocm/primus:v26.4).
#
#   Baseline for contrast: the previous default (MBS=8 / eager / recompute 12, and --
#   crucially -- NUMA silently OFF) was 64-80 TFLOP/s (iters 45-50, still rising) on
#   docker.io/tasimage/primus:pr-927. A "Stream-K Data Parallel does not support GSU"
#   warning FLOODS stderr on v26.4; it is COSMETIC (the 211 reference logged 869510 of
#   them and still hit 211) -- filter it, do not chase it.
#
# DATA (portable by default)
#   Defaults to MOCK data so the perf reproduction has no external dependency.
#   For the real-data CONVERGENCE curve (wikitext-103 w/ the Qwen2.5 tokenizer,
#   the turbo-OFF control the committed kimi_k3-BF16-curve.yaml describes), set
#     MOCK_DATA=0  PRIMUS_TRAIN_DATA=<idx_prefix>  PRIMUS_VALID_DATA=<idx_prefix> \
#     PRIMUS_TOKENIZER_MODEL=<hf_snapshot_dir>  MoE_Features="3 6 7"  MBS=2
#   (dropping feature 2 and the Turbo args restores the control's ms/iter).
#
# LAUNCH PATH (single-node: primus-cli direct)
#   ./primus-cli direct -- train pretrain --config $EXP <overrides>
#   Mirrors examples/moe_package/run_glm5_4layers_proxy.sh: for ONE node we drive
#   primus-cli directly rather than the multi-node examples/run_slurm_pretrain.sh
#   wrapper. primus-cli (repo root) -> runner/primus-cli; `direct` runs the
#   pretrain entrypoint (primus/cli/main.py train pretrain --config $EXP) in the
#   container it brings up from DOCKER_IMAGE on this node. The 8-layer official
#   sibling stays on run_slurm_pretrain.sh because it is multi-node (>= 4 nodes).
#
# USAGE (run from the Primus repo root, INSIDE the rocm/primus:v26.4 container)
#   # Perf headline -- defaults are the measured winner; add the reference allocator
#   # env + a longer run for a steady-state number:
#   HSA_NO_SCRATCH_RECLAIM=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_ITERS=450 EVAL_ITERS=0 \
#     bash examples/models/kimi-k3/run_kimi_k3_curve_pretrain_mi355x.sh
#   # Quick smoke (defaults, 50 iters):
#   bash examples/models/kimi-k3/run_kimi_k3_curve_pretrain_mi355x.sh
###############################################################################

######################### Training Docker and Variables #########################
# fla (flash-linear-attention) is installed by the megatron pretrain hook, not
# the base image.
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v26.4"}
export CLEAN_DOCKER_CONTAINER=${CLEAN_DOCKER_CONTAINER:-1}
export SKIP_TRAIN=${SKIP_TRAIN:-0}

######################### Training Environment Variables #########################
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
# HSA_NO_SCRATCH_RECLAIM=0 is the fla fix (matches run_pretrain.sh's own default,
# which its comment calls the fla fix); =1 regressed fla throughput, so keep 0.
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}

# Kimi K3 KDA backend selection (self-contained; run_pretrain.sh no longer sets it):
#   PRIMUS_KDA_BACKEND=fla -> kda_backend=fla (fla Triton chunk kernel; config
#     field, forwarded into the container as a PRIMUS_-prefixed var).
#   K3P_KDA_CONV=fla       -> fla causal_conv1d in the KDA depthwise conv, read
#     from the environment inside the container by kimi_delta_attention.py.
export PRIMUS_KDA_BACKEND=${PRIMUS_KDA_BACKEND:-fla}
export K3P_KDA_CONV=${K3P_KDA_CONV:-fla}

# Kimi K3 attention-residual mixer backend (attention_residual.py / attn_res_kernels):
#   eager  -> pure-PyTorch reference; the config DEFAULT and the permanent oracle.
#             Materialises six full-size candidate intermediates, five of them fp32,
#             so at MBS=16 it OOMs inside iteration 1 at the 1024 MiB fp32 up-cast in
#             attention_residual.py (measured), and on rocm/primus:v26.4 that same
#             path NaNs at MBS=16.
#   flydsl -> one fused FlyDSL kernel per direction (gfx950 / CDNA4 only; needs the
#             `flydsl` pip package present in the image, loaded lazily). Deletes that
#             fp32 up-cast, which is exactly what lets MBS=16 fit and train clean --
#             the ~211 TFLOP/s single-node reference below used it.
# Forwarded as a CLI override to the pretrain call. Defaults to flydsl: it is the
# measured perf winner AND the only backend on which the default MBS=16 trains clean
# on rocm/primus:v26.4. Set ATTN_RES_BACKEND=eager for the pure-PyTorch oracle (then
# also drop to MBS<=8, since eager MBS=16 NaNs/OOMs at the fp32 up-cast).
export ATTN_RES_BACKEND=${ATTN_RES_BACKEND:-flydsl}

# Single node. primus-cli `direct` does NOT allocate via SLURM -- it runs torchrun
# on the CURRENT host (only `primus-cli slurm` submits via srun/sbatch). So on a
# reserved cluster you must already BE on a GPU node before running this, e.g.
#   srun --reservation=<name> -w <node> -N1 --exclusive --pty \
#        bash examples/models/kimi-k3/run_kimi_k3_curve_pretrain_mi355x.sh
# (or switch the launch below to `primus-cli slurm`). The reservation here is on
# the `amd-spur` partition; export SBATCH_RESERVATION=<name> and SLURM_EXCLUSIVE=0
# for that srun. The repo must also live on a WRITABLE path -- a read-only checkout
# makes the `mkdir output/...` below fail. The SLURM_* vars below are only consulted
# if you switch this launcher to `primus-cli slurm`; the `direct` path ignores them.
export NNODES=${NNODES:-1}
export USING_AINIC=${USING_AINIC:-0}
export SLURM_TIME=${SLURM_TIME:-01:00:00}
export SLURM_PARTITION=${SLURM_PARTITION:-amd-spur}

# Node-local Triton cache (avoids the shared-NFS fla causal_conv1d 'hsaco' KeyError).
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_k3_curve}

# Select the curve preset (24L / hidden 2048 / 32 experts / top-4).
export PRIMUS_MODEL=${PRIMUS_MODEL:-kimi_k3_curve}

######################### Training Config (single-node perf) #########################
export MBS=${MBS:-16}                              # 16 = the measured perf winner; clean (0 NaN) with attn_res_backend=flydsl (see header). On eager MBS=16 NaNs/OOMs.
export GBS=${GBS:-128}                             # 128 = MBS(16) * DP(8) * 1 -> 1 micro-batch (grad accum 1)
export SEQ_LENGTH=${SEQ_LENGTH:-2048}
export TP=${TP:-1}
export ETP=${ETP:-1}
export PP=${PP:-1}
export EP=${EP:-8}                                 # 32 experts / 8 GPUs = 4 local experts/GPU
export CP=${CP:-1}
export OPTIMIZER=${OPTIMIZER:-adam}
export FP8=${FP8:-False}                           # False = bf16
export TRAIN_ITERS=${TRAIN_ITERS:-50}
export MOCK_DATA=${MOCK_DATA:-True}                # True = portable perf; False = real-data convergence
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-6}     # full/block/6: fewer recompute layers -> higher reported (model-FLOP) TFLOP; MBS=16+flydsl fits at ~210GB peak (measured). 0=off -> OOM.

# MoE_Features legend (K3-applicable, contiguous ids):
#   0 baseline | 1 turbo grouped GEMM | 2 cross-entropy loss fusion |
#   3 NUMA binding | 4 manual GC
# Default = 1 2 3 4, the K3 "measured winner": grouped GEMM (+ RMSNorm/permute in
# K3_TURBO_ARGS) + CE loss fusion + NUMA + manual GC. Upstream turbo attention /
# DeepEP / sync-free MoE / UCCL-EP are intentionally NOT offered -- they are NO-OP
# or unsafe for K3 (see the K3_TURBO_ARGS note below), so they are absent from this
# legend and the case handler and cannot be enabled.
MoE_Features=(1 2 3 4)

FEATURE_ARGS=()
PRIMUS_TURBO_ENABLED="False"
ensure_primus_turbo() {
    if [ "$PRIMUS_TURBO_ENABLED" = "False" ]; then
        FEATURE_ARGS+=("--enable_primus_turbo" "True")
        PRIMUS_TURBO_ENABLED="True"
    fi
}

for feature in "${MoE_Features[@]}"; do
    case "$feature" in
    0) ;;
    1)
        ensure_primus_turbo
        FEATURE_ARGS+=("--use_turbo_grouped_gemm" "True")
        ;;
    2)
        FEATURE_ARGS+=("--cross_entropy_fusion_impl" "te")
        FEATURE_ARGS+=("--cross_entropy_loss_fusion" "True")
        ;;
    3)
        export ENABLE_NUMA_BINDING=1
        export HSA_KERNARG_POOL_SIZE=12582912
        ;;
    4)
        FEATURE_ARGS+=("--manual_gc" "True")
        FEATURE_ARGS+=("--manual_gc_interval" "1")
        ;;
    *) ;;
    esac
done

FEATURE_LIST="${MoE_Features[*]}"
FEATURE_TAG=$(printf "%s" "${FEATURE_LIST}" | tr ' ' '-')

# K3-specific Turbo settings not covered by the feature legend above.
# kimi_k3-BF16-curve.yaml ships Turbo OFF (it is the convergence CONTROL), so
# turn on the "measured winner" kernels here for the perf run. rms_norm +
# grouped GEMM (feature 2) + permute are checkpoint-safe at EP=8. The four flags
# pinned OFF below (attention / deepep / shared_expert_overlap / sync-free) were
# dropped from the MoE_Features legend/case above because they are inapplicable or
# unsafe for K3 (same reasons as the official recipe); they are held off here so
# they can never be enabled.
K3_TURBO_ARGS=()
ensure_primus_turbo
K3_TURBO_ARGS+=("--use_turbo_rms_norm" "True")
K3_TURBO_ARGS+=("--moe_permute_fusion" "True")
K3_TURBO_ARGS+=("--use_turbo_attention" "False")
K3_TURBO_ARGS+=("--use_turbo_deepep" "False")
K3_TURBO_ARGS+=("--moe_shared_expert_overlap" "False")
K3_TURBO_ARGS+=("--turbo_sync_free_moe_stage" "0")

# Activation recompute, so the larger micro-batches fit (measured: MBS=8 with
# full/block/12 -> 139.6 GB peak). Kept in the SCRIPT, not the convergence-control
# yaml. RECOMPUTE_LAYERS=0 disables it (MBS=16 then OOMs at 276 GB).
RECOMPUTE_ARGS=()
if [ "${RECOMPUTE_LAYERS}" -gt 0 ]; then
    RECOMPUTE_ARGS+=("--recompute_granularity" "full")
    RECOMPUTE_ARGS+=("--recompute_method" "block")
    RECOMPUTE_ARGS+=("--recompute_num_layers" "${RECOMPUTE_LAYERS}")
fi

# Distributed + precision-aware optimizer with bf16 grads/moments (env-gated).
# ON by default (PERF_OPT=1) -- it is part of the measured perf winner; set PERF_OPT=0
# to restore plain fp32 Adam with no distributed optimizer.
# PERF_OPT=1 mirrors the 8L official recipe's MEM_ARGS and the ~211 TFLOP/s
# single-node reference: shard the optimizer state across DP (EP=8 -> only the
# 1.5 B non-expert params actually shard, but distributed-optimizer also stores
# grads/moments in bf16), overlap the grad reduce-scatter and param all-gather with
# compute, and keep bf16 gradient + Adam-moment states while fp32 master weights are
# retained. This is the second half of the MBS=16 story: flydsl removes the eager
# attn-residual up-cast, and the bf16 optimizer state frees the memory the larger
# micro-batch needs. Numerically validated over 450 iters (0 NaN) in the reference.
PERF_OPT=${PERF_OPT:-1}
MEM_ARGS=()
if [ "${PERF_OPT}" = "1" ]; then
    MEM_ARGS+=("--use_distributed_optimizer" "True")
    MEM_ARGS+=("--overlap_grad_reduce" "True")
    MEM_ARGS+=("--overlap_param_gather" "True")
    MEM_ARGS+=("--use_precision_aware_optimizer" "True")
    MEM_ARGS+=("--main_grads_dtype" "bf16")
    MEM_ARGS+=("--exp_avg_dtype" "bf16")
    MEM_ARGS+=("--exp_avg_sq_dtype" "bf16")
fi

FP8_ARGS=()
if [ "$FP8" = "True" ]; then
    FP8_ARGS+=("--fp8" "hybrid")
fi

# NUMA binding for the single-node `primus-cli direct` launch path.
# `direct` has its OWN NUMA control (--numa / --no-numa) that defaults to AUTO=OFF
# and does NOT consult ENABLE_NUMA_BINDING -- that env is only read by
# examples/run_pretrain.sh's torchrun wrapper (the multi-node path). MoE feature 3
# sets ENABLE_NUMA_BINDING=1 to get the per-rank `numactl --cpunodebind --membind`
# wrap that perf/FINDINGS measured as the single biggest lever on this overhead-bound
# shape, so without forwarding it to `direct` as --numa the curve run silently loses
# NUMA binding (torchrun launches unwrapped). Gate on the same env feature 3 sets so
# turning feature 3 off (or ENABLE_NUMA_BINDING=0) still disables it.
DIRECT_ARGS=()
if [ "${ENABLE_NUMA_BINDING:-0}" = "1" ]; then
    DIRECT_ARGS+=("--numa")
fi

# NOTE: no MLA/MTP CLI args (K3 builds MLA from its own specs; multi_latent_attention
# stays false). The distributed + precision-aware optimizer IS now passed via MEM_ARGS
# (PERF_OPT=1, the perf default). Activation recompute (RECOMPUTE_ARGS) IS passed via
# CLI so the micro-batch fits (measured: MBS=16 + full/block/6 + flydsl -> ~210 GB
# peak); it is kept OUT of kimi_k3-BF16-curve.yaml because that file is the convergence
# CONTROL, where recompute would distort the ms/iter and FLOPs figures.

######################### Training Experiments #########################
PRIMUS_TEAM="date-$(date +%Y%m%d)-KimiK3-Curve"
export PRIMUS_TEAM
PRIMUS_USER=${PRIMUS_USER:-user-kimi-k3}
export PRIMUS_USER
export PRIMUS_EXP_NAME="KimiK3_Curve_MI355X_FP8${FP8}_MBS${MBS}_GBS${GBS}_SEQ${SEQ_LENGTH}_TP${TP}_ETP${ETP}_PP${PP}_EP${EP}_CP${CP}_Mock${MOCK_DATA}_Features${FEATURE_TAG}"

# Writable workspace: the checkout may sit on a read-only mount, so default output
# to a writable HOME path (env-overridable). primus reads PRIMUS_WORKSPACE for the
# yaml `workspace`, forwarded into the container as a PRIMUS_-prefixed var.
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/home/$USER/primus_output}
LOG_DIR=$PRIMUS_WORKSPACE/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export LOG_FILE=$LOG_DIR/training.log
mkdir -p "$LOG_DIR"
rm -rf "$LOG_FILE"

# The curve experiment YAML (selects PRIMUS_MODEL=kimi_k3_curve). Real-data by
# design; MOCK_DATA=True below overrides it to NullTokenizer/mock for a portable
# perf run.
export EXP="examples/megatron/configs/MI355X/kimi_k3-BF16-curve.yaml"

echo "--------------------------------" | tee -a "$LOG_FILE"
echo "Begin Training... $(date +%Y%m%d_%H%M%S)" | tee -a "$LOG_FILE"
echo "Training Config: $EXP (PRIMUS_MODEL=${PRIMUS_MODEL}, 24L/hidden2048/32e/top4, ~3.85B)" | tee -a "$LOG_FILE"
echo "NNODES=${NNODES}  TP=${TP} PP=${PP} EP=${EP}  MBS=${MBS} GBS=${GBS} SEQ=${SEQ_LENGTH}  MOCK_DATA=${MOCK_DATA}" | tee -a "$LOG_FILE"
echo "LOG_DIR=${LOG_DIR}" | tee -a "$LOG_FILE"
echo "FEATURE_ARGS=${FEATURE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "K3_TURBO_ARGS=${K3_TURBO_ARGS[*]}" | tee -a "$LOG_FILE"
echo "ATTN_RES_BACKEND=${ATTN_RES_BACKEND}" | tee -a "$LOG_FILE"
echo "MEM_ARGS=${MEM_ARGS[*]}  (PERF_OPT=${PERF_OPT})" | tee -a "$LOG_FILE"
echo "DIRECT_ARGS=${DIRECT_ARGS[*]}  (ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-0})" | tee -a "$LOG_FILE"
echo "RECOMPUTE_ARGS=${RECOMPUTE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

######################### Training Job (single-node: primus-cli direct) #########################
# Mirrors examples/moe_package/run_glm5_4layers_proxy.sh -- drive primus-cli
# directly on ONE node instead of the multi-node run_slurm_pretrain.sh wrapper.
# Same args, same $EXP; only the launch entrypoint differs.
mkdir -p "$LOG_DIR"
./primus-cli direct \
    "${DIRECT_ARGS[@]}" \
    -- train pretrain --config "$EXP" \
    --micro_batch_size "$MBS" \
    --global_batch_size "$GBS" \
    --seq_length "$SEQ_LENGTH" \
    --max_position_embeddings "$SEQ_LENGTH" \
    --tensor_model_parallel_size "$TP" \
    --expert_tensor_parallel_size "$ETP" \
    --pipeline_model_parallel_size "$PP" \
    --expert_model_parallel_size "$EP" \
    --context_parallel_size "$CP" \
    --optimizer "$OPTIMIZER" \
    --mock_data "$MOCK_DATA" \
    --attn_res_backend "$ATTN_RES_BACKEND" \
    "${FEATURE_ARGS[@]}" \
    "${K3_TURBO_ARGS[@]}" \
    "${MEM_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
