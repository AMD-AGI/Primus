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
#   The model + optimizer fit comfortably on 8 x 288 GB (no distributed optimizer
#   needed). Activation recompute IS used for the MBS=16 perf recipe below
#   (full/block/12 -> 224.71 GB peak); without it MBS=16 OOMs (~276 GB).
#
# PERFORMANCE (measured winner; see the perf-config header in the repo history)
#   MBS=16 / seq 2048 / recompute full/block/12 / Turbo grouped_gemm + rms_norm +
#   permute / ENABLE_NUMA_BINDING=1 (fla, bf16) reaches ~130.7 TFLOP/s/GPU
#   single-node -- 2361 ms/iter, 224.71 GB peak of 288, measured over a FULL 450
#   iterations (0 NaN, 0 skipped). ENABLE_NUMA_BINDING alone is worth +28.2%.
#   NOTE: this shape needs ~150 iterations to reach steady state (2789 ms @ iter20
#   -> 2353 @ iter200), so a short run (<= 50 iters) legitimately reads BELOW 130.7
#   (~9% low at 20 iters) -- that is warmup, not a regression. (There is no
#   ~190 TFLOP/s single-node number; ~191 is the 8L official 4-node figure.)
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
# USAGE (run from the Primus repo root)
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
export MBS=${MBS:-16}
export GBS=${GBS:-128}                             # 128 = MBS(16) * DP(8) -> 1 micro-batch, no grad accum
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
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-12}    # winner: full/block/12 (224.71GB); 8=271GB(riskier); 0=off(mbs16 OOMs)
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-12}    # winner: full/block/12 (224.71GB); 8=271GB(riskier); 0=off(mbs16 OOMs)

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

# Activation recompute so MBS=16 fits (measured winner: full/block/12 -> 224.71 GB
# peak). Kept in the SCRIPT, not the convergence-control yaml. RECOMPUTE_LAYERS=0
# disables it (mbs16 then OOMs at ~276 GB).
RECOMPUTE_ARGS=()
if [ "${RECOMPUTE_LAYERS}" -gt 0 ]; then
    RECOMPUTE_ARGS+=("--recompute_granularity" "full")
    RECOMPUTE_ARGS+=("--recompute_method" "block")
    RECOMPUTE_ARGS+=("--recompute_num_layers" "${RECOMPUTE_LAYERS}")
fi

# Activation recompute so MBS=16 fits (measured winner: full/block/12 -> 224.71 GB
# peak). Kept in the SCRIPT, not the convergence-control yaml. RECOMPUTE_LAYERS=0
# disables it (mbs16 then OOMs at ~276 GB).
RECOMPUTE_ARGS=()
if [ "${RECOMPUTE_LAYERS}" -gt 0 ]; then
    RECOMPUTE_ARGS+=("--recompute_granularity" "full")
    RECOMPUTE_ARGS+=("--recompute_method" "block")
    RECOMPUTE_ARGS+=("--recompute_num_layers" "${RECOMPUTE_LAYERS}")
fi

FP8_ARGS=()
if [ "$FP8" = "True" ]; then
    FP8_ARGS+=("--fp8" "hybrid")
fi

# NOTE: no MLA/MTP CLI args (K3 builds MLA from its own specs; multi_latent_attention
# stays false) and no distributed optimizer (3.85 B fits on one node). Activation
# recompute (RECOMPUTE_ARGS) IS passed via CLI so MBS=16 fits (~224.71 GB peak);
# it is kept OUT of kimi_k3-BF16-curve.yaml because that file is the convergence
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
echo "RECOMPUTE_ARGS=${RECOMPUTE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

######################### Training Job (single-node: primus-cli direct) #########################
# Mirrors examples/moe_package/run_glm5_4layers_proxy.sh -- drive primus-cli
# directly on ONE node instead of the multi-node run_slurm_pretrain.sh wrapper.
# Same args, same $EXP; only the launch entrypoint differs.
mkdir -p "$LOG_DIR"
./primus-cli direct \
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
    "${FEATURE_ARGS[@]}" \
    "${K3_TURBO_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
