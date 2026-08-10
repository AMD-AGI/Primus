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
#   At EP=8 this fits comfortably on 8 x 288 GB (the debug bring-up peaked at
#   ~14.6 GB), so NO distributed optimizer / recompute is needed.
#
# PERFORMANCE
#   In PERF mode (mbs 16, seq 2048, fla + Turbo, bf16) curve reaches ~190
#   TFLOP/s/GPU single-node -- close to the official 8-layer 4-node number, which
#   is what makes it a useful throughput proxy without a multi-node reservation.
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
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
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

FP8_ARGS=()
if [ "$FP8" = "True" ]; then
    FP8_ARGS+=("--fp8" "hybrid")
fi

# NOTE: no MLA/MTP CLI args (K3 builds MLA from its own specs; multi_latent_attention
# stays false), no distributed-optimizer / recompute args (3.85 B fits on one node).

######################### Training Experiments #########################
PRIMUS_TEAM="date-$(date +%Y%m%d)-KimiK3-Curve"
export PRIMUS_TEAM
PRIMUS_USER=${PRIMUS_USER:-user-kimi-k3}
export PRIMUS_USER
export PRIMUS_EXP_NAME="KimiK3_Curve_MI355X_FP8${FP8}_MBS${MBS}_GBS${GBS}_SEQ${SEQ_LENGTH}_TP${TP}_ETP${ETP}_PP${PP}_EP${EP}_CP${CP}_Mock${MOCK_DATA}_Features${FEATURE_TAG}"

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
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
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

######################### Training Job (single-node: primus-cli direct) #########################
# Mirrors examples/moe_package/run_glm5_4layers_proxy.sh -- drive primus-cli
# directly on ONE node instead of the multi-node run_slurm_pretrain.sh wrapper.
# Same args, same $EXP; only the launch entrypoint differs.
mkdir -p "output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME"
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
    "${FP8_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
