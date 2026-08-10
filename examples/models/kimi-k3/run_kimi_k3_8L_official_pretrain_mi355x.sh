#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Kimi K3 -- 8-LAYER OFFICIAL-WIDTH BF16 pretraining, MULTI-NODE MI355X.
#
# WHAT THIS RUNS
#   The Kimi K3 text backbone at the RELEASED per-tensor width (every dim ==
#   moonshotai/Kimi-K3: hidden 7168, 96 heads, 896 routed + 2 shared experts,
#   top-16, MoE FFN 3072, Stable-Latent-MoE latent 3584 = hidden/2, MLA q_head_dim
#   192), with only the DEPTH compressed from the production 93 layers to 8
#   (1 dense + 7 MoE). This is the shape our 4-node throughput tuning measured at
#   ~191 TFLOP/s/GPU, 0-NaN, ~96% of 288 GB HBM on 4 x 8 MI355X (bf16).
#
#   The 8-layer depth is encoded by the kimi_k3_8L_official preset
#   (primus/configs/models/megatron/kimi_k3_8L_official.yaml), which extends the
#   full 93-layer kimi_k3.yaml (~2.8 T params) and overrides only the three
#   depth-dependent keys:
#       num_layers 8
#       linear_attention_freq "([1]*3+[0])*2"   # = [1,1,1,0,1,1,1,0]  (6 KDA / 2 full)
#       moe_layer_freq        "([0]*1+[1]*7)"   # = [0,1,1,1,1,1,1,1]  (layer 0 dense)
#   Both patterns are the first-8 truncation of the production patterns
#   ("([1]*3+[0])*22+[1]*3+[0]*2" and "([0]*1+[1]*92)"), so the interleave under
#   test is faithful. They live in the preset (not CLI overrides) so num_layers and
#   the two length-checked patterns can never desync.
#
# PARAMETER COUNT (derived from kimi_k3.yaml + kimi_k3_base.yaml, not measured)
#   Each routed expert is a SwiGLU MLP living in the 3584-dim Stable-Latent-MoE
#   space: 2*3584*3072 (gate+up) + 3072*3584 (down) ~= 33.0 M params.
#     routed experts   896 * 7 MoE layers * 33.0 M           ~= 207 B   (dominant)
#     untied embed+head 2 * 163840 * 7168                    ~= 2.35 B
#     KDA (6 layers)    6 * (q,k,v,gate,out = 5 * 7168*12288) ~= 2.64 B
#     shared experts    7 * (2*7168*6144 + 6144*7168)        ~= 0.92 B
#     dense layer 0     2*7168*33792 + 33792*7168            ~= 0.73 B
#     latent proj       7 * (7168*3584 + 3584*7168)          ~= 0.36 B
#     MLA (2 layers)    2 * ~144 M                            ~= 0.29 B
#     -------------------------------------------------------------------
#     TOTAL                                                   ~= 215 B
#   Per GPU at EP=8 (TP=1/PP=1): the 896 experts shard across EP, the rest is
#   replicated -> 207 B / 8 (~26 B) + ~7 B replicated ~= 33 B params/GPU.
#   Activated / token (top-16): routed 16 * 33.0 M * 7 ~= 3.7 B, plus the
#   always-on attention (KDA is FULL width, no GQA -> heavy), shared experts,
#   dense layer and LM head ~= 6 B  ->  ~10 B activated/token.
#
# WHY >= 4 NODES
#   The 896-expert optimizer state only shards across expert-DP = DP / EP, so you
#   need DP/EP >= 2 (i.e. >= 2 nodes at EP=8) for ANY optimizer sharding, and the
#   measured 191 TFLOP/s headroom needs 4: even with the distributed +
#   precision-aware optimizer, bf16 grad/moments and recompute full/block/8, the
#   footprint sits at ~96% of 288 GB on 4 nodes. The memory recipe is folded into
#   kimi_k3-BF16-8L-official.yaml (distributed + precision-aware optimizer, bf16
#   grads/moments, recompute full/block/8); this launcher also re-asserts it via
#   MEM_ARGS below so the knobs are visible at the call site.
#
# LAUNCH PATH (the PR's own primus-cli slurm entry -- NOT run_slurm_pretrain.sh)
#   ./primus-cli slurm srun -N N -p amd-spur --reservation=... -- container --
#     train pretrain --config $EXP
#   --(srun)--> runner/primus-cli-slurm-entry.sh (Spur-safe pure-bash nodelist
#     fallback, no `srun --export`, no `scontrol show hostnames`) -->
#     primus-cli-container.sh --(docker)--> primus-cli-direct.sh --> torchrun
#     python primus/cli/main.py train pretrain --config $EXP <overrides>
#   run_slurm_pretrain.sh is avoided on purpose: this cluster's SLURM rejects its
#   `srun --export ALL` and lacks `scontrol show hostnames`.
#
# USAGE
#   NNODES=4 bash examples/models/kimi-k3/run_kimi_k3_8L_official_pretrain_mi355x.sh
#   Override any knob from the environment, e.g. MBS=1 SEQ_LENGTH=4096 TRAIN_ITERS=20 ...
#   Pin the allocation with SLURM_NODELIST="nodeA,nodeB,..." (forwarded to srun -w).
###############################################################################

######################### Training Docker and Variables #########################
# fla (flash-linear-attention) is not in the base image; the megatron pretrain
# hook installs it (runner/helpers/hooks/train/pretrain/.../requirements-megatron.txt).
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/primus:v26.4"}
export CLEAN_DOCKER_CONTAINER=${CLEAN_DOCKER_CONTAINER:-1}
export SKIP_TRAIN=${SKIP_TRAIN:-0}

######################### Training Environment Variables #########################
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
# HSA_NO_SCRATCH_RECLAIM=0 is the fla fix (matches run_pretrain.sh's own default,
# whose comment calls it the fla fix); =1 regressed fla throughput, so keep 0.
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}

# Kimi K3 KDA backend selection (self-contained; run_pretrain.sh no longer sets it):
#   PRIMUS_KDA_BACKEND=fla -> kda_backend=fla, the fused fla Triton chunk kernel
#     (config field, forwarded into the container because it is PRIMUS_-prefixed).
#   K3P_KDA_CONV=fla       -> fla causal_conv1d replaces nn.Conv1d in the KDA
#     depthwise conv. It is read directly from the environment inside the
#     container by kimi_delta_attention.py, so it must be present in the container
#     env (run_local_pretrain.sh forwards PRIMUS_*, NCCL_*, ENABLE_NUMA_BINDING,
#     ...; if your site launcher does not forward K3P_*, export it there too).
export PRIMUS_KDA_BACKEND=${PRIMUS_KDA_BACKEND:-fla}
export K3P_KDA_CONV=${K3P_KDA_CONV:-fla}
# Unified KDA backend selector (supersedes the two above): fla = fused chunk kernel
# + fla causal_conv1d. kimi_k3-BF16-8L-official.yaml reads it as PRIMUS_ATTN_BACKEND.
export PRIMUS_ATTN_BACKEND=${PRIMUS_ATTN_BACKEND:-fla}

export NNODES=${NNODES:-4}

# Multi-node cluster networking -- PROVEN ionic RoCE values for this MI355X cluster
# (env VALUES copied from the validated 8L perf run). Forwarded into the training
# container both by the .primus.yaml env whitelist AND explicitly as --env in the
# launch below (belt-and-suspenders, in case srun does not propagate the env).
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens3}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens3}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-20}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-300}

# SLURM allocation targets for the `primus-cli slurm` launch below.
# IMPORTANT: this cluster's scheduler (Spur) does NOT honor srun's --reservation
# FLAG -- it reads the reservation from the ENVIRONMENT. So export BOTH names and
# request shared (non-exclusive) access; the reserved nodes are shared with other
# users, so --exclusive would never get an allocation. These are the exact env
# vars every working submission on this cluster used.
export SLURM_PARTITION=${SLURM_PARTITION:-amd-spur}
export SBATCH_RESERVATION=${SBATCH_RESERVATION:-primus-deepseek-v4-reserved}
export SLURM_RESERVATION=${SLURM_RESERVATION:-$SBATCH_RESERVATION}
export SLURM_EXCLUSIVE=${SLURM_EXCLUSIVE:-0}
export SLURM_TIME=${SLURM_TIME:-04:00:00}

# Node-local Triton cache (avoids the shared-NFS fla causal_conv1d 'hsaco' KeyError).
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_k3_8L}

# Select the 8-layer official-width preset (extends kimi_k3.yaml, num_layers=8).
export PRIMUS_MODEL=${PRIMUS_MODEL:-kimi_k3_8L_official}

######################### Training Config (measured 4-node best) #########################
export NUM_LAYERS=${NUM_LAYERS:-8}                 # display label; depth is fixed at 8 by the preset
export MBS=${MBS:-2}
export GBS=${GBS:-128}                             # must be a multiple of MBS * DP (DP=32 at 4 nodes)
export SEQ_LENGTH=${SEQ_LENGTH:-7168}
export TP=${TP:-1}
export ETP=${ETP:-1}
export PP=${PP:-1}
export EP=${EP:-8}
export CP=${CP:-1}
export OPTIMIZER=${OPTIMIZER:-adam}
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-8}     # full/block over all 8 layers
export FP8=${FP8:-False}                           # False = bf16 (bf16 matched fp8 here)
export TRAIN_ITERS=${TRAIN_ITERS:-50}

# Depth (num_layers 8 + the attention/MoE interleave patterns) is encoded by the
# kimi_k3_8L_official preset, so no --num_layers / --*_freq CLI slicing is needed.

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
        # NUMA binding: worth ~+28% on K3, only chooses NUMA node for CPU/host mem.
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

# K3-specific Turbo settings that are NOT in the feature legend above.
#   ON : the other two of the three "measured winner" kernels (grouped GEMM is
#        feature 2). Both are checkpoint-safe at EP=8.
#   OFF (explicit guard): the flags below were dropped from the MoE_Features
#        legend/case above because they are inapplicable or unsafe for K3; they are
#        pinned off here so they can never be turned on --
#          use_turbo_attention        NO-OP: K3 attention is KDA (fla kernels) plus
#                                      its own KimiK3MLASelfAttention, neither of
#                                      which the Turbo flash-attn path touches.
#          use_turbo_deepep           breaks K3 numerics (first backward went
#                                      non-finite) even after the shape fix.
#          moe_shared_expert_overlap  moe_layer.py asserts NOT-overlap on the live
#                                      K3 latent path -> dies at forward.
#          turbo_sync_free_moe_stage  stage>=2 force-enables DeepEP; stage 1 is
#                                      unvalidated for K3's Stable-Latent-MoE.
K3_TURBO_ARGS=()
ensure_primus_turbo
K3_TURBO_ARGS+=("--use_turbo_rms_norm" "True")
K3_TURBO_ARGS+=("--moe_permute_fusion" "True")
K3_TURBO_ARGS+=("--use_turbo_attention" "False")
K3_TURBO_ARGS+=("--use_turbo_deepep" "False")
K3_TURBO_ARGS+=("--moe_shared_expert_overlap" "False")
K3_TURBO_ARGS+=("--turbo_sync_free_moe_stage" "0")

# 896-expert MEMORY recipe (REQUIRED at scale). kimi_k3-BF16-8L-official.yaml
# already sets these; this block re-asserts them at the call site. Distributed +
# precision-aware optimizer with bf16 grads/moments; fp32 master weights are kept.
MEM_ARGS=(
    "--use_distributed_optimizer" "True"
    "--overlap_grad_reduce" "True"
    "--overlap_param_gather" "True"
    "--use_precision_aware_optimizer" "True"
    "--main_grads_dtype" "bf16"
    "--exp_avg_dtype" "bf16"
    "--exp_avg_sq_dtype" "bf16"
)

RECOMPUTE_ARGS=()
if [ "$RECOMPUTE_LAYERS" -gt 0 ]; then
    RECOMPUTE_ARGS+=("--recompute_granularity" "full")
    RECOMPUTE_ARGS+=("--recompute_method" "block")
    RECOMPUTE_ARGS+=("--recompute_num_layers" "${RECOMPUTE_LAYERS}")
fi

FP8_ARGS=()
if [ "$FP8" = "True" ]; then
    FP8_ARGS+=("--fp8" "hybrid")
fi

# NOTE: no MLA/MTP CLI args. K3 builds its MLA from its own module specs and
# multi_latent_attention MUST stay false (kimi_k3_base.yaml); MTP is off by
# default (num_nextn_predict_layers: null).

######################### Training Experiments #########################
PRIMUS_TEAM="date-$(date +%Y%m%d)-KimiK3-8L-Official"
export PRIMUS_TEAM
PRIMUS_USER=${PRIMUS_USER:-user-kimi-k3}
export PRIMUS_USER
export PRIMUS_EXP_NAME="KimiK3_8L_Official_MI355X_FP8${FP8}_MBS${MBS}_GBS${GBS}_SEQ${SEQ_LENGTH}_L${NUM_LAYERS}_REC${RECOMPUTE_LAYERS}_TP${TP}_ETP${ETP}_PP${PP}_EP${EP}_CP${CP}_NN${NNODES}_Features${FEATURE_TAG}"

# Writable workspace: the checkout may sit on a read-only mount (e.g. /shared_nfs
# from the submit node), so default output to a writable HOME path (env-overridable).
# primus reads PRIMUS_WORKSPACE for the yaml `workspace` (PRIMUS_-prefixed -> also
# forwarded into the container).
export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/home/$USER/primus_output}
LOG_DIR=$PRIMUS_WORKSPACE/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export LOG_FILE=$LOG_DIR/training.log
mkdir -p "$LOG_DIR"
rm -rf "$LOG_FILE"

# The official 8-layer experiment YAML (selects PRIMUS_MODEL=kimi_k3_8L_official,
# mock data + NullTokenizer, the 896-expert memory recipe, sigmoid+noaux_tc
# router, situ, MLA/KDA specs).
export EXP="examples/megatron/configs/MI355X/kimi_k3-BF16-8L-official.yaml"

echo "--------------------------------" | tee -a "$LOG_FILE"
echo "Begin Training... $(date +%Y%m%d_%H%M%S)" | tee -a "$LOG_FILE"
echo "Training Config: $EXP (PRIMUS_MODEL=${PRIMUS_MODEL}, num_layers=${NUM_LAYERS}, official width)" | tee -a "$LOG_FILE"
echo "NNODES=${NNODES}  TP=${TP} PP=${PP} EP=${EP}  MBS=${MBS} GBS=${GBS} SEQ=${SEQ_LENGTH}" | tee -a "$LOG_FILE"
echo "LOG_DIR=${LOG_DIR}" | tee -a "$LOG_FILE"
echo "FEATURE_ARGS=${FEATURE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "K3_TURBO_ARGS=${K3_TURBO_ARGS[*]}" | tee -a "$LOG_FILE"
echo "MEM_ARGS=${MEM_ARGS[*]}" | tee -a "$LOG_FILE"
echo "RECOMPUTE_ARGS=${RECOMPUTE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

######################### Training Job (primus-cli slurm -> container) #########################
# Extra container mounts. --volume is a cumulative passthrough option accepted by
# runner/primus-cli-container.sh, so the K3 script can inject mounts WITHOUT editing
# any shared script:
#   * whole /shared_nfs so the checkout (and any data) is visible on every node;
#   * the writable HOME workspace;
#   * host libionic (ionic RoCE ABI-4 provider) over the container's own provider
#     lib so NCCL IB init sees the ABI the ionic NICs need. NOTE: if the container's
#     path is a symlink to a versioned .so, a plain mount-over-name may not fully
#     swap it; the proven fallback is an in-container `cp` over `readlink -f` of the
#     provider, which needs a pre-run hook (primus-cli-direct `--patch`). See report.
CONTAINER_VOL_ARGS=("--volume" "/shared_nfs:/shared_nfs" "--volume" "/home/$USER:/home/$USER")
# Destination path inside the container (where libibverbs looks for the provider).
LIBIONIC_HOST=${LIBIONIC_HOST:-/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so}
# Source file to bind over it. Two traps, both of which fail SILENTLY and cost ~12x
# throughput (measured ~16 TFLOP/s/GPU instead of ~190 on 4 nodes) with *identical*
# numerics, so they read like a compute regression rather than a networking one:
#   1. This script runs on the SUBMIT node, but the mount is consumed on the COMPUTE
#      node. On a CPU-only submit host the provider does not exist, so a plain
#      `[ -e ]` guard here skips the mount for every node in the job.
#   2. The provider path is a SYMLINK to a versioned .so that lives outside the
#      mounted name, so bind-mounting the link itself lands a dangling link.
# Either way libibverbs loads no provider, `ibv_devices` is empty, NCCL_IB_HCA
# matches nothing and NCCL falls back to TCP over the socket interface.
# So: dereference, and when the submit host has no copy, point LIBIONIC_SRC at one
# staged on a shared mount (cp -L from any compute node).
LIBIONIC_SRC=${LIBIONIC_SRC:-$(readlink -f "$LIBIONIC_HOST" 2>/dev/null || echo "$LIBIONIC_HOST")}
if [ -f "$LIBIONIC_SRC" ]; then
    CONTAINER_VOL_ARGS+=("--volume" "${LIBIONIC_SRC}:${LIBIONIC_HOST}:ro")
else
    echo "WARNING: ionic RoCE provider not found at ${LIBIONIC_SRC}; NCCL will fall" \
         "back to TCP and throughput will collapse. Stage it with 'cp -L" \
         "${LIBIONIC_HOST} <shared path>' on a compute node and re-run with" \
         "LIBIONIC_SRC=<shared path>." | tee -a "$LOG_FILE"
fi

# The verbs char devices themselves; without them the provider above still
# enumerates nothing.
CONTAINER_DEV_ARGS=()
[ -e /dev/infiniband ] && CONTAINER_DEV_ARGS+=("--device" "/dev/infiniband")

# Env forwarded explicitly into the container (robust even if srun does not
# propagate the submit environment; PRIMUS_*/NCCL_* also auto-forward).
CONTAINER_ENV_ARGS=(
    "--env" "USING_AINIC=${USING_AINIC}"
    "--env" "NCCL_IB_HCA=${NCCL_IB_HCA}"
    "--env" "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    "--env" "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
    "--env" "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
    "--env" "NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT}"
    "--env" "NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT}"
    "--env" "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM}"
    "--env" "PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF}"
    "--env" "ENABLE_NUMA_BINDING=${ENABLE_NUMA_BINDING:-1}"
    "--env" "HSA_KERNARG_POOL_SIZE=${HSA_KERNARG_POOL_SIZE:-12582912}"
    "--env" "GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES}"
    "--env" "NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3}"
    "--env" "PRIMUS_KDA_BACKEND=${PRIMUS_KDA_BACKEND}"
    "--env" "PRIMUS_ATTN_BACKEND=${PRIMUS_ATTN_BACKEND}"
    "--env" "K3P_KDA_CONV=${K3P_KDA_CONV}"
    "--env" "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    "--env" "PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE}"
)

# NOTE: NO --exclusive (reserved nodes are shared with other users; exclusive
# never allocates) and NO -p (the reservation determines the partition; SLURM_*
# env above carries it). The reservation is applied via the SBATCH_RESERVATION /
# SLURM_RESERVATION env exported above; --reservation is kept only as a hint.
# stdout goes to per-node files under the writable HOME workspace, because the
# submit CWD may be a read-only /shared_nfs checkout (Spur writes spur-<jobid>.out
# into CWD otherwise). %j = job id, %t = task/node rank -> last node is ..._r3.out.
SLURM_FLAGS=("-N" "$NNODES" "-t" "$SLURM_TIME" "--output=${PRIMUS_WORKSPACE}/8L_%j_r%t.out")
[ -n "${SBATCH_RESERVATION:-}" ] && SLURM_FLAGS+=("--reservation=${SBATCH_RESERVATION}")
# Node pinning. The reservation is env-driven on Spur, but SLURM_NODELIST is NOT:
# exporting it is silently ignored and the job lands on an arbitrary subset of the
# reservation. srun's -w/--nodelist flag IS honored, so forward the env var as a
# flag. Give the scheduler's full node names (e.g. "nodeA,nodeB"), not bare
# suffixes, and keep NNODES consistent with the list length.
[ -n "${SLURM_NODELIST:-}" ] && SLURM_FLAGS+=("-w" "$SLURM_NODELIST")

./primus-cli slurm srun "${SLURM_FLAGS[@]}" \
    -- container --shm-size 64g "${CONTAINER_VOL_ARGS[@]}" "${CONTAINER_DEV_ARGS[@]}" "${CONTAINER_ENV_ARGS[@]}" \
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
    --mock_data True \
    "${FEATURE_ARGS[@]}" \
    "${K3_TURBO_ARGS[@]}" \
    "${MEM_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
