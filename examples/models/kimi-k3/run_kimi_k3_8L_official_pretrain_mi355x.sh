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
#   The PR ships a single official-width preset, primus/configs/models/megatron/
#   kimi_k3.yaml, which defaults to the full 93-layer production stack. That stack
#   is ~2.8 T parameters and does not fit on a small reservation, so this launcher
#   slices it to 8 layers via CLI overrides:
#       --num_layers 8
#       --linear_attention_freq "([1]*3+[0])*2"   # = [1,1,1,0,1,1,1,0]  (6 KDA / 2 full)
#       --moe_layer_freq        "([0]*1+[1]*7)"   # = [0,1,1,1,1,1,1,1]  (layer 0 dense)
#   Both patterns are the first-8 truncation of the production patterns
#   ("([1]*3+[0])*22+[1]*3+[0]*2" and "([0]*1+[1]*92)"), so the interleave under
#   test is faithful. They MUST be overridden together with num_layers: K3's
#   normalize_linear_attention_freq / moe_layer_freq hard-error unless the
#   pattern length equals num_layers (kimi_k3_transformer_config.py:115,
#   kimi_k3_layer_specs.py:123).
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
#   dense layer and LM head ~= 5 B  ->  ~8-9 B activated/token.
#
# WHY >= 4 NODES
#   The 896-expert optimizer state only shards across expert-DP = DP / EP, so you
#   need DP/EP >= 2 (i.e. >= 2 nodes at EP=8) for ANY optimizer sharding, and the
#   measured 191 TFLOP/s headroom needs 4: even with the distributed +
#   precision-aware optimizer, bf16 grad/moments and recompute full/block/8, the
#   footprint sits at ~96% of 288 GB on 4 nodes. kimi_k3-BF16-pretrain.yaml ships
#   the optimizer knobs OFF (it must also run as a 1-GPU smoke), so this launcher
#   turns them on -- see MEM_ARGS below.
#
# LAUNCH PATH (this IS the primus-cli path)
#   run_slurm_pretrain.sh --(srun)--> run_local_pretrain.sh --(docker)-->
#   run_pretrain.sh --> python primus/cli/main.py train pretrain --config $EXP <overrides>
#
# USAGE
#   NNODES=4 bash examples/models/kimi-k3/run_kimi_k3_8L_official_pretrain_mi355x.sh
#   Override any knob from the environment, e.g. MBS=1 SEQ_LENGTH=4096 TRAIN_ITERS=20 ...
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
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-1}

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

# Multi-node cluster wiring -- ADJUST per cluster (these are ionic/Vultr examples).
export NNODES=${NNODES:-4}
export USING_AINIC=${USING_AINIC:-0}
# export NCCL_IB_HCA="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
# export NCCL_SOCKET_IFNAME="enp193s0f1np1"
# export GLOO_SOCKET_IFNAME="enp193s0f1np1"
# export SLURM_PARTITION=${SLURM_PARTITION:-}
# export SLURM_NODELIST=${SLURM_NODELIST:-}

# Per-rank LOCAL Triton cache: the fla causal_conv1d 'hsaco' KeyError only shows
# up under a shared-NFS Triton cache, so keep it node-local.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_k3_8L}

# Select the official-width preset (93-layer file, sliced to 8 layers below).
export PRIMUS_MODEL=${PRIMUS_MODEL:-kimi_k3}

######################### Training Config (measured 4-node best) #########################
export NUM_LAYERS=${NUM_LAYERS:-8}                 # 1 dense + 7 MoE, official width
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

# 8-layer interleave patterns (first-8 slice of the production patterns). Length
# MUST equal NUM_LAYERS or K3 raises at config build.
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-"([1]*3+[0])*2"}
export MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-"([0]*1+[1]*7)"}

# MoE_Features legend (K3-applicable subset of examples/moe_package/*):
#   0 baseline | 2 turbo grouped GEMM | 3 loss fusion | 6 NUMA binding | 7 manual GC
# K3 measured winner: grouped GEMM (+RMSNorm/permute below) + loss fusion + NUMA +
# manual GC. The upstream options 1 (turbo attention), 4 (DeepEP), 5 (sync-free
# MoE) and 8 (UCCL-EP) are intentionally NOT offered here -- they are NO-OP or
# unsafe for K3 (see the K3_TURBO_ARGS note below), so they were dropped from both
# this legend and the case handler and cannot be enabled.
MoE_Features=(2 3 6 7)

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
    2)
        ensure_primus_turbo
        FEATURE_ARGS+=("--use_turbo_grouped_gemm" "True")
        ;;
    3)
        FEATURE_ARGS+=("--cross_entropy_fusion_impl" "te")
        FEATURE_ARGS+=("--cross_entropy_loss_fusion" "True")
        ;;
    6)
        # NUMA binding: worth ~+28% on K3, only chooses NUMA node for CPU/host mem.
        export ENABLE_NUMA_BINDING=1
        export HSA_KERNARG_POOL_SIZE=12582912
        ;;
    7)
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

# 896-expert MEMORY recipe (REQUIRED at scale; kimi_k3-BF16-pretrain.yaml ships
# these OFF so it can also run as a 1-GPU smoke). Distributed + precision-aware
# optimizer with bf16 grads/moments; fp32 master weights are still kept.
MEM_ARGS=(
    "--use_distributed_optimizer" "True"
    "--overlap_grad_reduce" "True"
    "--overlap_param_gather" "True"
    "--use_precision_aware_optimizer" "True"
    "--main_grads_dtype" "bf16"
    "--exp_avg_dtype" "bf16"
    "--exp_avg_sq_dtype" "bf16"
)

# Depth override: 8-layer official-width slice of the 93-layer kimi_k3 preset.
DEPTH_ARGS=(
    "--num_layers" "$NUM_LAYERS"
    "--linear_attention_freq" "$LINEAR_ATTENTION_FREQ"
    "--moe_layer_freq" "$MOE_LAYER_FREQ"
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

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export LOG_FILE=$LOG_DIR/training.log
mkdir -p "$LOG_DIR"
rm -rf "$LOG_FILE"

# The official-width experiment YAML (selects PRIMUS_MODEL=kimi_k3, mock data,
# NullTokenizer, sigmoid+noaux_tc router, situ, MLA/KDA specs).
export EXP="examples/megatron/configs/MI355X/kimi_k3-BF16-pretrain.yaml"

echo "--------------------------------" | tee -a "$LOG_FILE"
echo "Begin Training... $(date +%Y%m%d_%H%M%S)" | tee -a "$LOG_FILE"
echo "Training Config: $EXP (PRIMUS_MODEL=${PRIMUS_MODEL}, num_layers=${NUM_LAYERS}, official width)" | tee -a "$LOG_FILE"
echo "NNODES=${NNODES}  TP=${TP} PP=${PP} EP=${EP}  MBS=${MBS} GBS=${GBS} SEQ=${SEQ_LENGTH}" | tee -a "$LOG_FILE"
echo "LOG_DIR=${LOG_DIR}" | tee -a "$LOG_FILE"
echo "FEATURE_ARGS=${FEATURE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "K3_TURBO_ARGS=${K3_TURBO_ARGS[*]}" | tee -a "$LOG_FILE"
echo "MEM_ARGS=${MEM_ARGS[*]}" | tee -a "$LOG_FILE"
echo "DEPTH_ARGS=${DEPTH_ARGS[*]}" | tee -a "$LOG_FILE"
echo "RECOMPUTE_ARGS=${RECOMPUTE_ARGS[*]}" | tee -a "$LOG_FILE"
echo "FP8_ARGS=${FP8_ARGS[*]}" | tee -a "$LOG_FILE"
echo "--------------------------------" | tee -a "$LOG_FILE"

######################### Training Job #########################
bash ./examples/run_slurm_pretrain.sh \
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
    "${DEPTH_ARGS[@]}" \
    "${FEATURE_ARGS[@]}" \
    "${K3_TURBO_ARGS[@]}" \
    "${MEM_ARGS[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${FP8_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee -a "$LOG_FILE"
