#!/bin/bash
#
# DeepSeek-V4-Flash pretraining on AMD Instinct MI355X.
#
# Six families of optimization, each behind one switch, all on by default, so
# running this script with nothing but an image reproduces the best measured
# configuration:
#
#   PRIMUS_OPT_FUSION     (1) small kernel fusions (RMSNorm, RoPE, Sinkhorn,
#                             hyper-connections, compressor pooling, indexer
#                             tail, router tail, grouped-weight stack, plus the
#                             Megatron permute / CE / grad-accum fusions)
#   PRIMUS_OPT_ATTENTION  (2) attention backend: FlyDSL sparse-MLA ("turbo")
#   PRIMUS_OPT_DEEPEP     (3) DeepEP dispatch/combine + Turbo grouped GEMM
#   PRIMUS_OPT_SYNC_FREE  (4) sync-free MoE stages
#   PRIMUS_OPT_MEGA_MOE   (5) MegaMoE -- a REPLACEMENT for (3) and (4)
#   PRIMUS_OPT_LAYOUT     (6) pipeline layout + recompute depth
#
# Set any of them to 0 to turn that family off. Individual knobs inside a family
# can still be overridden one at a time: every export below is
# `${VAR:-<value from the switch>}`, so the environment always wins.
#
# -----------------------------------------------------------------------------
# Reproducing the speedup curve
# -----------------------------------------------------------------------------
# Each step keeps everything from the steps above it and adds one change. Set
# these once, then run any step below unchanged:
#
#   export DOCKER_IMAGE=<primus-image>
#   export SLURM_ALLOC_JOB_ID=<jobid>   # optional: join an allocation you hold
#   export TRAIN_ITERS=10
#   cd <primus-root>
#
# The throughput after each command is what a 4-node MI355X run measured at
# 10 iterations with router load balancing forced to uniform (TFLOP/s per GPU).
#
# step 0 -- baseline, every optimization off
#   PRIMUS_OPT_FUSION=0 PRIMUS_OPT_ATTENTION=0 PRIMUS_OPT_DEEPEP=0 \
#   PRIMUS_OPT_SYNC_FREE=0 PRIMUS_OPT_MEGA_MOE=0 PRIMUS_OPT_LAYOUT=0 \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 1 -- + small kernel fusions
#   PRIMUS_OPT_FUSION=1 PRIMUS_OPT_ATTENTION=0 PRIMUS_OPT_DEEPEP=0 \
#   PRIMUS_OPT_SYNC_FREE=0 PRIMUS_OPT_MEGA_MOE=0 PRIMUS_OPT_LAYOUT=0 \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 2 -- attention to Gluon. The ATTENTION switch means FlyDSL, so the
#           Gluon backend is named explicitly on both paths.
#   PRIMUS_OPT_FUSION=1 PRIMUS_OPT_ATTENTION=0 PRIMUS_OPT_DEEPEP=0 \
#   PRIMUS_OPT_SYNC_FREE=0 PRIMUS_OPT_MEGA_MOE=0 PRIMUS_OPT_LAYOUT=0 \
#   USE_V4_ATTENTION_BACKEND=gluon_v3 USE_V4_CSA_ATTENTION_BACKEND=gluon_v3 \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 3 -- attention to FlyDSL sparse-MLA
#   PRIMUS_OPT_FUSION=1 PRIMUS_OPT_ATTENTION=1 PRIMUS_OPT_DEEPEP=0 \
#   PRIMUS_OPT_SYNC_FREE=0 PRIMUS_OPT_MEGA_MOE=0 PRIMUS_OPT_LAYOUT=0 \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 4 -- + DeepEP alone. The DEEPEP switch also turns on the Turbo grouped
#           GEMM, so the GEMM is held off to measure the two apart.
#   PRIMUS_OPT_FUSION=1 PRIMUS_OPT_ATTENTION=1 PRIMUS_OPT_DEEPEP=1 \
#   PRIMUS_OPT_SYNC_FREE=0 PRIMUS_OPT_MEGA_MOE=0 PRIMUS_OPT_LAYOUT=0 \
#   TURBO_USE_GROUPED_MLP=False YAML_TURBO_GROUPED_GEMM=false \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 5 -- + Turbo grouped GEMM
#   PRIMUS_OPT_FUSION=1 PRIMUS_OPT_ATTENTION=1 PRIMUS_OPT_DEEPEP=1 \
#   PRIMUS_OPT_SYNC_FREE=0 PRIMUS_OPT_MEGA_MOE=0 PRIMUS_OPT_LAYOUT=0 \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 6 -- MegaMoE, which replaces DeepEP and the grouped GEMM
#   PRIMUS_OPT_MEGA_MOE=1 PRIMUS_OPT_LAYOUT=0 \
#     bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# step 7 -- + tuned pipeline layout and no recompute. This is the default,
#           so it needs no switches at all.
#   bash examples/deepseek-v4/run_deepseek_v4_flash.sh
#
# Steps 6 and 7 need no explicit PRIMUS_OPT_DEEPEP / PRIMUS_OPT_SYNC_FREE:
# MegaMoE turns both off itself, since it replaces rather than stacks on them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Optimization switches
# =============================================================================
PRIMUS_OPT_FUSION=${PRIMUS_OPT_FUSION:-1}
PRIMUS_OPT_ATTENTION=${PRIMUS_OPT_ATTENTION:-1}
PRIMUS_OPT_DEEPEP=${PRIMUS_OPT_DEEPEP:-1}
PRIMUS_OPT_SYNC_FREE=${PRIMUS_OPT_SYNC_FREE:-1}
PRIMUS_OPT_MEGA_MOE=${PRIMUS_OPT_MEGA_MOE:-1}
PRIMUS_OPT_LAYOUT=${PRIMUS_OPT_LAYOUT:-1}

# (5) replaces (3) and (4) rather than stacking on them: MegaMoE builds its own
# experts, brings its own all-to-all, and never reaches token_dispatcher or
# grouped_experts, so DeepEP, the Turbo grouped GEMM and the sync-free stages
# have nothing left to accelerate. run_deepseek_v4.sh already forces
# USE_TURBO_DEEPEP=False when MegaMoE is on; the rest is turned off here so the
# summary printed at the end matches what actually runs.
if [ "$PRIMUS_OPT_MEGA_MOE" = 1 ]; then
    PRIMUS_OPT_DEEPEP=0
    PRIMUS_OPT_SYNC_FREE=0
fi

# =============================================================================
# Cluster wiring -- site-specific, not an optimization
# =============================================================================
if command -v spur >/dev/null 2>&1; then
    export PRIMUS_LAUNCHER=slurm
    export SLURM_LAUNCH_CMD="${SLURM_LAUNCH_CMD:-srun}"
    # Partition / QOS / account are site-specific and only forwarded when
    # non-empty, so set them in the environment for your cluster.
    export SLURM_PARTITION="${SLURM_PARTITION:-}"
    export SLURM_QOS="${SLURM_QOS:-}"
    export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
    # Empty = let the scheduler allocate nodes. An incoming SLURM_NODELIST is
    # honored so callers can pin to specific known-good nodes.
    export SLURM_NODELIST="${SLURM_NODELIST:-}"
    # Path to an ABI-4 libionic provider .so to swap into the container at launch
    # (fixes ionic RDMA on images whose bundled libionic only advertises uverbs
    # ABI 1). tools/patches/fix_libionic_abi4.sh reads it; empty disables it.
    export PRIMUS_LIBIONIC_SRC_ABI4_SO="${PRIMUS_LIBIONIC_SRC_ABI4_SO:-}"
    export NCCL_DEBUG="${NCCL_DEBUG:-}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens3}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens3}"
    # Registry login, needed only when the image is not public. Each node runs
    # `docker login` before the pull, and sbatch --export=ALL propagates these.
    # Empty means no login is attempted, which keeps credentials out of this
    # git-tracked file.
    export DOCKER_LOGIN_USER="${DOCKER_LOGIN_USER:-}"
    export DOCKER_LOGIN_KEY="${DOCKER_LOGIN_KEY:-}"
else
    # dccs cluster. Partition / nodelist are not pinned here; export
    # SLURM_PARTITION / SLURM_NODELIST to target specific hardware.
    #
    # Socket interface: run_deepseek_v4.sh falls back to `lo`, which leaves
    # multi-node rendezvous hanging. The dccs front-end NIC is `fenic` (the RDMA
    # devices are benic1p1..benic8p1 / ionic_0..7). runner/helpers/hooks/
    # 10_auto_nccl_net.sh would auto-detect it, but only when these are unset, and
    # the `lo` fallback sets them -- so pin them here.
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-fenic}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-fenic}"

    # Optionally reuse one held allocation instead of queueing per run, so a sweep
    # keeps landing on the same (known-good) nodes:
    #   salloc --no-shell -N 4 --exclusive --partition=<partition> --mem=0
    #   SLURM_ALLOC_JOB_ID=<granted id> bash examples/deepseek-v4/run_deepseek_v4_flash.sh
    # srun then joins that allocation rather than asking for new nodes;
    # --no-shell keeps it alive independently of any terminal, and
    # `scancel <id>` releases it. Left unset by default: a job id is only valid
    # for the life of its allocation.
    SLURM_ALLOC_JOB_ID="${SLURM_ALLOC_JOB_ID:-}"
    if [ -n "$SLURM_ALLOC_JOB_ID" ]; then
        export SLURM_JOB_ID="${SLURM_JOB_ID:-$SLURM_ALLOC_JOB_ID}"
        export SLURM_JOBID="${SLURM_JOBID:-$SLURM_ALLOC_JOB_ID}"
        # Let the step share nodes the allocation already holds.
        export SLURM_OVERLAP="${SLURM_OVERLAP:-1}"
        # srun rejects a --nodelist that is not inside the allocation, so take the
        # allocation's own list. Empty (allocation still pending) is fine; srun
        # then just uses every node it was given.
        export SLURM_NODELIST="${SLURM_NODELIST:-$(squeue -h -j "$SLURM_ALLOC_JOB_ID" -o '%N' 2>/dev/null || true)}"
    fi
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-fenic}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-fenic}"
fi

# =============================================================================
# Model shape and parallelism -- must stay identical across configurations
# being compared
# =============================================================================
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-43}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-256}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-6}
export PRIMUS_MOE_FFN_HIDDEN_SIZE=${PRIMUS_MOE_FFN_HIDDEN_SIZE:-2048}
export PRIMUS_INDEX_TOPK=${PRIMUS_INDEX_TOPK:-512}
export PRIMUS_COMPRESS_RATIOS=${PRIMUS_COMPRESS_RATIOS:-'[0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]'}
export MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}
export NNODES=${NNODES:-4}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-4}
export PRIMUS_EP=${PRIMUS_EP:-8}

export MBS=${MBS:-1}
export GBS=${GBS:-$((64 * NNODES * MBS))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}

# Force load balancing so the expert GEMM shapes are step-invariant. This keeps
# a comparison between two configurations from being polluted by routing jitter;
# `uniform` is closer to real routing than `even`, which hands every expert an
# identical token count and flatters the unfused grouped-GEMM path.
export MOE_FORCE_LB_TYPE=${MOE_FORCE_LB_TYPE:-uniform}

# =============================================================================
# (1) Small kernel fusions
# =============================================================================
# Each of these replaces a chain of small elementwise ops -- and the HBM round
# trip per op -- with a single Triton kernel. Individually none is dramatic;
# together they are the largest single step in the speedup curve.
if [ "$PRIMUS_OPT_FUSION" = 1 ]; then
    _F=1; _FY=true; _FB=True
else
    _F=0; _FY=false; _FB=False
fi

export PRIMUS_RMSNORM_TRITON=${PRIMUS_RMSNORM_TRITON:-$_F}                          # RMSNorm
export PRIMUS_ROPE_TRITON=${PRIMUS_ROPE_TRITON:-$_F}                                # interleaved partial RoPE
export PRIMUS_SINKHORN_TRITON=${PRIMUS_SINKHORN_TRITON:-$_F}                        # Sinkhorn-Knopp
export PRIMUS_HC_TRITON=${PRIMUS_HC_TRITON:-$_F}                                    # hyper-connection glue
export PRIMUS_HC_COLLAPSE_TRITON=${PRIMUS_HC_COLLAPSE_TRITON:-$_F}                  # hyper-connection collapse
export PRIMUS_HC_EXPAND_TRITON=${PRIMUS_HC_EXPAND_TRITON:-$_F}                      # hyper-connection expand
export PRIMUS_COMPRESS_POOL_TRITON=${PRIMUS_COMPRESS_POOL_TRITON:-$_F}              # compressor pooling
export PRIMUS_INDEXER_TRITON=${PRIMUS_INDEXER_TRITON:-$_F}                          # indexer scoring tail
export PRIMUS_V4_ROUTER_TRITON=${PRIMUS_V4_ROUTER_TRITON:-$_F}                      # MoE router tail
export PRIMUS_STACK_GROUPED_WEIGHT_TRITON=${PRIMUS_STACK_GROUPED_WEIGHT_TRITON:-$_F}  # grouped weight stack
export PRIMUS_INDEXER_FUSE_PROJ=${PRIMUS_INDEXER_FUSE_PROJ:-$_F}                    # indexer q/k projection fuse
export PRIMUS_INDEXER_MASK_CACHE=${PRIMUS_INDEXER_MASK_CACHE:-$_F}                  # indexer causal-mask cache
export PRIMUS_COMPRESS_ROPE_CACHE=${PRIMUS_COMPRESS_ROPE_CACHE:-$_F}                # compressed RoPE cache
# Backward-side fusions inside the Triton/FlyDSL attention kernels.
export PRIMUS_V4_ATTN_BWD_USE_SPLIT=${PRIMUS_V4_ATTN_BWD_USE_SPLIT:-$_F}
export PRIMUS_V4_CSA_BWD_SEGREDUCE=${PRIMUS_V4_CSA_BWD_SEGREDUCE:-$_F}
# An alternative, wider indexer fusion. Off by default even with (1) on: it
# supersedes PRIMUS_INDEXER_TRITON and has not been the faster of the two here.
export PRIMUS_INDEXER_TRITON_FULL=${PRIMUS_INDEXER_TRITON_FULL:-0}
# torch.compile path for Sinkhorn, an alternative to the Triton kernel above.
export USE_V4_COMPILED_SINKHORN=${USE_V4_COMPILED_SINKHORN:-$_FB}

# Fusions that live in the experiment yaml (see the derivation at the end).
# use_turbo_rms_norm additionally needs the primus_turbo master gate, which only
# comes on with (3) or (5).
YAML_MOE_PERMUTE_FUSION=${YAML_MOE_PERMUTE_FUSION:-$_FY}
YAML_CE_LOSS_FUSION=${YAML_CE_LOSS_FUSION:-$_FY}
YAML_GRAD_ACC_FUSION=${YAML_GRAD_ACC_FUSION:-$_FY}
YAML_TURBO_RMS_NORM=${YAML_TURBO_RMS_NORM:-$_FY}

# =============================================================================
# (2) Attention backend
# =============================================================================
# The dense/HCA path and the CSA path are selected separately because CSA's
# indexer and top-k selection make it a different kernel problem. Backends, in
# increasing order of speed: eager, triton_v1, triton_v2, gluon_v2, gluon_v3,
# turbo (Primus-Turbo native FlyDSL sparse-MLA).
#
# `eager` cannot run at this scale: it materialises a [B, H, S, S] tensor, about
# 16 GiB per microbatch per layer at V4-Flash dimensions, and fails to allocate
# at any recompute setting. The runnable zero point is triton_v1.
if [ "$PRIMUS_OPT_ATTENTION" = 1 ]; then _ATTN=turbo; else _ATTN=triton_v1; fi
export USE_V4_ATTENTION_BACKEND=${USE_V4_ATTENTION_BACKEND:-$_ATTN}
export USE_V4_CSA_ATTENTION_BACKEND=${USE_V4_CSA_ATTENTION_BACKEND:-$_ATTN}
# A separate flash-attention path that takes dispatch precedence over the V4
# backend on dense layers; measured far slower here because it cannot do SWA.
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
# Keep the indexer QK in high precision. The indexer decides which compressed KV
# entries each query attends to, so quantization error there changes the
# selection itself rather than merely perturbing a value. It is also a fake
# quant (quantize/dequantize around a BF16 GEMM), so leaving it off removes pure
# overhead. Set True to opt back in for QAT experiments.
export USE_V4_FP8_INDEXER=${USE_V4_FP8_INDEXER:-False}

# =============================================================================
# (3) DeepEP + Turbo grouped GEMM
# =============================================================================
# DeepEP moves the expert-parallel dispatch and combine into dedicated kernels
# instead of PyTorch permutation around two all-to-alls. The Turbo grouped GEMM
# issues the local experts' GEMMs as one ragged-batch kernel rather than a loop
# of small ones -- measured separately, the grouped GEMM is worth several times
# what DeepEP is.
if [ "$PRIMUS_OPT_DEEPEP" = 1 ]; then _EP=True; _EPY=true; else _EP=False; _EPY=false; fi
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-$_EP}
export TURBO_USE_GROUPED_MLP=${TURBO_USE_GROUPED_MLP:-$_EP}
YAML_TURBO_GROUPED_GEMM=${YAML_TURBO_GROUPED_GEMM:-$_EPY}
export LEGACY_GG=${LEGACY_GG:-False}
# run_deepseek_v4.sh only sets this when TURBO_USE_GROUPED_MLP=True but then
# dereferences it unconditionally under `set -u`, so it must exist either way.
export PRIMUS_BIAS_SWIGLU_FUSION=${PRIMUS_BIAS_SWIGLU_FUSION:-False}

# =============================================================================
# (4) Sync-free MoE
# =============================================================================
# Stage 1 fuses the router with the aux score and enables permutation fusion;
# stage 2 additionally implies DeepEP and the grouped GEMM. Note that the fused
# router drives Megatron's TopKRouter, while V4 uses its own learned router, so
# on this model stage 1 reduces to permutation fusion alone.
#
# Stage > 1 with the grouped GEMM off raises ValueError before Megatron can
# auto-enable it, which is why (4) implies (3) below.
if [ "$PRIMUS_OPT_SYNC_FREE" = 1 ]; then _SF=1; _SFY=true; else _SF=0; _SFY=false; fi
YAML_SYNC_FREE_STAGE=${YAML_SYNC_FREE_STAGE:-$_SF}
YAML_FUSED_ROUTER=${YAML_FUSED_ROUTER:-$_SFY}

# =============================================================================
# (5) MegaMoE
# =============================================================================
# Fuses the expert-parallel all-to-all into the grouped GEMM itself, so the
# ideal cost becomes max(comm, gemm) rather than their sum. EP-only: needs TP=1,
# BF16 and an EP process group. Mutually exclusive with (3), which
# run_deepseek_v4.sh enforces by forcing USE_TURBO_DEEPEP=False.
if [ "$PRIMUS_OPT_MEGA_MOE" = 1 ]; then _MM=True; else _MM=False; fi
export USE_TURBO_MEGA_MOE=${USE_TURBO_MEGA_MOE:-$_MM}

# Master gate for every primus_turbo patch (DeepEP dispatcher, Turbo RMSNorm,
# MegaMoE, Turbo grouped GEMM, sync-free auto-enable). run_deepseek_v4.sh turns
# it on automatically when any turbo feature is requested; pinned here so the
# state is explicit when every MoE optimization is off.
if [ "$PRIMUS_OPT_DEEPEP" = 1 ] || [ "$PRIMUS_OPT_MEGA_MOE" = 1 ]; then
    export ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-True}
else
    export ENABLE_PRIMUS_TURBO=${ENABLE_PRIMUS_TURBO:-False}
fi

# =============================================================================
# (6) Pipeline layout + recompute
# =============================================================================
# An even split is not balanced: the last stage also carries the MTP module and
# the loss, while 1F1B leaves stage 0 holding the most microbatches in flight.
# The tuned layout moves layers off the last stage onto the middle ones and
# keeps stage 0 small. Recompute can then go to zero, but only because the
# optimizations above have freed the memory for it -- dropping it first will
# run out of memory.
if [ "$PRIMUS_OPT_LAYOUT" = 1 ]; then
    export PRIMUS_PP_LAYOUT="${PRIMUS_PP_LAYOUT:-Et*10|t*12|t*12|t*9mL}"
    export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-0}
else
    export PRIMUS_PP_LAYOUT="${PRIMUS_PP_LAYOUT:-Et*10|t*11|t*11|t*11mL}"
    export PRIMUS_RECOMPUTE_LAYERS=${PRIMUS_RECOMPUTE_LAYERS:-3}
fi
export RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block}
export PRIMUS_RECOMPUTE_LAYER_IDS=${PRIMUS_RECOMPUTE_LAYER_IDS:-}

# =============================================================================
# Experiment config: derived yaml
# =============================================================================
# Seven of the switches above land on fields the shipped experiment yaml
# hardcodes, and run_deepseek_v4.sh passes no CLI override for them, so no
# environment variable can reach them. Derive a yaml rather than editing the
# original, tagged so concurrent configurations never share a file.
YAML_TAG=${YAML_TAG:-flash}
SRC_YAML="$PRIMUS_ROOT/examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml"
GEN_YAML="$PRIMUS_ROOT/examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-${YAML_TAG}.yaml"

if [ ! -f "$SRC_YAML" ]; then
    echo "[flash] ERROR: source config not found: $SRC_YAML" >&2
    exit 1
fi

# Always regenerate: the same tag must never carry values from a previous run.
{
    echo "# GENERATED by examples/deepseek-v4/run_deepseek_v4_flash.sh -- do not edit."
    echo "# switches: fusion=$PRIMUS_OPT_FUSION attention=$PRIMUS_OPT_ATTENTION deepep=$PRIMUS_OPT_DEEPEP"
    echo "#           sync_free=$PRIMUS_OPT_SYNC_FREE mega_moe=$PRIMUS_OPT_MEGA_MOE layout=$PRIMUS_OPT_LAYOUT"
    sed \
        -e "s/^\( *\)turbo_sync_free_moe_stage: *[0-9]*/\1turbo_sync_free_moe_stage: $YAML_SYNC_FREE_STAGE/" \
        -e "s/^\( *\)use_turbo_rms_norm: *\(true\|false\)/\1use_turbo_rms_norm: $YAML_TURBO_RMS_NORM/" \
        -e "s/^\( *\)use_turbo_grouped_gemm: *\(true\|false\)/\1use_turbo_grouped_gemm: $YAML_TURBO_GROUPED_GEMM/" \
        -e "s/^\( *\)moe_use_fused_router_with_aux_score: *\(true\|false\)/\1moe_use_fused_router_with_aux_score: $YAML_FUSED_ROUTER/" \
        -e "s/^\( *\)moe_permute_fusion: *\(true\|false\)/\1moe_permute_fusion: $YAML_MOE_PERMUTE_FUSION/" \
        -e "s/^\( *\)cross_entropy_loss_fusion: *\(true\|false\)/\1cross_entropy_loss_fusion: $YAML_CE_LOSS_FUSION/" \
        -e "s/^\( *\)gradient_accumulation_fusion: *\(true\|false\)/\1gradient_accumulation_fusion: $YAML_GRAD_ACC_FUSION/" \
        "$SRC_YAML"
} > "$GEN_YAML"

# Fail loudly rather than silently training a config that is not the one asked
# for: a rename upstream would otherwise pass through unnoticed.
for kv in "turbo_sync_free_moe_stage: $YAML_SYNC_FREE_STAGE" \
          "use_turbo_rms_norm: $YAML_TURBO_RMS_NORM" \
          "use_turbo_grouped_gemm: $YAML_TURBO_GROUPED_GEMM" \
          "moe_use_fused_router_with_aux_score: $YAML_FUSED_ROUTER" \
          "moe_permute_fusion: $YAML_MOE_PERMUTE_FUSION" \
          "cross_entropy_loss_fusion: $YAML_CE_LOSS_FUSION" \
          "gradient_accumulation_fusion: $YAML_GRAD_ACC_FUSION"; do
    if ! grep -q "$kv" "$GEN_YAML"; then
        echo "[flash] ERROR: '$kv' missing from $GEN_YAML -- did the source yaml change shape?" >&2
        exit 1
    fi
done

# Sync-free stage > 1 needs the grouped GEMM already on, or Megatron raises
# before it gets a chance to auto-enable it.
if [ "$YAML_SYNC_FREE_STAGE" -gt 1 ] && [ "$YAML_TURBO_GROUPED_GEMM" != "true" ] \
   && [ "$TURBO_USE_GROUPED_MLP" != "True" ]; then
    echo "[flash] ERROR: sync-free stage $YAML_SYNC_FREE_STAGE requires the grouped GEMM" >&2
    exit 1
fi

export EXP="${EXP:-$GEN_YAML}"

export PROFILE=${PROFILE:-False}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v4_flash_nodes${NNODES}_pp${PRIMUS_PP}_ep${PRIMUS_EP}_seq${PRIMUS_SEQ_LENGTH}}

echo "[flash] (1) fusion=$PRIMUS_OPT_FUSION  (2) attention=$PRIMUS_OPT_ATTENTION  (3) deepep=$PRIMUS_OPT_DEEPEP  (4) sync_free=$PRIMUS_OPT_SYNC_FREE  (5) mega_moe=$PRIMUS_OPT_MEGA_MOE  (6) layout=$PRIMUS_OPT_LAYOUT"
echo "[flash] attention=$USE_V4_ATTENTION_BACKEND/$USE_V4_CSA_ATTENTION_BACKEND turbo_gate=$ENABLE_PRIMUS_TURBO"
echo "[flash] deepep=$USE_TURBO_DEEPEP grouped_gemm=$TURBO_USE_GROUPED_MLP mega_moe=$USE_TURBO_MEGA_MOE sync_free_stage=$YAML_SYNC_FREE_STAGE"
if [ -n "$PRIMUS_RECOMPUTE_LAYER_IDS" ]; then
    _RECOMPUTE_DESC="layer_ids=$PRIMUS_RECOMPUTE_LAYER_IDS"
else
    _RECOMPUTE_DESC="$PRIMUS_RECOMPUTE_LAYERS"
fi
echo "[flash] layout=${PRIMUS_PP_LAYOUT:-<even split>} recompute=$_RECOMPUTE_DESC"
echo "[flash] nodes=$NNODES tp=$PRIMUS_TP pp=$PRIMUS_PP ep=$PRIMUS_EP gbs=$GBS seq=$PRIMUS_SEQ_LENGTH iters=$TRAIN_ITERS"
echo "[flash] exp=$EXP"

# Resolve the configuration and print it without launching anything, so a switch
# combination can be checked before an allocation is spent on it.
if [ "${PRIMUS_DRY_RUN:-0}" = 1 ]; then
    echo "[flash] dry run: configuration resolved, not launching"
    exit 0
fi

exec "${SCRIPT_DIR}/run_deepseek_v4.sh" 2>&1 | tee "train_flash_${YAML_TAG}.log"
