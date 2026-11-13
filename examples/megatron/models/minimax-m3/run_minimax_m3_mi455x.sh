#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# MiniMax-M3 proxy pretraining on one MI455X node.
#
#   bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh                         # on the host
#   PRIMUS_LAUNCHER=direct bash examples/megatron/models/minimax-m3/run_minimax_m3_mi455x.sh  # in a container
#
# One GPU unless GPUS_PER_NODE is set; extra GPUs are data and expert parallel.
# Settings are the environment variables below. Extra arguments are appended
# to the Primus command line.
###############################################################################
set -euo pipefail

SCRIPT=$(realpath "$0")
# relative to the caller's directory
if [[ -n "${TURBO_DIR:-}" ]]; then
    TURBO_DIR=$(realpath -e "${TURBO_DIR}")
fi
if [[ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE:-}" ]]; then
    HIPBLASLT_TUNING_OVERRIDE_FILE=$(realpath -e "${HIPBLASLT_TUNING_OVERRIDE_FILE}")
fi
cd "$(dirname "${SCRIPT}")/../../../.."

# container: re-run this script in a new container of DOCKER_IMAGE; direct: run in this container
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261109}
export EXP=${EXP:-examples/megatron/configs/MI455X/minimax_m3-BF16-pretrain.yaml}
# Primus-Turbo checkout built in place for DOCKER_IMAGE (see README.md); empty: the image's primus_turbo
export TURBO_DIR=${TURBO_DIR:-}

# One GPU until the caller asks for more. Do not default to every device on a
# shared node.
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-1}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-4096}
export PRIMUS_MAX_POSITION_EMBEDDINGS=${PRIMUS_MAX_POSITION_EMBEDDINGS:-${PRIMUS_SEQ_LENGTH}}
# flydsl or eager (the PyTorch reference)
export PRIMUS_MSA_BACKEND=${PRIMUS_MSA_BACKEND:-flydsl}
# 0: no recomputation. Every proxy shape the launcher allows fits without it (one GPU: 234 of 432 GB),
# and on M3.1-Flash the recomputed first layer is an MSA layer, so recomputing it repeats its
# whole forward (projections, indexer, MSA kernels) inside the backward.
export PRIMUS_RECOMPUTE_NUM_LAYERS=${PRIMUS_RECOMPUTE_NUM_LAYERS:-0}
# Primus-Turbo attention for the three dense layers. Empty: on with the flydsl backend when the
# Primus-Turbo in use has the gfx1250 head-dim-128 kernels (see README.md), off otherwise; decided in
# the container, below. The default backend fails in fmha_bwd here.
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-}
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-}
# Primus-Turbo grouped GEMM for the experts; False: TE's grouped GEMM (one hipBLASLt GEMM per expert)
export USE_TURBO_GROUPED_GEMM=${USE_TURBO_GROUPED_GEMM:-True}
# bf16:flydsl (Primus-Turbo's FlyDSL kernels; Primus raises their 64-expert weight-gradient cap to
# the 128 local experts of one GPU on gfx1250) or default (Primus-Turbo's choice: Triton)
export TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND:-bf16:flydsl}
TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND,,}
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}
# 1: switch Triton's gfx1250 expert scheduling off in the container's Triton (needs root), see below
export PATCH_TRITON_EXPERT_SCHEDULING=${PATCH_TRITON_EXPERT_SCHEDULING:-1}
# hipBLASLt kernels tuned for this example's GEMMs (README.md); empty: hipBLASLt's own choice. hipBLASLt
# uses a row only while its solution index still names the kernel it was tuned with. No default with
# PRIMUS_HIPBLASLT_TUNING=1: its stage 3 would take this file instead of the stage-2 result.
if [[ "${PRIMUS_HIPBLASLT_TUNING:-0}" == 1 ]]; then
    export HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE-}
else
    export HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE-$(dirname "${SCRIPT}")/hipblaslt_tuning/minimax_m3-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt}
fi
if [[ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE}" && ! -r "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
    echo "cannot read HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE}" >&2
    exit 2
fi
# Primus's HIP Adam applies the clip coefficient in-kernel and writes the BF16 parameters, so the
# step has no host sync, no separate grad scale and no parameter copy. Megatron then logs no grad
# norm. The other two need the first.
export PRIMUS_FUSED_ADAM_CLIP=${PRIMUS_FUSED_ADAM_CLIP:-1}
export PRIMUS_FUSED_ADAM_BF16_WRITEBACK=${PRIMUS_FUSED_ADAM_BF16_WRITEBACK:-${PRIMUS_FUSED_ADAM_CLIP}}
export PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=${PRIMUS_STATIC_BF16_LOSS_SCALE_CPU:-${PRIMUS_FUSED_ADAM_CLIP}}
# With one micro-batch per step on one data-parallel rank the gradient buffer is never accumulated
# into nor reduced: its FP32 value is exactly the BF16 weight gradient. Keep it in BF16 there
# (--grad_reduce_in_bf16) and let the HIP Adam read it directly (PRIMUS_FUSED_ADAM_BF16_GRAD):
# bit-identical, and it halves the gradient zeroing, accumulation, norm and Adam gradient
# traffic. With PRIMUS_FUSED_ADAM_DIRECT_GRAD (also on) Adam reads each parameter's autograd
# gradient and the buffer is neither zeroed nor accumulated into. Decided below, once the
# data-parallel size is known. 0 keeps FP32 gradients.
export PRIMUS_BF16_GRAD_SINGLE_MICROBATCH=${PRIMUS_BF16_GRAD_SINGLE_MICROBATCH:-1}
# The forward is host-bound on the Python main thread (instruction-fetch bound, ~70% frontend
# stalls). From the second step on, pin that thread to its own core on the GPU's NUMA node and keep
# the other threads off it (primus/backends/megatron/patches/host_step_overhead_patches.py). Host
# scheduling only. Empty disables it.
export PRIMUS_MAIN_THREAD_CPU=${PRIMUS_MAIN_THREAD_CPU-auto}
# Turbo's RMSNorm backward Triton kernels are autotuned by timing in step 1 over num_warps 4/8 ×
# num_stages 1/2, and num_warps changes the bits of dx (the row reduction). Which config wins the
# timing race decided the gradient bits of the whole run. Pin the config the reference run picks
# (primus/backends/megatron/patches/turbo_rmsnorm_bwd_pin_patches.py). Empty restores autotuning.
export PRIMUS_TURBO_RMSNORM_BWD_CONFIG=${PRIMUS_TURBO_RMSNORM_BWD_CONFIG-4,2}
# Run training_log(N) after step N+1's forward is enqueued, so the host launches the next forward
# while the GPU still runs step N's optimizer (primus/backends/megatron/patches/deferred_step_log_patches.py).
# Same kernels in the same order; the interval timer still reads step N's GPU end. 0 disables it.
export PRIMUS_DEFERRED_STEP_LOG=${PRIMUS_DEFERRED_STEP_LOG:-1}

case "${PRIMUS_MSA_BACKEND}" in
    flydsl | eager) ;;
    *)
        echo "PRIMUS_MSA_BACKEND must be flydsl or eager, not '${PRIMUS_MSA_BACKEND}'" >&2
        exit 2
        ;;
esac
case "${PRIMUS_TURBO_ATTN_BACKEND,,}" in
    "" | flydsl | triton) ;;
    *)
        echo "PRIMUS_TURBO_ATTN_BACKEND must be flydsl or triton, not '${PRIMUS_TURBO_ATTN_BACKEND}'" >&2
        exit 2
        ;;
esac
# FlyDSL's expert weight-gradient GEMM has no beta=1 epilogue, which gradient_accumulation_fusion
# needs; Primus-Turbo would raise at the first backward.
GAF_RE=' --gradient_accumulation_fusion[ =](true|1) '
if [[ "${TURBO_GROUPED_GEMM_BACKEND}" == *flydsl* && " ${*,,} " =~ ${GAF_RE} ]]; then
    echo "--gradient_accumulation_fusion True needs TURBO_GROUPED_GEMM_BACKEND=default," \
        "not '${TURBO_GROUPED_GEMM_BACKEND}'" >&2
    exit 2
fi
if [[ "${PRIMUS_TP}" != 1 ]]; then
    echo "MSA has no tensor parallel; PRIMUS_TP must be 1, not '${PRIMUS_TP}'" >&2
    exit 2
fi
if (( PRIMUS_TOTAL_LAYERS < 4 )); then
    echo "PRIMUS_TOTAL_LAYERS must be >= 4 so the first three layers stay dense (got ${PRIMUS_TOTAL_LAYERS})" >&2
    exit 2
fi
if (( GPUS_PER_NODE % PRIMUS_PP != 0 )); then
    echo "GPUS_PER_NODE=${GPUS_PER_NODE} is not divisible by PRIMUS_PP=${PRIMUS_PP}" >&2
    exit 2
fi
DP=$((GPUS_PER_NODE / PRIMUS_PP))
# expert parallel over the whole data-parallel group unless set
export PRIMUS_EP=${PRIMUS_EP:-${DP}}
if (( DP % PRIMUS_EP != 0 || 128 % PRIMUS_EP != 0 )); then
    echo "PRIMUS_EP=${PRIMUS_EP} must divide the data-parallel size ${DP} and the 128 experts" >&2
    exit 2
fi
export GBS=${GBS:-$((MBS * DP))}
if (( GBS % (MBS * DP) != 0 )); then
    echo "GBS=${GBS} must be divisible by MBS*DP=$((MBS * DP))" >&2
    exit 2
fi
BF16_GRAD_ARGS=()
if [[ "${PRIMUS_BF16_GRAD_SINGLE_MICROBATCH}" == 1 && "${DP}" == 1 && "${GBS}" == "${MBS}" ]]; then
    BF16_GRAD_ARGS=(--grad_reduce_in_bf16 True)
    if [[ "${PRIMUS_FUSED_ADAM_CLIP}" == 1 ]]; then
        export PRIMUS_FUSED_ADAM_BF16_GRAD=${PRIMUS_FUSED_ADAM_BF16_GRAD:-1}
        # and Adam reads each param.grad itself: no buffer zeroing, no main_grad.add_
        export PRIMUS_FUSED_ADAM_DIRECT_GRAD=${PRIMUS_FUSED_ADAM_DIRECT_GRAD:-1}
        # and keeps each FP32 master as its BF16 parameter + a 16-bit remainder (lossless;
        # Adam moves 26 instead of 28 B/param, bit-identical)
        export PRIMUS_FUSED_ADAM_SPLIT_MASTER=${PRIMUS_FUSED_ADAM_SPLIT_MASTER:-1}
    fi
fi
# the yaml's moe_layer_freq, sparse_attention_freq and sparse_disable_index_value
export PRIMUS_LAYER_FREQ="([0]*3+[1]*$((PRIMUS_TOTAL_LAYERS - 3)))"

export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(id -un)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-minimax_m3_${GPUS_PER_NODE}gpu_ep${PRIMUS_EP}_$(date +%Y%m%d_%H%M%S)}

case "${PRIMUS_LAUNCHER}" in
    container)
        # The settings above, plus the single-node variables `primus-cli container` passes through
        ENV_ARGS=(-e PRIMUS_LAUNCHER=direct)
        while IFS= read -r v; do
            case "${v}" in
                PRIMUS_LAUNCHER) ;;
                EXP | TURBO_DIR | GPUS_PER_NODE | HIP_VISIBLE_DEVICES | TRAIN_ITERS | MBS | GBS | USE_TURBO_* | \
                    TURBO_GROUPED_GEMM_BACKEND | MOCK_DATA | PROFILE | PATCH_TRITON_EXPERT_SCHEDULING | \
                    MASTER_ADDR | MASTER_PORT | HF_TOKEN | WANDB_API_KEY | ENABLE_NUMA_BINDING | GPU_MAX_HW_QUEUES | \
                    HSA_* | NVTE_* | PRIMUS_* | NCCL_* | RCCL_* | GLOO_* | IONIC_* | HIPBLASLT_*)
                    ENV_ARGS+=(-e "${v}")
                    ;;
            esac
        done < <(compgen -e)
        DOCKER_ARGS=(--device /dev/kfd --device /dev/dri -v "${PWD}:${PWD}")
        if [[ -e /dev/infiniband ]]; then
            DOCKER_ARGS+=(--device /dev/infiniband)
        fi
        if [[ -n "${TURBO_DIR}" ]]; then
            DOCKER_ARGS+=(-v "${TURBO_DIR}:${TURBO_DIR}")
        fi
        if [[ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
            DOCKER_ARGS+=(-v "${HIPBLASLT_TUNING_OVERRIDE_FILE}:${HIPBLASLT_TUNING_OVERRIDE_FILE}:ro")
        fi
        # Same runtime options as `primus-cli container` (runner/.primus.yaml)
        exec docker run --rm --name "minimax-m3-mi455x-$$" \
            --ipc host --network host --privileged --group-add video \
            --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN --security-opt seccomp=unconfined \
            --ulimit nofile=1048576:1048576 "${DOCKER_ARGS[@]}" -w "${PWD}" "${ENV_ARGS[@]}" \
            "${DOCKER_IMAGE}" bash "${SCRIPT}" "$@"
        ;;
    direct) ;;
    *)
        echo "PRIMUS_LAUNCHER must be container or direct, not '${PRIMUS_LAUNCHER}'" >&2
        exit 2
        ;;
esac

# The flydsl backend needs Primus-Turbo's gfx1250 head-dim-128 kernels (README.md). Without them the
# first attention call raises ValueError, after the model and optimizer are built; stop here instead.
if [[ -n "${TURBO_DIR}" ]]; then
    TURBO_PKG=${TURBO_DIR}/primus_turbo
else
    TURBO_PKG=$(python3 -c 'import importlib.util, os; s = importlib.util.find_spec("primus_turbo"); print(os.path.dirname(s.origin) if s and s.origin else "")')
fi
if [[ -d "${TURBO_PKG}/flydsl/attention/gfx1250" ]]; then
    USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
    PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
else
    USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-False}
    PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-triton}
    if [[ "${USE_TURBO_ATTENTION,,}" =~ ^(true|1|yes)$ && "${PRIMUS_TURBO_ATTN_BACKEND,,}" == flydsl ]]; then
        echo "PRIMUS_TURBO_ATTN_BACKEND=flydsl needs a Primus-Turbo with primus_turbo/flydsl/attention/gfx1250/," \
            "which ${TURBO_PKG:-this Python environment} lacks. Set TURBO_DIR to such a checkout (README.md)" \
            "or PRIMUS_TURBO_ATTN_BACKEND=triton." >&2
        exit 2
    fi
fi
export USE_TURBO_ATTENTION PRIMUS_TURBO_ATTN_BACKEND

# Triton 3.8 turns on AMDGPU expert scheduling for gfx1250, which makes Triton kernels fault or
# return NaN at random. TRITON_HIP_USE_EXPERT_SCHEDULING=0 (below) turns it off when a kernel is
# compiled, but Inductor loads kernels from its own cache without compiling them; changing Triton's
# default also invalidates that cache.
if [[ "${PATCH_TRITON_EXPERT_SCHEDULING}" == 1 ]]; then
    TRITON_AMD_COMPILER=$(python3 -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "amd", "compiler.py"))')
    if ! grep -q '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}"; then
        echo "no is_expert_scheduling_enabled(arch) in ${TRITON_AMD_COMPILER}; Triton not patched" >&2
    elif ! grep -A1 '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}" | grep -q '^    return False$'; then
        if [[ ! -w "$(dirname "${TRITON_AMD_COMPILER}")" ]]; then
            echo "cannot patch ${TRITON_AMD_COMPILER} as uid $(id -u); run as root or set PATCH_TRITON_EXPERT_SCHEDULING=0" >&2
            exit 2
        fi
        sed -i '/^def is_expert_scheduling_enabled(arch):/a\    return False' "${TRITON_AMD_COMPILER}"
        echo "patched ${TRITON_AMD_COMPILER}: is_expert_scheduling_enabled() -> False"
    fi
fi

# hipBLASLt takes a set but empty HIPBLASLT_TUNING_OVERRIDE_FILE as an override file named ""
if [[ -z "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
    unset HIPBLASLT_TUNING_OVERRIDE_FILE
fi

ENV_ARGS=(
    --env PRIMUS_GPU_MODEL=MI455X
    --env NNODES=1
    --env GPUS_PER_NODE="${GPUS_PER_NODE}"
    --env HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    --env TRITON_HIP_USE_EXPERT_SCHEDULING=0
    --env HSA_COREDUMP_PATTERN=/dev/null
)
if [[ -n "${TURBO_DIR}" ]]; then
    # --env is applied after Primus builds PYTHONPATH, so this checkout comes first.
    ENV_ARGS+=(
        --env "PYTHONPATH=${TURBO_DIR}:${TURBO_DIR}/build/pydeps:${PWD}:${PWD}/third_party/Megatron-LM"
        --env PYTHONDONTWRITEBYTECODE=1
    )
fi

LOG_DIR="${PRIMUS_WORKSPACE:-output}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOG_DIR}"
./primus-cli direct "${ENV_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --train_iters "${TRAIN_ITERS}" \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --tensor_model_parallel_size "${PRIMUS_TP}" \
    --pipeline_model_parallel_size "${PRIMUS_PP}" \
    --expert_model_parallel_size "${PRIMUS_EP}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --use_turbo_grouped_gemm "${USE_TURBO_GROUPED_GEMM}" \
    --turbo_grouped_gemm_backend "${TURBO_GROUPED_GEMM_BACKEND}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "${BF16_GRAD_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
