#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Llama-3.1-8B pretraining on one MI455X node.
#
#   TURBO_DIR=/path/to/Primus-Turbo bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh  # on the host
#   PRIMUS_LAUNCHER=direct TURBO_DIR=... bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh  # in a container
#
# Settings are the environment variables below. Extra arguments are appended
# to the Primus command line.
###############################################################################
set -euo pipefail

SCRIPT=$(realpath "$0")
# relative to the caller's directory
if [[ -n "${TURBO_DIR:-}" ]]; then
    TURBO_DIR=$(realpath -e "${TURBO_DIR}")
fi
cd "$(dirname "${SCRIPT}")/../../../.."

# container: re-run this script in a new container of DOCKER_IMAGE; direct: run in this container
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261019}
export EXP=${EXP:-examples/megatron/configs/MI455X/llama3.1_8B-BF16-pretrain.yaml}
# Primus-Turbo checkout built in place for DOCKER_IMAGE (see README.md); empty: the image's primus_turbo
export TURBO_DIR=${TURBO_DIR:-}

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-4}
export GBS=${GBS:-$((MBS * GPUS_PER_NODE))}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
# layers to recompute in full, e.g. 0,3; empty: no recompute
export PRIMUS_RECOMPUTE_LAYER_IDS=${PRIMUS_RECOMPUTE_LAYER_IDS:-}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
export USE_TURBO_RMS_NORM=${USE_TURBO_RMS_NORM:-True}
# Primus-Turbo attention backend: flydsl (needs the gfx1250 head-dim-128 kernels, see README.md)
# or triton
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
case "${PRIMUS_TURBO_ATTN_BACKEND,,}" in
    flydsl | triton) ;;
    *)
        echo "PRIMUS_TURBO_ATTN_BACKEND must be flydsl or triton, not '${PRIMUS_TURBO_ATTN_BACKEND}'" >&2
        exit 2
        ;;
esac
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}
# 1: switch Triton's gfx1250 expert scheduling off in the container's Triton (needs root), see below
export PATCH_TRITON_EXPERT_SCHEDULING=${PATCH_TRITON_EXPERT_SCHEDULING:-1}

export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(id -un)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-llama3.1_8b_${GPUS_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)}

case "${PRIMUS_LAUNCHER}" in
    container)
        # The settings above, plus the single-node variables `primus-cli container` passes through
        ENV_ARGS=(-e PRIMUS_LAUNCHER=direct)
        while IFS= read -r v; do
            case "${v}" in
                PRIMUS_LAUNCHER) ;;
                EXP | TURBO_DIR | GPUS_PER_NODE | HIP_VISIBLE_DEVICES | TRAIN_ITERS | MBS | GBS | USE_TURBO_* | \
                    MOCK_DATA | PROFILE | PATCH_TRITON_EXPERT_SCHEDULING | \
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
        # Same runtime options as `primus-cli container` (runner/.primus.yaml)
        exec docker run --rm --name "llama3.1-8b-mi455x-$$" \
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

ENV_ARGS=(
    --env PRIMUS_GPU_MODEL=MI455X
    --env NNODES=1
    --env GPUS_PER_NODE="${GPUS_PER_NODE}"
    --env HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    --env TRITON_HIP_USE_EXPERT_SCHEDULING=0
    --env PRIMUS_TURBO_ATTN_BACKEND="${PRIMUS_TURBO_ATTN_BACKEND}"
    --env HSA_COREDUMP_PATTERN=/dev/null
)
if [[ -n "${TURBO_DIR}" ]]; then
    # --env is applied after Primus builds PYTHONPATH, so this checkout comes first.
    ENV_ARGS+=(
        --env "PYTHONPATH=${TURBO_DIR}:${TURBO_DIR}/build/pydeps:${PWD}:${PWD}/third_party/Megatron-LM"
        --env PYTHONDONTWRITEBYTECODE=1
    )
fi

if [[ -n "${PRIMUS_RECOMPUTE_LAYER_IDS}" ]]; then
    # as a list: Primus reads a single id as an int
    RECOMPUTE_ARGS=(--recompute_granularity full --recompute_layer_ids "[${PRIMUS_RECOMPUTE_LAYER_IDS//[][]/}]")
else
    RECOMPUTE_ARGS=(--recompute_granularity null --recompute_layer_ids null)
fi

LOG_DIR="${PRIMUS_WORKSPACE:-output}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOG_DIR}"
./primus-cli direct "${ENV_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --train_iters "${TRAIN_ITERS}" \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --tensor_model_parallel_size "${PRIMUS_TP}" \
    --pipeline_model_parallel_size "${PRIMUS_PP}" \
    "${RECOMPUTE_ARGS[@]}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --use_turbo_rms_norm "${USE_TURBO_RMS_NORM}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
