#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# DeepSeek-V3 proxy pretraining on one MI455X node.
#
#   bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh                         # on the host
#   PRIMUS_LAUNCHER=direct bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh  # in a container
#
# Settings are the environment variables below. Extra arguments are appended
# to the Primus command line.
###############################################################################
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/../../../.."

# container: start a container from DOCKER_IMAGE; direct: run in this container
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261015}
export EXP=${EXP:-examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml}
# Primus-Turbo checkout built by prepare_mi455x.sh; empty: the image's primus_turbo
export TURBO_DIR=${TURBO_DIR:-}

export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MBS=${MBS:-2}
export GBS=${GBS:-64}
export PRIMUS_TP=${PRIMUS_TP:-1}
export PRIMUS_PP=${PRIMUS_PP:-1}
export PRIMUS_EP=${PRIMUS_EP:-4}
export PRIMUS_TOTAL_LAYERS=${PRIMUS_TOTAL_LAYERS:-4}
export PRIMUS_NUM_EXPERTS=${PRIMUS_NUM_EXPERTS:-128}
export PRIMUS_MOE_TOPK=${PRIMUS_MOE_TOPK:-8}
# layers to recompute in full, e.g. 0,3; empty: no recompute
export PRIMUS_RECOMPUTE_LAYER_IDS=${PRIMUS_RECOMPUTE_LAYER_IDS:-}
export USE_TURBO_ATTENTION=${USE_TURBO_ATTENTION:-True}
export USE_TURBO_GROUPED_GEMM=${USE_TURBO_GROUPED_GEMM:-True}
export USE_TURBO_DEEPEP=${USE_TURBO_DEEPEP:-False}
# Primus-Turbo backends. Attention: flydsl or triton; grouped GEMM: bf16:flydsl or default
# (Primus-Turbo's choice: Triton, or auto-tuning with PRIMUS_TURBO_AUTO_TUNE=1)
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-flydsl}
export TURBO_GROUPED_GEMM_BACKEND=${TURBO_GROUPED_GEMM_BACKEND:-bf16:flydsl}
case "${PRIMUS_TURBO_ATTN_BACKEND,,}" in
    flydsl | triton) ;;
    *)
        echo "PRIMUS_TURBO_ATTN_BACKEND must be flydsl or triton, not '${PRIMUS_TURBO_ATTN_BACKEND}'" >&2
        exit 2
        ;;
esac
# Primus overwrites PRIMUS_TURBO_GROUPED_GEMM_BACKEND from TURBO_GROUPED_GEMM_BACKEND unless that is default
if [[ -n "${PRIMUS_TURBO_GROUPED_GEMM_BACKEND:-}" ]]; then
    echo "set TURBO_GROUPED_GEMM_BACKEND instead of PRIMUS_TURBO_GROUPED_GEMM_BACKEND" >&2
    exit 2
fi
# Primus opt-ins (all bit-identical or FP32-equivalent; set to 0 for the previous behaviour):
# MoE all-to-all without the self round, no blocking host copies in get_batch/TE CE backward,
# Primus-Turbo cross entropy, Triton GEMMs for the FP32 router
export PRIMUS_MOE_A2A_SKIP_SELF=${PRIMUS_MOE_A2A_SKIP_SELF:-1}
export PRIMUS_NO_HOST_SYNC=${PRIMUS_NO_HOST_SYNC:-1}
export PRIMUS_TURBO_CROSS_ENTROPY=${PRIMUS_TURBO_CROSS_ENTROPY:-1}
export PRIMUS_ROUTER_TRITON_GEMM=${PRIMUS_ROUTER_TRITON_GEMM:-1}
export MOCK_DATA=${MOCK_DATA:-True}
export PROFILE=${PROFILE:-False}

export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(id -un)}
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-deepseek_v3_proxy_${GPUS_PER_NODE}gpu_$(date +%Y%m%d_%H%M%S)}

ENV_ARGS=(
    --env PRIMUS_GPU_MODEL=MI455X
    --env NNODES=1
    --env GPUS_PER_NODE="${GPUS_PER_NODE}"
    --env HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
    --env TRITON_HIP_USE_EXPERT_SCHEDULING=0
    --env PRIMUS_TURBO_ATTN_BACKEND="${PRIMUS_TURBO_ATTN_BACKEND}"
    --env HSA_COREDUMP_PATTERN=/dev/null
)
# extra environment for the training processes, e.g. EXTRA_ENV="GPU_MAX_HW_QUEUES=4 CUDA_DEVICE_MAX_CONNECTIONS=8"
for kv in ${EXTRA_ENV:-}; do
    ENV_ARGS+=(--env "${kv}")
done
if [[ -n "${TURBO_DIR}" ]]; then
    TURBO_DIR=$(realpath -e "${TURBO_DIR}")
    # --env is applied after Primus builds PYTHONPATH, so this checkout comes first.
    ENV_ARGS+=(
        --env "PYTHONPATH=${TURBO_DIR}:${TURBO_DIR}/build/pydeps:${PWD}:${PWD}/third_party/Megatron-LM"
        --env PYTHONDONTWRITEBYTECODE=1
    )
fi
PATCH_ARGS=(--patch runner/helpers/patches/12_fix_triton_expert_scheduling.sh)

case "${PRIMUS_LAUNCHER}" in
    container)
        LAUNCHER_ARGS=(container --image "${DOCKER_IMAGE}" "${ENV_ARGS[@]}")
        if [[ -n "${TURBO_DIR}" ]]; then
            LAUNCHER_ARGS+=(--volume "${TURBO_DIR}:${TURBO_DIR}")
        fi
        LAUNCHER_ARGS+=(-- "${PATCH_ARGS[@]}")
        ;;
    direct)
        LAUNCHER_ARGS=(direct "${ENV_ARGS[@]}" "${PATCH_ARGS[@]}")
        ;;
    *)
        echo "PRIMUS_LAUNCHER must be container or direct, not '${PRIMUS_LAUNCHER}'" >&2
        exit 2
        ;;
esac

if [[ -n "${PRIMUS_RECOMPUTE_LAYER_IDS}" ]]; then
    RECOMPUTE_ARGS=(--recompute_granularity full --recompute_layer_ids "${PRIMUS_RECOMPUTE_LAYER_IDS}")
else
    RECOMPUTE_ARGS=(--recompute_granularity null --recompute_layer_ids null)
fi

LOG_DIR="${PRIMUS_WORKSPACE:-output}/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOG_DIR}"
./primus-cli "${LAUNCHER_ARGS[@]}" -- train pretrain --config "${EXP}" \
    --train_iters "${TRAIN_ITERS}" \
    --micro_batch_size "${MBS}" \
    --global_batch_size "${GBS}" \
    --tensor_model_parallel_size "${PRIMUS_TP}" \
    --pipeline_model_parallel_size "${PRIMUS_PP}" \
    --expert_model_parallel_size "${PRIMUS_EP}" \
    --num_layers "${PRIMUS_TOTAL_LAYERS}" \
    --num_experts "${PRIMUS_NUM_EXPERTS}" \
    --moe_router_topk "${PRIMUS_MOE_TOPK}" \
    "${RECOMPUTE_ARGS[@]}" \
    --use_turbo_attention "${USE_TURBO_ATTENTION}" \
    --use_turbo_grouped_gemm "${USE_TURBO_GROUPED_GEMM}" \
    --turbo_grouped_gemm_backend "${TURBO_GROUPED_GEMM_BACKEND}" \
    --use_turbo_deepep "${USE_TURBO_DEEPEP}" \
    --mock_data "${MOCK_DATA}" \
    --profile "${PROFILE}" \
    --use_pytorch_profiler "${PROFILE}" \
    --profile_step_start 6 \
    --profile_step_end 8 \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
