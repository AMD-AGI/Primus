#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# GPT-OSS BF16 pretrain on one MI455X (gfx1250) node, 4 GPUs, mock data.
# v0 baseline for dev/mi455x_v1; results and known issues are in README.md.
#
#   From the host (starts DOCKER_IMAGE and re-runs this script inside it):
#     MODEL=20B  bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
#     MODEL=120B bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
#   Already inside the container:
#     PRIMUS_LAUNCHER=direct MODEL=20B bash examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh
#
# MODEL=20B  is the full 24-layer model, EP4, MBS8 / GBS64.
# MODEL=120B is an 8-of-36-layer proxy, EP4, MBS1 / GBS8, full recompute.
# Extra Megatron overrides go in EXTRA_ARGS, e.g. EXTRA_ARGS="--sink_sliding_window 128".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO}"

export MODEL=${MODEL:-20B}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261011}
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(whoami)}

if [ "${PRIMUS_LAUNCHER}" = "container" ]; then
  ENV_ARGS=()
  for v in MODEL PRIMUS_TEAM PRIMUS_USER PRIMUS_EXP_NAME PRIMUS_EP HIP_VISIBLE_DEVICES GPUS_PER_NODE \
           TRAIN_ITERS MBS GBS NUM_LAYERS CE_IMPL EXTRA_ARGS PATCH_TRITON_EXPERT_SCHEDULING \
           TRITON_HIP_USE_EXPERT_SCHEDULING PRIMUS_TURBO_ATTN_BACKEND HSA_COREDUMP_PATTERN HF_TOKEN; do
    if [ -n "${!v+x}" ]; then ENV_ARGS+=(-e "${v}"); fi
  done
  DEVICE_ARGS=(--device /dev/kfd --device /dev/dri)
  if [ -e /dev/infiniband ]; then DEVICE_ARGS+=(--device /dev/infiniband); fi
  echo "[gpt-oss] launching ${MODEL} in ${DOCKER_IMAGE}"
  # Same runtime options as `primus-cli container` (runner/.primus.yaml).
  exec docker run --rm --name "gpt-oss-mi455x-$$" \
    --ipc host --network host --privileged --group-add video \
    --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN --security-opt seccomp=unconfined \
    --ulimit nofile=1048576:1048576 "${DEVICE_ARGS[@]}" \
    -v "${REPO}:${REPO}" -w "${REPO}" "${ENV_ARGS[@]}" -e PRIMUS_LAUNCHER=direct \
    "${DOCKER_IMAGE}" bash "${HERE}/$(basename "${BASH_SOURCE[0]}")"
fi

########################## gfx1250 runtime settings ##########################
# Primus reports the GPU as "AMD Radeon Graphics" and would skip runner/helpers/envs/MI455X.sh.
export PRIMUS_GPU_MODEL=${PRIMUS_GPU_MODEL:-MI455X}
# Triton 3.8 enables amdgpu-expert-scheduling-mode for gfx1250 by default, which causes random
# memory faults / NaN in Turbo Triton kernels. The env var alone misses some Inductor kernels,
# so Triton's default is also patched below (PATCH_TRITON_EXPERT_SCHEDULING=0 to skip).
export TRITON_HIP_USE_EXPERT_SCHEDULING=${TRITON_HIP_USE_EXPERT_SCHEDULING:-0}
# The default Turbo attention backend tries aiter, which the image does not ship.
export PRIMUS_TURBO_ATTN_BACKEND=${PRIMUS_TURBO_ATTN_BACKEND:-triton}
# A GPU fault otherwise writes a multi-GB core file per rank into the working directory.
export HSA_COREDUMP_PATTERN=${HSA_COREDUMP_PATTERN:-/dev/null}

if [ "${PATCH_TRITON_EXPERT_SCHEDULING:-1}" = "1" ]; then
  TRITON_AMD_COMPILER="$(python -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "amd", "compiler.py"))')"
  if grep -q '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}" &&
     ! grep -A1 '^def is_expert_scheduling_enabled(arch):' "${TRITON_AMD_COMPILER}" | grep -q '^    return False$'; then
    sed -i '/^def is_expert_scheduling_enabled(arch):/a\    return False' "${TRITON_AMD_COMPILER}"
    echo "[gpt-oss] patched ${TRITON_AMD_COMPILER}: is_expert_scheduling_enabled() -> False"
  fi
fi

################################ Parallelism #################################
export NNODES=1
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=${PRIMUS_EP:-4}

case "${MODEL}" in
  20B)
    EXP=examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml
    MBS=${MBS:-8}
    GBS=${GBS:-64}
    # The native fused CE is compiled by Inductor, whose autotuner backs up the
    # 8x4096x131200 fp32 logits (~17 GB per rank) into pinned host memory; four ranks
    # exceed the amdgpu pinned-memory limit and step 1 fails with "CUDA error: out of memory".
    CE_IMPL=${CE_IMPL:-te}
    MODEL_ARGS=()
    ;;
  120B)
    EXP=examples/megatron/configs/MI355X/gpt_oss_120B-BF16-pretrain.yaml
    MBS=${MBS:-1}
    GBS=${GBS:-8}
    CE_IMPL=${CE_IMPL:-native}
    # The yaml enables the torch profiler, which kills the process in gilstate_tss_set
    # after training on this image.
    MODEL_ARGS=(
      --pipeline_model_parallel_size 1 --expert_model_parallel_size "${PRIMUS_EP}"
      --num_layers "${NUM_LAYERS:-8}"
      --recompute_granularity full --recompute_method uniform --recompute_num_layers 1
      --profile False --use_pytorch_profiler False
    )
    ;;
  *)
    echo "[gpt-oss] unknown MODEL=${MODEL}, expected 20B or 120B" >&2
    exit 1
    ;;
esac

export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-gpt_oss_${MODEL}_mi455x_v0_ep${PRIMUS_EP}_mbs${MBS}_gbs${GBS}}
LOGDIR="${REPO}/output/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOGDIR}"
echo "[gpt-oss] model=${MODEL} gpus=${GPUS_PER_NODE} ep=${PRIMUS_EP} mbs=${MBS} gbs=${GBS} ce=${CE_IMPL} log=${LOGDIR}"

# shellcheck disable=SC2086
./primus-cli direct -- train pretrain --config "${EXP}" \
  --train_iters "${TRAIN_ITERS:-10}" --mock_data True \
  --micro_batch_size "${MBS}" --global_batch_size "${GBS}" \
  --cross_entropy_fusion_impl "${CE_IMPL}" \
  --disable_wandb True --disable_tensorboard True \
  "${MODEL_ARGS[@]}" ${EXTRA_ARGS:-} \
  2>&1 | tee "${LOGDIR}/log_node0.txt"
