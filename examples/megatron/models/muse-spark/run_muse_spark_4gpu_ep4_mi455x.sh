#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Muse-Spark-Open-816B-A42B BF16 pretraining proxy on one MI455X (gfx1250) node: 4 GPUs, EP 4, mock data.
# The first 4 layers with all 256 experts (examples/megatron/configs/MI455X/
# muse_spark_816B_A42B-BF16-pretrain-4gpu-ep4-proxy.yaml). Notes are in README.md.
#
#   From the host (starts DOCKER_IMAGE and re-runs this script inside it):
#     MOE=deepep  MBS=1 bash examples/megatron/models/muse-spark/run_muse_spark_4gpu_ep4_mi455x.sh
#     MOE=megamoe MBS=2 bash examples/megatron/models/muse-spark/run_muse_spark_4gpu_ep4_mi455x.sh
#   Already inside the container:
#     PRIMUS_LAUNCHER=direct MOE=alltoall bash examples/megatron/models/muse-spark/run_muse_spark_4gpu_ep4_mi455x.sh
#
# MOE selects the routed-expert path:
#   alltoall -- Megatron's alltoall token dispatcher + Turbo grouped GEMM experts (no DeepEP)
#   deepep   -- Primus-Turbo DeepEP dispatcher + Turbo grouped GEMM experts (the yaml's default)
#   megamoe  -- Primus-Turbo Mega MoE with the post-norm stage2 (muse_spark_mega_moe); needs a
#               Primus-Turbo with fused_mega_moe_stage2_post_norm
# GBS defaults to 8 x MBS (two micro-batches per GPU). Extra Megatron overrides go in EXTRA_ARGS.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
cd "${REPO}"

export MOE=${MOE:-deepep}
export MBS=${MBS:-1}
export GBS=${GBS:-$((8 * MBS))}
export TRAIN_ITERS=${TRAIN_ITERS:-40}
export DOCKER_IMAGE=${DOCKER_IMAGE:-amdprimus/amdprimus:tas-20261109-triton-83c9228329}
export PRIMUS_LAUNCHER=${PRIMUS_LAUNCHER:-container}
export PRIMUS_TEAM=${PRIMUS_TEAM:-amd}
export PRIMUS_USER=${PRIMUS_USER:-$(whoami)}

if [ "${PRIMUS_LAUNCHER}" = "container" ]; then
  ENV_ARGS=()
  for v in MOE MBS GBS TRAIN_ITERS PRIMUS_TEAM PRIMUS_USER PRIMUS_EXP_NAME HIP_VISIBLE_DEVICES EXTRA_ARGS \
           HSA_COREDUMP_PATTERN; do
    if [ -n "${!v+x}" ]; then ENV_ARGS+=(-e "${v}"); fi
  done
  echo "[muse-spark] launching MOE=${MOE} MBS=${MBS} in ${DOCKER_IMAGE}"
  # Same runtime options as `primus-cli container` (runner/.primus.yaml).
  exec docker run --rm --name "muse-spark-mi455x-$$" \
    --ipc host --network host --privileged --group-add video \
    --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN --security-opt seccomp=unconfined \
    --ulimit nofile=1048576:1048576 --device /dev/kfd --device /dev/dri \
    -v "${REPO}:${REPO}" -w "${REPO}" "${ENV_ARGS[@]}" -e PRIMUS_LAUNCHER=direct \
    "${DOCKER_IMAGE}" bash "${HERE}/$(basename "${BASH_SOURCE[0]}")"
fi

########################## gfx1250 runtime settings ##########################
export PRIMUS_GPU_MODEL=${PRIMUS_GPU_MODEL:-MI455X}
# The README numbers were measured with Triton's own default (the image's Triton >= 3.9), not the
# image's TRITON_HIP_USE_EXPERT_SCHEDULING=0.
unset TRITON_HIP_USE_EXPERT_SCHEDULING
# The image pins its torch / Triton; Primus's pip hook would otherwise replace them.
export PRIMUS_SKIP_PIP=${PRIMUS_SKIP_PIP:-1}
# A GPU fault otherwise writes a multi-GB core file per rank into the working directory.
export HSA_COREDUMP_PATTERN=${HSA_COREDUMP_PATTERN:-/dev/null}

################################ Parallelism #################################
export NNODES=1
export GPUS_PER_NODE=4
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}

case "${MOE}" in
  alltoall) MOE_ARGS=(--use_turbo_deepep False) ;;
  deepep) MOE_ARGS=() ;;
  megamoe) MOE_ARGS=(--muse_spark_mega_moe True --use_turbo_deepep False) ;;
  *)
    echo "[muse-spark] unknown MOE=${MOE}, expected alltoall, deepep or megamoe" >&2
    exit 1
    ;;
esac

EXP=examples/megatron/configs/MI455X/muse_spark_816B_A42B-BF16-pretrain-4gpu-ep4-proxy.yaml
export PRIMUS_EXP_NAME=${PRIMUS_EXP_NAME:-muse_spark_4gpu_ep4_${MOE}_mbs${MBS}_gbs${GBS}}
LOGDIR="${REPO}/output/${PRIMUS_TEAM}/${PRIMUS_USER}/${PRIMUS_EXP_NAME}"
mkdir -p "${LOGDIR}"
echo "[muse-spark] moe=${MOE} mbs=${MBS} gbs=${GBS} iters=${TRAIN_ITERS} log=${LOGDIR}"

# shellcheck disable=SC2086
./primus-cli direct -- train pretrain --config "${EXP}" \
  --train_iters "${TRAIN_ITERS}" \
  --micro_batch_size "${MBS}" --global_batch_size "${GBS}" \
  "${MOE_ARGS[@]}" ${EXTRA_ARGS:-} \
  2>&1 | tee "${LOGDIR}/log_node0.txt"
