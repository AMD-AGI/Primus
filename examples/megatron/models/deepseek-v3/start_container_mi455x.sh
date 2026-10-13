#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Host side: start a long-lived container for the DeepSeek-V3 proxy runs that
# sees only the selected GPUs (/dev/kfd plus their /dev/dri/renderD* nodes, not
# privileged), so the other GPUs of a shared node stay untouched. GPU i is the
# i-th GPU node of the KFD topology, the order HIP enumerates.
#
#   GPUS=0,1,2,3 MOUNTS=/path/to/Primus-Turbo bash examples/megatron/models/deepseek-v3/start_container_mi455x.sh
#
# Env: GPUS (required)   comma-separated GPU indices
#      IMAGE             default amdprimus/amdprimus:tas-20261011 (runbook v3)
#      NAME              default primus-dsv3-gpu<indices>
#      MOUNTS            ':'-separated host dirs mounted at the same path, e.g. a
#                        Primus-Turbo checkout (for a git worktree also its main
#                        repository, or the build cannot read the commit)
#      DOCKER_ARGS       extra `docker run` options, split on spaces
#      FORCE=1           start even if another process has one of those GPUs open
###############################################################################
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"

GPUS=${GPUS:?set GPUS, e.g. GPUS=3 or GPUS=0,1,2,3}
IMAGE=${IMAGE:-amdprimus/amdprimus:tas-20261011}
NAME=${NAME:-primus-dsv3-gpu${GPUS//,/}}
TOPO=/sys/class/kfd/kfd/topology/nodes

# "node gpu_id render_minor" of every GPU node, in node order (CPU nodes have gpu_id 0).
mapfile -t GPU_NODES < <(
    for n in "${TOPO}"/*; do
        id=$(cat "${n}/gpu_id")
        [[ "${id}" == 0 ]] && continue
        echo "$(basename "${n}") ${id} $(awk '$1 == "drm_render_minor" {print $2}' "${n}/properties")"
    done | sort -n
)
[[ ${#GPU_NODES[@]} -gt 0 ]] || { echo "no GPU nodes under ${TOPO}" >&2; exit 1; }

DEV_ARGS=(--device /dev/kfd)
RENDER_GID=""
IFS=',' read -r -a SEL <<< "${GPUS}"
for i in "${SEL[@]}"; do
    if ! [[ "${i}" =~ ^[0-9]+$ ]] || (( i >= ${#GPU_NODES[@]} )); then
        echo "GPU index ${i} out of range: this node has ${#GPU_NODES[@]} GPUs" >&2
        exit 1
    fi
    read -r node gpu_id minor <<< "${GPU_NODES[${i}]}"
    dev="/dev/dri/renderD${minor}"
    pci=$(basename "$(readlink -f "/sys/class/drm/renderD${minor}/device")")
    echo "GPU ${i}: KFD node ${node}, gpu_id ${gpu_id}, ${dev}, PCI ${pci}"
    # Every process that has the GPU open has a KFD vram_<gpu_id> entry. Its byte count is no
    # usable threshold: it read 0 while the process already held ~20 GiB (4 x MI455X node).
    for f in /sys/class/kfd/kfd/proc/*/vram_"${gpu_id}"; do
        [[ -e "${f}" ]] || continue
        if [[ "${FORCE:-0}" != 1 ]]; then
            echo "GPU ${i} is in use: pid $(basename "$(dirname "${f}")") has it open (FORCE=1 to start anyway)" >&2
            exit 1
        fi
    done
    DEV_ARGS+=(--device "${dev}")
    RENDER_GID=$(stat -c %g "${dev}")
done

MOUNT_ARGS=(-v "${REPO}:${REPO}")
if [[ -n "${MOUNTS:-}" ]]; then
    IFS=':' read -r -a EXTRA_MOUNTS <<< "${MOUNTS}"
    for m in "${EXTRA_MOUNTS[@]}"; do
        MOUNT_ARGS+=(-v "${m}:${m}")
    done
fi

read -r -a EXTRA_DOCKER_ARGS <<< "${DOCKER_ARGS:-}"
docker run -d --name "${NAME}" \
    "${DEV_ARGS[@]}" --group-add video --group-add "${RENDER_GID}" \
    --ipc=host --shm-size 64g --security-opt seccomp=unconfined --security-opt label=disable \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${MOUNT_ARGS[@]}" -w "${REPO}" "${EXTRA_DOCKER_ARGS[@]}" \
    "${IMAGE}" sleep infinity > /dev/null

REL="${HERE#"${REPO}"/}"
RUN="bash ${REL}/run_deepseek_v3_proxy_mi455x.sh"
case ${#SEL[@]} in
    1) RUN="env PRESET=ep1_1gpu ${RUN}" ;;
    4) ;;
    *) RUN="${RUN}   (the presets use 1 or 4 GPUs, not ${#SEL[@]})" ;;
esac
cat <<EOF
started ${NAME} (${IMAGE}) with GPUs ${GPUS}; inside it they are devices 0..$((${#SEL[@]} - 1)).
Once, as root (Triton expert-scheduling patch; with TURBO_DIR also builds that Primus-Turbo checkout):
  docker exec [-e TURBO_DIR=/path/to/Primus-Turbo] ${NAME} bash ${REL}/prepare_container_mi455x.sh
Then, as yourself:
  docker exec -u $(id -u):$(id -g) [-e TURBO_DIR=...] ${NAME} ${RUN}
EOF
