#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli --patch script: build RCCL from ROCm/rocm-systems at a pinned
# commit and install it over the librccl that PyTorch loads, at container
# launch (before torchrun starts).
#
# Why: an image's RCCL can lack a feature a recipe needs -- for example the
# copy-engine ReduceScatter that rocm/primus:v26.9's RCCL 2.32.3 does not have.
# Rebuilding RCCL at launch avoids rebuilding the image.
#
# Controlled by PRIMUS_RCCL_REF (a full 40-character rocm-systems commit). If
# unset/empty the patch is skipped. Optional settings:
#   PRIMUS_RCCL_REPO        Git remote (default: https://github.com/ROCm/rocm-systems.git)
#   PRIMUS_RCCL_GPU_ARCHS   ';'-separated GPU targets (default: GPU_ARCHS, else gfx942;gfx950)
#   PRIMUS_RCCL_BUILD_DIR   Node-local build root (default: /tmp/primus_rccl)
#   PRIMUS_RCCL_BUILD_JOBS  Parallel compile jobs (default: nproc)
#
# The build uses the image's ROCm (ROCM_PATH) and installs under
# PRIMUS_RCCL_BUILD_DIR, never into /opt/rocm, which launchers may bind-mount
# read-only from the host. A finished build of the same commit and targets is
# reused, so later launches in the same container only verify the library. The
# PRIMUS_ prefix makes the settings auto-forward into the container (see
# primus-cli-container.sh env passthrough).
###############################################################################
set -euo pipefail

if [[ -z "${PRIMUS_RCCL_REF:-}" ]]; then
    echo "[rebuild_rccl] PRIMUS_RCCL_REF not set -- skipping"
    exit 2  # 2 = skip (not an error), per runner/helpers/execute_patches.sh
fi

REF="${PRIMUS_RCCL_REF}"
REPO="${PRIMUS_RCCL_REPO:-https://github.com/ROCm/rocm-systems.git}"
ARCHS="${PRIMUS_RCCL_GPU_ARCHS:-${GPU_ARCHS:-gfx942;gfx950}}"
BUILD_DIR="${PRIMUS_RCCL_BUILD_DIR:-/tmp/primus_rccl}"
# nproc honors OMP_NUM_THREADS, which training configs often set to 1.
JOBS="${PRIMUS_RCCL_BUILD_JOBS:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}"

if [[ ! "${REF}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[rebuild_rccl] PRIMUS_RCCL_REF must be a full 40-character commit, got '${REF}'" >&2
    exit 1
fi
if [[ -z "${ROCM_PATH:-}" || ! -d "${ROCM_PATH}" ]]; then
    echo "[rebuild_rccl] ROCM_PATH must point to the image's ROCm, got '${ROCM_PATH:-}'" >&2
    exit 1
fi

# ROCm SDK wheels preload their own librccl by absolute path, so ask a torch
# process which file it maps instead of trusting ldd.
mapfile -t LOADED < <(python3 - 2>/dev/null <<'EOF'
import os
import torch
torch.cuda.nccl.version()
for path in sorted({os.path.realpath(l.split()[-1]) for l in open("/proc/self/maps") if "librccl.so" in l}):
    print(path)
EOF
)
if (( ${#LOADED[@]} != 1 )); then
    echo "[rebuild_rccl] expected torch to map one librccl, got: ${LOADED[*]:-none}" >&2
    exit 1
fi
DST="${LOADED[0]}"
if [[ ! -w "${DST}" ]]; then
    echo "[rebuild_rccl] ${DST} is not writable" >&2
    exit 1
fi

BUILD_ROOT="${BUILD_DIR}/${REF}-${ARCHS//;/_}"
PREFIX="${BUILD_ROOT}/install"

if [[ ! -f "${BUILD_ROOT}/.done" ]]; then
    echo "[rebuild_rccl] building RCCL ${REF} for ${ARCHS} in ${BUILD_ROOT} (log: ${BUILD_ROOT}.log)"
    rm -rf "${BUILD_ROOT}"
    SRC="${BUILD_ROOT}/src"
    git init -q "${SRC}"
    git -C "${SRC}" remote add origin "${REPO}"
    git -C "${SRC}" sparse-checkout init --cone
    git -C "${SRC}" sparse-checkout set projects/rccl
    git -C "${SRC}" fetch -q --depth 1 origin "${REF}"
    git -C "${SRC}" checkout -q --detach FETCH_HEAD
    if [[ "$(git -C "${SRC}" rev-parse HEAD)" != "${REF}" ]]; then
        echo "[rebuild_rccl] fetched $(git -C "${SRC}" rev-parse HEAD), expected ${REF}" >&2
        exit 1
    fi
    start=${SECONDS}
    if ! (cd "${SRC}/projects/rccl" && ROCM_PATH="${ROCM_PATH}" ./install.sh \
        --prefix "${PREFIX}" --amdgpu_targets "${ARCHS}" --jobs "${JOBS}") \
        >"${BUILD_ROOT}.log" 2>&1; then
        tail -n 40 "${BUILD_ROOT}.log" >&2
        echo "[rebuild_rccl] RCCL build failed; full log: ${BUILD_ROOT}.log" >&2
        exit 1
    fi
    rm -rf "${SRC}"
    touch "${BUILD_ROOT}/.done"
    echo "[rebuild_rccl] built RCCL in $((SECONDS - start)) s"
fi

SRC_SO="$(readlink -f "${PREFIX}/lib/librccl.so")"
if ! cmp -s "${SRC_SO}" "${DST}"; then
    [[ -e "${DST}.orig" ]] || cp -p "${DST}" "${DST}.orig"
    cp --remove-destination "${SRC_SO}" "${DST}"
fi
if ! cmp -s "${SRC_SO}" "${DST}"; then
    echo "[rebuild_rccl] ${DST} does not match ${SRC_SO} after install" >&2
    exit 1
fi
echo "[rebuild_rccl] ${DST} <- RCCL ${REF} (sha256 $(sha256sum "${DST}" | awk '{print $1}'))"
