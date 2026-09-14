#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

MORI_REPO="${MORI_REPO:-https://github.com/ROCm/mori.git}"
MORI_REF="${MORI_REF:-12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4}"
MORI_SOURCE_DIR="${MORI_SOURCE_DIR:-/opt/mori}"
MAX_JOBS="${MAX_JOBS:-32}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "[MORI:Install] run as root inside the training container" >&2
    exit 1
fi
if [[ -z "${MORI_SOURCE_DIR}" || "${MORI_SOURCE_DIR}" == "/" ]]; then
    echo "[MORI:Install] unsafe MORI_SOURCE_DIR=${MORI_SOURCE_DIR@Q}" >&2
    exit 1
fi
if [[ ! -d "${ROCM_PATH}" ]]; then
    echo "[MORI:Install] ROCM_PATH does not exist: ${ROCM_PATH}" >&2
    exit 1
fi

echo "[MORI:Install] install dependencies"
apt-get update
apt-get install -y --no-install-recommends \
    git ibverbs-utils libibverbs-dev libnuma-dev libpci-dev
python3 -m pip install --no-cache-dir \
    "packaging<26" "setuptools==81.0.0" setuptools_scm Cython pybind11 ninja

echo "[MORI:Install] clone ${MORI_REPO}@${MORI_REF}"
rm -rf "${MORI_SOURCE_DIR}"
git clone --filter=blob:none "${MORI_REPO}" "${MORI_SOURCE_DIR}"
cd "${MORI_SOURCE_DIR}"
git checkout "${MORI_REF}"
git submodule update --init --depth 1 3rdparty/msgpack-c 3rdparty/spdlog

if [[ ! -e /usr/lib64/libc.so ]]; then
    mkdir -p /usr/lib64
    ln -s /usr/lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so
fi

echo "[MORI:Install] build and verify"
export ROCM_PATH
export CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/rocm_sysdeps${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CMAKE_LIBRARY_PATH="/opt/mori-host-libs${CMAKE_LIBRARY_PATH:+:${CMAKE_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/opt/mori-host-libs:${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset MORI_DEVICE_NIC
MAX_JOBS="${MAX_JOBS}" python3 -m pip install --no-build-isolation .
python3 -c \
    "import mori; from mori.ccl import HierAllGather; print(f'MORI {mori.__version__}: {HierAllGather}')"
echo "[MORI:Install] PASS"
