#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: optionally rebuild Primus-Turbo before running the main command.
#
# Trigger:
#   export REBUILD_PRIMUS_TURBO=1
#
# Notes:
# - Hooks run before the main training command, and are the right place for
#   environment preparation steps (vs. patching the training runtime).
# - Inline implementation (no separate helper script).
###############################################################################

set -euo pipefail

if [[ "${REBUILD_PRIMUS_TURBO:-0}" != "1" ]]; then
    exit 0
fi

if [[ -z "${PRIMUS_PATH:-}" ]]; then
    # Infer PRIMUS_PATH from this file location (runner/helpers/hooks/)
    PRIMUS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export PRIMUS_PATH
fi

# Use a node-local temporary directory by default to avoid multi-node conflicts
# when /workspace is shared across hosts.
PRIMUS_TURBO_BUILD_DIR="${PRIMUS_TURBO_BUILD_DIR:-/tmp/primus_turbo_${HOSTNAME:-$(hostname)}}"
GPU_ARCHS="${GPU_ARCHS:-gfx942;gfx950}"
PRIMUS_TURBO_REF="${PRIMUS_TURBO_REF:-}"

LOG_INFO_RANK0 "[hook system] REBUILD_PRIMUS_TURBO=1 → rebuilding Primus-Turbo"
LOG_INFO_RANK0 "  Build directory : ${PRIMUS_TURBO_BUILD_DIR}"
LOG_INFO_RANK0 "  GPU_ARCHS       : ${GPU_ARCHS}"

mkdir -p "${PRIMUS_TURBO_BUILD_DIR}"
cd "${PRIMUS_TURBO_BUILD_DIR}" || exit 1

# Clean up old checkout to avoid conflicts
if [[ -d "Primus-Turbo" ]]; then
    LOG_INFO_RANK0 "Removing existing Primus-Turbo directory..."
    rm -rf "Primus-Turbo"
fi

: "${GIT_SSL_VERIFY:=true}"
LOG_INFO_RANK0 "Cloning Primus-Turbo with GIT_SSL_VERIFY=${GIT_SSL_VERIFY}"
git clone https://github.com/AMD-AGI/Primus-Turbo.git --recursive
cd "Primus-Turbo" || exit 1

# Optionally checkout a specific branch/tag/commit if PRIMUS_TURBO_REF is set.
if [[ -n "$PRIMUS_TURBO_REF" ]]; then
    LOG_INFO_RANK0 "Checking out Primus-Turbo ref: ${PRIMUS_TURBO_REF}"
    git fetch --all --tags
    git checkout "${PRIMUS_TURBO_REF}"
fi

LOG_INFO_RANK0 "Installing Primus-Turbo build dependencies..."
pip3 install -r requirements.txt

LOG_INFO_RANK0 "Installing Primus-Turbo with GPU_ARCHS=${GPU_ARCHS} ..."
GPU_ARCHS="${GPU_ARCHS}" pip3 install --no-build-isolation .

cd "${PRIMUS_PATH}" || exit 1
LOG_INFO_RANK0 "Rebuilding Primus-Turbo done."
