#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Rebuild Primus-Turbo from source, intended to be run via:
#   runner/helpers/execute_patches.sh runner/helpers/rebuild_primus_turbo.sh
#
# Control via environment variables:
#   PRIMUS_TURBO_BUILD_DIR=/path    # optional build workspace (default: /tmp/primus_turbo_$HOSTNAME)
#   GPU_ARCHS="gfx942;gfx950"       # optional override for target GPU architectures
#   PRIMUS_TURBO_REF=<branch-or-sha> # optional git ref (branch/tag/commit) to checkout
###############################################################################

set -euo pipefail

# Use a node-local temporary directory by default to avoid multi-node conflicts
# when /workspace is shared across hosts.
PRIMUS_TURBO_BUILD_DIR="${PRIMUS_TURBO_BUILD_DIR:-/tmp/primus_turbo_${HOSTNAME:-$(hostname)}}"
GPU_ARCHS="${GPU_ARCHS:-gfx942;gfx950}"
PRIMUS_TURBO_REF="${PRIMUS_TURBO_REF:-}"

LOG_INFO "Rebuilding Primus Turbo from source..."
LOG_INFO "  Build directory : ${PRIMUS_TURBO_BUILD_DIR}"
LOG_INFO "  GPU_ARCHS       : ${GPU_ARCHS}"

mkdir -p "${PRIMUS_TURBO_BUILD_DIR}"
cd "${PRIMUS_TURBO_BUILD_DIR}" || exit 1

# Clean up old checkout to avoid conflicts
if [[ -d "Primus-Turbo" ]]; then
    LOG_INFO "Removing existing Primus-Turbo directory..."
    rm -rf "Primus-Turbo"
fi

git clone https://github.com/AMD-AGI/Primus-Turbo.git --recursive
cd "Primus-Turbo" || exit 1

# Optionally checkout a specific branch/tag/commit if PRIMUS_TURBO_REF is set.
if [[ -n "$PRIMUS_TURBO_REF" ]]; then
    LOG_INFO "Checking out Primus-Turbo ref: ${PRIMUS_TURBO_REF}"
    git fetch --all --tags
    git checkout "${PRIMUS_TURBO_REF}"
fi

LOG_INFO "Installing Primus-Turbo build dependencies..."
pip3 install -r requirements.txt

LOG_INFO "Installing Primus-Turbo with GPU_ARCHS=${GPU_ARCHS} ..."
GPU_ARCHS="${GPU_ARCHS}" pip3 install --no-build-isolation .

# Return to Primus root if available
cd "${PRIMUS_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}" || exit 1
LOG_INFO "Rebuilding Primus Turbo from source done."

exit 0
