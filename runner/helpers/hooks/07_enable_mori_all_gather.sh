#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: opt into MORI FSDP all-gather.
#
# Trigger:
#
#   export FSDP_ALL_GATHER_BACKEND=mori
#   primus-cli direct -- train pretrain --config <existing yaml>
#
# When enabled, this hook installs MORI when needed and propagates MORI_* env
# vars into torchrun children. The Python patches then attach MoriAllGather to
# FSDP2 modules.

set -euo pipefail

if [[ "${FSDP_ALL_GATHER_BACKEND:-}" != "mori" ]]; then
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mori_installer="${SCRIPT_DIR}/../mori/install_mori.sh"

if ! python3 -c "from mori.ccl import HierAllGather" >/dev/null 2>&1; then
    echo "[MORI] HierAllGather is unavailable; installing MORI before launch." >&2
    if [[ ! -x "${mori_installer}" ]]; then
        echo "[ERROR] MORI installer is not executable: ${mori_installer}" >&2
        exit 1
    fi
    "${mori_installer}" >&2
    if ! python3 -c "from mori.ccl import HierAllGather" >/dev/null 2>&1; then
        echo "[ERROR] MORI installation completed but HierAllGather is unavailable." >&2
        exit 1
    fi
fi

# MORI all-gather uses SDMA for the intra-node leg. Allow an explicit caller
# value to win, but default it on for this feature.
export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"

# MORI's single-node eager path is the correctness-safe default on the ROCm
# versions used by Primus v26.4. Explicit user settings still win.
export MORI_HIER_CUDA_GRAPH="${MORI_HIER_CUDA_GRAPH:-0}"

if [[ -z "${MORI_SOCKET_IFNAME:-}" && -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export MORI_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME#=}"
fi

# primus-cli direct does not implicitly propagate host env into torchrun
# children. Emit every MORI_* variable as env.* so user-selected MORI tuning
# knobs (host-proxy, RDMA devices, async, graph/debug flags, etc.) survive.
echo "env.FSDP_ALL_GATHER_BACKEND=mori"
for name in "${!MORI_@}"; do
    echo "env.${name}=${!name}"
done
