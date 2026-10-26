#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Enable the RCCL copy-engine AllGather path. Keep the established FSDP selector
# working while also accepting the backend-neutral SDMA_ALL_GATHER trigger.

set -euo pipefail

sdma_trigger="${SDMA_ALL_GATHER:-0}"
fsdp_backend="${FSDP_ALL_GATHER_BACKEND:-}"
if [[ "${sdma_trigger}" != "1" && "${fsdp_backend}" != "rccl_sdma" ]]; then
    exit 0
fi

kernel_release="$(uname -r)"
if [[ ! "${kernel_release}" =~ ^([0-9]+)\.([0-9]+) ]]; then
    echo "[ERROR] RCCL-SDMA could not parse Linux kernel version: ${kernel_release}" >&2
    exit 2
fi
if (( BASH_REMATCH[1] < 6 || (BASH_REMATCH[1] == 6 && BASH_REMATCH[2] < 8) )); then
    echo "[ERROR] RCCL-SDMA requires Linux kernel 6.8 or newer; found ${kernel_release}." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "env.NCCL_CTA_POLICY=2"
echo "env.NCCL_CUMEM_ENABLE=1"
echo "env.NCCL_LOCAL_REGISTER=0"
if [[ "${TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE:-}" != "unset" ]]; then
    echo "env.TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=${TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE:-true}"
fi

if [[ "${sdma_trigger}" == "1" ]]; then
    echo "env.SDMA_ALL_GATHER=1"
fi
if [[ "${fsdp_backend}" == "rccl_sdma" ]]; then
    echo "env.FSDP_ALL_GATHER_BACKEND=rccl_sdma"
fi

SRC="${SCRIPT_DIR}/sdma/hip_attr_drain_preload.c"
SO=/tmp/libhip_attr_drain.so
if [[ ! -f "${SRC}" ]]; then
    echo "[ERROR] [Hooks/sdma] interposer source not found: ${SRC}" >&2
    exit 1
fi
gcc -O2 -fPIC -shared "${SRC}" -o "${SO}" -ldl
echo "env.LD_PRELOAD=${SO}"
