#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: opt into an SDMA/RCCL AllGather path.
#
# Backend selectors:
#
#   export FSDP_ALL_GATHER_BACKEND=rccl_sdma
#   export MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma  # direct symmetric buffers
#   primus-cli direct -- train pretrain --config <any existing yaml>
#
# The FSDP selector uses global NCCL_CTA_POLICY=2 because FSDP's custom
# collective owns the relevant communicator. The Megatron distributed-
# optimizer selector creates a dedicated zero-CTA process group in Python, so
# this hook deliberately does not change the global policy; ReduceScatter and
# gradient-norm AllReduce remain on their stock RCCL path.
#
# For either selector, this hook:
#   1. Exports cuMem and allocator prerequisites used by RCCL's copy-engine path.
#   2. Propagates the selector into torchrun children.
#   3. Rebuilds the bundled LD_PRELOAD interposer
#      (hooks/sdma/hip_attr_drain_preload.c) into /tmp and exports
#      LD_PRELOAD. The interposer is a workaround for the ROCm
#      cuDeviceGetAttribute hipErrorInvalidValue TLS-leak hit by RCCL's
#      cuMem code path on builds that don't have the upstream fix.
#      Recompiled every time so it never goes stale relative to the
#      source.
#      Lorri: Check JIRA ticket ROCM-24832 for more details.
#
# Anything else (HSA_SDMA_LINEAR_B2B, NCCL_DEBUG, ...) can be set
# independently via the normal `export` / `primus-cli --env` flow.

set -euo pipefail

fsdp_backend="${FSDP_ALL_GATHER_BACKEND:-}"
megatron_backend="${MEGATRON_PARAM_GATHER_BACKEND:-}"

if [[ "${fsdp_backend}" != "rccl_sdma" && "${megatron_backend}" != "rccl_sdma" ]]; then
    exit 0
fi

if [[ "${fsdp_backend}" == "rccl_sdma" && "${megatron_backend}" == "rccl_sdma" ]]; then
    echo "[ERROR] FSDP and Megatron RCCL-SDMA cannot be enabled together: " \
         "the FSDP path requires global NCCL_CTA_POLICY=2, while the Megatron " \
         "path requires zero CTA only on its dedicated parameter-AllGather group." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Common cuMem and allocator prerequisites for the RCCL copy-engine path.
echo "env.NCCL_CUMEM_ENABLE=1"
echo "env.NCCL_LOCAL_REGISTER=0"
if [[ "${TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE:-}" != "unset" ]]; then
    echo "env.TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=${TORCH_NCCL_ALLOCATOR_HOOK_OVERRIDE:-true}"
fi

# 2) Make the active selector visible to torchrun children. primus-cli direct
#    does not otherwise inherit arbitrary host environment variables.
if [[ "${fsdp_backend}" == "rccl_sdma" ]]; then
    echo "env.NCCL_CTA_POLICY=2"
    echo "env.FSDP_ALL_GATHER_BACKEND=rccl_sdma"
    if [[ -n "${PRIMUS_TURBO_GROUPED_GEMM_BACKEND:-}" ]]; then
        echo "env.PRIMUS_TURBO_GROUPED_GEMM_BACKEND=${PRIMUS_TURBO_GROUPED_GEMM_BACKEND}"
    fi
fi
if [[ "${megatron_backend}" == "rccl_sdma" ]]; then
    echo "env.MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma"
    echo "env.ENABLE_SDMA_ALLGATHER=0"
    echo "env.MEGATRON_RCCL_SDMA_CTA_POLICY=${MEGATRON_RCCL_SDMA_CTA_POLICY:-2}"
    echo "env.MEGATRON_RCCL_SDMA_EAGER_INIT=${MEGATRON_RCCL_SDMA_EAGER_INIT:-1}"
    for name in MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES MEGATRON_RCCL_SDMA_LOG; do
        if [[ -n "${!name:-}" ]]; then
            echo "env.${name}=${!name}"
        fi
    done
fi
# The native Primus RCCL build used by Megatron has the cuMem probe fix and
# does not need the legacy attribute-drain interposer.
if [[ "${fsdp_backend}" != "rccl_sdma" ]]; then
    exit 0
fi

# 3) Always (re)build the interposer. The source is tiny and gcc is
#    typically <1s; we don't bother with a staleness check so the .so
#    can never lag behind the source.
SRC="${SCRIPT_DIR}/sdma/hip_attr_drain_preload.c"
SO=/tmp/libhip_attr_drain.so
if [[ ! -f "${SRC}" ]]; then
    echo "[ERROR] [Hooks/sdma] interposer source not found: ${SRC}" >&2
    exit 1
fi
gcc -O2 -fPIC -shared "${SRC}" -o "${SO}" -ldl
echo "env.LD_PRELOAD=${SO}"
