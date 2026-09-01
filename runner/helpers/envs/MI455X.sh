#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# AMD MI455X (gfx1250) GPU-specific settings.
# Common settings are in base_env.sh. Values derive from world size so one file
# serves single- and multi-GPU runs.
#
LOG_INFO_RANK0 "Loading MI455X (gfx1250) specific settings..."

_PRIMUS_WORLD_SIZE=$(( ${GPUS_PER_NODE:-1} * ${NNODES:-1} ))

# 3 is a V4-Pro workaround that no longer reproduces here and costs 38% on Pro itself.
export AMD_SERIALIZE_COPY=${AMD_SERIALIZE_COPY:-0}

# Cards in a node are XGMI 1-hop, so P2P is the right transport above one rank.
if [[ -z "${NCCL_P2P_DISABLE:-}" ]]; then
    if [[ "${_PRIMUS_WORLD_SIZE}" -gt 1 ]]; then
        export NCCL_P2P_DISABLE=0
    else
        export NCCL_P2P_DISABLE=1
    fi
fi

# Single node needs no IB. Clear NCCL_IB_HCA rather than inherit an ionic_* list from a
# multi-node recipe: naming devices this host lacks stalls RCCL init.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-}
export USING_AINIC=${USING_AINIC:-0}

# Escape hatch for an image whose librccl was built without gfx1250 device code (its
# .hip_fatbin is NOBITS, and every collective then faults). Unset is correct otherwise.
if [[ -n "${RCCL_LIB_DIR:-}" && -f "${RCCL_LIB_DIR}/lib/librccl.so.1" ]]; then
    export LD_LIBRARY_PATH="${RCCL_LIB_DIR}/lib:${LD_LIBRARY_PATH:-}"
    LOG_INFO_RANK0 "MI455X: using private RCCL from ${RCCL_LIB_DIR}"
elif [[ "${_PRIMUS_WORLD_SIZE}" -gt 1 ]]; then
    LOG_INFO_RANK0 "MI455X: RCCL_LIB_DIR unset at world size ${_PRIMUS_WORLD_SIZE}; if this image's librccl lacks gfx1250 device code, the first collective will fault."
fi

unset _PRIMUS_WORLD_SIZE
