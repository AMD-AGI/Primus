#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: opt into the SDMA/RCCL dispatch path for FSDP
# all-gather.
#
# Single trigger -- the only knob:
#
#   export SDMA_ALL_GATHER=1
#   primus-cli direct -- train pretrain --config <any existing yaml>
#
# When SDMA_ALL_GATHER=1, this hook:
#   1. Exports the zero-CTA env that RCCL needs to actually take the
#      copy-engine path (NCCL_CTA_POLICY=2, NCCL_CUMEM_ENABLE=1, ...).
#   2. Propagates SDMA_ALL_GATHER=1 into the launched torchrun children
#      so the companion Python patch's gate fires there too. See
#      primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py.
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

if [[ "${SDMA_ALL_GATHER:-0}" != "1" ]]; then
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Zero-CTA env for the RCCL copy-engine path. FSDP requests CTA=ZERO
#    via pg_options too, but setting these explicitly here makes the
#    dispatch path observable. 
#Lorri: Check the header in sdma_symm_mem_collectives.py for more details.
echo "env.NCCL_CTA_POLICY=2"
echo "env.NCCL_CUMEM_ENABLE=1"
echo "env.NCCL_LOCAL_REGISTER=0"
echo "env.TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"

# 2) Make the trigger visible to torchrun children so the Python patch
#    fires there. primus-cli direct doesn't inherit the host env into
#    the child unless it's either CLI-passed via --env or hook-emitted
#    via env.*.
echo "env.SDMA_ALL_GATHER=1"

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
