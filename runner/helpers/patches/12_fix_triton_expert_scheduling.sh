#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# primus-cli --patch script: turn off Triton's expert scheduling for gfx1250
# (MI455X) before torchrun starts.
#
# Triton 3.8 enables AMDGPU expert scheduling on gfx1250 by default, which
# makes Triton kernels fault or return NaN at random. Setting
# TRITON_HIP_USE_EXPERT_SCHEDULING=0 misses some compile paths (torch inductor),
# so the default is switched off in Triton itself. Skips (exit 2) when Triton
# has no such switch or already has it off. Needs write access to Triton
# (root in the container).
###############################################################################
set -euo pipefail

state() {
    env -u TRITON_HIP_USE_EXPERT_SCHEDULING python3 -c '
from triton.backends.amd import compiler as c
f = getattr(c, "is_expert_scheduling_enabled", None)
print("n/a" if f is None else "on" if f("gfx1250") else "off")' 2>/dev/null || echo n/a
}

if [[ "$(state)" != on ]]; then
    echo "[fix_triton_expert_scheduling] nothing to do -- skipping"
    exit 2
fi
COMPILER=$(python3 -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "amd", "compiler.py"))')
sed -i '/^def is_expert_scheduling_enabled(arch):/a\    return False' "${COMPILER}"
if [[ "$(state)" != off ]]; then
    echo "[fix_triton_expert_scheduling] failed to patch ${COMPILER}" >&2
    exit 1
fi
echo "[fix_triton_expert_scheduling] expert scheduling off in ${COMPILER}"
