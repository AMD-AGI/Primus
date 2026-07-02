###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared AMD Triton compiler-knob helpers for the triton_v2 sparse-MLA kernels.

primus_turbo globally enables ``TRITON_HIP_USE_BLOCK_PINGPONG`` /
``TRITON_HIP_USE_ASYNC_COPY`` (via ``set_triton_knobs_gfx950``) as soon as it is
imported by the training runtime. Those knobs double-buffer (ping-pong) a
kernel's LDS operand tiles. Measured on gfx950 (bench_v4_attention, flash) they
are a *pessimization* for the V4 sparse-MLA kernels — the forward is ~16-29%
slower and the backward is slightly slower with them on — and they also overflow
the 160 KB LDS limit for the wide (BH=64/TK=128) dKV tiling.

These knobs are read at *compile time* and are NOT part of Triton's compile
cache key (``HIPOptions.hash`` does not hash them), so compiling a kernel inside
:func:`amd_pingpong_disabled` pins that kernel to the non-ping-pong schedule for
the whole process, while restoring the knobs on exit leaves every other kernel
(compiled outside the scope) exactly as primus_turbo configured it.
"""

import contextlib

import triton


@contextlib.contextmanager
def amd_pingpong_disabled():
    """Temporarily disable the AMD Triton ping-pong / async-copy LDS knobs."""
    amd = getattr(getattr(triton, "knobs", None), "amd", None)
    if amd is None:
        yield
        return
    prev_pp = amd.use_block_pingpong
    prev_ac = amd.use_async_copy
    try:
        amd.use_block_pingpong = False
        amd.use_async_copy = False
        yield
    finally:
        amd.use_block_pingpong = prev_pp
        amd.use_async_copy = prev_ac
