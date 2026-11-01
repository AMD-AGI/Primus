###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""In-place scaling of a contiguous gradient run by a factor that stays on the device.

Every product is taken in fp32 and rounded once to the run's dtype, as TE's ``multi_tensor_scale_tensor``
does, so the scaled gradients are bit for bit the ones of the kernel it replaces.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 16384


@triton.jit
def _scale_by_device_scalar_kernel(X, S, n, BLOCK: tl.constexpr):
    """X[block] *= S[0] (fp32 product, one rounding to X's dtype)."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.load(S)
    x = tl.load(X + offs, mask=mask, other=0.0)
    tl.store(X + offs, (x.to(tl.float32) * s).to(X.dtype.element_ty), mask=mask)


def scale_by_device_scalar(x: torch.Tensor, scale: torch.Tensor) -> None:
    """``x *= scale`` in place for a contiguous ``x`` and a one-element fp32 ``scale`` on its device."""
    n = x.numel()
    if n:
        _scale_by_device_scalar_kernel[(triton.cdiv(n, _BLOCK),)](x, scale, n, BLOCK=_BLOCK, num_warps=8)
