###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""In-place scaling of contiguous gradient buffers, for the Primus Megatron gradient patches.

Every product is taken in fp32 and rounded once to the buffer's dtype, as ``grad_data *= factor``
does, so the scaled gradients are bit for bit the ones of the PyTorch op they replace.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 16384


@triton.jit
def _scale_sumsq_kernel(X, PARTIAL, n, factor, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    """PARTIAL[pid] = sum of X[block]^2 in fp32, and X[block] *= factor (fp32 product, one rounding)."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(PARTIAL + pid, tl.sum(x * x, axis=0))
    if SCALE:
        tl.store(X + offs, (x * factor).to(X.dtype.element_ty), mask=mask)


def scale_and_norm(x: torch.Tensor, factor: float) -> torch.Tensor:
    """``x *= factor`` in place; returns the L2 norm of ``x`` before the scaling, in ``x.dtype``."""
    n = x.numel()
    if n == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    grid = triton.cdiv(n, _BLOCK)
    partial = torch.empty(grid, dtype=torch.float32, device=x.device)
    _scale_sumsq_kernel[(grid,)](x, partial, n, float(factor), BLOCK=_BLOCK, SCALE=factor != 1.0, num_warps=8)
    return partial.sum().sqrt().to(x.dtype)


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
