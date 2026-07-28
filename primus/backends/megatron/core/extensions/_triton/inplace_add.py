###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""In-place ``dst += src`` Triton kernel for the Primus Megatron extensions.

Used by the FP8/FP4 weight-gradient bridge in
:mod:`primus.backends.megatron.core.extensions.primus_turbo` to accumulate a
freshly produced weight gradient into the persistent ``main_grad`` buffer with
a single Triton launch (fp32 accumulate), instead of Torch's ``add_`` which
tiles a large consolidated grouped-expert ``main_grad`` of ``[E, N, K]`` into
multiple ~528M-element ``vectorized_elementwise_kernel`` launches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _inplace_add_kernel(dst_ptr, src_ptr, n_elements, BLOCK: tl.constexpr):
    """In-place ``dst += src`` over a flat buffer, accumulating in fp32.

    int64 offsets so a single launch covers tensors with > 2**31 elements
    (e.g. the consolidated grouped-expert ``main_grad`` of [E, N, K]),
    instead of Torch's ``add_`` which tiles into multiple ~528M-element
    ``vectorized_elementwise_kernel`` launches.
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    d = tl.load(dst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + offs, (d + s).to(dst_ptr.dtype.element_ty), mask=mask)


def inplace_add_triton_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``dst.add_(src)`` via a single Triton launch (fp32 accumulate).

    Falls back to Torch's ``add_`` when the layout is unsupported (not on CUDA
    / non-contiguous / shape mismatch). The write is in-place on ``dst``'s
    storage, so ``dst`` must be contiguous.
    """
    if not dst.is_cuda or not dst.is_contiguous() or dst.numel() != src.numel():
        return dst.add_(src)

    dst_flat = dst.view(-1)
    # reshape (not view) so a non-contiguous grad is materialized contiguously.
    src_flat = src.reshape(-1)
    n_elements = dst_flat.numel()
    BLOCK = 8192
    grid = (triton.cdiv(n_elements, BLOCK),)
    _inplace_add_kernel[grid](dst_flat, src_flat, n_elements, BLOCK=BLOCK)
    return dst


__all__ = [
    "inplace_add_triton_",
]
