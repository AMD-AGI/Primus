###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bandwidth-bound (batched) 2-D transpose for 16-bit tensors, used to give hipBLASLt a K-contiguous operand.

``torch``'s generic strided copy (``x.t().contiguous()``) runs at ~1.3 TB/s on MI455X (1.6 ms for a 1 GB bf16
matrix); this tiled Triton kernel goes through LDS and runs near HBM bandwidth.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose2d_kernel(X, Y, M, N, sxb, sxm, syb, syn, BM: tl.constexpr, BN: tl.constexpr):
    # X[b, m, n] (unit stride in n) -> Y[b, n, m] (unit stride in m)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(X + b * sxb + rm[:, None].to(tl.int64) * sxm + rn[None, :], mask=msk)
    xt = tl.trans(x)
    tl.store(Y + b * syb + rn[:, None].to(tl.int64) * syn + rm[None, :], xt, mask=tl.trans(msk))


def transpose_last2(x: torch.Tensor, out: torch.Tensor | None = None, BM: int = 64, BN: int = 64,
                    num_warps: int = 4) -> torch.Tensor:
    """``x[..., M, N]`` (last dim unit stride) -> contiguous ``[..., N, M]`` (same as ``x.transpose(-1, -2).contiguous()``)."""
    if x.dim() == 2:
        xb = x.unsqueeze(0)
    elif x.dim() == 3:
        xb = x
    else:
        raise ValueError("transpose_last2: 2-D or 3-D input")
    if xb.stride(-1) != 1:
        raise ValueError("transpose_last2: last dim must have unit stride")
    B, M, N = xb.shape
    if out is None:
        out = torch.empty(x.shape[:-2] + (N, M), dtype=x.dtype, device=x.device)
    yb = out.view(B, N, M)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), B)
    _transpose2d_kernel[grid](xb, yb, M, N, xb.stride(0), xb.stride(1), yb.stride(0), yb.stride(1),
                              BM=BM, BN=BN, num_warps=num_warps)
    return out
