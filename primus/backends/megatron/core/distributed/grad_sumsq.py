###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bandwidth-bound sum of squares over a flat gradient bucket (fp32 accumulation).

``bucket.grad_data.norm()`` (ATen ``reduce_kernel<512,1>`` with ``NormTwoOps``) reads a
1.06 GB bf16 bucket in ~161 us on MI455X (6.6 TB/s, 36% of HBM). This two-stage reduction
(grid-stride partial sums per program, then a tiny ``sum`` of the partials) is
deterministic and streams at close to the HBM rate. Used only for the DDP NaN/Inf grad
check, where the value itself is not consumed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sumsq_partial_kernel(X_PTR, N, OUT_PTR, BLOCK: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, NPROG * BLOCK):
        idx = start + offs
        x = tl.load(X_PTR + idx, mask=idx < N, other=0.0).to(tl.float32)
        acc += x * x
    tl.store(OUT_PTR + pid, tl.sum(acc, axis=0))


def grad_sumsq(x: torch.Tensor, nprog: int = 4096, block: int = 2048) -> torch.Tensor:
    """fp32 sum of squares of a contiguous 1-D CUDA tensor (0-dim fp32 result, no sync)."""
    x = x.reshape(-1)
    n = x.numel()
    if n < (1 << 31):
        nprog = max(1, min(nprog, triton.cdiv(n, block)))
        part = torch.empty((nprog,), dtype=torch.float32, device=x.device)
        _sumsq_partial_kernel[(nprog,)](x, n, part, BLOCK=block, NPROG=nprog)
        return part.sum()
    return x.float().pow(2).sum()
