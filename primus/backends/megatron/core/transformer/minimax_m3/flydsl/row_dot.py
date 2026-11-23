###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``delta = rowsum(dO * O)`` of the MSA backward in one Triton launch (fp32 sum of bf16 rows)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_dot_kernel(a_ptr, b_ptr, out_ptr, R, D: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    m = rows < R
    off = rows[:, None] * D + tl.arange(0, D)[None, :]
    a = tl.load(a_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
    b = tl.load(b_ptr + off, mask=m[:, None], other=0.0).to(tl.float32)
    tl.store(out_ptr + rows, tl.sum(a * b, axis=1), mask=m)


def row_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a.float() * b.float()).sum(-1)`` for contiguous bf16 ``[..., D]`` tensors, fp32."""
    D = a.shape[-1]
    out = torch.empty(a.shape[:-1], dtype=torch.float32, device=a.device)
    R = out.numel()
    BR = 32
    _row_dot_kernel[(triton.cdiv(R, BR),)](a, b, out, R, D=D, BR=BR, num_warps=4)
    return out
