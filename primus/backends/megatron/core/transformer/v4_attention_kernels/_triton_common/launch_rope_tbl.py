# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""launch line, PRIMUS_LAUNCH_ROPE_TBL=1 (default off): RoPECache.forward_pos_cast's table in ONE launch.

Replaces ``position_ids.float() * inv_freq -> cos / sin -> .to(dtype)`` (6 kernels: int->fp32 cast, mul, cos, sin, 2
casts) by one Triton kernel doing the same fp32 ops (RNE int->fp32, fp32 mul, ocml cos / sin = the functions torch's
cos / sin kernels call, RNE cast). Bit-identity is checked by test_launch6 on the real positions / inv_freq.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


def enabled() -> bool:
    return os.environ.get("PRIMUS_LAUNCH_ROPE_TBL", "0") == "1"


@triton.jit
def _rope_tbl_kernel(POS, INV, COS, SIN, n, R: tl.constexpr, BR: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BR)
    rm = r < n
    cm = c < R
    p = tl.load(POS + r, mask=rm, other=0).to(tl.float32)
    inv = tl.load(INV + c, mask=cm, other=0.0).to(tl.float32)  # bf16 buffer (Float16Module) -> fp32, exact
    f = p[:, None] * inv[None, :]
    o = r.to(tl.int64)[:, None] * R + c[None, :]
    m = rm[:, None] & cm[None, :]
    tl.store(COS + o, libdevice.cos(f).to(COS.dtype.element_ty), mask=m)
    tl.store(SIN + o, libdevice.sin(f).to(SIN.dtype.element_ty), mask=m)


def rope_table(position_ids: torch.Tensor, inv_freq: torch.Tensor, dtype):
    """(cos, sin) == (cos(pos.float()[..., None] * inv).to(dtype), sin(...).to(dtype)), contiguous; or None."""
    if not (position_ids.is_cuda and inv_freq.is_cuda and inv_freq.dim() == 1
            and inv_freq.dtype in (torch.float32, torch.bfloat16, torch.float16)

            and inv_freq.is_contiguous() and dtype in (torch.bfloat16, torch.float16, torch.float32)
            and not position_ids.dtype.is_floating_point and position_ids.numel() > 0):
        return None
    pos = position_ids.contiguous()
    R = inv_freq.numel()
    n = pos.numel()
    shape = tuple(pos.shape) + (R,)
    cos = torch.empty(shape, dtype=dtype, device=pos.device)
    sin = torch.empty(shape, dtype=dtype, device=pos.device)
    rows = 16
    _rope_tbl_kernel[(triton.cdiv(n, rows),)](pos, inv_freq, cos, sin, n, R=R, BR=triton.next_power_of_2(R),
                                               ROWS=rows, num_warps=4)
    return cos, sin
