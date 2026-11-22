# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""fuse2 line (PRIMUS_FUSE2_IDX_ROT=1): the CSA indexer's ``rotate_activation(apply_interleaved_partial_rope(x))``
in one Triton kernel.

Unfused: two cos/sin casts, ``_apply_rope_fwd_kernel`` (bf16 out), a hipBLASLt GEMM ``x_rot @ H`` with the
Sylvester Hadamard matrix (fp32 accumulate, bf16 out) and an elementwise ``* n**-0.5`` (bf16 out) -- 5 launches and
three HBM round trips of a [rows, 128] tensor (the query side is 64 MB per CSA layer). Here every row tile is
rotated in registers with the same bf16 expressions as ``_apply_rope_fwd_kernel``, multiplied by the bf16 Hadamard
matrix on the matrix cores (fp32 accumulate over K = 128), rounded to bf16, scaled in fp32 and rounded again --
the unfused path's roundings. Forward only (the indexer runs without autograd here); anything else falls back.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_hadamard_kernel(X, COS, SIN, HM, OUT, R, H, NORM,
                          HD: tl.constexpr, NOPE: tl.constexpr, RD_HALF: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    pos = rows // H
    dt = X.dtype.element_ty
    kk = tl.arange(0, RD_HALF)
    cos = tl.load(COS + pos[:, None].to(tl.int64) * RD_HALF + kk[None, :], mask=rmask[:, None], other=0.0).to(dt)
    sin = tl.load(SIN + pos[:, None].to(tl.int64) * RD_HALF + kk[None, :], mask=rmask[:, None], other=0.0).to(dt)
    base = X + rows[:, None].to(tl.int64) * HD
    even = tl.load(base + NOPE + 2 * kk[None, :], mask=rmask[:, None], other=0.0)
    odd = tl.load(base + NOPE + 2 * kk[None, :] + 1, mask=rmask[:, None], other=0.0)
    # same bf16 expressions as _apply_rope_fwd_kernel
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    tail = tl.reshape(tl.join(rot_even, rot_odd), (BR, 2 * RD_HALF))  # interleaved pairs back in place
    nn = tl.arange(0, NOPE)
    nope = tl.load(base + nn[None, :], mask=rmask[:, None], other=0.0)
    xr = tl.reshape(tl.permute(tl.join(nope, tail), (0, 2, 1)), (BR, HD))  # [nope | rotated tail]
    cols = tl.arange(0, HD)
    hm = tl.load(HM + cols[:, None] * HD + cols[None, :])
    acc = tl.dot(xr, hm)  # fp32 accumulate over K = HD, like the bf16 GEMM
    y = acc.to(dt).to(tl.float32) * NORM
    tl.store(OUT + rows[:, None].to(tl.int64) * HD + cols[None, :], y.to(dt), mask=rmask[:, None])


@triton.jit
def _rope_hadamard_v2_kernel(X, COS, SIN, HM, OUT, R, H, NORM,
                             HD: tl.constexpr, NOPE: tl.constexpr, RD_HALF: tl.constexpr, BR: tl.constexpr):
    """Same math as _rope_hadamard_kernel with contiguous row loads: the rope partner of every tail element comes
    from a register-level pair swap; rot = x*c + sgn*(partner*s) is, element by element, the same bf16 expression
    (even: e*c - o*s, odd: o*c + e*s)."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    pos = rows // H
    dt = X.dtype.element_ty
    cols = tl.arange(0, HD)
    x = tl.load(X + rows[:, None].to(tl.int64) * HD + cols[None, :], mask=rmask[:, None], other=0.0)
    a, b = tl.split(tl.reshape(x, (BR, HD // 2, 2)))
    partner = tl.reshape(tl.join(b, a), (BR, HD))
    is_tail = cols >= NOPE
    ci = tl.where(is_tail, (cols - NOPE) // 2, 0)
    cm = rmask[:, None] & is_tail[None, :]
    c = tl.load(COS + pos[:, None].to(tl.int64) * RD_HALF + ci[None, :], mask=cm, other=0.0).to(dt)
    s = tl.load(SIN + pos[:, None].to(tl.int64) * RD_HALF + ci[None, :], mask=cm, other=0.0).to(dt)
    ps = partner * s
    ps = tl.where((cols % 2 == 0)[None, :], -ps, ps)
    rot = x * c + ps
    xr = tl.where(is_tail[None, :], rot, x)
    hm = tl.load(HM + cols[:, None] * HD + cols[None, :])
    acc = tl.dot(xr, hm)
    y = acc.to(dt).to(tl.float32) * NORM
    tl.store(OUT + rows[:, None].to(tl.int64) * HD + cols[None, :], y.to(dt), mask=rmask[:, None])


_HM = {}


def _hmat(n, device, dtype):
    key = (n, str(device), dtype)
    h = _HM.get(key)
    if h is None:
        from primus.backends.megatron.core.transformer.hadamard_rotation import _sylvester

        h = _sylvester(n, device, dtype).contiguous()
        _HM[key] = h
    return h


def fuse2_idx_rot_ok(x: torch.Tensor, cos: torch.Tensor, rotary_dim: int) -> bool:
    if os.environ.get("PRIMUS_FUSE2_IDX_ROT", "0") != "1":
        return False
    if os.environ.get("PRIMUS_BF16G2_IDXQ", "0") == "1":  # bf16g2's GEMM+RoPE+Hadamard fusion wins when both are on
        return False
    hd = x.shape[-1]
    nope = hd - rotary_dim
    return (
        x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.dim() >= 3
        and not (x.requires_grad and torch.is_grad_enabled())
        and hd == 128 and rotary_dim == nope == 64
        and cos.is_cuda
    )


def rope_hadamard(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, norm=None) -> torch.Tensor:
    """``rotate_activation(apply_interleaved_partial_rope(x, cos, sin, rotary_dim=rotary_dim))`` for
    ``x [..., H, 128]`` bf16 and ``cos / sin`` with one ``rotary_dim // 2`` row per leading position."""
    hd = x.shape[-1]
    H = x.shape[-2]
    R = x.numel() // hd
    N = R // H
    rd_half = rotary_dim // 2
    c = cos.reshape(N, rd_half)
    s = sin.reshape(N, rd_half)
    if not c.is_contiguous():
        c = c.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    out = torch.empty_like(x)
    hm = _hmat(hd, x.device, x.dtype)
    norm = hd ** -0.5 if norm is None else norm
    BR = int(os.environ.get("PRIMUS_FUSE2_IDX_ROT_BR", "64"))
    nw = int(os.environ.get("PRIMUS_FUSE2_IDX_ROT_NW", "4"))
    kern = _rope_hadamard_v2_kernel if os.environ.get("PRIMUS_FUSE2_IDX_ROT_V", "2") == "2" else _rope_hadamard_kernel
    kern[(triton.cdiv(R, BR),)](x, c, s, hm, out, R, H, float(norm), HD=hd, NOPE=hd - rotary_dim,
                                RD_HALF=rd_half, BR=BR, num_warps=nw)
    return out
