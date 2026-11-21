###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Weighted RMSNorm forward/backward in Triton (g90mem, 2026-11-19), used by PrimusTurboRMSNorm.

Forward: 2 rows per program. Backward: 256 programs x row blocks (dx + a per-program dgamma partial in
registers), then a column-parallel deterministic reduction of the partials. MI455X, [4096, H] bf16, fwd+bwd:
H=4096 53.3 -> 32.5 us, H=1024 44.1 -> 29.3 us, H=512 38.8 -> 25.3 us vs primus_turbo.ops.rmsnorm.
``PRIMUS_RMSNORM_G90=0`` restores the Turbo op.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def rmsnorm_g90_enabled() -> bool:
    return os.environ.get("PRIMUS_RMSNORM_G90", "1") != "0"


def _glue_main_grad(gamma):
    """PRIMUS_GLUE_RMSNORM=1: ``gamma.main_grad`` when dgamma can be written into it directly, else None."""
    if os.environ.get("PRIMUS_GLUE_RMSNORM", "0") != "1":
        return None
    mg = getattr(gamma, "main_grad", None)
    if (
        isinstance(gamma, torch.nn.Parameter)
        and hasattr(gamma, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.is_cuda
        and mg.shape == gamma.shape
        and mg.is_contiguous()
        and gamma.dim() == 1
        and not getattr(gamma, "shared", False)
    ):
        return mg
    return None


@triton.jit
def _rms_fwd_kernel(X, G, Y, RSTD, M, H: tl.constexpr, eps, BH: tl.constexpr, RB: tl.constexpr):
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    cols = tl.arange(0, BH)
    rm = rows < M
    cm = cols < H
    m2 = rm[:, None] & cm[None, :]
    off = rows[:, None].to(tl.int64) * H + cols[None, :]
    x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / H + eps)
    g = tl.load(G + cols, mask=cm, other=0.0).to(tl.float32)
    tl.store(Y + off, (x * rstd[:, None] * g[None, :]).to(Y.dtype.element_ty), mask=m2)
    tl.store(RSTD + rows, rstd, mask=rm)


@triton.jit
def _rms_bwd_kernel(DY, X, G, RSTD, DX, DGP, M, H: tl.constexpr, BH: tl.constexpr, RB: tl.constexpr, ROWS_PER_PROG):
    pid = tl.program_id(0)
    cols = tl.arange(0, BH)
    cm = cols < H
    g = tl.load(G + cols, mask=cm, other=0.0).to(tl.float32)
    dg = tl.zeros((RB, BH), dtype=tl.float32)
    r_lo = pid * ROWS_PER_PROG
    for r0 in range(r_lo, r_lo + ROWS_PER_PROG, RB):
        rows = r0 + tl.arange(0, RB)
        rm = rows < M
        m2 = rm[:, None] & cm[None, :]
        off = rows[:, None].to(tl.int64) * H + cols[None, :]
        x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask=m2, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
        xh = x * rstd[:, None]
        dxh = dy * g[None, :]
        mm = tl.sum(dxh * xh, axis=1) / H
        tl.store(DX + off, ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DX.dtype.element_ty), mask=m2)
        dg += dy * xh
    tl.store(DGP + pid * H + cols, tl.sum(dg, axis=0), mask=cm)


@triton.jit
def _rms_dg_reduce_kernel(DGP, DG, P, H: tl.constexpr, BH: tl.constexpr, BP: tl.constexpr):
    cols = tl.program_id(0) * BH + tl.arange(0, BH)
    cm = cols < H
    acc = tl.zeros((BP, BH), dtype=tl.float32)
    for p0 in range(0, P, BP):
        pr = p0 + tl.arange(0, BP)
        acc += tl.load(DGP + pr[:, None] * H + cols[None, :], mask=(pr[:, None] < P) & cm[None, :], other=0.0)
    tl.store(DG + cols, tl.sum(acc, axis=0).to(DG.dtype.element_ty), mask=cm)


@triton.jit
def _rms_dx_kernel(DY, X, G, RSTD, DX, M, H: tl.constexpr, BH: tl.constexpr, RB: tl.constexpr):
    """dx only, row-parallel (no loop)."""
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    cols = tl.arange(0, BH)
    rm = rows < M
    cm = cols < H
    m2 = rm[:, None] & cm[None, :]
    off = rows[:, None].to(tl.int64) * H + cols[None, :]
    g = tl.load(G + cols, mask=cm, other=0.0).to(tl.float32)
    x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
    dy = tl.load(DY + off, mask=m2, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
    xh = x * rstd[:, None]
    dxh = dy * g[None, :]
    mm = tl.sum(dxh * xh, axis=1) / H
    tl.store(DX + off, ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DX.dtype.element_ty), mask=m2)


@triton.jit
def _rms_dg_partial_kernel(DY, X, RSTD, DGP, M, H: tl.constexpr, BC: tl.constexpr, RS: tl.constexpr):
    """dgamma partial over RS rows x BC columns: sum_t dy * x * rstd (column-parallel, no loop)."""
    pc = tl.program_id(0)
    pr = tl.program_id(1)
    cols = pc * BC + tl.arange(0, BC)
    rows = pr * RS + tl.arange(0, RS)
    rm = rows < M
    cm = cols < H
    m2 = rm[:, None] & cm[None, :]
    off = rows[:, None].to(tl.int64) * H + cols[None, :]
    x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
    dy = tl.load(DY + off, mask=m2, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
    tl.store(DGP + pr * H + cols, tl.sum(dy * x * rstd[:, None], axis=0), mask=cm)


class RMSNormG90Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps):  # type: ignore[override]
        H = x.shape[-1]
        x2 = x.reshape(-1, H)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        BH = triton.next_power_of_2(H)
        rb = 2 if BH <= 8192 else 1
        y = torch.empty_like(x2)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        g = gamma.contiguous()
        _rms_fwd_kernel[(triton.cdiv(M, rb),)](x2, g, y, rstd, M, H, float(eps), BH=BH, RB=rb, num_warps=4)
        ctx.save_for_backward(x2, g, rstd)
        ctx.x_shape = x.shape
        # PRIMUS_GLUE_RMSNORM=1: dgamma goes straight into gamma.main_grad (no DDP hook copy / add); flagged in
        # forward like the o_a / Turbo fused-wgrad writers.
        ctx.wparam = None
        if _glue_main_grad(gamma) is not None:
            gamma.grad_added_to_main_grad = True
            gamma._primus_fwd_claim = True
            ctx.wparam = gamma
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):  # type: ignore[override]
        x2, g, rstd = ctx.saved_tensors
        M, H = x2.shape
        dy2 = dy.reshape(-1, H)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()
        BH = triton.next_power_of_2(H)
        rb = 2 if BH <= 8192 else 1
        dx = torch.empty_like(x2)
        if os.environ.get("PRIMUS_RMSNORM_G90_BWD", "loop") == "split":
            # dx row-parallel + dgamma column-parallel partials (two passes over x, dy; no serial row loops)
            _rms_dx_kernel[(triton.cdiv(M, rb),)](dy2, x2, g, rstd, dx, M, H, BH=BH, RB=rb, num_warps=4)
            RS = 64
            P = triton.cdiv(M, RS)
            dgp = torch.empty((P, H), dtype=torch.float32, device=x2.device)
            _rms_dg_partial_kernel[(triton.cdiv(H, 128), P)](dy2, x2, rstd, dgp, M, H, BC=128, RS=RS, num_warps=4)
        else:
            P = min(256, triton.cdiv(M, rb))
            rpp = triton.cdiv(triton.cdiv(M, P), rb) * rb
            P = triton.cdiv(M, rpp)
            dgp = torch.empty((P, H), dtype=torch.float32, device=x2.device)
            _rms_bwd_kernel[(P,)](dy2, x2, g, rstd, dx, dgp, M, H, BH=BH, RB=rb, ROWS_PER_PROG=rpp, num_warps=8)
        wparam, ctx.wparam = ctx.wparam, None
        mg = _glue_main_grad(wparam) if wparam is not None else None
        if mg is not None:
            from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

            first = main_grad_first_write(wparam)
            if first and mg.dtype == g.dtype:
                # beta 0: the reduce stores into main_grad exactly the bits the DDP hook would have copied
                _rms_dg_reduce_kernel[(triton.cdiv(H, 64),)](dgp, mg.view(-1), P, H, BH=64, BP=64, num_warps=4)
            else:
                dg = torch.empty_like(g)
                _rms_dg_reduce_kernel[(triton.cdiv(H, 64),)](dgp, dg, P, H, BH=64, BP=64, num_warps=4)
                mg.view(-1).copy_(dg) if first else mg.view(-1).add_(dg)
            return dx.view(ctx.x_shape), _get_dummy_wgrad(list(g.shape), g.dtype), None
        dg = torch.empty_like(g)
        _rms_dg_reduce_kernel[(triton.cdiv(H, 64),)](dgp, dg, P, H, BH=64, BP=64, num_warps=4)
        return dx.view(ctx.x_shape), dg, None


def rmsnorm_g90(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return RMSNormG90Fn.apply(x, gamma, eps)
