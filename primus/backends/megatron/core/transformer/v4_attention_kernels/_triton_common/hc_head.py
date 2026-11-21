###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HC head / tail glue as Triton kernels (hc track, 2026-11-21).

The trunk enters HC through ``_lift_streams_in`` (``h.unsqueeze(2).expand(B, S, 4, D).contiguous()``, an 88 us
torch copy) and leaves through ``HyperHead`` (``(pre.unsqueeze(-1) * x).sum(-2).to(bf16)``: a 268 MB fp32 product
tensor, a reduce and a cast in the forward, two more 268 MB fp32 products, a reduce and a cast in the backward).
Both become one streaming pass each way:

* :func:`hc_lift` -- fwd: x[t, k, :] = h[t, :]; bwd: dh = bf16(sum_k float(dx_k)) in torch's summation order.
* :func:`hc_head_collapse` -- fwd: y = bf16(sum_k pre_k * float(x_k)) (fp32 products rounded separately, no FMA,
  torch's summation order); bwd: dx_k = bf16(float(dy) * pre_k) (bit-identical to the eager autograd) and
  dpre_k = sum_d float(dy) * float(x_k): with ``PRIMUS_HC_HEAD_FUSED=1`` the exact fp32 products are written and
  reduced by the same torch reduce as the eager path (bit-identical); ``=2`` sums them in registers (one pass,
  different fp32 order: loss@20 9.8672 in one e2e run, outside the 9.868-9.872 spread, so not the default).

Opt-in: ``PRIMUS_HC_LIFT_FUSED=1`` / ``PRIMUS_HC_HEAD_FUSED=1|2``; bf16 x, K = 4, CUDA, contiguous only.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def hc_lift_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_LIFT_FUSED", "0") == "1"


def hc_head_fused_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_HEAD_FUSED", "0") in ("1", "2")


@triton.jit
def _sum4(a, b, c, d, ORDER: tl.constexpr):
    if ORDER == 0:  # (((0 + a) + b) + c) + d: torch's reduce starts from +0.0, so all-(-0.0) streams sum to +0.0
        r = (((a + 0.0) + b) + c) + d
    elif ORDER == 1:  # (a + b) + (c + d)
        r = (a + b) + (c + d)
    else:  # (a + c) + (b + d)
        r = (a + c) + (b + d)
    return r


@triton.jit
def _hc_lift_fwd_kernel(H, X, D, BD: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    m = c < D
    h = tl.load(H + t * D + c, mask=m)
    base = X + t * (4 * D) + c
    tl.store(base, h, mask=m)
    tl.store(base + D, h, mask=m)
    tl.store(base + 2 * D, h, mask=m)
    tl.store(base + 3 * D, h, mask=m)


@triton.jit
def _hc_lift_bwd_kernel(G, DH, D, BD: tl.constexpr, ORDER: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    m = c < D
    base = G + t * (4 * D) + c
    g0 = tl.load(base, mask=m, other=0.0).to(tl.float32)
    g1 = tl.load(base + D, mask=m, other=0.0).to(tl.float32)
    g2 = tl.load(base + 2 * D, mask=m, other=0.0).to(tl.float32)
    g3 = tl.load(base + 3 * D, mask=m, other=0.0).to(tl.float32)
    tl.store(DH + t * D + c, _sum4(g0, g1, g2, g3, ORDER).to(DH.dtype.element_ty), mask=m)


@triton.jit
def _hc_head_fwd_kernel(X, PRE, Y, D, BD: tl.constexpr, ORDER: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    m = c < D
    base = X + t * (4 * D) + c
    p0 = tl.load(PRE + t * 4)
    p1 = tl.load(PRE + t * 4 + 1)
    p2 = tl.load(PRE + t * 4 + 2)
    p3 = tl.load(PRE + t * 4 + 3)
    a = p0 * tl.load(base, mask=m, other=0.0).to(tl.float32)
    b = p1 * tl.load(base + D, mask=m, other=0.0).to(tl.float32)
    cc = p2 * tl.load(base + 2 * D, mask=m, other=0.0).to(tl.float32)
    d = p3 * tl.load(base + 3 * D, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + t * D + c, _sum4(a, b, cc, d, ORDER).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _hc_head_bwd_kernel(G, X, PRE, DX, DPRE, D, BD: tl.constexpr):
    """One row per program (BD = next_pow2(D)): dx_k = bf16(dy * pre_k), dpre_k = sum_d dy * x_k."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BD)
    m = c < D
    g = tl.load(G + t * D + c, mask=m, other=0.0).to(tl.float32)
    base = X + t * (4 * D) + c
    dbase = DX + t * (4 * D) + c
    for k in tl.static_range(4):
        pk = tl.load(PRE + t * 4 + k)
        xk = tl.load(base + k * D, mask=m, other=0.0).to(tl.float32)
        tl.store(dbase + k * D, (g * pk).to(DX.dtype.element_ty), mask=m)
        tl.store(DPRE + t * 4 + k, tl.sum(g * xk, axis=0))


@triton.jit
def _hc_head_bwd_prod_kernel(G, X, PRE, DX, PROD, D, BD: tl.constexpr):
    """dx_k = bf16(dy * pre_k) and the fp32 products dy * x_k (exact: bf16 x bf16 fits fp32) for torch's own
    reduce over D, so dpre is summed in exactly the eager order (PRIMUS_HC_HEAD_FUSED=1)."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    m = c < D
    g = tl.load(G + t * D + c, mask=m, other=0.0).to(tl.float32)
    base = X + t * (4 * D) + c
    dbase = DX + t * (4 * D) + c
    pbase = PROD + t * (4 * D) + c
    for k in tl.static_range(4):
        pk = tl.load(PRE + t * 4 + k)
        xk = tl.load(base + k * D, mask=m, other=0.0).to(tl.float32)
        tl.store(dbase + k * D, (g * pk).to(DX.dtype.element_ty), mask=m)
        tl.store(pbase + k * D, g * xk, mask=m)


# summation orders matching torch's reduce over the size-4 stream dim (calibrated in bench_head.py)
_LIFT_ORDER = 0  # ((g0 + g1) + g2) + g3: bit-identical to torch's expand backward (calib_lift.py, spread exponents)
_HEAD_ORDER = 0  # ((a + b) + c) + d: bit-identical to torch's fp32 sum over the stream dim (bench_head.py, 0 of 67M differ)


def _bd(D):
    return min(4096, triton.next_power_of_2(D))


def _ok_ptr(*ts):
    for t in ts:
        if not (t.is_cuda and t.numel() > 0 and t.data_ptr() != 0):
            raise RuntimeError("hc_head: refusing to launch on an empty / non-CUDA tensor")


class HCLiftFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):  # h [B, S, D] -> [B, S, 4, D]
        B, S, D = h.shape
        hc = h.contiguous()
        x = torch.empty((B, S, 4, D), dtype=h.dtype, device=h.device)
        _ok_ptr(hc, x)
        M = B * S
        bd = _bd(D)
        _hc_lift_fwd_kernel[(M, triton.cdiv(D, bd))](hc, x, D, BD=bd, num_warps=8 if bd >= 4096 else 4)
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        B, S, K, D = g.shape
        dh = torch.empty((B, S, D), dtype=g.dtype, device=g.device)
        _ok_ptr(g, dh)
        bd = _bd(D)
        _hc_lift_bwd_kernel[(B * S, triton.cdiv(D, bd))](g, dh, D, BD=bd, ORDER=_LIFT_ORDER,
                                                          num_warps=8 if bd >= 4096 else 4)
        return dh


class HCHeadCollapseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pre):  # x [..., 4, D] bf16, pre [..., 4] fp32 -> [..., D] bf16
        lead = x.shape[:-2]
        D = x.shape[-1]
        x3 = x.reshape(-1, 4, D)
        p2 = pre.reshape(-1, 4).contiguous()
        M = x3.shape[0]
        y = torch.empty((M, D), dtype=x.dtype, device=x.device)
        _ok_ptr(x3, p2, y)
        bd = _bd(D)
        _hc_head_fwd_kernel[(M, triton.cdiv(D, bd))](x3, p2, y, D, BD=bd, ORDER=_HEAD_ORDER,
                                                      num_warps=8 if bd >= 4096 else 4, enable_fp_fusion=False)
        ctx.save_for_backward(x3, p2)
        ctx.lead = lead
        ctx.pre_shape = pre.shape
        return y.view(*lead, D)

    @staticmethod
    def backward(ctx, gy):
        x3, p2 = ctx.saved_tensors
        M, K, D = x3.shape
        g = gy.contiguous().view(M, D)
        dx = torch.empty_like(x3)
        if os.environ.get("PRIMUS_HC_HEAD_FUSED", "0") == "2":
            # one pass: dpre summed per row in registers (fp32, different order than torch's reduce)
            dpre = torch.empty((M, 4), dtype=torch.float32, device=x3.device)
            _ok_ptr(g, dx, dpre)
            _hc_head_bwd_kernel[(M,)](g, x3, p2, dx, dpre, D, BD=triton.next_power_of_2(D), num_warps=4,
                                      enable_fp_fusion=False)
            return dx.view(*ctx.lead, K, D), dpre.view(ctx.pre_shape).to(p2.dtype)
        # exact: the fp32 products go through the same torch reduce as the eager autograd (sum_to over the last dim
        # of a contiguous [..., 4, D] fp32 tensor), so dpre is bit-identical
        prod = torch.empty((*ctx.lead, K, D), dtype=torch.float32, device=x3.device)
        _ok_ptr(g, dx, prod)
        bd = _bd(D)
        _hc_head_bwd_prod_kernel[(M, triton.cdiv(D, bd))](g, x3, p2, dx, prod, D, BD=bd,
                                                           num_warps=8 if bd >= 4096 else 4, enable_fp_fusion=False)
        dpre = prod.sum(dim=(prod.dim() - 1,), keepdim=True)
        return dx.view(*ctx.lead, K, D), dpre.view(ctx.pre_shape).to(p2.dtype)


def hc_lift_supported(h: torch.Tensor, K: int) -> bool:
    return h.is_cuda and h.dim() == 3 and K == 4 and h.dtype == torch.bfloat16 and h.numel() > 0


def hc_head_supported(x: torch.Tensor, pre: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-2] == 4 and x.is_contiguous() and x.numel() > 0
            and pre.dtype == torch.float32 and tuple(pre.shape) == tuple(x.shape[:-1]))


def hc_lift(h: torch.Tensor) -> torch.Tensor:
    return HCLiftFn.apply(h)


def hc_head_collapse(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    return HCHeadCollapseFn.apply(x, pre)


__all__ = ["hc_lift", "hc_head_collapse", "hc_lift_enabled", "hc_head_fused_enabled", "hc_lift_supported",
           "hc_head_supported"]
