###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HC head / tail glue as Triton kernels.

The trunk enters HC through ``_lift_streams_in`` (``h.unsqueeze(2).expand(B, S, 4, D).contiguous()``, an 88 us
torch copy) and leaves through ``HyperHead`` (``(pre.unsqueeze(-1) * x).sum(-2).to(bf16)``: a 268 MB fp32 product
tensor, a reduce and a cast in the forward, two more 268 MB fp32 products, a reduce and a cast in the backward).
Both become one streaming pass each way:

* :func:`hc_lift` -- fwd: x[t, k, :] = h[t, :]; bwd: dh = bf16(sum_k float(dx_k)) in torch's summation order.
* :func:`hc_head_collapse` -- fwd: y = bf16(sum_k pre_k * float(x_k)) (fp32 products rounded separately, no FMA,
  torch's summation order); bwd: dx_k = bf16(float(dy) * pre_k) (bit-identical to the eager autograd) and
  dpre_k = sum_d float(dy) * float(x_k) in the order of torch's reduce over the exact fp32 products
  (bit-identical): for D = 4096 summed in registers in one pass (``_hc_head_bwd_fused_kernel``), otherwise the
  products are written and reduced by torch itself.

Switches: ``PRIMUS_HC_LIFT_FUSED=1`` / ``PRIMUS_HC_HEAD_FUSED=1``; bf16 x, K = 4, CUDA, contiguous only.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def hc_lift_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_LIFT_FUSED", "0") == "1"


def hc_head_fused_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_HEAD_FUSED", "0") == "1"


@triton.jit
def _sum4(a, b, c, d):
    # (((0 + a) + b) + c) + d: torch's reduce starts from +0.0, so all-(-0.0) streams sum to +0.0; bit-identical to
    # torch's fp32 sum over the stream dim and to its expand backward
    return (((a + 0.0) + b) + c) + d


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
def _hc_lift_bwd_kernel(G, DH, D, BD: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    m = c < D
    base = G + t * (4 * D) + c
    g0 = tl.load(base, mask=m, other=0.0).to(tl.float32)
    g1 = tl.load(base + D, mask=m, other=0.0).to(tl.float32)
    g2 = tl.load(base + 2 * D, mask=m, other=0.0).to(tl.float32)
    g3 = tl.load(base + 3 * D, mask=m, other=0.0).to(tl.float32)
    tl.store(DH + t * D + c, _sum4(g0, g1, g2, g3).to(DH.dtype.element_ty), mask=m)


@triton.jit
def _hc_head_fwd_kernel(X, PRE, Y, D, BD: tl.constexpr):
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
    tl.store(Y + t * D + c, _sum4(a, b, cc, d).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _hc_head_bwd_prod_kernel(G, X, PRE, DX, PROD, D, BD: tl.constexpr):
    """dx_k = bf16(dy * pre_k) and the fp32 products dy * x_k (exact: bf16 x bf16 fits fp32) for torch's own
    reduce over D, so dpre is summed in exactly the eager order."""
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


@triton.jit
def _pair(s, R: tl.constexpr, W: tl.constexpr):
    """[R, W] -> [R, W/2]: s[2m] + s[2m+1] (one shfl_down level)."""
    a, b = tl.split(tl.reshape(s, [R, W // 2, 2]))
    return a + b


@triton.jit
def _hc_head_bwd_fused_kernel(G, X, PRE, DX, DPRE, D, BR: tl.constexpr):
    """dx_k as ``_hc_head_bwd_prod_kernel`` and dpre_k = sum_d dy * x_k for D = 4096, without the fp32 product tensor.

    dpre follows torch's ``reduce_kernel<512, 1>`` for the contiguous inner reduction ``prod.sum(-1)`` over 4096 fp32
    with at least 16 outputs (32 x 16 blocks, 4-wide vector loads): lane l keeps acc_i += prod[128 j + 4 l + i]
    (i < 4, j < 32), the lane value is ((acc_0 + acc_1) + acc_2) + acc_3 and the 32 lanes combine as a shfl_down
    tree with offsets 1, 2, ..., 16.
    The products are exact in fp32 (bf16 x bf16), so acc + dy * x rounds as torch's add of the stored product
    (``enable_fp_fusion=False`` keeps dx's multiply unfused too).
    """
    pid = tl.program_id(0)
    R: tl.constexpr = BR * 4
    rr = pid * BR + tl.arange(0, BR)
    kk = tl.arange(0, 4)
    ll = tl.arange(0, 32)
    ii = tl.arange(0, 4)
    r64 = rr.to(tl.int64)
    # [BR, 4, 32, 4] element offsets (row, stream, lane, vector slot) inside one 128-column slice j
    col = ll[None, None, :, None] * 4 + ii[None, None, None, :]
    xoff = (r64[:, None, None, None] * 4 + kk[None, :, None, None]) * D + col
    goff = r64[:, None, None, None] * D + col
    pk = tl.load(PRE + r64[:, None, None, None] * 4 + kk[None, :, None, None])
    acc = tl.zeros([BR, 4, 32, 4], dtype=tl.float32)
    for j in range(0, D // 128):
        g = tl.load(G + goff + j * 128).to(tl.float32)
        x = tl.load(X + xoff + j * 128).to(tl.float32)
        tl.store(DX + xoff + j * 128, (g * pk).to(DX.dtype.element_ty))
        acc = acc + g * x
    a = tl.reshape(acc, [R, 32, 2, 2])
    e0, e1 = tl.split(a)  # e0: i in {0, 2}, e1: i in {1, 3}
    a0, a2 = tl.split(e0)
    a1, a3 = tl.split(e1)
    s = ((a0 + a1) + a2) + a3  # [R, 32]
    s = _pair(s, R, 32)
    s = _pair(s, R, 16)
    s = _pair(s, R, 8)
    s = _pair(s, R, 4)
    s = _pair(s, R, 2)
    tl.store(DPRE + (pid * BR) * 4 + tl.arange(0, R), tl.reshape(s, [R]))


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
        _hc_lift_bwd_kernel[(B * S, triton.cdiv(D, bd))](g, dh, D, BD=bd, num_warps=8 if bd >= 4096 else 4)
        return dh


class HeadGradSlot:
    """PRIMUS_FUSE2_HEAD_GADD: the HyperHead input feeds the rms-linear mix (flattened) and the collapse. The
    collapse backward (autograd runs it first) leaves its x-gradient here; the rms-linear backward adds its own into
    that buffer in-kernel (bf16 + bf16 -> fp32 -> bf16, autograd's add) instead of the separate 268 MB-traffic add.
    """

    __slots__ = ("g", "fused")

    def __init__(self):
        self.g = self.fused = None

    def take(self, like):
        g, self.g = self.g, None
        if g is None or g.dtype != like.dtype or not g.is_contiguous() or g.numel() != like.numel():
            return None
        return g.view(like.shape)


class HeadFanOut(torch.autograd.Function):
    """``x -> (x_rms, x_collapse)`` aliases: their two gradients meet here instead of in autograd's add."""

    @staticmethod
    def forward(ctx, x, slot):
        ctx.slot = slot
        return x.view_as(x), x.view_as(x)

    @staticmethod
    def backward(ctx, g_r, g_c):
        slot, ctx.slot = ctx.slot, None
        fused = slot.fused if slot is not None else None
        if slot is not None:
            slot.fused = slot.g = None
        if (
            fused is not None
            and g_r is not None
            and g_c is not None
            and g_r.data_ptr() == fused.data_ptr()
            and g_c.data_ptr() == fused.data_ptr()
        ):
            return g_r, None  # the rms-linear backward already added the collapse's gradient (in its buffer)
        if g_r is None:
            return g_c, None
        if g_c is None:
            return g_r, None
        return g_c + g_r, None  # autograd's order: collapse first, then the rms-linear


def head_gadd_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSE2_HEAD_GADD", "0") == "1"


class HCHeadCollapseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pre, gslot=None):  # x [..., 4, D] bf16, pre [..., 4] fp32 -> [..., D] bf16
        lead = x.shape[:-2]
        D = x.shape[-1]
        x3 = x.reshape(-1, 4, D)
        p2 = pre.reshape(-1, 4).contiguous()
        M = x3.shape[0]
        y = torch.empty((M, D), dtype=x.dtype, device=x.device)
        _ok_ptr(x3, p2, y)
        bd = _bd(D)
        _hc_head_fwd_kernel[(M, triton.cdiv(D, bd))](
            x3, p2, y, D, BD=bd, num_warps=8 if bd >= 4096 else 4, enable_fp_fusion=False
        )
        ctx.save_for_backward(x3, p2)
        ctx.lead = lead
        ctx.pre_shape = pre.shape
        ctx.gslot = gslot
        return y.view(*lead, D)

    @staticmethod
    def backward(ctx, gy):
        x3, p2 = ctx.saved_tensors
        M, K, D = x3.shape
        g = gy.contiguous().view(M, D)
        dx = torch.empty_like(x3)
        # torch reduces 16+ outputs with 32 x 16 blocks (the order the one-pass kernel follows); fewer outputs widen
        # the block to 64 lanes, a different order
        if D == 4096 and M * K >= 16:
            dpre = torch.empty((M, K), dtype=torch.float32, device=x3.device)
            _ok_ptr(g, x3, p2, dx, dpre)
            _hc_head_bwd_fused_kernel[(M,)](g, x3, p2, dx, dpre, D, BR=1, num_warps=4, enable_fp_fusion=False)
            if ctx.gslot is not None:
                ctx.gslot.g, ctx.gslot = dx, None
            return dx.view(*ctx.lead, K, D), dpre.view(ctx.pre_shape).to(p2.dtype), None
        # exact: the fp32 products go through the same torch reduce as the eager autograd (sum_to over the last dim
        # of a contiguous [..., 4, D] fp32 tensor), so dpre is bit-identical
        prod = torch.empty((*ctx.lead, K, D), dtype=torch.float32, device=x3.device)
        _ok_ptr(g, dx, prod)
        bd = _bd(D)
        _hc_head_bwd_prod_kernel[(M, triton.cdiv(D, bd))](
            g, x3, p2, dx, prod, D, BD=bd, num_warps=8 if bd >= 4096 else 4, enable_fp_fusion=False
        )
        dpre = prod.sum(dim=(prod.dim() - 1,), keepdim=True)
        if ctx.gslot is not None:
            ctx.gslot.g, ctx.gslot = dx, None
        return dx.view(*ctx.lead, K, D), dpre.view(ctx.pre_shape).to(p2.dtype), None


def hc_lift_supported(h: torch.Tensor, K: int) -> bool:
    return h.is_cuda and h.dim() == 3 and K == 4 and h.dtype == torch.bfloat16 and h.numel() > 0


def hc_head_supported(x: torch.Tensor, pre: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape[-2] == 4
        and x.is_contiguous()
        and x.numel() > 0
        and pre.dtype == torch.float32
        and tuple(pre.shape) == tuple(x.shape[:-1])
    )


def hc_lift(h: torch.Tensor) -> torch.Tensor:
    return HCLiftFn.apply(h)


def hc_head_collapse(x: torch.Tensor, pre: torch.Tensor, gslot=None) -> torch.Tensor:
    return HCHeadCollapseFn.apply(x, pre, gslot)


__all__ = [
    "hc_lift",
    "hc_head_collapse",
    "hc_lift_enabled",
    "hc_head_fused_enabled",
    "hc_lift_supported",
    "hc_head_supported",
]
