###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""q_a / kv norms + kv RoPE of the V4 attention input as ONE autograd op (fuseattn, ``PRIMUS_V4_QKV_POST_FUSED=1``).

Unfused, each layer runs after the wq_a || wkv GEMM:
  fwd  q_layernorm (_rms_fwd), kv_layernorm (_rms_fwd), kv RoPE (_rope_gen_row)                   3 launches
  bwd  kv RoPE adjoint (_rope_gen_row), kv rms bwd + dgamma reduce, q rms bwd + dgamma reduce       5 launches
Here:
  fwd  ``_qkv_post_fwd_kernel``: both RMSNorms and the kv RoPE (+ the cos / sin table of the RoPE adjoint)  1 launch
  bwd  ``_qkv_post_bwd_kernel``: both RMSNorm backwards per row partition (programs [0, P) q, [P, 2P) kv) with the
       kv RoPE adjoint applied on the fly; ``_dg_reduce2_kernel``: both dgamma reductions                   2 launches
The concat op's [dq_a | dkv] quant is unchanged.

Bit-identical to the unfused chain (rmsnorm_g90 fwd / bwd incl. PRIMUS_GLUE_RMSNORM's main_grad writes, and
RoPEFromPositionsFn with the row kernel): every reduction runs on the original kernel's tile shape and warp count
(same reduction tree), every elementwise expression is the original's (the kv RoPE adjoint uses partner-column
loads instead of tl.split so its result keeps the load layout the RMSNorm reduction expects), and the dgamma
partials keep the original row partition and 2-row accumulation. Checked byte for byte by tools/test_qkvpost.py
(outputs, both input grads, both dgammas, plain and main_grad paths, both concat orders).
"""

import os

import torch
import triton
import triton.language as tl

from .rmsnorm_g90 import _glue_main_grad
from .rope_interleaved_partial import rope_kernel_vectors



def _rpp(M: int) -> int:
    """rows per dgamma partial exactly as RMSNormG90Fn.backward picks them (RB = 2): 16 at M = 4096 (256 partials)."""
    P = min(256, triton.cdiv(M, 2))
    return triton.cdiv(triton.cdiv(M, P), 2) * 2


@triton.jit
def _qkv_post_fwd_kernel(XQ, XKV, GQ, GKV, YQ, YKV, RSTDQ, RSTDKV, POS, INV, CSB, EPSQ, EPSKV,
                         HQ: tl.constexpr, HKV: tl.constexpr, NOPE: tl.constexpr, RB: tl.constexpr):
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    r64 = rows.to(tl.int64)
    # ---- q_a RMSNorm: _rms_fwd_kernel's tile [RB, HQ], 4 warps ----
    cq = tl.arange(0, HQ)
    x = tl.load(XQ + r64[:, None] * HQ + cq[None, :]).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / HQ + EPSQ)
    g = tl.load(GQ + cq).to(tl.float32)
    tl.store(YQ + r64[:, None] * HQ + cq[None, :], (x * rstd[:, None] * g[None, :]).to(YQ.dtype.element_ty))
    tl.store(RSTDQ + rows, rstd)
    # ---- kv RMSNorm: tile [RB, HKV] ----
    ck = tl.arange(0, HKV)
    x = tl.load(XKV + r64[:, None] * HKV + ck[None, :]).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / HKV + EPSKV)
    g = tl.load(GKV + ck).to(tl.float32)
    y = (x * rstd[:, None] * g[None, :]).to(YKV.dtype.element_ty)
    tl.store(RSTDKV + rows, rstd)
    # ---- kv RoPE: _rope_gen_row_kernel (INVERSE False) on the rounded norm output ----
    HALF: tl.constexpr = HKV // 2
    RDH: tl.constexpr = HALF - NOPE // 2
    fi = tl.arange(0, HALF) - NOPE // 2
    is_rope = fi >= 0
    ang = tl.load(POS + rows).to(tl.float32)[:, None] * tl.load(INV + fi, mask=is_rope, other=0.0).to(tl.float32)[None, :]
    c = tl.where(is_rope[None, :], tl.cos(ang), 1.0)
    s = tl.where(is_rope[None, :], tl.sin(ang), 0.0)
    a, b = tl.split(tl.reshape(y.to(tl.float32), (RB, HALF, 2)))
    o = tl.reshape(tl.join(a * c - b * s, a * s + b * c), (RB, HKV))
    tl.store(YKV + r64[:, None] * HKV + ck[None, :], o.to(YKV.dtype.element_ty))
    # backward table: cos / sin of -angle (the adjoint = row kernel with INVERSE) for the RD/2 rotated pairs
    angn = -ang
    m2 = is_rope[None, :] & (rows[:, None] >= 0)
    tl.store(CSB + r64[:, None] * (2 * RDH) + fi[None, :], tl.cos(angn), mask=m2)
    tl.store(CSB + r64[:, None] * (2 * RDH) + RDH + fi[None, :], tl.sin(angn), mask=m2)


@triton.jit
def _kv_dy(DYKV, CSB, r64, c0, W: tl.constexpr, HKV: tl.constexpr, NOPE: tl.constexpr, SDY=0,
           STRIDED: tl.constexpr = 0):
    """dy of the kv RMSNorm = RoPE adjoint of the incoming kv gradient on columns [c0, c0 + W), rounded to the
    activation dtype like _rope_gen_row_kernel's store (INVERSE: even = a c - b s, odd = a s + b c with (a, b) the
    (even, odd) pair). Partner-column loads instead of tl.split keep the plain load layout, so the RMSNorm
    reductions below see the same layout as _rms_bwd_kernel."""
    RDH: tl.constexpr = HKV // 2 - NOPE // 2
    cols = c0 + tl.arange(0, W)
    if STRIDED:  # membound PRIMUS_MB_SMALLS: incoming kv gradient with row stride SDY (no contiguous copy)
        row = DYKV + r64[:, None] * SDY
    else:
        row = DYKV + r64[:, None] * HKV
    gk = tl.load(row + cols[None, :])
    g = gk.to(tl.float32)
    gp = tl.load(row + (cols ^ 1)[None, :]).to(tl.float32)
    fi = cols // 2 - NOPE // 2
    rm = (fi >= 0)[None, :]
    cs = CSB + r64[:, None] * (2 * RDH) + fi[None, :]
    c = tl.load(cs, mask=rm, other=1.0)
    s = tl.load(cs + RDH, mask=rm, other=0.0)
    ev = ((cols % 2) == 0)[None, :]
    o = tl.where(ev, g * c - gp * s, gp * s + g * c)
    return o.to(gk.dtype).to(tl.float32)


@triton.jit
def _rms_bwd_part(DY, X, G, RSTD, CSB, DX, DGP, pid, H: tl.constexpr, IS_KV: tl.constexpr, HKV: tl.constexpr,
                  NOPE: tl.constexpr, RPP: tl.constexpr, SDY=0, STRIDED: tl.constexpr = 0):
    """_rms_bwd_kernel's body for one row partition (RB = 2, BH = H, same tile shapes -> same reduction trees); the
    kv dy is the RoPE adjoint of the incoming gradient, rounded like the separate RoPE kernel's store."""
    cols = tl.arange(0, H)
    g = tl.load(G + cols).to(tl.float32)
    dg = tl.zeros((2, H), dtype=tl.float32)
    r_lo = pid * RPP
    for r0 in range(r_lo, r_lo + RPP, 2):
        r64 = (r0 + tl.arange(0, 2)).to(tl.int64)
        off = r64[:, None] * H + cols[None, :]
        x = tl.load(X + off).to(tl.float32)
        if IS_KV:
            dy = _kv_dy(DY, CSB, r64, 0, H, HKV, NOPE, SDY, STRIDED)
        elif STRIDED:
            dy = tl.load(DY + r64[:, None] * SDY + cols[None, :]).to(tl.float32)
        else:
            dy = tl.load(DY + off).to(tl.float32)
        rstd = tl.load(RSTD + r64)
        xh = x * rstd[:, None]
        dxh = dy * g[None, :]
        mm = tl.sum(dxh * xh, axis=1) / H
        tl.store(DX + off, ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DX.dtype.element_ty))
        dg += dy * xh
    tl.store(DGP + pid * H + cols, tl.sum(dg, axis=0))


@triton.jit
def _qkv_post_bwd_kernel(DYQ, DYKV, XQ, XKV, GQ, GKV, RSTDQ, RSTDKV, CSB, DGPQ, DGPKV, DXQ, DXKV,
                         HQ: tl.constexpr, HKV: tl.constexpr, NOPE: tl.constexpr, RPP: tl.constexpr, SDQ=0, SDKV=0,
                         STRIDED: tl.constexpr = 0):
    """Both RMSNorm backwards (+ the kv RoPE adjoint) for one RPP-row partition: programs [0, P) do q, [P, 2P) kv."""
    pid = tl.program_id(0)
    P = tl.num_programs(0) // 2
    if pid < P:
        _rms_bwd_part(DYQ, XQ, GQ, RSTDQ, CSB, DXQ, DGPQ, pid, HQ, False, HKV, NOPE, RPP, SDQ, STRIDED)
    else:
        _rms_bwd_part(DYKV, XKV, GKV, RSTDKV, CSB, DXKV, DGPKV, pid - P, HKV, True, HKV, NOPE, RPP, SDKV, STRIDED)


@triton.jit
def _dg_reduce2_kernel(DGPQ, DGQ, DGPKV, DGKV, P, HQ: tl.constexpr, HKV: tl.constexpr, BH: tl.constexpr,
                       BP: tl.constexpr):
    """_rms_dg_reduce_kernel for both partial sets in one launch (programs [0, HQ/BH) -> q, the rest -> kv)."""
    pid = tl.program_id(0)
    NQ: tl.constexpr = HQ // BH
    if pid < NQ:
        cols = pid * BH + tl.arange(0, BH)
        acc = tl.zeros((BP, BH), dtype=tl.float32)
        for p0 in range(0, P, BP):
            pr = p0 + tl.arange(0, BP)
            acc += tl.load(DGPQ + pr[:, None] * HQ + cols[None, :], mask=(pr[:, None] < P), other=0.0)
        tl.store(DGQ + cols, tl.sum(acc, axis=0).to(DGQ.dtype.element_ty))
    else:
        cols = (pid - NQ) * BH + tl.arange(0, BH)
        acc = tl.zeros((BP, BH), dtype=tl.float32)
        for p0 in range(0, P, BP):
            pr = p0 + tl.arange(0, BP)
            acc += tl.load(DGPKV + pr[:, None] * HKV + cols[None, :], mask=(pr[:, None] < P), other=0.0)
        tl.store(DGKV + cols, tl.sum(acc, axis=0).to(DGKV.dtype.element_ty))


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def qkv_post_supported(q_a, kv_a, q_ln, kv_ln, rotary_dim) -> bool:
    """PRIMUS_V4_QKV_POST_FUSED=1 and the shapes / modules the fused kernels reproduce bit for bit."""
    if os.environ.get("PRIMUS_V4_QKV_POST_FUSED", "0") != "1":
        return False
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm
        from .rmsnorm_g90 import rmsnorm_g90_enabled
    except Exception:  # noqa: BLE001
        return False
    if not (type(q_ln) is PrimusTurboRMSNorm and type(kv_ln) is PrimusTurboRMSNorm and rmsnorm_g90_enabled()):
        return False
    if q_ln.zero_centered_gamma or kv_ln.zero_centered_gamma:
        return False
    HQ, HKV = q_a.shape[-1], kv_a.shape[-1]
    N = q_a.numel() // HQ
    return (
        q_a.is_cuda and kv_a.is_cuda and q_a.dtype == torch.bfloat16 and kv_a.dtype == torch.bfloat16
        and q_a.is_contiguous() and kv_a.is_contiguous() and kv_a.numel() // HKV == N
        and N % 128 == 0 and N >= 128 and 128 % _rpp(N) == 0
        and _pow2(HQ) and _pow2(HKV) and HQ % 128 == 0 and HKV % 128 == 0 and HQ <= 8192 and HKV <= 8192
        and rotary_dim >= 2 and _pow2(rotary_dim) and (HKV - rotary_dim) % 2 == 0 and rotary_dim <= HKV
        and q_ln.weight.dtype == torch.bfloat16 and kv_ln.weight.dtype == torch.bfloat16
        and q_ln.weight.numel() == HQ and kv_ln.weight.numel() == HKV
    )


class QKVPostFn(torch.autograd.Function):
    """(q_a [.., HQ], kv_a [.., HKV]) -> (rmsnorm(q_a) * gq, rope(rmsnorm(kv_a) * gkv)); see the module doc."""

    @staticmethod
    def forward(ctx, q_a, kv_a, gq, gkv, pos_flat, inv_freq, eps_q, eps_kv, rotary_dim, q_first):  # type: ignore[override]
        HQ, HKV = q_a.shape[-1], kv_a.shape[-1]
        N = q_a.numel() // HQ
        xq, xkv = q_a.reshape(N, HQ), kv_a.reshape(N, HKV)
        pos_c, inv_c = rope_kernel_vectors(pos_flat.reshape(-1), inv_freq)
        yq, ykv = torch.empty_like(xq), torch.empty_like(xkv)
        rq = torch.empty((N,), dtype=torch.float32, device=q_a.device)
        rkv = torch.empty((N,), dtype=torch.float32, device=q_a.device)
        csb = torch.empty((N, rotary_dim), dtype=torch.float32, device=q_a.device)
        gqc, gkvc = gq.contiguous(), gkv.contiguous()
        _qkv_post_fwd_kernel[(N // 2,)](xq, xkv, gqc, gkvc, yq, ykv, rq, rkv, pos_c, inv_c, csb, float(eps_q),
                                        float(eps_kv), HQ=HQ, HKV=HKV, NOPE=HKV - rotary_dim, RB=2, num_warps=4)
        ctx.save_for_backward(xq, xkv, gqc, gkvc, rq, rkv, csb)
        ctx.meta = (q_a.shape, kv_a.shape, HQ, HKV, rotary_dim, bool(q_first))
        # PRIMUS_GLUE_RMSNORM: dgamma straight into main_grad, claimed in forward like RMSNormG90Fn
        ctx.wparams = []
        for gm in (gq, gkv):
            if _glue_main_grad(gm) is not None:
                gm.grad_added_to_main_grad = True
                gm._primus_fwd_claim = True
                ctx.wparams.append(gm)
            else:
                ctx.wparams.append(None)
        return yq.view(q_a.shape), ykv.view(kv_a.shape)

    @staticmethod
    def backward(ctx, dyq, dykv):  # type: ignore[override]
        xq, xkv, gq, gkv, rq, rkv, csb = ctx.saved_tensors
        qshape, kvshape, HQ, HKV, rd, q_first = ctx.meta
        N = xq.shape[0]
        dev = xq.device
        strided = 0
        if os.environ.get("PRIMUS_MB_SMALLS", "0") == "1":
            # membound: row-strided incoming gradients are read in place (a [N, H] view with unit column stride),
            # no .contiguous() copy; same values, same tiles
            try:
                dq_v, dkv_v = dyq.view(N, HQ), dykv.view(N, HKV)
            except RuntimeError:
                dq_v = dkv_v = None
            if dq_v is not None and dq_v.stride(1) == 1 and dkv_v.stride(1) == 1 and (
                    not dq_v.is_contiguous() or not dkv_v.is_contiguous()):
                dyq2, dykv2, strided = dq_v, dkv_v, 1
        if not strided:
            dyq2 = dyq.reshape(N, HQ).contiguous()
            dykv2 = dykv.reshape(N, HKV).contiguous()
        rpp = _rpp(N)
        P = triton.cdiv(N, rpp)
        dgpq = torch.empty((P, HQ), dtype=torch.float32, device=dev)
        dgpkv = torch.empty((P, HKV), dtype=torch.float32, device=dev)
        dxq = torch.empty(qshape, dtype=xq.dtype, device=dev)
        dxkv = torch.empty(kvshape, dtype=xkv.dtype, device=dev)
        for t in (dyq2, dykv2, xq, xkv, gq, gkv, rq, rkv, csb, dgpq, dgpkv, dxq, dxkv):
            assert t.is_cuda and t.data_ptr() != 0
        if strided:
            _qkv_post_bwd_kernel[(2 * P,)](dyq2, dykv2, xq, xkv, gq, gkv, rq, rkv, csb, dgpq, dgpkv, dxq, dxkv,
                                           HQ=HQ, HKV=HKV, NOPE=HKV - rd, RPP=rpp, SDQ=dyq2.stride(0),
                                           SDKV=dykv2.stride(0), STRIDED=1, num_warps=8)
        else:
            _qkv_post_bwd_kernel[(2 * P,)](dyq2, dykv2, xq, xkv, gq, gkv, rq, rkv, csb, dgpq, dgpkv, dxq, dxkv,
                                           HQ=HQ, HKV=HKV, NOPE=HKV - rd, RPP=rpp, num_warps=8)
        # dgamma: one reduce launch for both (into main_grad on the glue first-write path, like RMSNormG90Fn)
        outs, posts, grads = [], [], []
        from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

        for wp, g in zip(ctx.wparams, (gq, gkv)):
            mg = _glue_main_grad(wp) if wp is not None else None
            if mg is not None:
                first = main_grad_first_write(wp)
                if first and mg.dtype == g.dtype:
                    outs.append(mg.view(-1)); posts.append(None)
                else:
                    t = torch.empty_like(g); outs.append(t); posts.append((mg, first))
                grads.append(_get_dummy_wgrad(list(g.shape), g.dtype))
            else:
                t = torch.empty_like(g); outs.append(t); posts.append(None); grads.append(t)
        ctx.wparams = None
        _dg_reduce2_kernel[(HQ // 64 + HKV // 64,)](dgpq, outs[0], dgpkv, outs[1], P, HQ=HQ, HKV=HKV, BH=64, BP=64,
                                                     num_warps=4)
        for o, pst in zip(outs, posts):
            if pst is not None:
                mg, first = pst
                mg.view(-1).copy_(o) if first else mg.view(-1).add_(o)
        return dxq, dxkv, grads[0], grads[1], None, None, None, None, None, None


def qkv_post(q_a, kv_a, gq, gkv, pos_flat, inv_freq, *, eps_q, eps_kv, rotary_dim, q_first=True):
    """-> (rmsnorm(q_a) * gq, rope(rmsnorm(kv_a) * gkv)); ``q_first`` is unused by the math (kept for callers)."""
    return QKVPostFn.apply(q_a, kv_a, gq, gkv, pos_flat, inv_freq, eps_q, eps_kv, rotary_dim, q_first)
