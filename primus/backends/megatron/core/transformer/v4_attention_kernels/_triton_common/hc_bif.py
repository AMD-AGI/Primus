###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HC streaming kernels with more bytes in flight (opt3-mem, 2026-11-20; opt-in HCF_CFG["bif"] / PRIMUS_HC_BIF=1).

g90mem ATT (prof/hcrows): ``_hc_collapse2d`` (s_wait_xcnt 71%), ``_hc_dpre2d`` (xcnt 56% + loadcnt 16%),
``_hc_post_bwd2d`` (loadcnt 46% + xcnt 21% + LDS/barrier 24%) and ``_hc_expand_fwd`` (loadcnt 51% + xcnt 24%)
are VMEM-latency bound at 43-65% of the HBM floor with DRAM traffic already at compulsory (GL2C): one small tile
per program, loads issued then waited on, cross-warp reductions.

Same math and the same outputs / partial layouts as the 2-D kernels in ``hc_fused.py`` (drop-in), but:

* persistent grid-stride programs (``NPROG`` programs, several [BT, BC] tiles each, column tile fastest so the
  programs running together stream neighbouring addresses);
* the NEXT tile's global loads are issued before the current tile is consumed (software pipeline; the prefetch
  is kept as raw bf16 so it costs half the VGPRs of the fp32 working set. These kernels need far fewer VGPRs than
  the WMMA dx+dW kernel where the register double buffer failed, so they stay <= 256 VGPRs -> 2+ waves/SIMD);
* K = 4 unrolled into four 2-D [BT, BC] planes: rows map to warps, columns to lanes, so every reduction over
  the column tile stays inside one wave (no LDS round trip / barrier, the 24% of post_bwd2d);
* all per-row scalars (pre / post / comb) are loaded with the tile prefetch, not inside the h loop.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _tile_coords(i, n_c, M, BT: tl.constexpr, BC: tl.constexpr, NT):
    ti = i // n_c
    ci = i - ti * n_c
    t = ti * BT + tl.arange(0, BT)
    c = ci * BC + tl.arange(0, BC)
    tm = (t < M) & (i < NT)
    return t.to(tl.int64), c, tm, ci


@triton.jit
def _ld4(X, t64, c, tm, D):
    base = X + t64[:, None] * (4 * D) + c[None, :]
    m = tm[:, None]
    x0 = tl.load(base, mask=m, other=0.0)
    x1 = tl.load(base + D, mask=m, other=0.0)
    x2 = tl.load(base + 2 * D, mask=m, other=0.0)
    x3 = tl.load(base + 3 * D, mask=m, other=0.0)
    return x0, x1, x2, x3


@triton.jit
def _ld_row4(P, t64, tm):
    p0 = tl.load(P + t64 * 4, mask=tm, other=0.0).to(tl.float32)
    p1 = tl.load(P + t64 * 4 + 1, mask=tm, other=0.0).to(tl.float32)
    p2 = tl.load(P + t64 * 4 + 2, mask=tm, other=0.0).to(tl.float32)
    p3 = tl.load(P + t64 * 4 + 3, mask=tm, other=0.0).to(tl.float32)
    return p0, p1, p2, p3


@triton.jit
def _pack4(a, b, c, d):
    """Four [BT] row reductions -> one [BT, 4] tile, so the store needs one layout conversion, not four."""
    # join(X, Y)[t, m, n] = (X if n == 0 else Y)[t, m]  ->  flat index 2m + n: a, b, c, d
    return tl.reshape(tl.join(tl.join(a, c), tl.join(b, d)), (a.shape[0], 4))


# ------------------------------------------------------------------------------------------------
# F2 collapse: coll[t, d] = sum_k pre[t, k] x[t, k, d]
# ------------------------------------------------------------------------------------------------
@triton.jit
def _hc_collapse_bif_kernel(X, PRE, COLL, M, D, BT: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    n_c = D // BC
    NT = tl.cdiv(M, BT) * n_c
    t, c, tm, _ci = _tile_coords(pid, n_c, M, BT, BC, NT)
    a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
    p0, p1, p2, p3 = _ld_row4(PRE, t, tm)
    for i in range(pid, NT, NPROG):
        x0, x1, x2, x3 = a0, a1, a2, a3
        q0, q1, q2, q3 = p0, p1, p2, p3
        tc, cc, tmc = t, c, tm
        # issue the next tile's loads before touching the current one
        t, c, tm, _ci = _tile_coords(i + NPROG, n_c, M, BT, BC, NT)
        a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
        p0, p1, p2, p3 = _ld_row4(PRE, t, tm)
        acc = q0[:, None] * x0.to(tl.float32) + q1[:, None] * x1.to(tl.float32)
        acc += q2[:, None] * x2.to(tl.float32) + q3[:, None] * x3.to(tl.float32)
        tl.store(COLL + tc[:, None] * D + cc[None, :], acc.to(COLL.dtype.element_ty), mask=tmc[:, None])


# ------------------------------------------------------------------------------------------------
# B2 dpre partials: dprep[c_tile, t, k] = <x[t, k, c_tile], dcoll[t, c_tile]>
# ------------------------------------------------------------------------------------------------
@triton.jit
def _hc_dpre_bif_kernel(X, DC, DPREP, M, D, BT: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    n_c = D // BC
    NT = tl.cdiv(M, BT) * n_c
    t, c, tm, ci = _tile_coords(pid, n_c, M, BT, BC, NT)
    a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
    g = tl.load(DC + t[:, None] * D + c[None, :], mask=tm[:, None], other=0.0)
    for i in range(pid, NT, NPROG):
        x0, x1, x2, x3, gc = a0, a1, a2, a3, g
        tc, tmc, cic = t, tm, ci
        t, c, tm, ci = _tile_coords(i + NPROG, n_c, M, BT, BC, NT)
        a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
        g = tl.load(DC + t[:, None] * D + c[None, :], mask=tm[:, None], other=0.0)
        gf = gc.to(tl.float32)
        k4 = tl.arange(0, 4)
        v = _pack4(tl.sum(x0.to(tl.float32) * gf, axis=1), tl.sum(x1.to(tl.float32) * gf, axis=1),
                   tl.sum(x2.to(tl.float32) * gf, axis=1), tl.sum(x3.to(tl.float32) * gf, axis=1))
        tl.store(DPREP + cic.to(tl.int64) * M * 4 + tc[:, None] * 4 + k4[None, :], v, mask=tmc[:, None])


# ------------------------------------------------------------------------------------------------
# B1 expand backward without dx: dout = sum_h post_h dnew_h; per column tile dpost_h = <out, dnew_h>,
# dcomb^T[h, k] = <dnew_h, x_k>
# ------------------------------------------------------------------------------------------------
@triton.jit
def _dcomb_row(gh, x0, x1, x2, x3):
    return _pack4(tl.sum(gh * x0, axis=1), tl.sum(gh * x1, axis=1), tl.sum(gh * x2, axis=1), tl.sum(gh * x3, axis=1))


@triton.jit
def _hc_post_bwd_bif_kernel(G, X, OUT, POST, DOUT, DPOSTP, DCOMBTP, M, D,
                            BT: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    n_c = D // BC
    NT = tl.cdiv(M, BT) * n_c
    t, c, tm, ci = _tile_coords(pid, n_c, M, BT, BC, NT)
    a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
    b0, b1, b2, b3 = _ld4(G, t, c, tm, D)
    ob = tl.load(OUT + t[:, None] * D + c[None, :], mask=tm[:, None], other=0.0)
    p0, p1, p2, p3 = _ld_row4(POST, t, tm)
    for i in range(pid, NT, NPROG):
        x0, x1, x2, x3 = a0, a1, a2, a3
        g0, g1, g2, g3 = b0.to(tl.float32), b1.to(tl.float32), b2.to(tl.float32), b3.to(tl.float32)
        o = ob.to(tl.float32)
        q0, q1, q2, q3 = p0, p1, p2, p3
        tc, cc, tmc, cic = t, c, tm, ci
        t, c, tm, ci = _tile_coords(i + NPROG, n_c, M, BT, BC, NT)
        a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
        b0, b1, b2, b3 = _ld4(G, t, c, tm, D)
        ob = tl.load(OUT + t[:, None] * D + c[None, :], mask=tm[:, None], other=0.0)
        p0, p1, p2, p3 = _ld_row4(POST, t, tm)

        dout = q0[:, None] * g0 + q1[:, None] * g1 + q2[:, None] * g2 + q3[:, None] * g3
        tl.store(DOUT + tc[:, None] * D + cc[None, :], dout.to(DOUT.dtype.element_ty), mask=tmc[:, None])
        rowp = cic.to(tl.int64) * M + tc
        k4 = tl.arange(0, 4)
        k16 = tl.arange(0, 16)
        dp = _pack4(tl.sum(o * g0, axis=1), tl.sum(o * g1, axis=1), tl.sum(o * g2, axis=1), tl.sum(o * g3, axis=1))
        tl.store(DPOSTP + rowp[:, None] * 4 + k4[None, :], dp, mask=tmc[:, None])
        y0, y1, y2, y3 = x0.to(tl.float32), x1.to(tl.float32), x2.to(tl.float32), x3.to(tl.float32)
        # [t, k, m, n] with h = 2m + n -> permute to [t, m, n, k] -> flat h * 4 + k (dcomb^T[h, k])
        dj = tl.join(tl.join(_dcomb_row(g0, y0, y1, y2, y3), _dcomb_row(g2, y0, y1, y2, y3)),
                     tl.join(_dcomb_row(g1, y0, y1, y2, y3), _dcomb_row(g3, y0, y1, y2, y3)))
        dct = tl.reshape(tl.permute(dj, (0, 2, 3, 1)), (BT, 16))
        tl.store(DCOMBTP + rowp[:, None] * 16 + k16[None, :], dct, mask=tmc[:, None])


# ------------------------------------------------------------------------------------------------
# F3 expand: new[t, h, d] = post[t, h] out[t, d] + sum_k C[t, h, k] x[t, k, d]  (C = the COMB argument as
# passed by hc_fused, i.e. comb^T: identical indexing to _hc_expand_fwd_kernel)
# ------------------------------------------------------------------------------------------------
@triton.jit
def _ld_c4(C, t64, tm, h: tl.constexpr):
    b = C + t64 * 16 + h * 4
    c0 = tl.load(b, mask=tm, other=0.0).to(tl.float32)
    c1 = tl.load(b + 1, mask=tm, other=0.0).to(tl.float32)
    c2 = tl.load(b + 2, mask=tm, other=0.0).to(tl.float32)
    c3 = tl.load(b + 3, mask=tm, other=0.0).to(tl.float32)
    return c0, c1, c2, c3


@triton.jit
def _expand_h(NEWP, ph, o, ch0, ch1, ch2, ch3, x0, x1, x2, x3, m2):
    v = ph[:, None] * o + ch0[:, None] * x0 + ch1[:, None] * x1 + ch2[:, None] * x2 + ch3[:, None] * x3
    tl.store(NEWP, v.to(NEWP.dtype.element_ty), mask=m2)


@triton.jit
def _hc_expand_bif_kernel(X, OUT, POST, COMB, NEW, M, D, BT: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    n_c = D // BC
    NT = tl.cdiv(M, BT) * n_c
    t, c, tm, _ci = _tile_coords(pid, n_c, M, BT, BC, NT)
    a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
    ob = tl.load(OUT + t[:, None] * D + c[None, :], mask=tm[:, None], other=0.0)
    p0, p1, p2, p3 = _ld_row4(POST, t, tm)
    c00, c01, c02, c03 = _ld_c4(COMB, t, tm, 0)
    c10, c11, c12, c13 = _ld_c4(COMB, t, tm, 1)
    c20, c21, c22, c23 = _ld_c4(COMB, t, tm, 2)
    c30, c31, c32, c33 = _ld_c4(COMB, t, tm, 3)
    for i in range(pid, NT, NPROG):
        x0, x1, x2, x3 = a0.to(tl.float32), a1.to(tl.float32), a2.to(tl.float32), a3.to(tl.float32)
        o = ob.to(tl.float32)
        q0, q1, q2, q3 = p0, p1, p2, p3
        e00, e01, e02, e03, e10, e11, e12, e13 = c00, c01, c02, c03, c10, c11, c12, c13
        e20, e21, e22, e23, e30, e31, e32, e33 = c20, c21, c22, c23, c30, c31, c32, c33
        tc, cc, tmc = t, c, tm
        t, c, tm, _ci = _tile_coords(i + NPROG, n_c, M, BT, BC, NT)
        a0, a1, a2, a3 = _ld4(X, t, c, tm, D)
        ob = tl.load(OUT + t[:, None] * D + c[None, :], mask=tm[:, None], other=0.0)
        p0, p1, p2, p3 = _ld_row4(POST, t, tm)
        c00, c01, c02, c03 = _ld_c4(COMB, t, tm, 0)
        c10, c11, c12, c13 = _ld_c4(COMB, t, tm, 1)
        c20, c21, c22, c23 = _ld_c4(COMB, t, tm, 2)
        c30, c31, c32, c33 = _ld_c4(COMB, t, tm, 3)
        base = NEW + tc[:, None] * (4 * D) + cc[None, :]
        m2 = tmc[:, None]
        _expand_h(base, q0, o, e00, e01, e02, e03, x0, x1, x2, x3, m2)
        _expand_h(base + D, q1, o, e10, e11, e12, e13, x0, x1, x2, x3, m2)
        _expand_h(base + 2 * D, q2, o, e20, e21, e22, e23, x0, x1, x2, x3, m2)
        _expand_h(base + 3 * D, q3, o, e30, e31, e32, e33, x0, x1, x2, x3, m2)


def bif_grid(M, D, BT, BC, nprog):
    n_tiles = triton.cdiv(M, BT) * (D // BC)
    return (min(nprog, n_tiles),)


__all__ = ["_hc_collapse_bif_kernel", "_hc_dpre_bif_kernel", "_hc_post_bwd_bif_kernel", "_hc_expand_bif_kernel",
           "bif_grid"]
