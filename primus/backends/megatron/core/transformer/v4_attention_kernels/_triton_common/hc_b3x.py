###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HyperConnection backward (dx + dW partials) with all 4 streams in one program.

One program owns (BC columns, row split) for all 4 streams: per BT-row step it loads x (4 planes), dnew (4) and
dcollapsed (1) once and writes the 4 dx planes (a per-stream program would re-read dcollapsed and the dnew planes
4x). W^T hi/lo of the 4 streams sit in LDS; DIRECT_W reads bf16 fn.weight [N, KD] directly (no W^T split; the "lo"
part of a bf16 W is zero, so its WMMA is skipped). _hc_b3x_u2_kernel is the same computation unrolled by 2 row steps
for DIRECT_W with an even step count (the launched case); _hc_b3x_kernel covers the rest.
"""

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _b3x_ldw(W, WTH, WTL, k: gl.constexpr, nb, cb, D, N, BN: gl.constexpr, DIRECT_W: gl.constexpr):
    if DIRECT_W:
        KD = 4 * D
        off = gl.expand_dims(nb.to(gl.int64) * KD, 1) + gl.expand_dims(k * D + cb, 0)
        m = gl.expand_dims(nb < N, 1) & gl.expand_dims(cb >= 0, 0)
        wh = gl.load(W + off, mask=m, other=0.0)
        return wh, wh
    else:
        woff = gl.expand_dims((k * D + cb) * BN, 0) + gl.expand_dims(nb, 1)
        return gl.load(WTH + woff), gl.load(WTL + woff)


@gluon.jit
def _b3x_load(
    t0,
    X,
    DC,
    COEF,
    DNEW,
    GH,
    GL,
    GTH,
    GTL,
    tb,
    cr,
    na_r,
    na_c,
    M,
    D,
    BT: gl.constexpr,
    BN: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    OPA: gl.constexpr,
):
    KD = 4 * D
    tr = t0 + tb
    tr64 = tr.to(gl.int64)
    rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
    x0 = gl.load(X + rowoff)
    x1 = gl.load(X + rowoff + D)
    x2 = gl.load(X + rowoff + 2 * D)
    x3 = gl.load(X + rowoff + 3 * D)
    dcb = gl.load(DC + gl.expand_dims(tr64 * D, 1) + gl.expand_dims(cr, 0))
    cf = gl.expand_dims(gl.load(COEF + tr), 1)
    if HAS_DNEW:
        n0 = gl.load(DNEW + rowoff)
        n1 = gl.load(DNEW + rowoff + D)
        n2 = gl.load(DNEW + rowoff + 2 * D)
        n3 = gl.load(DNEW + rowoff + 3 * D)
    else:
        n0 = x0
        n1 = x0
        n2 = x0
        n3 = x0
    ta = t0 + gl.arange(0, BT, layout=gl.SliceLayout(1, OPA))
    goff = gl.expand_dims(ta, 1) * BN + gl.expand_dims(na_r, 0)
    gh = gl.load(GH + goff)
    tc = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, OPA))
    gto = gl.expand_dims(na_c, 1) * M + gl.expand_dims(tc, 0)
    gth = gl.load(GTH + gto)
    gq = gl.load(GL + goff)
    gtl = gl.load(GTL + gto)
    return x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl


@gluon.jit
def _b3x_rv(t0, tb, PRE, COMBT, k: gl.constexpr, HAS_DNEW: gl.constexpr):
    """Raw (bf16, unconverted) per-row vectors of stream k for row step t0: pre_k and comb^T[h, k], h = 0..3."""
    tr = t0 + tb
    pk = gl.load(PRE + tr * 4 + k)
    if HAS_DNEW:
        q0 = gl.load(COMBT + tr * 16 + 0 + k)
        q1 = gl.load(COMBT + tr * 16 + 4 + k)
        q2 = gl.load(COMBT + tr * 16 + 8 + k)
        q3 = gl.load(COMBT + tr * 16 + 12 + k)
    else:
        q0 = pk
        q1 = pk
        q2 = pk
        q3 = pk
    return pk, q0, q1, q2, q3


@gluon.jit
def _b3x_mm(
    gh,
    gq,
    wsh,
    wsl,
    BT: gl.constexpr,
    BC: gl.constexpr,
    WMMA: gl.constexpr,
    OPB: gl.constexpr,
    SKIP_LO: gl.constexpr,
):
    """dx GEMM part of one stream, exactly as in _b3x_dx (same WMMA chain)."""
    wh = wsh.load(OPB)
    acc = gl.zeros([BT, BC], gl.float32, layout=WMMA)
    acc = gl.amd.gfx1250.wmma(gh, wh, acc)
    if not SKIP_LO:
        wl = wsl.load(OPB)
        acc = gl.amd.gfx1250.wmma(gh, wl, acc)
    acc = gl.amd.gfx1250.wmma(gq, wh, acc)
    return acc


@gluon.jit
def _b3x_ew(
    k: gl.constexpr,
    acs,
    xk,
    cf,
    dcv,
    n0,
    n1,
    n2,
    n3,
    DX,
    rowoff,
    D,
    BLK: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    pkr,
    q0r,
    q1r,
    q2r,
    q3r,
):
    """dx elementwise part of one stream from its GEMM result read back from LDS in BLK (as _b3x_dx, RV form)."""
    pk = gl.expand_dims(pkr.to(gl.float32), 1)
    out = acs.load(BLK) - xk.to(gl.float32) * cf + pk * dcv
    if HAS_DNEW:
        out = (
            out
            + gl.expand_dims(q0r.to(gl.float32), 1) * n0.to(gl.float32)
            + gl.expand_dims(q1r.to(gl.float32), 1) * n1.to(gl.float32)
            + gl.expand_dims(q2r.to(gl.float32), 1) * n2.to(gl.float32)
            + gl.expand_dims(q3r.to(gl.float32), 1) * n3.to(gl.float32)
        )
    gl.store(DX + rowoff + k * D, out.to(gl.bfloat16))


@gluon.jit
def _b3x_step_bat(
    t0,
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    dcb,
    cf,
    gh,
    gq,
    gth,
    gtl,
    wsh0,
    wsh1,
    wsh2,
    wsh3,
    wsl0,
    wsl1,
    wsl2,
    wsl3,
    xs0,
    xs1,
    xs2,
    xs3,
    as0,
    as1,
    as2,
    as3,
    accw0,
    accw1,
    accw2,
    accw3,
    DX,
    tb,
    cr,
    D,
    BT: gl.constexpr,
    BC: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    WMMA: gl.constexpr,
    OPB: gl.constexpr,
    BLK: gl.constexpr,
    DIRECT_W: gl.constexpr,
    p0,
    a0,
    b0,
    c0,
    d0,
    p1,
    a1,
    b1,
    c1,
    d1,
    p2,
    a2,
    b2,
    c2,
    d2,
    p3,
    a3,
    b3,
    c3,
    d3,
):
    """The 4 dx GEMM results and the 4 x planes go through LDS in ONE store / barrier / load round
    (2 barriers per row step instead of one convert_layout round trip per stream plus the x staging)."""
    KD = 4 * D
    tr = t0 + tb
    tr64 = tr.to(gl.int64)
    rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
    m0 = _b3x_mm(gh, gq, wsh0, wsl0, BT, BC, WMMA, OPB, DIRECT_W)
    m1 = _b3x_mm(gh, gq, wsh1, wsl1, BT, BC, WMMA, OPB, DIRECT_W)
    m2 = _b3x_mm(gh, gq, wsh2, wsl2, BT, BC, WMMA, OPB, DIRECT_W)
    m3 = _b3x_mm(gh, gq, wsh3, wsl3, BT, BC, WMMA, OPB, DIRECT_W)
    gl.barrier()  # the previous step's LDS reads must be done (single buffer)
    as0.store(m0)
    as1.store(m1)
    as2.store(m2)
    as3.store(m3)
    xs0.store(x0)
    xs1.store(x1)
    xs2.store(x2)
    xs3.store(x3)
    gl.barrier()
    xb = xs0.load(OPB)
    accw0 = gl.amd.gfx1250.wmma(gth, xb, accw0)
    accw0 = gl.amd.gfx1250.wmma(gtl, xb, accw0)
    xb = xs1.load(OPB)
    accw1 = gl.amd.gfx1250.wmma(gth, xb, accw1)
    accw1 = gl.amd.gfx1250.wmma(gtl, xb, accw1)
    xb = xs2.load(OPB)
    accw2 = gl.amd.gfx1250.wmma(gth, xb, accw2)
    accw2 = gl.amd.gfx1250.wmma(gtl, xb, accw2)
    xb = xs3.load(OPB)
    accw3 = gl.amd.gfx1250.wmma(gth, xb, accw3)
    accw3 = gl.amd.gfx1250.wmma(gtl, xb, accw3)
    dcv = dcb.to(gl.float32)
    _b3x_ew(0, as0, x0, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p0, a0, b0, c0, d0)
    _b3x_ew(1, as1, x1, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p1, a1, b1, c1, d1)
    _b3x_ew(2, as2, x2, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p2, a2, b2, c2, d2)
    _b3x_ew(3, as3, x3, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p3, a3, b3, c3, d3)
    return accw0, accw1, accw2, accw3


@gluon.jit
def _hc_b3x_kernel(
    X,
    W,
    WTH,
    WTL,
    GH,
    GL,
    GTH,
    GTL,
    COEF,
    PRE,
    COMBT,
    DC,
    DNEW,
    DX,
    DWP,
    M,
    D,
    N,
    T_PER,
    N_D,
    N_TILES,
    BT: gl.constexpr,
    BC: gl.constexpr,
    BN: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    WMMA: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    BLK: gl.constexpr,
    SMEM: gl.constexpr,
    SMEM_W: gl.constexpr,
    DIRECT_W: gl.constexpr,
    SMEM_F: gl.constexpr,
):
    tile = gl.program_id(0)
    if tile < N_TILES:
        pid_d = tile % N_D
        pid_s = tile // N_D
        c0 = pid_d * BC
        KD = 4 * D
        nb = gl.arange(0, BN, layout=gl.SliceLayout(1, OPB))
        cb = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, OPB))
        wsh0 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wsh1 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wsh2 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wsh3 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        a, b = _b3x_ldw(W, WTH, WTL, 0, nb, cb, D, N, BN, DIRECT_W)
        wsh0.store(a)
        a1, b1 = _b3x_ldw(W, WTH, WTL, 1, nb, cb, D, N, BN, DIRECT_W)
        wsh1.store(a1)
        a2, b2 = _b3x_ldw(W, WTH, WTL, 2, nb, cb, D, N, BN, DIRECT_W)
        wsh2.store(a2)
        a3, b3 = _b3x_ldw(W, WTH, WTL, 3, nb, cb, D, N, BN, DIRECT_W)
        wsh3.store(a3)
        if DIRECT_W:
            wsl0 = wsh0
            wsl1 = wsh1
            wsl2 = wsh2
            wsl3 = wsh3
        else:
            wsl0 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
            wsl1 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
            wsl2 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
            wsl3 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
            wsl0.store(b)
            wsl1.store(b1)
            wsl2.store(b2)
            wsl3.store(b3)
        na_r = gl.arange(0, BN, layout=gl.SliceLayout(0, OPA))
        na_c = gl.arange(0, BN, layout=gl.SliceLayout(1, OPA))
        tb = gl.arange(0, BT, layout=gl.SliceLayout(1, BLK))
        cr = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, BLK))
        accw0 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        accw1 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        accw2 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        accw3 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        xs0 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        xs1 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        xs2 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        xs3 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        t_lo = pid_s * T_PER
        t_hi = t_lo + T_PER
        as0 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        as1 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        as2 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        as3 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        # register prefetch of EVERYTHING the step reads (incl. pre / comb^T / coef row vectors, kept raw), so no
        # wait inside the step covers the next step's loads
        x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl = _b3x_load(
            t_lo, X, DC, COEF, DNEW, GH, GL, GTH, GTL, tb, cr, na_r, na_c, M, D, BT, BN, HAS_DNEW, OPA
        )
        pv0, qa0, qb0, qc0, qd0 = _b3x_rv(t_lo, tb, PRE, COMBT, 0, HAS_DNEW)
        pv1, qa1, qb1, qc1, qd1 = _b3x_rv(t_lo, tb, PRE, COMBT, 1, HAS_DNEW)
        pv2, qa2, qb2, qc2, qd2 = _b3x_rv(t_lo, tb, PRE, COMBT, 2, HAS_DNEW)
        pv3, qa3, qb3, qc3, qd3 = _b3x_rv(t_lo, tb, PRE, COMBT, 3, HAS_DNEW)
        for t0 in range(t_lo, t_hi, BT):
            c_x0, c_x1, c_x2, c_x3, c_n0, c_n1, c_n2, c_n3 = x0, x1, x2, x3, n0, n1, n2, n3
            c_dcb, c_cf, c_gh, c_gq, c_gth, c_gtl = dcb, cf, gh, gq, gth, gtl
            cpv0, cqa0, cqb0, cqc0, cqd0 = pv0, qa0, qb0, qc0, qd0
            cpv1, cqa1, cqb1, cqc1, cqd1 = pv1, qa1, qb1, qc1, qd1
            cpv2, cqa2, cqb2, cqc2, cqd2 = pv2, qa2, qb2, qc2, qd2
            cpv3, cqa3, cqb3, cqc3, cqd3 = pv3, qa3, qb3, qc3, qd3
            tn = t_lo + (t0 + BT - t_lo) % (
                t_hi - t_lo
            )  # wraps to the first step on the last (harmless reload)
            x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl = _b3x_load(
                tn, X, DC, COEF, DNEW, GH, GL, GTH, GTL, tb, cr, na_r, na_c, M, D, BT, BN, HAS_DNEW, OPA
            )
            pv0, qa0, qb0, qc0, qd0 = _b3x_rv(tn, tb, PRE, COMBT, 0, HAS_DNEW)
            pv1, qa1, qb1, qc1, qd1 = _b3x_rv(tn, tb, PRE, COMBT, 1, HAS_DNEW)
            pv2, qa2, qb2, qc2, qd2 = _b3x_rv(tn, tb, PRE, COMBT, 2, HAS_DNEW)
            pv3, qa3, qb3, qc3, qd3 = _b3x_rv(tn, tb, PRE, COMBT, 3, HAS_DNEW)
            accw0, accw1, accw2, accw3 = _b3x_step_bat(
                t0,
                c_x0,
                c_x1,
                c_x2,
                c_x3,
                c_n0,
                c_n1,
                c_n2,
                c_n3,
                c_dcb,
                c_cf,
                c_gh,
                c_gq,
                c_gth,
                c_gtl,
                wsh0,
                wsh1,
                wsh2,
                wsh3,
                wsl0,
                wsl1,
                wsl2,
                wsl3,
                xs0,
                xs1,
                xs2,
                xs3,
                as0,
                as1,
                as2,
                as3,
                accw0,
                accw1,
                accw2,
                accw3,
                DX,
                tb,
                cr,
                D,
                BT,
                BC,
                HAS_DNEW,
                WMMA,
                OPB,
                BLK,
                DIRECT_W,
                cpv0,
                cqa0,
                cqb0,
                cqc0,
                cqd0,
                cpv1,
                cqa1,
                cqb1,
                cqc1,
                cqd1,
                cpv2,
                cqa2,
                cqb2,
                cqc2,
                cqd2,
                cpv3,
                cqa3,
                cqb3,
                cqc3,
                cqd3,
            )
        nr = gl.arange(0, BN, layout=gl.SliceLayout(1, WMMA))
        cw = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, WMMA))
        dwo = DWP + pid_s * BN * KD + gl.expand_dims(nr.to(gl.int64) * KD, 1) + gl.expand_dims(cw, 0)
        gl.store(dwo, accw0)
        gl.store(dwo + D, accw1)
        gl.store(dwo + 2 * D, accw2)
        gl.store(dwo + 3 * D, accw3)


# ---------------------------------------------------------------------------------------------------------------------
# Unrolled-by-2 B3X (DIRECT_W, even number of row steps per program). Same ops, arithmetic and order as
# _hc_b3x_kernel, so dx and the dW partials are byte-identical; it differs in how the memory traffic overlaps:
# - no explicit gl.barrier(): it lowers to a fence on ALL address spaces (s_wait_loadcnt 0 + s_wait_storecnt 0 +
#   s_barrier), which drained the next step's prefetch and this step's dx stores twice per row step. The LDS hazards
#   (WAR on as* / xs* across steps, store -> load) are left to the Membar pass, which inserts LOCAL barriers only;
# - two register sets (steps t0 and t0 + BT), so a prefetch lands in the registers its consumer reads (no
#   loop-carried copies, which forced a full load wait mid-step);
# - issue order g^T of the current step, then the next step's row vectors, then its planes: the early waits (g^T for
#   the dW WMMAs, the bf16 unpacks of the row vectors) leave the planes in flight (loadcnt is in-order); g^T is
#   loaded in-step, not prefetched, which keeps the kernel at 256 VGPRs with waves_per_eu=4 (2 programs per CU).


@gluon.jit
def _b3x_ldgt(t0, GTH, GTL, na_c, M, BT: gl.constexpr, OPA: gl.constexpr):
    """The dW A operands (g^T hi / lo) of row step t0 (as in _b3x_load)."""
    tc = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, OPA))
    gto = gl.expand_dims(na_c, 1) * M + gl.expand_dims(tc, 0)
    gth = gl.load(GTH + gto)
    gtl = gl.load(GTL + gto)
    return gth, gtl


@gluon.jit
def _b3x_load_ng(
    t0,
    X,
    DC,
    COEF,
    DNEW,
    GH,
    GL,
    tb,
    cr,
    na_r,
    M,
    D,
    BT: gl.constexpr,
    BN: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    OPA: gl.constexpr,
):
    """_b3x_load without the g^T operands (same loads, same layouts)."""
    KD = 4 * D
    tr = t0 + tb
    tr64 = tr.to(gl.int64)
    rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
    x0 = gl.load(X + rowoff)
    x1 = gl.load(X + rowoff + D)
    x2 = gl.load(X + rowoff + 2 * D)
    x3 = gl.load(X + rowoff + 3 * D)
    dcb = gl.load(DC + gl.expand_dims(tr64 * D, 1) + gl.expand_dims(cr, 0))
    cf = gl.expand_dims(gl.load(COEF + tr), 1)
    if HAS_DNEW:
        n0 = gl.load(DNEW + rowoff)
        n1 = gl.load(DNEW + rowoff + D)
        n2 = gl.load(DNEW + rowoff + 2 * D)
        n3 = gl.load(DNEW + rowoff + 3 * D)
    else:
        n0 = x0
        n1 = x0
        n2 = x0
        n3 = x0
    ta = t0 + gl.arange(0, BT, layout=gl.SliceLayout(1, OPA))
    goff = gl.expand_dims(ta, 1) * BN + gl.expand_dims(na_r, 0)
    gh = gl.load(GH + goff)
    gq = gl.load(GL + goff)
    return x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq


@gluon.jit
def _b3x_dw_only(x0, x1, x2, x3, gth, gtl, xs0, xs1, xs2, xs3, accw0, accw1, accw2, accw3, OPB: gl.constexpr):
    xs0.store(x0)
    xs1.store(x1)
    xs2.store(x2)
    xs3.store(x3)
    xb = xs0.load(OPB)
    accw0 = gl.amd.gfx1250.wmma(gth, xb, accw0)
    accw0 = gl.amd.gfx1250.wmma(gtl, xb, accw0)
    xb = xs1.load(OPB)
    accw1 = gl.amd.gfx1250.wmma(gth, xb, accw1)
    accw1 = gl.amd.gfx1250.wmma(gtl, xb, accw1)
    xb = xs2.load(OPB)
    accw2 = gl.amd.gfx1250.wmma(gth, xb, accw2)
    accw2 = gl.amd.gfx1250.wmma(gtl, xb, accw2)
    xb = xs3.load(OPB)
    accw3 = gl.amd.gfx1250.wmma(gth, xb, accw3)
    accw3 = gl.amd.gfx1250.wmma(gtl, xb, accw3)
    return accw0, accw1, accw2, accw3


@gluon.jit
def _b3x_step_u2(
    t0,
    x0,
    x1,
    x2,
    x3,
    n0,
    n1,
    n2,
    n3,
    dcb,
    cf,
    gh,
    gq,
    gth,
    gtl,
    wsh0,
    wsh1,
    wsh2,
    wsh3,
    xs0,
    xs1,
    xs2,
    xs3,
    as0,
    as1,
    as2,
    as3,
    accw0,
    accw1,
    accw2,
    accw3,
    DX,
    tb,
    cr,
    D,
    BT: gl.constexpr,
    BC: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    WMMA: gl.constexpr,
    OPB: gl.constexpr,
    BLK: gl.constexpr,
    p0,
    a0,
    b0,
    c0,
    d0,
    p1,
    a1,
    b1,
    c1,
    d1,
    p2,
    a2,
    b2,
    c2,
    d2,
    p3,
    a3,
    b3,
    c3,
    d3,
):
    """_b3x_step_bat (DIRECT_W) without the explicit gl.barrier() calls."""
    KD = 4 * D
    tr = t0 + tb
    tr64 = tr.to(gl.int64)
    rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
    m0 = _b3x_mm(gh, gq, wsh0, wsh0, BT, BC, WMMA, OPB, True)
    m1 = _b3x_mm(gh, gq, wsh1, wsh1, BT, BC, WMMA, OPB, True)
    m2 = _b3x_mm(gh, gq, wsh2, wsh2, BT, BC, WMMA, OPB, True)
    m3 = _b3x_mm(gh, gq, wsh3, wsh3, BT, BC, WMMA, OPB, True)
    as0.store(m0)
    as1.store(m1)
    as2.store(m2)
    as3.store(m3)
    accw0, accw1, accw2, accw3 = _b3x_dw_only(
        x0, x1, x2, x3, gth, gtl, xs0, xs1, xs2, xs3, accw0, accw1, accw2, accw3, OPB
    )
    dcv = dcb.to(gl.float32)
    _b3x_ew(0, as0, x0, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p0, a0, b0, c0, d0)
    _b3x_ew(1, as1, x1, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p1, a1, b1, c1, d1)
    _b3x_ew(2, as2, x2, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p2, a2, b2, c2, d2)
    _b3x_ew(3, as3, x3, cf, dcv, n0, n1, n2, n3, DX, rowoff, D, BLK, HAS_DNEW, p3, a3, b3, c3, d3)
    return accw0, accw1, accw2, accw3


@gluon.jit
def _hc_b3x_u2_kernel(
    X,
    W,
    GH,
    GL,
    GTH,
    GTL,
    COEF,
    PRE,
    COMBT,
    DC,
    DNEW,
    DX,
    DWP,
    M,
    D,
    N,
    T_PER,
    N_D,
    N_TILES,
    BT: gl.constexpr,
    BC: gl.constexpr,
    BN: gl.constexpr,
    HAS_DNEW: gl.constexpr,
    WMMA: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    BLK: gl.constexpr,
    SMEM: gl.constexpr,
    SMEM_W: gl.constexpr,
    SMEM_F: gl.constexpr,
):
    """B3X for DIRECT_W with T_PER / BT even (see the section comment above)."""
    tile = gl.program_id(0)
    if tile < N_TILES:
        pid_d = tile % N_D
        pid_s = tile // N_D
        c0 = pid_d * BC
        nb = gl.arange(0, BN, layout=gl.SliceLayout(1, OPB))
        cb = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, OPB))
        KD = 4 * D
        wsh0 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wsh1 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wsh2 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wsh3 = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
        wm = gl.expand_dims(nb < N, 1) & gl.expand_dims(cb >= 0, 0)
        woff = gl.expand_dims(nb.to(gl.int64) * KD, 1) + gl.expand_dims(cb, 0)
        wsh0.store(gl.load(W + woff, mask=wm, other=0.0))
        wsh1.store(gl.load(W + woff + D, mask=wm, other=0.0))
        wsh2.store(gl.load(W + woff + 2 * D, mask=wm, other=0.0))
        wsh3.store(gl.load(W + woff + 3 * D, mask=wm, other=0.0))
        na_r = gl.arange(0, BN, layout=gl.SliceLayout(0, OPA))
        na_c = gl.arange(0, BN, layout=gl.SliceLayout(1, OPA))
        tb = gl.arange(0, BT, layout=gl.SliceLayout(1, BLK))
        cr = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, BLK))
        accw0 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        accw1 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        accw2 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        accw3 = gl.zeros([BN, BC], gl.float32, layout=WMMA)
        xs0 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        xs1 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        xs2 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        xs3 = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
        as0 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        as1 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        as2 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        as3 = gl.allocate_shared_memory(gl.float32, [BT, BC], SMEM_F)
        t_lo = pid_s * T_PER
        t_hi = t_lo + T_PER
        pv0, qa0, qb0, qc0, qd0 = _b3x_rv(t_lo, tb, PRE, COMBT, 0, HAS_DNEW)
        pv1, qa1, qb1, qc1, qd1 = _b3x_rv(t_lo, tb, PRE, COMBT, 1, HAS_DNEW)
        pv2, qa2, qb2, qc2, qd2 = _b3x_rv(t_lo, tb, PRE, COMBT, 2, HAS_DNEW)
        pv3, qa3, qb3, qc3, qd3 = _b3x_rv(t_lo, tb, PRE, COMBT, 3, HAS_DNEW)
        x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq = _b3x_load_ng(
            t_lo, X, DC, COEF, DNEW, GH, GL, tb, cr, na_r, M, D, BT, BN, HAS_DNEW, OPA
        )
        for t0 in range(t_lo, t_hi, 2 * BT):
            t1 = t0 + BT
            tn = t_lo + (t0 + 2 * BT - t_lo) % (
                t_hi - t_lo
            )  # wraps to the first pair on the last (harmless reload)
            gth, gtl = _b3x_ldgt(t0, GTH, GTL, na_c, M, BT, OPA)
            rv0, ra0, rb0, rc0, rd0 = _b3x_rv(t1, tb, PRE, COMBT, 0, HAS_DNEW)
            rv1, ra1, rb1, rc1, rd1 = _b3x_rv(t1, tb, PRE, COMBT, 1, HAS_DNEW)
            rv2, ra2, rb2, rc2, rd2 = _b3x_rv(t1, tb, PRE, COMBT, 2, HAS_DNEW)
            rv3, ra3, rb3, rc3, rd3 = _b3x_rv(t1, tb, PRE, COMBT, 3, HAS_DNEW)
            y0, y1, y2, y3, m0_, m1_, m2_, m3_, ydcb, ycf, ygh, ygq = _b3x_load_ng(
                t1, X, DC, COEF, DNEW, GH, GL, tb, cr, na_r, M, D, BT, BN, HAS_DNEW, OPA
            )
            accw0, accw1, accw2, accw3 = _b3x_step_u2(
                t0,
                x0,
                x1,
                x2,
                x3,
                n0,
                n1,
                n2,
                n3,
                dcb,
                cf,
                gh,
                gq,
                gth,
                gtl,
                wsh0,
                wsh1,
                wsh2,
                wsh3,
                xs0,
                xs1,
                xs2,
                xs3,
                as0,
                as1,
                as2,
                as3,
                accw0,
                accw1,
                accw2,
                accw3,
                DX,
                tb,
                cr,
                D,
                BT,
                BC,
                HAS_DNEW,
                WMMA,
                OPB,
                BLK,
                pv0,
                qa0,
                qb0,
                qc0,
                qd0,
                pv1,
                qa1,
                qb1,
                qc1,
                qd1,
                pv2,
                qa2,
                qb2,
                qc2,
                qd2,
                pv3,
                qa3,
                qb3,
                qc3,
                qd3,
            )
            gth, gtl = _b3x_ldgt(t1, GTH, GTL, na_c, M, BT, OPA)
            pv0, qa0, qb0, qc0, qd0 = _b3x_rv(tn, tb, PRE, COMBT, 0, HAS_DNEW)
            pv1, qa1, qb1, qc1, qd1 = _b3x_rv(tn, tb, PRE, COMBT, 1, HAS_DNEW)
            pv2, qa2, qb2, qc2, qd2 = _b3x_rv(tn, tb, PRE, COMBT, 2, HAS_DNEW)
            pv3, qa3, qb3, qc3, qd3 = _b3x_rv(tn, tb, PRE, COMBT, 3, HAS_DNEW)
            x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq = _b3x_load_ng(
                tn, X, DC, COEF, DNEW, GH, GL, tb, cr, na_r, M, D, BT, BN, HAS_DNEW, OPA
            )
            accw0, accw1, accw2, accw3 = _b3x_step_u2(
                t1,
                y0,
                y1,
                y2,
                y3,
                m0_,
                m1_,
                m2_,
                m3_,
                ydcb,
                ycf,
                ygh,
                ygq,
                gth,
                gtl,
                wsh0,
                wsh1,
                wsh2,
                wsh3,
                xs0,
                xs1,
                xs2,
                xs3,
                as0,
                as1,
                as2,
                as3,
                accw0,
                accw1,
                accw2,
                accw3,
                DX,
                tb,
                cr,
                D,
                BT,
                BC,
                HAS_DNEW,
                WMMA,
                OPB,
                BLK,
                rv0,
                ra0,
                rb0,
                rc0,
                rd0,
                rv1,
                ra1,
                rb1,
                rc1,
                rd1,
                rv2,
                ra2,
                rb2,
                rc2,
                rd2,
                rv3,
                ra3,
                rb3,
                rc3,
                rd3,
            )
        nr = gl.arange(0, BN, layout=gl.SliceLayout(1, WMMA))
        cw = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, WMMA))
        dwo = DWP + pid_s * BN * KD + gl.expand_dims(nr.to(gl.int64) * KD, 1) + gl.expand_dims(cw, 0)
        gl.store(dwo, accw0)
        gl.store(dwo + D, accw1)
        gl.store(dwo + 2 * D, accw2)
        gl.store(dwo + 3 * D, accw3)


def b3x_u2_ok(direct_w, t_per, bt) -> bool:
    """_hc_b3x_u2_kernel covers DIRECT_W with an even number of BT-row steps per program; the rest runs
    _hc_b3x_kernel."""
    return bool(direct_w) and t_per % bt == 0 and (t_per // bt) % 2 == 0


# launch hint: waves_per_eu=4 caps the kernel at 256 VGPRs (2 programs per CU, 0 spills)
B3X_U2_LAUNCH = dict(waves_per_eu=4)


_LAYOUTS = {}


def b3x_smem_f(bt, bc, nbuf=1):
    """fp32 [bt, bc] (or [2, bt, bc]) shared layout for the batched dx GEMM round trip (BAT), 16-byte row
    padding."""
    key = ("f", bt, bc, nbuf)
    if key not in _LAYOUTS:
        if nbuf == 1:
            _LAYOUTS[key] = gl.PaddedSharedLayout.with_identity_for([[bc, 4]], [bt, bc], [1, 0])
        else:
            _LAYOUTS[key] = gl.PaddedSharedLayout.with_identity_for([[bc, 4]], [nbuf, bt, bc], [2, 1, 0])
    return _LAYOUTS[key]


def b3x_layouts(bt, bc, nw):
    """(wmma, opa, opb, blk, smem, smem_w) for B3X; warps along columns (16 each) then rows."""
    key = (bt, bc, nw)
    if key not in _LAYOUTS:
        ncol_t = bc // 16
        wc = min(nw, ncol_t)
        wr = nw // wc
        assert wc * wr == nw and wr <= 2 and bt % (16 * wr) == 0, (bt, bc, nw)
        wb = [[0, 1 << i] for i in range(wc.bit_length() - 1)] + [
            [1 << i, 0] for i in range(wr.bit_length() - 1)
        ]
        wm = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=wb, instr_shape=[16, 16, 32])
        opa = gl.DotOperandLayout(operand_index=0, parent=wm, k_width=8)
        opb = gl.DotOperandLayout(operand_index=1, parent=wm, k_width=8)
        tpw = [256 // bc, bc // 8] if bc < 256 else [1, 32]
        blk = gl.BlockedLayout([1, 8], tpw, [nw, 1], [1, 0])
        smem = gl.PaddedSharedLayout.with_identity_for([[bc, 16]], [bt, bc], [1, 0])
        smem_w = gl.PaddedSharedLayout.with_identity_for([[bc, 16]], [32, bc], [1, 0])
        _LAYOUTS[key] = (wm, opa, opb, blk, smem, smem_w)
    return _LAYOUTS[key]
