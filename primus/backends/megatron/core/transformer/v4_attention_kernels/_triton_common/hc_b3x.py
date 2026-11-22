###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""fuse-hc: B3 (HC backward dx + dW partials) with all 4 streams in one program (PRIMUS_HC_B3X=1).

``_hc_b3_gl_kernel`` runs one program per (stream k, BC columns, row split); the 4 stream programs of a tile each
load dcollapsed and all 4 dnew planes (4x the L2 traffic of those tensors) plus their own x plane. Here one program
owns (BC columns, row split) for all 4 streams: per BT-row step it loads x (4 planes), dnew (4), dcollapsed (1)
once and writes the 4 dx planes. Per element the arithmetic is the production kernel's (same WMMA chain
gh*wh + gh*wl + gq*wh, same elementwise expression, same BT-step dW accumulation), so dx and the dW partials are
byte-identical. W^T hi/lo of the 4 streams sit in LDS (as PRIMUS_HC_B3_VAR=1,0); DIRECT_W reads bf16 fn.weight
[N, KD] directly (no _hc_wt_split; the "lo" part of a bf16 W is zero, its WMMA is skipped).
"""

from __future__ import annotations

import os

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
def _b3x_dx(k: gl.constexpr, gh, gq, wsh, wsl, xk, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
            BT: gl.constexpr, BC: gl.constexpr, HAS_DNEW: gl.constexpr, WMMA: gl.constexpr, OPB: gl.constexpr,
            BLK: gl.constexpr, SKIP_LO: gl.constexpr, ABL: gl.constexpr = 0, BUF: gl.constexpr = False,
            dDX=None, sxo=0):
    pk = gl.expand_dims(gl.load(PRE + tr * 4 + k).to(gl.float32), 1)
    if ABL & 2:
        out = gl.zeros([BT, BC], gl.float32, layout=BLK) - xk.to(gl.float32) * cf + pk * dcv
    else:
        wh = wsh.load(OPB)
        acc = gl.zeros([BT, BC], gl.float32, layout=WMMA)
        acc = gl.amd.gfx1250.wmma(gh, wh, acc)
        if not SKIP_LO:
            wl = wsl.load(OPB)
            acc = gl.amd.gfx1250.wmma(gh, wl, acc)
        acc = gl.amd.gfx1250.wmma(gq, wh, acc)
        out = gl.convert_layout(acc, BLK) - xk.to(gl.float32) * cf + pk * dcv
    if HAS_DNEW:
        q0 = gl.expand_dims(gl.load(COMBT + tr * 16 + 0 + k).to(gl.float32), 1)
        q1 = gl.expand_dims(gl.load(COMBT + tr * 16 + 4 + k).to(gl.float32), 1)
        q2 = gl.expand_dims(gl.load(COMBT + tr * 16 + 8 + k).to(gl.float32), 1)
        q3 = gl.expand_dims(gl.load(COMBT + tr * 16 + 12 + k).to(gl.float32), 1)
        out = (out + q0 * n0.to(gl.float32) + q1 * n1.to(gl.float32) + q2 * n2.to(gl.float32)
               + q3 * n3.to(gl.float32))
    if BUF:
        gl.amd.cdna5.buffer.store(out.to(gl.bfloat16), dDX, rowoff, soffset=sxo + k * D)
    else:
        gl.store(DX + rowoff + k * D, out.to(gl.bfloat16))


@gluon.jit
def _b3x_load(t0, X, DC, COEF, DNEW, GH, GL, GTH, GTL, tb, cr, na_r, na_c, M, D, BT: gl.constexpr, BN: gl.constexpr,
              HAS_DNEW: gl.constexpr, OPA: gl.constexpr):
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
    tc = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, OPA))
    gto = gl.expand_dims(na_c, 1) * M + gl.expand_dims(tc, 0)
    gth = gl.load(GTH + gto)
    gtl = gl.load(GTL + gto)
    return x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl


@gluon.jit
def _b3x_load_buf(t0, dX, dDC, COEF, dN, GH, GL, GTH, GTL, loc_x, loc_d, tb, na_r, na_c, M, D, BT: gl.constexpr,
                  BN: gl.constexpr, HAS_DNEW: gl.constexpr, OPA: gl.constexpr):
    """_b3x_load through buffer descriptors: loop-invariant int32 tile offsets + a scalar row base (soffset)."""
    KD = 4 * D
    sx = t0 * KD
    x0 = gl.amd.cdna5.buffer.load(dX, loc_x, soffset=sx)
    x1 = gl.amd.cdna5.buffer.load(dX, loc_x, soffset=sx + D)
    x2 = gl.amd.cdna5.buffer.load(dX, loc_x, soffset=sx + 2 * D)
    x3 = gl.amd.cdna5.buffer.load(dX, loc_x, soffset=sx + 3 * D)
    dcb = gl.amd.cdna5.buffer.load(dDC, loc_d, soffset=t0 * D)
    cf = gl.expand_dims(gl.load(COEF + t0 + tb), 1)
    if HAS_DNEW:
        n0 = gl.amd.cdna5.buffer.load(dN, loc_x, soffset=sx)
        n1 = gl.amd.cdna5.buffer.load(dN, loc_x, soffset=sx + D)
        n2 = gl.amd.cdna5.buffer.load(dN, loc_x, soffset=sx + 2 * D)
        n3 = gl.amd.cdna5.buffer.load(dN, loc_x, soffset=sx + 3 * D)
    else:
        n0 = x0
        n1 = x0
        n2 = x0
        n3 = x0
    ta = t0 + gl.arange(0, BT, layout=gl.SliceLayout(1, OPA))
    goff = gl.expand_dims(ta, 1) * BN + gl.expand_dims(na_r, 0)
    gh = gl.load(GH + goff)
    gq = gl.load(GL + goff)
    tc = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, OPA))
    gto = gl.expand_dims(na_c, 1) * M + gl.expand_dims(tc, 0)
    gth = gl.load(GTH + gto)
    gtl = gl.load(GTL + gto)
    return x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl


@gluon.jit
def _b3x_step(t0, x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl,
              wsh0, wsh1, wsh2, wsh3, wsl0, wsl1, wsl2, wsl3, xs0, xs1, xs2, xs3, accw0, accw1, accw2, accw3,
              PRE, COMBT, DX, tb, cr, D, BT: gl.constexpr, BC: gl.constexpr, HAS_DNEW: gl.constexpr,
              WMMA: gl.constexpr, OPB: gl.constexpr, BLK: gl.constexpr, DIRECT_W: gl.constexpr, ABL: gl.constexpr = 0,
              BUF: gl.constexpr = False, dDX=None, loc_x=None):
    KD = 4 * D
    tr = t0 + tb
    if BUF:
        rowoff = loc_x
        sxo = t0 * KD
    else:
        tr64 = tr.to(gl.int64)
        rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
        sxo = 0
    dcv = dcb.to(gl.float32)
    _b3x_dx(0, gh, gq, wsh0, wsl0, x0, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
            BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL, BUF, dDX, sxo)
    _b3x_dx(1, gh, gq, wsh1, wsl1, x1, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
            BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL, BUF, dDX, sxo)
    _b3x_dx(2, gh, gq, wsh2, wsl2, x2, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
            BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL, BUF, dDX, sxo)
    _b3x_dx(3, gh, gq, wsh3, wsl3, x3, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
            BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL, BUF, dDX, sxo)
    if not (ABL & 1):
        gl.barrier()
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
    return accw0, accw1, accw2, accw3


@gluon.jit
def _hc_b3x_kernel(X, W, WTH, WTL, GH, GL, GTH, GTL, COEF, PRE, COMBT, DC, DNEW, DX, DWP, M, D, N, T_PER, N_D, N_TILES,
                   BT: gl.constexpr, BC: gl.constexpr, BN: gl.constexpr, HAS_DNEW: gl.constexpr,
                   WMMA: gl.constexpr, OPA: gl.constexpr, OPB: gl.constexpr, BLK: gl.constexpr,
                   SMEM: gl.constexpr, SMEM_W: gl.constexpr, DIRECT_W: gl.constexpr, PF: gl.constexpr = False,
                   ABL: gl.constexpr = 0, BUF: gl.constexpr = False):
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
        if PF:
            # register prefetch: the next step's global loads are issued before this step's math (same arithmetic)
            x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl = _b3x_load(
                t_lo, X, DC, COEF, DNEW, GH, GL, GTH, GTL, tb, cr, na_r, na_c, M, D, BT, BN, HAS_DNEW, OPA)
            for t0 in range(t_lo, t_lo + T_PER, BT):
                c_x0, c_x1, c_x2, c_x3, c_n0, c_n1, c_n2, c_n3 = x0, x1, x2, x3, n0, n1, n2, n3
                c_dcb, c_cf, c_gh, c_gq, c_gth, c_gtl = dcb, cf, gh, gq, gth, gtl
                tn = t_lo + (t0 + BT - t_lo) % T_PER
                x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl = _b3x_load(
                    tn, X, DC, COEF, DNEW, GH, GL, GTH, GTL, tb, cr, na_r, na_c, M, D, BT, BN, HAS_DNEW, OPA)
                accw0, accw1, accw2, accw3 = _b3x_step(
                    t0, c_x0, c_x1, c_x2, c_x3, c_n0, c_n1, c_n2, c_n3, c_dcb, c_cf, c_gh, c_gq, c_gth, c_gtl,
                    wsh0, wsh1, wsh2, wsh3, wsl0, wsl1, wsl2, wsl3, xs0, xs1, xs2, xs3, accw0, accw1, accw2, accw3,
                    PRE, COMBT, DX, tb, cr, D, BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL)
        elif BUF:
            # buffer ops: int32 tile-local offsets (loop invariant) + scalar row base; x / dnew / dx < 2^31 elements
            nbytes = M * KD * 2
            dX = gl.amd.cdna5.buffer.make_buffer_descriptor(X, nbytes)
            dN = gl.amd.cdna5.buffer.make_buffer_descriptor(DNEW, nbytes)
            dDX = gl.amd.cdna5.buffer.make_buffer_descriptor(DX, nbytes)
            dDC = gl.amd.cdna5.buffer.make_buffer_descriptor(DC, M * D * 2)
            loc_x = gl.expand_dims(tb * KD, 1) + gl.expand_dims(cr, 0)
            loc_d = gl.expand_dims(tb * D, 1) + gl.expand_dims(cr, 0)
            for t0 in range(t_lo, t_lo + T_PER, BT):
                x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl = _b3x_load_buf(
                    t0, dX, dDC, COEF, dN, GH, GL, GTH, GTL, loc_x, loc_d, tb, na_r, na_c, M, D, BT, BN, HAS_DNEW, OPA)
                accw0, accw1, accw2, accw3 = _b3x_step(
                    t0, x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl,
                    wsh0, wsh1, wsh2, wsh3, wsl0, wsl1, wsl2, wsl3, xs0, xs1, xs2, xs3, accw0, accw1, accw2, accw3,
                    PRE, COMBT, DX, tb, cr, D, BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL, True, dDX, loc_x)
        else:
            for t0 in range(t_lo, t_lo + T_PER, BT):
                x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl = _b3x_load(
                    t0, X, DC, COEF, DNEW, GH, GL, GTH, GTL, tb, cr, na_r, na_c, M, D, BT, BN, HAS_DNEW, OPA)
                accw0, accw1, accw2, accw3 = _b3x_step(
                    t0, x0, x1, x2, x3, n0, n1, n2, n3, dcb, cf, gh, gq, gth, gtl,
                    wsh0, wsh1, wsh2, wsh3, wsl0, wsl1, wsl2, wsl3, xs0, xs1, xs2, xs3, accw0, accw1, accw2, accw3,
                    PRE, COMBT, DX, tb, cr, D, BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W, ABL)
        nr = gl.arange(0, BN, layout=gl.SliceLayout(1, WMMA))
        cw = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, WMMA))
        dwo = DWP + pid_s * BN * KD + gl.expand_dims(nr.to(gl.int64) * KD, 1) + gl.expand_dims(cw, 0)
        gl.store(dwo, accw0)
        gl.store(dwo + D, accw1)
        gl.store(dwo + 2 * D, accw2)
        gl.store(dwo + 3 * D, accw3)


def b3x_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_B3X", "0") == "1"


def b3x_cfg():
    """dict(bc, w, bt, pf, ewm); PRIMUS_HC_B3X_CFG="bc=64,w=8,bt=32" style (any subset). Default bc=64, w=8, bt=32 (bench_b3x: 76 us vs 89 at w=4, 110 for _hc_b3_gl). bt=64 changes the dW bits.
    Warps go along the columns first (16 per WMMA tile) and then along the rows (at most 2: the dW tile has 32 rows).
    pf=1: register prefetch of the next step's loads; ewm=1: elementwise in the WMMA layout (both measured slower)."""
    c = dict(bc=64, w=8, bt=32, pf=0, ewm=0, buf=0)
    v = os.environ.get("PRIMUS_HC_B3X_CFG", "").strip()
    if v:
        if "=" in v:
            for kv in v.split(","):
                k, x = kv.split("=")
                c[k.strip()] = int(x)
        else:  # legacy bc,warps[,pf[,ewm]]
            t = [int(a) for a in v.split(",")]
            c.update(bc=t[0], w=t[1])
            if len(t) > 2:
                c["pf"] = t[2]
            if len(t) > 3:
                c["ewm"] = t[3]
    return c


_LAYOUTS = {}


def b3x_layouts(bt, bc, nw):
    """(wmma, opa, opb, blk, smem, smem_w) for B3X; warps along columns (16 each) then rows."""
    key = (bt, bc, nw)
    if key not in _LAYOUTS:
        ncol_t = bc // 16
        wc = min(nw, ncol_t)
        wr = nw // wc
        assert wc * wr == nw and wr <= 2 and bt % (16 * wr) == 0, (bt, bc, nw)
        wb = [[0, 1 << i] for i in range(wc.bit_length() - 1)] + [[1 << i, 0] for i in range(wr.bit_length() - 1)]
        wm = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=wb, instr_shape=[16, 16, 32])
        opa = gl.DotOperandLayout(operand_index=0, parent=wm, k_width=8)
        opb = gl.DotOperandLayout(operand_index=1, parent=wm, k_width=8)
        tpw = [256 // bc, bc // 8] if bc < 256 else [1, 32]
        blk = gl.BlockedLayout([1, 8], tpw, [nw, 1], [1, 0])
        smem = gl.PaddedSharedLayout.with_identity_for([[bc, 16]], [bt, bc], [1, 0])
        smem_w = gl.PaddedSharedLayout.with_identity_for([[bc, 16]], [32, bc], [1, 0])
        _LAYOUTS[key] = (wm, opa, opb, blk, smem, smem_w)
    return _LAYOUTS[key]
