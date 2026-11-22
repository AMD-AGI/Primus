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
            BLK: gl.constexpr, SKIP_LO: gl.constexpr):
    pk = gl.expand_dims(gl.load(PRE + tr * 4 + k).to(gl.float32), 1)
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
    gl.store(DX + rowoff + k * D, out.to(gl.bfloat16))


@gluon.jit
def _hc_b3x_kernel(X, W, WTH, WTL, GH, GL, GTH, GTL, COEF, PRE, COMBT, DC, DNEW, DX, DWP, M, D, N, T_PER, N_D, N_TILES,
                   BT: gl.constexpr, BC: gl.constexpr, BN: gl.constexpr, HAS_DNEW: gl.constexpr,
                   WMMA: gl.constexpr, OPA: gl.constexpr, OPB: gl.constexpr, BLK: gl.constexpr,
                   SMEM: gl.constexpr, SMEM_W: gl.constexpr, DIRECT_W: gl.constexpr):
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
        for t0 in range(t_lo, t_lo + T_PER, BT):
            tr = t0 + tb
            tr64 = tr.to(gl.int64)
            rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
            x0 = gl.load(X + rowoff)
            x1 = gl.load(X + rowoff + D)
            x2 = gl.load(X + rowoff + 2 * D)
            x3 = gl.load(X + rowoff + 3 * D)
            dcv = gl.load(DC + gl.expand_dims(tr64 * D, 1) + gl.expand_dims(cr, 0)).to(gl.float32)
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
            _b3x_dx(0, gh, gq, wsh0, wsl0, x0, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
                    BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W)
            _b3x_dx(1, gh, gq, wsh1, wsl1, x1, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
                    BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W)
            _b3x_dx(2, gh, gq, wsh2, wsl2, x2, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
                    BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W)
            _b3x_dx(3, gh, gq, wsh3, wsl3, x3, cf, PRE, COMBT, tr, dcv, n0, n1, n2, n3, DX, rowoff, D,
                    BT, BC, HAS_DNEW, WMMA, OPB, BLK, DIRECT_W)
            tc = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, OPA))
            gto = gl.expand_dims(na_c, 1) * M + gl.expand_dims(tc, 0)
            gth = gl.load(GTH + gto)
            gtl = gl.load(GTL + gto)
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
    """(BC, num_warps); PRIMUS_HC_B3X_CFG=bc,warps (default 128,8)."""
    v = os.environ.get("PRIMUS_HC_B3X_CFG", "").strip()
    if v:
        bc, nw = (int(t) for t in v.split(","))
        return bc, nw
    return 128, 8
