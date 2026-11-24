###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HyperConnection expand backward as a Gluon kernel (gfx1250).

``post_bwd_v2`` replaces ``_hc_post_bwd2d_kernel`` (1 row x 256 columns per 1-warp program) with a kernel whose 20
per-row reductions are in-thread partials followed by one LDS transpose and the same summation tree, so the outputs
(dout, dpost and dcomb^T partials) are byte-identical. Callers must check that Gluon is available.
"""

from __future__ import annotations

from primus.backends.megatron.core.dsv4_features import gluon_ok

_HAVE_GLUON = gluon_ok()
if _HAVE_GLUON:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl


if _HAVE_GLUON:
    # Expand backward with the same lane mapping as _hc_post_bwd2d_kernel (1 row x 256 cols per 1-warp program,
    # 8 contiguous columns per lane); the 20 per-row reductions are in-thread partials -> one LDS transpose -> the
    # DPP tree of _hc_post_bwd2d_kernel (row_shr 8/4/2/1 inside each 16-lane row, then the two rows) done in-thread
    # (ORDER 3). Byte-identical to _hc_post_bwd2d_kernel (dout, dpost, dcomb^T partials); no shuffle chains.
    @gluon.jit
    def _hcf_post_v2_kernel(
        G,
        X,
        OUT,
        POST,
        DOUT,
        DPOSTP,
        DCOMBTP,
        M,
        D,
        L: gl.constexpr,
        SQ: gl.constexpr,
        BC: gl.constexpr,
        LR: gl.constexpr,
    ):
        row = gl.program_id(0)
        pid_c = gl.program_id(1)
        li = gl.arange(0, 32, layout=gl.SliceLayout(1, L))
        ci = gl.arange(0, 8, layout=gl.SliceLayout(0, L))
        col = pid_c * BC + gl.expand_dims(li * 8, 1) + gl.expand_dims(ci, 0)
        r64 = row.to(gl.int64)
        KD = 4 * D
        xb = X + r64 * KD + col
        gb = G + r64 * KD + col
        x0 = gl.load(xb).to(gl.float32)
        x1 = gl.load(xb + D).to(gl.float32)
        x2 = gl.load(xb + 2 * D).to(gl.float32)
        x3 = gl.load(xb + 3 * D).to(gl.float32)
        o = gl.load(OUT + r64 * D + col).to(gl.float32)
        g0 = gl.load(gb).to(gl.float32)
        g1 = gl.load(gb + D).to(gl.float32)
        g2 = gl.load(gb + 2 * D).to(gl.float32)
        g3 = gl.load(gb + 3 * D).to(gl.float32)
        p0 = gl.load(POST + r64 * 4 + 0).to(gl.float32)
        p1 = gl.load(POST + r64 * 4 + 1).to(gl.float32)
        p2 = gl.load(POST + r64 * 4 + 2).to(gl.float32)
        p3 = gl.load(POST + r64 * 4 + 3).to(gl.float32)
        dout = p0 * g0
        dout += p1 * g1
        dout += p2 * g2
        dout += p3 * g3
        gl.store(DOUT + r64 * D + col, dout.to(gl.bfloat16))
        q = gl.allocate_shared_memory(gl.float32, [32, 32], SQ)
        q.slice(0, 1, 1).store(gl.expand_dims(gl.sum(g0 * x0, axis=1), 1))
        q.slice(1, 1, 1).store(gl.expand_dims(gl.sum(g0 * x1, axis=1), 1))
        q.slice(2, 1, 1).store(gl.expand_dims(gl.sum(g0 * x2, axis=1), 1))
        q.slice(3, 1, 1).store(gl.expand_dims(gl.sum(g0 * x3, axis=1), 1))
        q.slice(4, 1, 1).store(gl.expand_dims(gl.sum(g1 * x0, axis=1), 1))
        q.slice(5, 1, 1).store(gl.expand_dims(gl.sum(g1 * x1, axis=1), 1))
        q.slice(6, 1, 1).store(gl.expand_dims(gl.sum(g1 * x2, axis=1), 1))
        q.slice(7, 1, 1).store(gl.expand_dims(gl.sum(g1 * x3, axis=1), 1))
        q.slice(8, 1, 1).store(gl.expand_dims(gl.sum(g2 * x0, axis=1), 1))
        q.slice(9, 1, 1).store(gl.expand_dims(gl.sum(g2 * x1, axis=1), 1))
        q.slice(10, 1, 1).store(gl.expand_dims(gl.sum(g2 * x2, axis=1), 1))
        q.slice(11, 1, 1).store(gl.expand_dims(gl.sum(g2 * x3, axis=1), 1))
        q.slice(12, 1, 1).store(gl.expand_dims(gl.sum(g3 * x0, axis=1), 1))
        q.slice(13, 1, 1).store(gl.expand_dims(gl.sum(g3 * x1, axis=1), 1))
        q.slice(14, 1, 1).store(gl.expand_dims(gl.sum(g3 * x2, axis=1), 1))
        q.slice(15, 1, 1).store(gl.expand_dims(gl.sum(g3 * x3, axis=1), 1))
        q.slice(16, 1, 1).store(gl.expand_dims(gl.sum(o * g0, axis=1), 1))
        q.slice(17, 1, 1).store(gl.expand_dims(gl.sum(o * g1, axis=1), 1))
        q.slice(18, 1, 1).store(gl.expand_dims(gl.sum(o * g2, axis=1), 1))
        q.slice(19, 1, 1).store(gl.expand_dims(gl.sum(o * g3, axis=1), 1))
        gl.barrier()
        # the DPP tree of _hc_post_bwd2d_kernel over the 32 lane partials: row_shr 8/4/2/1 inside each 16-lane row,
        # then the two rows
        v0 = q.slice(0, 1, 0).load(LR)
        v1 = q.slice(1, 1, 0).load(LR)
        v2 = q.slice(2, 1, 0).load(LR)
        v3 = q.slice(3, 1, 0).load(LR)
        v4 = q.slice(4, 1, 0).load(LR)
        v5 = q.slice(5, 1, 0).load(LR)
        v6 = q.slice(6, 1, 0).load(LR)
        v7 = q.slice(7, 1, 0).load(LR)
        v8 = q.slice(8, 1, 0).load(LR)
        v9 = q.slice(9, 1, 0).load(LR)
        v10 = q.slice(10, 1, 0).load(LR)
        v11 = q.slice(11, 1, 0).load(LR)
        v12 = q.slice(12, 1, 0).load(LR)
        v13 = q.slice(13, 1, 0).load(LR)
        v14 = q.slice(14, 1, 0).load(LR)
        v15 = q.slice(15, 1, 0).load(LR)
        v16 = q.slice(16, 1, 0).load(LR)
        v17 = q.slice(17, 1, 0).load(LR)
        v18 = q.slice(18, 1, 0).load(LR)
        v19 = q.slice(19, 1, 0).load(LR)
        v20 = q.slice(20, 1, 0).load(LR)
        v21 = q.slice(21, 1, 0).load(LR)
        v22 = q.slice(22, 1, 0).load(LR)
        v23 = q.slice(23, 1, 0).load(LR)
        v24 = q.slice(24, 1, 0).load(LR)
        v25 = q.slice(25, 1, 0).load(LR)
        v26 = q.slice(26, 1, 0).load(LR)
        v27 = q.slice(27, 1, 0).load(LR)
        v28 = q.slice(28, 1, 0).load(LR)
        v29 = q.slice(29, 1, 0).load(LR)
        v30 = q.slice(30, 1, 0).load(LR)
        v31 = q.slice(31, 1, 0).load(LR)
        r0a = ((v15 + v7) + (v11 + v3)) + ((v13 + v5) + (v9 + v1))
        r0b = ((v14 + v6) + (v10 + v2)) + ((v12 + v4) + (v8 + v0))
        r1a = ((v31 + v23) + (v27 + v19)) + ((v29 + v21) + (v25 + v17))
        r1b = ((v30 + v22) + (v26 + v18)) + ((v28 + v20) + (v24 + v16))
        s2 = (r0a + r0b) + (r1a + r1b)
        s = gl.reshape(s2, [32])
        j = gl.arange(0, 32, layout=gl.SliceLayout(0, LR))
        s = gl.convert_layout(s, gl.SliceLayout(0, LR))
        base = pid_c * M + r64
        gl.store(DCOMBTP + base * 16 + j, s, mask=j < 16)
        gl.store(DPOSTP + base * 4 + (j - 16), s, mask=(j >= 16) & (j < 20))

    _LY2 = {}

    def _post_v2_layouts():
        if not _LY2:
            _LY2["L"] = gl.BlockedLayout([1, 8], [32, 1], [1, 1], [1, 0])
            _LY2["SQ"] = gl.PaddedSharedLayout.with_identity_for([[32, 1]], [32, 32], [1, 0])
            _LY2["LR"] = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
        return _LY2


def post_bwd_v2(dnew, x3, out_c, post_c, dout, dpost, dcombt, bc=256):
    """Byte-identical replacement of _hc_post_bwd2d_kernel (BT 1, 1 warp, BC 256). Needs Gluon."""
    M, K, D = x3.shape
    ly = _post_v2_layouts()
    _hcf_post_v2_kernel[(M, D // bc)](
        dnew,
        x3,
        out_c,
        post_c,
        dout,
        dpost,
        dcombt,
        M,
        D,
        L=ly["L"],
        SQ=ly["SQ"],
        BC=bc,
        LR=ly["LR"],
        num_warps=1,
    )
