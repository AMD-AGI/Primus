###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused per-head RMSNorm + interleaved partial RoPE (+ zero pad) for the V4 query.

The V4 query path used to run three HBM passes over ``[B, S, H, head_dim]`` per layer
in the forward pass -- per-head RMSNorm (``fused_rms_norm``), RoPE
(``RoPEFromPositionsFn``) and the sparse-MLA adapter copy into a zero-padded
``[B*S, H, head_dim + 64]`` buffer (``_rope_pad_q``) -- and two in the backward
(RoPE bwd, RMSNorm bwd). This module does each direction in ONE pass:

* forward: read ``x`` once, write the normalized + rotated query straight into a
  ``[B, S, H, head_dim + pad]`` buffer (pad block zeroed) and return the
  ``[..., :head_dim]`` view of it. The adapter recognises that buffer (see
  ``v4_sparse_mla_adapter._rope_pad_q``) and uses it as-is, so the pad copy is gone;
* backward: read ``dq`` (any row strides, e.g. the adapter's padded dq view) and ``x``
  once, apply the transposed rotation and the RMSNorm adjoint in registers, write ``dx``.

Numerics mirror the unfused kernels op for op: rstd in fp32, normalized value rounded
to the input dtype, rotation on the rounded values with cos/sin cast to the input
dtype, RMSNorm backward on the (rounded) rotation-adjoint gradient in fp32. Only the
fp32 summation order of the two row reductions can differ.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_TL = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}

try:  # FP8 block-quant epilogue helpers (Primus-Turbo); only the PRIMUS_GLUE_WQB_DYQ kernel uses them
    from primus_turbo.triton.quantization.quant_epi_gfx1250 import pow2_neg, store_scale, ue8m0_exp_g90

    _GLUE_QUANT_OK = True
except Exception:  # noqa: BLE001
    _GLUE_QUANT_OK = False


@triton.jit
def _cols_rope_coeffs(pos, INVFREQ_PTR, d, NOPE: tl.constexpr, RD_HALF: tl.constexpr, DTYPE: tl.constexpr):
    """Per-column (cos, sin, partner column, is-rope, is-even) for the interleaved rotation."""
    is_rope = d >= NOPE
    j = tl.where(is_rope, (d - NOPE) // 2, 0)
    inv = tl.load(INVFREQ_PTR + j, mask=is_rope & (j < RD_HALF), other=0.0).to(tl.float32)
    angle = pos * inv
    cos = tl.cos(angle).to(DTYPE)
    sin = tl.sin(angle).to(DTYPE)
    partner = tl.where(is_rope, NOPE + ((d - NOPE) ^ 1), d)
    is_even = ((d - NOPE) % 2) == 0
    return cos, sin, partner, is_rope, is_even


@triton.jit
def _q_norm_rope_fwd_kernel(
    X_PTR,  # [N, H, D] contiguous
    POS_PTR,  # [N] fp32
    INVFREQ_PTR,  # [RD_HALF] fp32
    OUT_PTR,  # [N, H, D + PAD] contiguous
    RSTD_PTR,  # [N, H] fp32
    H,
    EPS: tl.constexpr,
    D: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    DP: tl.constexpr = D + PAD
    x_rows = X_PTR + pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D
    o_rows = OUT_PTR + pid_n.to(tl.int64) * H * DP + h[:, None].to(tl.int64) * DP

    pos = tl.load(POS_PTR + pid_n).to(tl.float32)
    cos, sin, partner, is_rope, is_even = _cols_rope_coeffs(pos, INVFREQ_PTR, d, NOPE, RD_HALF, DTYPE)

    x = tl.load(x_rows + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + EPS)  # [BLOCK_H]
    tl.store(RSTD_PTR + pid_n.to(tl.int64) * H + h, rstd, mask=hm)
    y = (x * rstd[:, None]).to(DTYPE)
    # partner value (only rope columns are read), normalized + rounded exactly like y
    xp = tl.load(x_rows + partner[None, :], mask=hm[:, None] & is_rope[None, :], other=0.0).to(tl.float32)
    yp = (xp * rstd[:, None]).to(DTYPE)
    # same expressions (operand order included) as the RoPE kernel:
    # even: even*cos - odd*sin ; odd: even*sin + odd*cos
    rot_e = y * cos[None, :] - yp * sin[None, :]
    rot_o = yp * sin[None, :] + y * cos[None, :]
    rot = tl.where(is_even[None, :], rot_e, rot_o)
    out = tl.where(is_rope[None, :], rot, y)
    tl.store(o_rows + d[None, :], out, mask=hm[:, None])
    if PAD > 0:
        p = tl.arange(0, BLOCK_PAD)
        tl.store(
            o_rows + D + p[None, :],
            tl.zeros((BLOCK_H, BLOCK_PAD), dtype=DTYPE),
            mask=hm[:, None] & (p[None, :] < PAD),
        )


@triton.jit
def _q_norm_rope_bwd_kernel(
    DQ_PTR,  # [N, H, D] last dim unit stride, rows strided (SN, SH)
    X_PTR,  # [N, H, D] contiguous (pre-norm input)
    RSTD_PTR,  # [N, H] fp32
    POS_PTR,
    INVFREQ_PTR,
    DX_PTR,  # [N, H, D] contiguous
    H,
    SN,
    SH,
    D: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    dq_rows = DQ_PTR + pid_n.to(tl.int64) * SN + h[:, None].to(tl.int64) * SH
    x_rows = X_PTR + pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D

    pos = tl.load(POS_PTR + pid_n).to(tl.float32)
    cos, sin, partner, is_rope, is_even = _cols_rope_coeffs(pos, INVFREQ_PTR, d, NOPE, RD_HALF, DTYPE)

    g = tl.load(dq_rows + d[None, :], mask=hm[:, None], other=0.0)
    gp = tl.load(dq_rows + partner[None, :], mask=hm[:, None] & is_rope[None, :], other=0.0)
    # RoPE adjoint, same expressions as the RoPE bwd kernel:
    # even: dout_even*cos + dout_odd*sin ; odd: -dout_even*sin + dout_odd*cos
    g_e = g * cos[None, :] + gp * sin[None, :]
    g_o = -gp * sin[None, :] + g * cos[None, :]
    g_rot = tl.where(is_even[None, :], g_e, g_o)
    dy = tl.where(is_rope[None, :], g_rot, g).to(tl.float32)

    x = tl.load(x_rows + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_PTR + pid_n.to(tl.int64) * H + h, mask=hm, other=0.0)
    c = tl.sum(dy * x, axis=1)
    coef = rstd * rstd * rstd * c / D
    dx = rstd[:, None] * dy - x * coef[:, None]
    tl.store(
        DX_PTR + pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D + d[None, :],
        dx.to(DTYPE),
        mask=hm[:, None],
    )


@triton.jit
def _pair_cs(pos, INVFREQ_PTR, HALF: tl.constexpr, NOPE: tl.constexpr):
    p = tl.arange(0, HALF)
    fi = p - NOPE // 2
    is_rope = fi >= 0
    ang = pos * tl.load(INVFREQ_PTR + fi, mask=is_rope, other=0.0).to(tl.float32)
    return tl.where(is_rope, tl.cos(ang), 1.0), tl.where(is_rope, tl.sin(ang), 0.0)


@triton.jit
def _q_norm_rope_fwd2_kernel(X_PTR, POS_PTR, INVFREQ_PTR, OUT_PTR, RSTD_PTR, H, EPS: tl.constexpr, D: tl.constexpr,
                             NOPE: tl.constexpr, PAD: tl.constexpr, BLOCK_PAD: tl.constexpr, BLOCK_H: tl.constexpr):
    """Pair-split variant (g90mem): rotation on (even, odd) pairs from ``tl.split`` in fp32, one rounding;
    [4096, 64, 512] + pad 64: 67.0 -> 44.9 us (75% of HBM floor)."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    DP: tl.constexpr = D + PAD
    x = tl.load(X_PTR + pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D + d[None, :], mask=hm[:, None],
                other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + EPS)
    tl.store(RSTD_PTR + pid_n.to(tl.int64) * H + h, rstd, mask=hm)
    c, s = _pair_cs(tl.load(POS_PTR + pid_n).to(tl.float32), INVFREQ_PTR, HALF, NOPE)
    a, b = tl.split(tl.reshape(x * rstd[:, None], (BLOCK_H, HALF, 2)))
    o = tl.reshape(tl.join(a * c[None, :] - b * s[None, :], a * s[None, :] + b * c[None, :]), (BLOCK_H, D))
    orow = OUT_PTR + pid_n.to(tl.int64) * H * DP + h[:, None].to(tl.int64) * DP
    tl.store(orow + d[None, :], o.to(OUT_PTR.dtype.element_ty), mask=hm[:, None])
    if PAD > 0:
        p = tl.arange(0, BLOCK_PAD)
        tl.store(orow + D + p[None, :], tl.zeros((BLOCK_H, BLOCK_PAD), dtype=OUT_PTR.dtype.element_ty),
                 mask=hm[:, None] & (p[None, :] < PAD))


@triton.jit
def _q_norm_rope_bwd2_kernel(DQ_PTR, X_PTR, RSTD_PTR, POS_PTR, INVFREQ_PTR, DX_PTR, H, SN, SH, D: tl.constexpr,
                             NOPE: tl.constexpr, BLOCK_H: tl.constexpr):
    """Pair-split backward (g90mem): 69.3 -> 54.9 us (84% of HBM floor)."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    g = tl.load(DQ_PTR + pid_n.to(tl.int64) * SN + h[:, None].to(tl.int64) * SH + d[None, :], mask=hm[:, None],
                other=0.0).to(tl.float32)
    c, s = _pair_cs(tl.load(POS_PTR + pid_n).to(tl.float32), INVFREQ_PTR, HALF, NOPE)
    ga, gb = tl.split(tl.reshape(g, (BLOCK_H, HALF, 2)))
    dy = tl.reshape(tl.join(ga * c[None, :] + gb * s[None, :], gb * c[None, :] - ga * s[None, :]), (BLOCK_H, D))
    xrow = pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D + d[None, :]
    x = tl.load(X_PTR + xrow, mask=hm[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_PTR + pid_n.to(tl.int64) * H + h, mask=hm, other=0.0)
    coef = rstd * rstd * rstd * tl.sum(dy * x, axis=1) / D
    tl.store(DX_PTR + xrow, (rstd[:, None] * dy - x * coef[:, None]).to(DX_PTR.dtype.element_ty), mask=hm[:, None])


@triton.jit
def _q_norm_rope_bwd_dyq_kernel(
    DQ_PTR, X_PTR, RSTD_PTR, POS_PTR, INVFREQ_PTR, COEF_PTR,
    QR_PTR, ER_PTR, QT_PTR, ET_PTR, DX_PTR,
    NTOK, H, SN, SH,
    D: tl.constexpr, NOPE: tl.constexpr, BLOCK_H: tl.constexpr,
    FP8_MAX: tl.constexpr, MAX_EXP: tl.constexpr, SLR: tl.constexpr, SLT: tl.constexpr, WRITE_DX: tl.constexpr,
):
    """PRIMUS_GLUE_WQB_DYQ=1: ``_q_norm_rope_bwd2_kernel`` + the wq_b dY dual quant (``_quant_dual_t2_kernel``) in
    one pass, so the bf16 dX of the wq_b output is never written to / re-read from HBM.

    Program = (128 tokens, one head). Phase A walks the 128 tokens 8 at a time with the bwd2 tile shape
    ([BLOCK_H=8, D] rows, same warps) and stores coef = rstd^3 * sum(dy * x) / D per (token, head) -- the same
    reduction as bwd2. Phase B rebuilds dX per [128 tokens x 128 columns] tile with bwd2's elementwise expressions,
    rounds it to bf16 exactly like bwd2's store, and quantizes the tile exactly like ``_quant_dual_t2_kernel``:
    1x128 row blocks -> QR [NTOK, H*D] + ER (layout SLR); 128x1 column blocks -> QT [H*D, NTOK] + ET (layout SLT).
    """
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, D)
    # ---- phase A: coef per (token, head), bwd2's [BLOCK_H, D] reduction ----
    for i in range(0, 128 // BLOCK_H):
        t = pid_m * 128 + i * BLOCK_H + tl.arange(0, BLOCK_H)
        t64 = t.to(tl.int64)
        g = tl.load(DQ_PTR + t64[:, None] * SN + h.to(tl.int64) * SH + d[None, :]).to(tl.float32)
        pos = tl.load(POS_PTR + t).to(tl.float32)
        p = tl.arange(0, HALF)
        fi = p - NOPE // 2
        is_rope = fi >= 0
        ang = pos[:, None] * tl.load(INVFREQ_PTR + fi, mask=is_rope, other=0.0).to(tl.float32)[None, :]
        c = tl.where(is_rope[None, :], tl.cos(ang), 1.0)
        s = tl.where(is_rope[None, :], tl.sin(ang), 0.0)
        ga, gb = tl.split(tl.reshape(g, (BLOCK_H, HALF, 2)))
        dy = tl.reshape(tl.join(ga * c + gb * s, gb * c - ga * s), (BLOCK_H, D))
        xrow = t64[:, None] * H * D + h.to(tl.int64) * D + d[None, :]
        x = tl.load(X_PTR + xrow).to(tl.float32)
        rstd = tl.load(RSTD_PTR + t64 * H + h)
        coef = rstd * rstd * rstd * tl.sum(dy * x, axis=1) / D
        tl.store(COEF_PTR + t64 * H + h, coef)
    tl.debug_barrier()
    # ---- phase B: dX tiles [128 tok, 128 col] -> bf16 rounding -> dual 1x128 / 128x1 quant ----
    tm = pid_m * 128 + tl.arange(0, 128)
    tm64 = tm.to(tl.int64)
    rstd_b = tl.load(RSTD_PTR + tm64 * H + h)
    coef_b = tl.load(COEF_PTR + tm64 * H + h)
    pos_b = tl.load(POS_PTR + tm).to(tl.float32)
    NCOL = H * D
    nb_n = NCOL // 128
    nb_m = NTOK // 128
    for j in range(0, D // 128):
        dc = j * 128 + tl.arange(0, 128)
        g = tl.load(DQ_PTR + tm64[:, None] * SN + h.to(tl.int64) * SH + dc[None, :]).to(tl.float32)
        p = j * 64 + tl.arange(0, 64)
        fi = p - NOPE // 2
        is_rope = fi >= 0
        ang = pos_b[:, None] * tl.load(INVFREQ_PTR + fi, mask=is_rope, other=0.0).to(tl.float32)[None, :]
        c = tl.where(is_rope[None, :], tl.cos(ang), 1.0)
        s = tl.where(is_rope[None, :], tl.sin(ang), 0.0)
        ga, gb = tl.split(tl.reshape(g, (128, 64, 2)))
        dy = tl.reshape(tl.join(ga * c + gb * s, gb * c - ga * s), (128, 128))
        xrow = tm64[:, None] * H * D + h.to(tl.int64) * D + dc[None, :]
        x = tl.load(X_PTR + xrow).to(tl.float32)
        dxb = (rstd_b[:, None] * dy - x * coef_b[:, None]).to(DX_PTR.dtype.element_ty)
        if WRITE_DX:
            tl.store(DX_PTR + xrow, dxb)
        v = dxb.to(tl.float32)
        av = tl.abs(v)
        ncol = h * D + dc  # global column of the [NTOK, H*D] dY
        nblk = h * (D // 128) + j  # its 128-column block
        # row quant (1x128 along columns): identical to _quant_dual_t2_kernel WITH_ROW
        er = ue8m0_exp_g90(tl.max(av, axis=1), MAX_EXP)
        qr = tl.clamp(v * pow2_neg(er)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(QR_PTR + tm64[:, None] * NCOL + ncol[None, :], qr.to(QR_PTR.dtype.element_ty))
        store_scale(ER_PTR, tm, nblk, er, NTOK, nb_n, SLR)
        # column quant (128 tokens per block) + transposed store: identical to WITH_T, TMODE 0
        et = ue8m0_exp_g90(tl.max(av, axis=0), MAX_EXP)
        qt = tl.clamp(v * pow2_neg(et)[None, :], -FP8_MAX, FP8_MAX).to(QT_PTR.dtype.element_ty)
        tl.store(QT_PTR + ncol[:, None].to(tl.int64) * NTOK + tm64[None, :], tl.trans(qt))
        store_scale(ET_PTR, ncol, pid_m, et, NCOL, nb_m, SLT)


@triton.jit
def _qnr_coef_kernel(DQ_PTR, X_PTR, RSTD_PTR, POS_PTR, INVFREQ_PTR, COEF_PTR, CS_PTR, H, SN, SH, D: tl.constexpr,
                     NOPE: tl.constexpr, BLOCK_H: tl.constexpr):
    """``_q_norm_rope_bwd2_kernel`` up to its row reduction (same code, so the same bits); stores
    coef = rstd^3 * sum(dy * x) / D per (token, head) instead of dX, and (head block 0) the token's rope
    (cos, sin) for the RD/2 rotated pairs, so the dX/quant kernel needs no transcendental."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    RDH: tl.constexpr = HALF - NOPE // 2
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    g = tl.load(DQ_PTR + pid_n.to(tl.int64) * SN + h[:, None].to(tl.int64) * SH + d[None, :], mask=hm[:, None],
                other=0.0).to(tl.float32)
    c, s = _pair_cs(tl.load(POS_PTR + pid_n).to(tl.float32), INVFREQ_PTR, HALF, NOPE)
    ga, gb = tl.split(tl.reshape(g, (BLOCK_H, HALF, 2)))
    dy = tl.reshape(tl.join(ga * c[None, :] + gb * s[None, :], gb * c[None, :] - ga * s[None, :]), (BLOCK_H, D))
    xrow = pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D + d[None, :]
    x = tl.load(X_PTR + xrow, mask=hm[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_PTR + pid_n.to(tl.int64) * H + h, mask=hm, other=0.0)
    coef = rstd * rstd * rstd * tl.sum(dy * x, axis=1) / D
    tl.store(COEF_PTR + pid_n.to(tl.int64) * H + h, coef, mask=hm)
    if pid_h == 0:
        fi = tl.arange(0, HALF) - NOPE // 2
        cm = fi >= 0
        tl.store(CS_PTR + pid_n.to(tl.int64) * (2 * RDH) + fi, c, mask=cm)
        tl.store(CS_PTR + pid_n.to(tl.int64) * (2 * RDH) + RDH + fi, s, mask=cm)


@triton.jit
def _qnr_dx_dyq_kernel(
    DQ_PTR, X_PTR, RSTD_PTR, CS_PTR, COEF_PTR,
    QR_PTR, ER_PTR, QT_PTR, ET_PTR, DX_PTR,
    NTOK, H, SN, SH,
    D: tl.constexpr, NOPE: tl.constexpr,
    FP8_MAX: tl.constexpr, MAX_EXP: tl.constexpr, SLR: tl.constexpr, SLT: tl.constexpr, WRITE_DX: tl.constexpr,
    ABL: tl.constexpr = 0,
):
    """dX tile [128 tokens x 128 columns] rebuilt from (dq, x, rstd, coef, cos/sin table) with bwd2's pair-split
    expressions (non-rope pairs use c = 1, s = 0 exactly like bwd2's tl.where), rounded like bwd2's store, then
    quantized exactly like
    ``_quant_dual_t2_kernel`` (row 1x128 -> QR/ER, column 128x1 -> QT/ET, TMODE 0)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    NCOL = H * D
    RDH: tl.constexpr = D // 2 - NOPE // 2
    h = (pid_n * 128) // D
    dl = (pid_n * 128) % D + tl.arange(0, 128)  # column inside the head
    tm = pid_m * 128 + tl.arange(0, 128)
    tm64 = tm.to(tl.int64)
    base = tm64[:, None] * SN + h.to(tl.int64) * SH
    g = tl.load(DQ_PTR + base + dl[None, :]).to(tl.float32)
    fi = ((pid_n * 128) % D) // 2 + tl.arange(0, 64) - NOPE // 2  # rope pair index of the tile's 64 pairs
    rm = (fi >= 0)[None, :]
    csrow = CS_PTR + tm64[:, None] * (2 * RDH) + fi[None, :]
    c = tl.load(csrow, mask=rm, other=1.0)
    s = tl.load(csrow + RDH, mask=rm, other=0.0)
    if ABL == 1:  # ablation: no rotation
        dy = g
    else:
        ga, gb = tl.split(tl.reshape(g, (128, 64, 2)))
        dy = tl.reshape(tl.join(ga * c + gb * s, gb * c - ga * s), (128, 128))
    xrow = tm64[:, None] * NCOL + (pid_n * 128 + tl.arange(0, 128))[None, :]
    x = tl.load(X_PTR + xrow).to(tl.float32)
    rstd = tl.load(RSTD_PTR + tm64 * H + h)
    coef = tl.load(COEF_PTR + tm64 * H + h)
    dxb = (rstd[:, None] * dy - x * coef[:, None]).to(DX_PTR.dtype.element_ty)
    if WRITE_DX:
        tl.store(DX_PTR + xrow, dxb)
    v = dxb.to(tl.float32)
    av = tl.abs(v)
    ncol = pid_n * 128 + tl.arange(0, 128)
    if ABL != 3:  # ablation 3: no row quant
        er = ue8m0_exp_g90(tl.max(av, axis=1), MAX_EXP)
        qr = tl.clamp(v * pow2_neg(er)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(QR_PTR + tm64[:, None] * NCOL + ncol[None, :], qr.to(QR_PTR.dtype.element_ty))
        store_scale(ER_PTR, tm, pid_n, er, NTOK, NCOL // 128, SLR)
    if ABL != 2:  # ablation 2: no transposed column quant
        et = ue8m0_exp_g90(tl.max(av, axis=0), MAX_EXP)
        qt = tl.clamp(v * pow2_neg(et)[None, :], -FP8_MAX, FP8_MAX).to(QT_PTR.dtype.element_ty)
        tl.store(QT_PTR + ncol[:, None].to(tl.int64) * NTOK + tm64[None, :], tl.trans(qt))
        store_scale(ET_PTR, ncol, pid_m, et, NCOL, NTOK // 128, SLT)


def glue_wqb_dyq_enabled() -> bool:
    """PRIMUS_GLUE_WQB_DYQ=1 (default off): the q norm/rope backward also emits the wq_b dY FP8 copies + scales
    (Turbo ``FP8GemmBlockFunction`` dy-prequant slot), so the separate ``_quant_dual_t2`` pass and the bf16 dX
    HBM round trip disappear."""
    return os.environ.get("PRIMUS_GLUE_WQB_DYQ", "0") == "1"


def q_norm_rope_bwd_dyq(dq3, x3, rstd, pos_c, inv_c, rotary_dim, row_layout, t_layout, fp8_dtype, write_dx=False):
    """Fused backward: returns (q_row [N, H*D], e8m0_row, q_t [H*D, N], e8m0_t, dx or None)."""
    from primus_turbo.triton.quantization.quant_g90 import _scale_buf
    from primus_turbo.triton.quantization.quant_epi_gfx1250 import fp8_max_exp, scale_layout_id

    N, H, D = x3.shape
    assert N % 128 == 0 and D % 128 == 0 and dq3.stride(2) == 1 and x3.is_contiguous()
    NCOL = H * D
    dev = x3.device
    slr, slt = scale_layout_id(row_layout), scale_layout_id(t_layout)
    qr = torch.empty((N, NCOL), dtype=fp8_dtype, device=dev)
    er = _scale_buf(N, NCOL, slr, dev)
    qt = torch.empty((NCOL, N), dtype=fp8_dtype, device=dev)
    et = _scale_buf(NCOL, N, slt, dev)
    coef = torch.empty((N, H), dtype=torch.float32, device=dev)
    dx = torch.empty((N, H, D), dtype=x3.dtype, device=dev) if write_dx else None
    fmax = float(torch.finfo(fp8_dtype).max)
    mode = os.environ.get("PRIMUS_GLUE_WQB_DYQ_MODE", "two")
    if mode == "one":  # one kernel per (128 tokens, head): coef pass + dX/quant pass
        _q_norm_rope_bwd_dyq_kernel[(N // 128, H)](
            dq3, x3, rstd, pos_c, inv_c, coef, qr, er, qt, et, dx if write_dx else x3,
            N, H, dq3.stride(0), dq3.stride(1), D=D, NOPE=D - rotary_dim, BLOCK_H=8,
            FP8_MAX=fmax, MAX_EXP=fp8_max_exp(fmax), SLR=slr, SLT=slt, WRITE_DX=bool(write_dx),
            num_warps=int(os.environ.get("PRIMUS_GLUE_WQB_DYQ_NW", "4")),
        )
    else:  # "two": coef kernel (bwd2's reduction) + per-128x128-tile dX rebuild and dual quant
        bh2 = min(8, triton.next_power_of_2(max(H, 1)))
        cs = torch.empty((N, rotary_dim), dtype=torch.float32, device=dev)  # [cos | sin] of the rope pairs
        _qnr_coef_kernel[(N, triton.cdiv(H, bh2))](
            dq3, x3, rstd, pos_c, inv_c, coef, cs, H, dq3.stride(0), dq3.stride(1), D=D, NOPE=D - rotary_dim,
            BLOCK_H=bh2, num_warps=4,
        )
        _qnr_dx_dyq_kernel[(N // 128, NCOL // 128)](
            dq3, x3, rstd, cs, coef, qr, er, qt, et, dx if write_dx else x3,
            N, H, dq3.stride(0), dq3.stride(1), D=D, NOPE=D - rotary_dim,
            FP8_MAX=fmax, MAX_EXP=fp8_max_exp(fmax), SLR=slr, SLT=slt, WRITE_DX=bool(write_dx),
            ABL=int(os.environ.get("PRIMUS_GLUE_WQB_DYQ_ABL", "0")),
            num_warps=int(os.environ.get("PRIMUS_GLUE_WQB_DYQ_NW", "8")),
        )
    return qr, er, qt, et, dx


def _split_ok() -> bool:
    return os.environ.get("PRIMUS_V4_QNR_SPLIT", "1") != "0"


def _block_h(H: int) -> int:
    return min(triton.next_power_of_2(max(H, 1)), int(os.environ.get("PRIMUS_V4_QNR_BLOCK_H", "8")))


class QNormRopePadFn(torch.autograd.Function):
    """``x [B,S,H,D]`` -> ``rope(rmsnorm(x))`` as the ``[..., :D]`` view of a zero-padded buffer."""

    @staticmethod
    def forward(ctx, x, pos_flat, inv_freq, rotary_dim, eps, pad, dyq_slot=None):  # type: ignore[override]
        B, S, H, D = x.shape
        N = B * S
        x3 = x.reshape(N, H, D).contiguous()
        from .rope_interleaved_partial import rope_kernel_vectors

        pos_c, inv_c = rope_kernel_vectors(pos_flat.reshape(-1), inv_freq)  # PRIMUS_GLUE_ROPE_NOCAST: no casts
        buf = torch.empty((B, S, H, D + pad), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, H), dtype=torch.float32, device=x.device)
        bh = _block_h(H)
        if _split_ok():
            # PRIMUS_V4_QNR_FWD_BH / _NW (fuseattn): tile of the forward kernel. Any BLOCK_H with num_warps >= 2 keeps
            # the per-head reduction tree (8 elements x 32 lanes x 2 warps over D = 512), so the bits; [4096, 64, 512]:
            # BH 16 / 8 warps 44.9 us -> BH 8 / 2 warps 40.2 us (tools/sweep_qnr.py).
            bh2 = min(int(os.environ.get("PRIMUS_V4_QNR_FWD_BH", "16")), triton.next_power_of_2(max(H, 1)))
            nw2 = int(os.environ.get("PRIMUS_V4_QNR_FWD_NW", "0")) or (8 if bh2 >= 16 else 4)
            _q_norm_rope_fwd2_kernel[(N, triton.cdiv(H, bh2))](
                x3, pos_c, inv_c, buf, rstd, H, EPS=float(eps), D=D, NOPE=D - rotary_dim, PAD=pad,
                BLOCK_PAD=max(triton.next_power_of_2(max(pad, 1)), 1), BLOCK_H=bh2, num_warps=nw2,
            )
        else:
          _q_norm_rope_fwd_kernel[(N, triton.cdiv(H, bh))](
            x3,
            pos_c,
            inv_c,
            buf,
            rstd,
            H,
            EPS=float(eps),
            D=D,
            NOPE=D - rotary_dim,
            RD_HALF=rotary_dim // 2,
            PAD=pad,
            BLOCK_PAD=max(triton.next_power_of_2(max(pad, 1)), 1),
            BLOCK_H=bh,
            DTYPE=_TL[x.dtype],
        )
        ctx.save_for_backward(x3, rstd, pos_c, inv_c)
        ctx.meta = (B, S, H, D, rotary_dim, x.dtype)
        # PRIMUS_GLUE_WQB_DYQ=1: the slot of the FP8 linear that produced x (its backward takes our FP8 dY)
        ctx.dyq_slot = dyq_slot if (dyq_slot is not None and _split_ok() and _dyq_shape_ok(dyq_slot, N, H, D)) else None
        if pad:
            buf._v4_rope_pad = pad  # lets the sparse-MLA adapter use the buffer without copying
        return buf[..., :D]

    @staticmethod
    def backward(ctx, dq):  # type: ignore[override]
        x3, rstd, pos_c, inv_c = ctx.saved_tensors
        B, S, H, D, rotary_dim, dtype = ctx.meta
        N = B * S
        try:
            dq3 = dq.view(N, H, D)
        except RuntimeError:
            dq3 = dq.reshape(N, H, D)
        if dq3.stride(2) != 1:
            dq3 = dq3.contiguous()
        slot, ctx.dyq_slot = ctx.dyq_slot, None
        if slot is not None and _GLUE_QUANT_OK:
            # fused backward + wq_b dY dual quant: the FP8 copies go to the linear's backward through the slot;
            # autograd gets a bf16 carrier with dX's shape whose storage identifies it (never read).
            qr, er, qt, et, _ = q_norm_rope_bwd_dyq(
                dq3, x3, rstd, pos_c, inv_c, rotary_dim, slot.row_layout, slot.t_layout, slot.fp8_dtype)
            dx = torch.empty((B, S, H, D), dtype=dtype, device=dq.device)
            slot.fill(dx, (qr, er, qt, et))
            return dx, None, None, None, None, None, None
        dx = torch.empty((B, S, H, D), dtype=dtype, device=dq.device)
        bh = _block_h(H)
        if _split_ok():
            bh2 = min(8, triton.next_power_of_2(max(H, 1)))
            _q_norm_rope_bwd2_kernel[(N, triton.cdiv(H, bh2))](
                dq3, x3, rstd, pos_c, inv_c, dx, H, dq3.stride(0), dq3.stride(1), D=D, NOPE=D - rotary_dim,
                BLOCK_H=bh2, num_warps=4,
            )
        else:
          _q_norm_rope_bwd_kernel[(N, triton.cdiv(H, bh))](
            dq3,
            x3,
            rstd,
            pos_c,
            inv_c,
            dx,
            H,
            dq3.stride(0),
            dq3.stride(1),
            D=D,
            NOPE=D - rotary_dim,
            RD_HALF=rotary_dim // 2,
            BLOCK_H=bh,
            DTYPE=_TL[dtype],
        )
        return dx, None, None, None, None, None, None


def _dyq_shape_ok(slot, N, H, D) -> bool:
    return (
        getattr(slot, "M", None) == N and getattr(slot, "N", None) == H * D and N % 128 == 0 and D % 128 == 0
        and D <= 512
    )


def q_norm_rope_supported(x: torch.Tensor, rotary_dim: int) -> bool:
    D = x.shape[-1]
    return (
        os.environ.get("PRIMUS_V4_Q_NORM_ROPE_FUSED", "1") != "0"
        and x.is_cuda
        and x.dim() == 4
        and x.dtype in (torch.bfloat16, torch.float16)
        and D >= 2
        and (D & (D - 1)) == 0
        and 0 < rotary_dim <= D
        and rotary_dim % 2 == 0
    )


def q_norm_rope_pad(x, pos_flat, inv_freq, *, rotary_dim: int, eps: float, pad: int = 0, dyq_slot=None):
    """Fused per-head RMSNorm + interleaved partial RoPE; output optionally zero-padded by ``pad``.
    ``dyq_slot``: Turbo FP8GemmBlockFunction dy-prequant slot of the linear that produced ``x`` (PRIMUS_GLUE_WQB_DYQ)."""
    return QNormRopePadFn.apply(x, pos_flat, inv_freq, rotary_dim, eps, pad, dyq_slot)


__all__ = ["QNormRopePadFn", "q_norm_rope_pad", "q_norm_rope_supported"]
