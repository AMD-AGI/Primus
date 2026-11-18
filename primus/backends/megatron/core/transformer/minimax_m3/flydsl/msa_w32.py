###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 wave32 helpers for the MiniMax MSA kernels.

``v_wmma_f32_16x16x32_bf16`` on a 32-lane wave, measured on gfx1250:

* A, matrix_a row-major ``[16, 32]``: ``row = lane % 16``,
  ``k = (lane // 16) * 16 + i`` for ``i`` in ``0..15``. One lane holds 16
  contiguous K of one row; the two lane groups split K.
* B, matrix_b so that ``C = A @ B``: ``col = lane % 16``,
  ``k = (lane // 16) * 16 + i``. One lane holds 16 contiguous K of one column.
* C, ``[16, 16]`` fp32: ``col = lane % 16``,
  ``row = (lane // 16) * 8 + i`` for ``i`` in ``0..7``. Lanes ``0..15`` hold
  rows ``0..7``, lanes ``16..31`` hold rows ``8..15``.

A column reduction is one ``v_permlane16_swap``. With both operands equal to
the lane's value, result 0 is the low half's value and result 1 is the high
half's value, on every lane.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw


def _pair_type():
    return ir.Type.parse("!llvm.struct<(i32, i32)>")


def wmma_f32_16x16x32_bf16(a, b, c):
    """One ``v_wmma_f32_16x16x32_bf16``. ``a``/``b`` are 16 bf16, ``c`` is 8 f32.

    Emitted as the LLVM intrinsic. The ROCDL op's ``modC`` attribute does not
    round-trip through this flydsl build.
    """
    acc_ty = Vec.make_type(8, fx.Float32)
    i16 = ir.IntegerType.get_signless(16)
    i1 = ir.IntegerType.get_signless(1)
    cmod = arith.ConstantOp(i16, ir.IntegerAttr.get(i16, 0)).result
    no = arith.ConstantOp(i1, ir.IntegerAttr.get(i1, 0)).result
    return llvm.call_intrinsic(
        acc_ty,
        "llvm.amdgcn.wmma.f32.16x16x32.bf16",
        [_raw(a), _raw(b), cmod, _raw(c), no, no],
        [],
        [],
    )


def cat_bf16_16(lo8, hi8):
    """Concatenate two 8-bf16 vectors into the 16-wide WMMA operand."""
    lo, hi = Vec(lo8), Vec(hi8)
    return Vec.from_elements([lo[i] for i in range(8)] + [hi[i] for i in range(8)], fx.BFloat16)


def permlane16_halves(x):
    """``(low16, high16)`` of ``x`` across ``lane ^ 16``. Both lanes see both halves."""
    bits = _raw(ArithValue(_raw(x)).bitcast(fx.Int32.ir_type))
    swapped = rocdl.permlane16_swap(_pair_type(), bits, bits, False, True)
    low = llvm.extractvalue(fx.Int32.ir_type, swapped, [0])
    high = llvm.extractvalue(fx.Int32.ir_type, swapped, [1])

    def as_f32(v):
        return fx.Float32(_raw(ArithValue(v).bitcast(fx.Float32.ir_type)))

    return as_f32(low), as_f32(high)


def pack_low16_bf16(vals8, grp0):
    """B operand for a 16-long K, high half zero.

    ``vals8[i]`` is K ``half*8+i`` on this lane (f32). Every lane observes both
    halves through ``permlane16``. Group 0 packs K ``0..15``; group 1 packs zeros,
    which is the upper 16 of the K=32 WMMA.
    """
    zero = fx.BFloat16(0.0)
    lows, highs = [], []
    for value in vals8:
        low, high = permlane16_halves(value)
        lows.append(low)
        highs.append(high)
    elems = []
    for src in (lows, highs):
        for value in src:
            kept = fx.BFloat16(grp0.select(_raw(fx.BFloat16(_raw(value))), _raw(zero)))
            elems.append(kept)
    return Vec.from_elements(elems, fx.BFloat16)


def pack_pair_bf16(vals_a, vals_b):
    """B operand for a K=32 made of two 16-long K: ``a`` in K ``0..15``, ``b`` in K ``16..31``.

    ``vals_a[i]`` and ``vals_b[i]`` are K ``half*8+i`` of their 16 on this lane (f32). One
    ``v_permlane16_swap`` per ``i`` gives group 0 both halves of ``a`` and group 1 both halves of
    ``b``.
    """
    lows, highs = [], []
    for a, b in zip(vals_a, vals_b):
        a_bits = _raw(ArithValue(_raw(a)).bitcast(fx.Int32.ir_type))
        b_bits = _raw(ArithValue(_raw(b)).bitcast(fx.Int32.ir_type))
        swapped = rocdl.permlane16_swap(_pair_type(), a_bits, b_bits, False, True)
        for out, idx in ((lows, 0), (highs, 1)):
            bits = llvm.extractvalue(fx.Int32.ir_type, swapped, [idx])
            out.append(fx.BFloat16(_raw(fx.Float32(_raw(ArithValue(bits).bitcast(fx.Float32.ir_type))))))
    return Vec.from_elements(lows + highs, fx.BFloat16)


def crossgrp_max(x):
    low, high = permlane16_halves(x)
    return fx.Float32(arith.MaxNumFOp(_raw(low), _raw(high)).result)


def crossgrp_sum(x):
    low, high = permlane16_halves(x)
    return fx.Float32(arith.AddFOp(_raw(low), _raw(high)).result)
