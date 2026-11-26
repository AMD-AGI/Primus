# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

#!/usr/bin/env python3


import os
from typing import Dict, List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import (
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.triton._autotune_pinning import pinned_or_full
from generative_recommenders.ops.triton.gfx950_pk_opsel import no_packed_fp32_launch_opts
from generative_recommenders.ops.triton import fp8_tm_alloc
from generative_recommenders.ops.triton import wgrad_ck
from generative_recommenders.ops.utils import (
    arm_print_all_ranks,
    is_sm100_plus,
    is_sm90,
    maybe_register_custom_op,
)

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef, rsqrt as libdevice_rsqrt
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import (
            fast_dividef,
            rsqrt as libdevice_rsqrt,
        )
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef, rsqrt as libdevice_rsqrt


# ROCm 10.2 port (T7): Triton 3.9 removed tl.make_block_ptr and the
# boundary_check / padding_option arguments of tl.load / tl.store. The five
# kernels below that used block pointers (shape (N, D), strides (stride, 1),
# offsets (start_row, 0), order (1, 0)) and triton_preproc_tail now write out
# explicit pointer + mask, op for op the IR Triton 3.6's RewriteTensorPointer
# pass lowered each block pointer to:
#   * at the make_block_ptr site, the int64 casts of N, D, the row stride and
#     start_row (_bp_i64);
#   * at each load / store, pointers base + (sext(start_row) + sext(arange))
#     * stride + sext(cols) (_bp_ptrs) and a mask of its own,
#     0 <= offset < extent in int64 per checked dim (_bp_mask: (0, 1),
#     _bp_row_mask: (0,)); padding_option="zero" is other=0.0.
# So the addresses, masks and padding are the block pointers', and on Triton
# 3.6 the ported kernels compile to the same machine code as before.
# Do NOT simplify this to the shorter rows64 * stride + cols form with the
# kernels' int32 row_mask & col_mask as the load mask: the load mask then is
# the same SSA value as the mask of the tl.where(row_mask & col_mask, ...)
# that feeds the sum of squares, and LLVM contracts that sum differently for
# inputs over 2 GB (no buffer ops). That moved rstd by 1 ulp on up to 9% of
# the rows on both Triton 3.6 and 3.9, which breaks the bitwise contracts
# keyed on these kernels (_WLN_FWD_V2_EMU, triton_preproc_tail == shipped
# chain at N = 2.33M / 2.78M x 512 bf16).
@triton.jit
def _bp_i64(v):
    """make_block_ptr's int64 cast of a shape / stride / offset argument."""
    return tl.cast(v, tl.int64)


@triton.jit
def _bp_ptrs(base, bp_row0, bp_stride, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """(pointers [BLOCK_N, BLOCK_D], row offsets [BLOCK_N, 1] int64) of the
    block pointer at (bp_row0, 0) with strides (bp_stride, 1); bp_row0 and
    bp_stride are _bp_i64 values. RewriteTensorPointer's op order."""
    p = tl.broadcast_to(base, [BLOCK_N, BLOCK_D])
    offs = tl.full([BLOCK_N], bp_row0, tl.int64) + tl.arange(0, BLOCK_N).to(tl.int64)
    offs = offs[:, None]
    p = p + tl.broadcast_to(offs * bp_stride, [BLOCK_N, BLOCK_D])
    cols = tl.arange(0, BLOCK_D).to(tl.int64)[None, :]
    p = p + tl.broadcast_to(cols, [BLOCK_N, BLOCK_D])
    return p, offs


@triton.jit
def _bp_row_mask(offs, bp_n_rows, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """boundary_check=(0,): the rows 0 <= offs < bp_n_rows (int64)."""
    return tl.broadcast_to((offs >= 0) & (offs < bp_n_rows), [BLOCK_N, BLOCK_D])


@triton.jit
def _bp_mask(offs, bp_n_rows, bp_n_cols, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """boundary_check=(0, 1): rows as _bp_row_mask and the columns
    0 <= col < bp_n_cols (int64)."""
    m = _bp_row_mask(offs, bp_n_rows, BLOCK_N, BLOCK_D)
    c = tl.arange(0, BLOCK_D).to(tl.int64)[None, :]
    return m & tl.broadcast_to((c >= 0) & (c < bp_n_cols), [BLOCK_N, BLOCK_D])


def _get_layer_norm_fwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm kernels."""
    configs = []
    block_ns = [4, 8, 16] if is_sm100_plus() else [1, 2, 4, 8]
    for BLOCK_N in block_ns:
        for num_warps in [1, 2, 4, 8]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


def _bwd_pre_hook(nargs):
    nargs["DW"].zero_()
    if "DB" in nargs:
        nargs["DB"].zero_()


def _get_norm_bwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm kernels."""
    configs = []
    if is_sm100_plus():
        block_ns = [8, 16]
        num_shards_list = [8, 16]
        num_warps_list = [2, 4]
    elif is_sm90():
        block_ns = [2, 4]
        num_shards_list = [8]
        num_warps_list = [2, 4]
    else:
        block_ns = [1, 2]
        num_shards_list = [8]
        num_warps_list = [2, 4]
    for BLOCK_N in block_ns:
        for num_warps in num_warps_list:
            for num_shards in num_shards_list:
                configs.append(
                    triton.Config(
                        {"BLOCK_N": BLOCK_N, "SHARDS_PER_SM": num_shards},
                        num_warps=num_warps,
                    )
                )
    return configs


@triton_autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _layer_norm_fwd(
    X,
    Y,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # X / Y block pointers (shape (N, D), strides (stride, 1), offsets
    # (start_row, 0)) as pointer + mask, see _bp_ptrs (ROCm 10.2 port, T7).
    bp_n = _bp_i64(N)
    bp_d = _bp_i64(D)
    bp_sx = _bp_i64(stride_x)
    bp_row0 = _bp_i64(start_row)
    bp_sy = _bp_i64(stride_y)

    x_ptrs, x_offs = _bp_ptrs(X, bp_row0, bp_sx, BLOCK_N, BLOCK_D)
    x_block = tl.load(
        x_ptrs, mask=_bp_mask(x_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
    ).to(tl.float32)

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd

    y_out = y.to(Y.dtype.element_ty)
    y_ptrs, y_offs = _bp_ptrs(Y, bp_row0, bp_sy, BLOCK_N, BLOCK_D)
    tl.store(y_ptrs, y_out, mask=_bp_mask(y_offs, bp_n, bp_d, BLOCK_N, BLOCK_D))


@triton_autotune(
    configs=_get_layer_norm_fwd_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _weighted_layer_norm_fwd(
    X,
    Y,
    Y_FP8,
    W,
    B,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    stride_y_fp8,
    Y_INV_SCALE,
    # F2: optional residual folded into this kernel's epilogue, so that
    # `LN(h) + r` costs one pass instead of a full [M, D] read-modify-write
    # against HBM immediately after the norm has just written that same tensor.
    # Mirrors the machinery _weighted_layer_norm_bwd_dx already carries.
    RESIDUAL,
    stride_residual,
    IS_SWISH: tl.constexpr,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    OUTPUT_BOTH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
):
    # Get the block ID and calculate starting row
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Load weight and bias once (shared across all rows in this block)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    # X / Y / Y_FP8 / RESIDUAL block pointers (shape (N, D), strides
    # (stride, 1), offsets (start_row, 0)) as pointer + mask, see _bp_ptrs
    # (ROCm 10.2 port, T7).
    bp_n = _bp_i64(N)
    bp_d = _bp_i64(D)
    bp_sx = _bp_i64(stride_x)
    bp_row0 = _bp_i64(start_row)
    bp_sy = _bp_i64(stride_y)
    if OUTPUT_BOTH:
        bp_sy8 = _bp_i64(stride_y_fp8)

    x_ptrs, x_offs = _bp_ptrs(X, bp_row0, bp_sx, BLOCK_N, BLOCK_D)
    x_block = tl.load(
        x_ptrs, mask=_bp_mask(x_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
    ).to(tl.float32)

    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)

    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = libdevice_rsqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]

    if IS_SWISH:
        y = tl.sigmoid(y) * x_block

    if ADD_RESIDUAL:
        # ONE insertion point covers ALL THREE store sites below, because every
        # one of them derives from this same `y`. Patching the three stores
        # separately would triplicate this and get the OUTPUT_FP8 case wrong:
        # there `y` is rescaled by Y_INV_SCALE *after* this point, and the
        # residual belongs before that scaling (the fp8 tensor is a scaled copy
        # of the true output, which is LN(x) + r).
        #
        # ROUND BEFORE ADDING. The shipped path this replaces stores the norm
        # output at Y's dtype and only then adds the residual to it, so
        # rounding here reproduces that arithmetic BIT FOR BIT rather than
        # merely closely. _weighted_layer_norm_bwd_dx:487 already answers the
        # identical question the identical way. Adding in fp32 instead would be
        # marginally more accurate and would turn a free fusion into one owing
        # a convergence arm.
        bp_sr = _bp_i64(stride_residual)
        y_rt = y.to(Y.dtype.element_ty).to(tl.float32)
        r_ptrs, r_offs = _bp_ptrs(RESIDUAL, bp_row0, bp_sr, BLOCK_N, BLOCK_D)
        y = y_rt + tl.load(
            r_ptrs,
            mask=_bp_mask(r_offs, bp_n, bp_d, BLOCK_N, BLOCK_D),
            other=0.0,
        ).to(tl.float32)

    if OUTPUT_BOTH:
        y_out = y.to(Y.dtype.element_ty)
        y_ptrs, y_offs = _bp_ptrs(Y, bp_row0, bp_sy, BLOCK_N, BLOCK_D)
        tl.store(y_ptrs, y_out, mask=_bp_mask(y_offs, bp_n, bp_d, BLOCK_N, BLOCK_D))
        y_fp8 = y * Y_INV_SCALE
        y8_out = y_fp8.to(Y_FP8.dtype.element_ty)
        y8_ptrs, y8_offs = _bp_ptrs(Y_FP8, bp_row0, bp_sy8, BLOCK_N, BLOCK_D)
        tl.store(
            y8_ptrs, y8_out, mask=_bp_mask(y8_offs, bp_n, bp_d, BLOCK_N, BLOCK_D)
        )
    else:
        if OUTPUT_FP8:
            # Plan 08 Phi1: the normalized value is already resident in fp32.
            # Scale before the typed E4M3 store so no standalone cast pass is
            # needed by the downstream scaled GEMM.
            y = y * Y_INV_SCALE

        y_out = y.to(Y.dtype.element_ty)
        y_ptrs, y_offs = _bp_ptrs(Y, bp_row0, bp_sy, BLOCK_N, BLOCK_D)
        tl.store(y_ptrs, y_out, mask=_bp_mask(y_offs, bp_n, bp_d, BLOCK_N, BLOCK_D))


@triton.jit
def _wln_v_exp_f32(x):
    return tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _wln_v_rcp_f32(x):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _wln_v_mul_f32(a, b):
    """a * b as an opaque v_mul_f32, so it can never be contracted into an
    fma with a following add."""
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _wln_sigmoid_ieee_fast(u):
    """Same as triton_hstu_linear._sigmoid_ieee_fast (copied, not imported, to
    keep this module out of the hstu_linear import graph): bit-identical to
    tl.sigmoid(u) on gfx950 whenever d = 1 + exp(-u) is in [1, 2^126].
    Returns (sigmoid, d); the caller falls back to tl.sigmoid otherwise."""
    e = _wln_v_exp_f32(-u * 1.4426950408889634)
    d = 1.0 + e
    r = _wln_v_rcp_f32(d)
    t = tl.fma(-d, r, 1.0)
    r = tl.fma(t, r, r)
    t = tl.fma(-d, r, 1.0)
    q = tl.fma(t, r, r)
    t = tl.fma(-d, q, 1.0)
    q = tl.fma(t, r, q)
    return q, d


@triton.jit
def _wln_quad(v, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Split [BLOCK_N, BLOCK_D] into the four column phases e0..e3 of
    [BLOCK_N, BLOCK_D // 4]: element 4*l + j of a row is e_j[l]."""
    v4 = tl.reshape(v, [BLOCK_N, BLOCK_D // 4, 2, 2])
    a, bb = tl.split(v4)  # a = (e0, e2), bb = (e1, e3)
    e0, e2 = tl.split(a)
    e1, e3 = tl.split(bb)
    return e0, e1, e2, e3


@triton.jit
def _wln_oct(v, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Split [BLOCK_N, BLOCK_D] into the eight column phases e0..e7 of
    [BLOCK_N, BLOCK_D // 8]: element 8*l + j of a row is e_j[l]."""
    v8 = tl.reshape(v, [BLOCK_N, BLOCK_D // 8, 2, 2, 2])
    a, bb = tl.split(v8)  # a = (e0, e2, e4, e6), bb = (e1, e3, e5, e7)
    a0, a1 = tl.split(a)  # a0 = (e0, e4), a1 = (e2, e6)
    b0, b1 = tl.split(bb)  # b0 = (e1, e5), b1 = (e3, e7)
    e0, e4 = tl.split(a0)
    e2, e6 = tl.split(a1)
    e1, e5 = tl.split(b0)
    e3, e7 = tl.split(b1)
    return e0, e1, e2, e3, e4, e5, e6, e7


@triton.jit
def _wln_lane_tree64(s, BLOCK_N: tl.constexpr):
    """Row sum of [BLOCK_N, 64] per-lane partials in the order of the wave64
    DPP tl.sum of the (BLOCK_N=1, num_warps=1) config: row_shr 8, 4, 2, 1
    (lane bits 3, 2, 1, 0) inside each 16-lane row, then row_bcast15 (bit 4),
    then row_bcast31 (bit 5). Each step sums a size-2 axis, i.e. is exactly
    one fp32 add, so the result does not depend on the tile's register
    layout."""
    s6 = tl.reshape(s, [BLOCK_N, 2, 2, 2, 2, 2, 2])  # lane bits b5 .. b0
    s5 = tl.sum(s6, axis=3)  # b3 -> [N, b5, b4, b2, b1, b0]
    s4 = tl.sum(s5, axis=3)  # b2 -> [N, b5, b4, b1, b0]
    s3 = tl.sum(s4, axis=3)  # b1 -> [N, b5, b4, b0]
    s2 = tl.sum(s3, axis=3)  # b0 -> [N, b5, b4]
    s1 = tl.sum(s2, axis=2)  # b4 -> [N, b5]
    return tl.sum(s1, axis=1)  # b5


@triton.jit
def _wln_lane_tree32(s, BLOCK_N: tl.constexpr):
    """Row sum of [BLOCK_N, 32] per-lane partials in the order of the
    half-wave tl.sum of the 2-rows-per-wave configs: ds_swizzle SWAP16 (lane
    bit 4), DPP row_shr/shl 8 (bit 3), 4 (bit 2), quad_perm xor 2 (bit 1),
    xor 1 (bit 0). One fp32 add per step."""
    s5 = tl.reshape(s, [BLOCK_N, 2, 2, 2, 2, 2])  # lane bits b4 .. b0
    s4 = tl.sum(s5, axis=1)  # b4
    s3 = tl.sum(s4, axis=1)  # b3
    s2 = tl.sum(s3, axis=1)  # b2
    s1 = tl.sum(s2, axis=1)  # b1
    return tl.sum(s1, axis=1)  # b0


@triton.jit
def _wln_row_moments(
    x_block,
    mask_2d,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EMU: tl.constexpr,
):
    """(mean, masked x - mean, var) in the exact fp32 operation order of one
    class of old-kernel configs (see _weighted_layer_norm_fwd_v2).
    EMU 3 / 4 (ROCm 10.2 port, Triton 3.9): Triton 3.9 reduces the elements
    a lane holds as a pairwise tree of 2-wide vector adds instead of 3.6's
    sequential chain, and LLVM contracts the squares into v_pk_fma pairs; the
    cross-lane DPP / permlane trees are unchanged (same lane-bit order)."""
    if EMU == 0:
        e0, e1, e2, e3 = _wln_quad(x_block, BLOCK_N, BLOCK_D)
        mean = _wln_lane_tree64(((e0 + e1) + e2) + e3, BLOCK_N) / D
    elif EMU == 3:
        e0, e1, e2, e3 = _wln_quad(x_block, BLOCK_N, BLOCK_D)
        mean = _wln_lane_tree64((e0 + e2) + (e1 + e3), BLOCK_N) / D
    elif EMU == 4:
        e0, e1, e2, e3, e4, e5, e6, e7 = _wln_oct(x_block, BLOCK_N, BLOCK_D)
        m8 = ((e0 + e2) + (e4 + e6)) + ((e1 + e3) + (e5 + e7))
        mean = _wln_lane_tree32(m8, BLOCK_N) / D
    else:
        e0, e1, e2, e3, e4, e5, e6, e7 = _wln_oct(x_block, BLOCK_N, BLOCK_D)
        m8 = ((((((e0 + e1) + e2) + e3) + e4) + e5) + e6) + e7
        mean = _wln_lane_tree32(m8, BLOCK_N) / D
    x_mean = x_block - tl.expand_dims(mean, 1)
    x_mean = tl.where(mask_2d, x_mean, 0.0)
    if EMU == 0:
        q0, q1, q2, q3 = _wln_quad(x_mean, BLOCK_N, BLOCK_D)
        p = tl.fma(q3, q3, tl.fma(q2, q2, tl.fma(q0, q0, q1 * q1)))
        var = _wln_lane_tree64(p, BLOCK_N) / D
    elif EMU == 3:
        q0, q1, q2, q3 = _wln_quad(x_mean, BLOCK_N, BLOCK_D)
        # the addend squares stay rounded products (an fma operand is never
        # contracted), the outer add joins two fma results
        p = tl.fma(q0, q0, q2 * q2) + tl.fma(q1, q1, q3 * q3)
        var = _wln_lane_tree64(p, BLOCK_N) / D
    elif EMU == 4:
        q0, q1, q2, q3, q4, q5, q6, q7 = _wln_oct(x_mean, BLOCK_N, BLOCK_D)
        p = (tl.fma(q0, q0, q2 * q2) + tl.fma(q4, q4, q6 * q6)) + (
            tl.fma(q1, q1, q3 * q3) + tl.fma(q5, q5, q7 * q7)
        )
        var = _wln_lane_tree32(p, BLOCK_N) / D
    else:
        q0, q1, q2, q3, q4, q5, q6, q7 = _wln_oct(x_mean, BLOCK_N, BLOCK_D)
        p = tl.fma(q2, q2, tl.fma(q0, q0, q1 * q1))
        p = tl.fma(q5, q5, tl.fma(q4, q4, tl.fma(q3, q3, p)))
        if EMU == 1:
            p = tl.fma(q7, q7, tl.fma(q6, q6, p))
        else:
            # The squares must stay rounded products: written as q6 * q6 the
            # compiler is free to contract `p + q6 * q6` into an fma (it does,
            # depending on the surrounding code), which changes the bits.
            p = (p + _wln_v_mul_f32(q6, q6)) + _wln_v_mul_f32(q7, q7)
        var = _wln_lane_tree32(p, BLOCK_N) / D
    return mean, x_mean, var


@triton.jit
def _weighted_layer_norm_fwd_v2(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    IS_SWISH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EMU: tl.constexpr,
):
    """AMD (gfx950) v2 of _weighted_layer_norm_fwd for D == BLOCK_D == 256,
    TRAINING, COMPUTE_MEAN_AND_RSTD, no fp8 output / residual.

    The old kernel's bits depend on its autotune pick: the 16 configs fall
    into 8 classes with distinct y/mean/rstd bits, and at the preprocessor's
    swish LinearNorm the pick varies between cold starts ((1,1), (8,4) and
    (2,1) observed). v2 reproduces the bits of one class, selected by EMU:
      EMU=0: (BLOCK_N, num_warps) in {(1,1), (2,2), (4,4), (8,8)}: one row
             per wave, 4 elements per lane, 64-lane DPP tree;
      EMU=1: (2,1): two rows per wave, 8 elements per lane, 32-lane tree,
             fully contracted variance chain;
      EMU=2: (8,4), (4,2): as EMU=1 but the last two squares of the
             variance chain are rounded and added (the old ISA uses v_pk_mul
             for them).
    The classes above are the old kernel built by Triton 3.6. Built by Triton
    3.9 (ROCm 10.2 port) the per-lane order changes (_wln_row_moments):
      EMU=3: {(1,1), (2,2), (4,4), (8,8)}: 4 elements per lane, pairwise;
      EMU=4: {(2,1), (4,2), (8,4)}: 8 elements per lane, pairwise.
    _WLN_FWD_V2_EMU picks the table for the running Triton;
    note_weighted_layer_norm_swish_fwd_old maps the old autotuner's pick to
    its mode (the tables are verified offline; there is no runtime check).
    The tile layout here is whatever Triton picks for [BLOCK_N, 256] (16 B
    loads); the old per-lane and cross-lane fp32 operation order is written
    out explicitly (_wln_row_moments), so it does not depend on the layout.
    Other changes: persistent grid; exact fast sigmoid
    (_wln_sigmoid_ieee_fast, tile-uniform fallback to tl.sigmoid when any
    d > 2^126 or NaN); y * w + b written as the fma the old kernel contracts
    it to. Only launched after the old autotuner's pick is known to be in
    the emulated class (maybe_weighted_layer_norm_swish_fwd_v2).
    Gate: DLRM_HSTU_FP8_WLN_FWD_V2 (gin default on, yambda_5b.gin item (20);
    absent or =0 = the old kernel)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    cols = tl.arange(0, BLOCK_D)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    for blk in tl.range(pid, num_blocks, nprog, num_stages=NUM_STAGES):
        rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mask_2d = row_mask[:, None]
        r64 = rows.to(tl.int64)
        x_block = tl.load(
            X + r64[:, None] * stride_x + cols[None, :], mask=mask_2d, other=0.0
        ).to(tl.float32)
        mean, x_mean, var = _wln_row_moments(
            x_block, mask_2d, D, BLOCK_N, BLOCK_D, EMU
        )
        tl.store(Mean + rows, mean, row_mask)
        rstd = libdevice_rsqrt(var + eps)
        tl.store(Rstd + rows, rstd, row_mask)
        y = x_mean * tl.expand_dims(rstd, 1)
        y = tl.fma(y, w[None, :], b[None, :])
        if IS_SWISH:
            s, d_sig = _wln_sigmoid_ieee_fast(y)
            # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN
            if (
                tl.max(tl.max(d_sig.to(tl.uint32, bitcast=True), axis=1), axis=0)
                > 0x7E800000
            ):
                s = tl.sigmoid(y)
            y = s * x_block
        tl.store(
            Y + r64[:, None] * stride_y + cols[None, :],
            y.to(Y.dtype.element_ty),
            mask=mask_2d,
        )


@triton.jit
def _layer_norm_bwd_dx(
    DX,
    DY,
    X,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / D
    c2 = tl.sum(dy, axis=0) / D
    dx = (dy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)


@triton_autotune(
    configs=pinned_or_full(
        [
            triton.Config({"BLOCK_N": 1}, num_warps=1),  # bs=32 winner
            triton.Config({"BLOCK_N": 8}, num_warps=1),  # bs=1024 winner
        ],
        _get_layer_norm_fwd_configs,
    ),
    key=["BLOCK_D"],
)
@triton.jit
def _weighted_layer_norm_bwd_dx(
    DX,
    DY,
    RESIDUAL,
    DW,
    DB,
    DXSUM,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_residual,
    stride_x,
    D,
    eps,
    IS_SWISH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    SUM_DX: tl.constexpr,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # SUM_DX additionally emits sum(dx, axis=0) into DXSUM, using the same
    # [tile_num, D] partial-buffer scheme the kernel already runs for dw/db.
    # It exists so the preceding nn.Linear's bias gradient -- which ATen
    # computes as a separate aten::sum over this very dx, 3.9608 ms/rank/step
    # across the two preprocessor widths -- can be read off here instead, while
    # dx is still in registers. Zero extra global traffic on the dx side.
    #
    # It is a tl.constexpr, not a runtime flag, so with SUM_DX=False Triton
    # eliminates the accumulator entirely and the control path is the byte-
    # identical kernel it was before. That is what makes the A/B honest.
    #
    # NUMERICS: this sums the fp32 dx, whereas aten::sum reduces the bf16 dx
    # that was stored. Slightly more accurate, not bit-identical -- which is
    # why this needs the convergence arm, not just a timing arm.
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    if pid < num_blocks % tile_num:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum = tl.zeros([BLOCK_D], dtype=tl.float32)

    start_block = pid

    for idx in range(blocks_per_tile):
        current_block = start_block + idx * tile_num
        start_row = current_block * BLOCK_N

        # X / DX / DY / RESIDUAL block pointers (shape (N, D), strides
        # (stride, 1), offsets (start_row, 0)) as pointer + mask, see _bp_ptrs
        # (ROCm 10.2 port, T7).
        bp_n = _bp_i64(N)
        bp_d = _bp_i64(D)
        bp_sx = _bp_i64(stride_x)
        bp_row0 = _bp_i64(start_row)
        bp_sdx = _bp_i64(stride_dx)
        bp_sdy = _bp_i64(stride_dy)
        bp_sr = _bp_i64(stride_residual)

        # Load data blocks
        x_ptrs, x_offs = _bp_ptrs(X, bp_row0, bp_sx, BLOCK_N, BLOCK_D)
        x_block = tl.load(
            x_ptrs, mask=_bp_mask(x_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
        ).to(tl.float32)
        dy_ptrs, dy_offs = _bp_ptrs(DY, bp_row0, bp_sdy, BLOCK_N, BLOCK_D)
        dy_block = tl.load(
            dy_ptrs, mask=_bp_mask(dy_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
        ).to(tl.float32)

        # Load mean and rstd for all rows in this block
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

        # Expand dimensions for broadcasting
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)

        xhat = (x_block - mean) * rstd

        xhat = tl.where(row_mask[:, None] & col_mask[None, :], xhat, 0.0)
        wdy = w[None, :] * dy_block
        wdy = tl.where(row_mask[:, None] & col_mask[None, :], wdy, 0.0)

        # Compute dx
        if IS_SWISH:
            b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
            sigmoid_layer_norm = tl.sigmoid(xhat * w[None, :] + b[None, :])
            sigmoid_layer_norm = tl.where(
                row_mask[:, None] & col_mask[None, :], sigmoid_layer_norm, 0.0
            )

            sigmoid_deriv = sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = wdy * x_block * sigmoid_deriv
            x_ = tl.where(row_mask[:, None] & col_mask[None, :], x_, 0.0)

            c1 = tl.sum(xhat * x_, axis=1) / D
            c2 = tl.sum(x_, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (x_ - (xhat * c1 + c2)) * rstd

            dx = dy_block * sigmoid_layer_norm + dx
            if ADD_RESIDUAL:
                dx_rt = dx.to(DX.dtype.element_ty).to(tl.float32)
                r_ptrs, r_offs = _bp_ptrs(RESIDUAL, bp_row0, bp_sr, BLOCK_N, BLOCK_D)
                dx = dx_rt + tl.load(
                    r_ptrs,
                    mask=_bp_mask(r_offs, bp_n, bp_d, BLOCK_N, BLOCK_D),
                    other=0.0,
                ).to(tl.float32)
            # Write dx
            dx_out = dx.to(DX.dtype.element_ty)
            dx_ptrs, dx_offs = _bp_ptrs(DX, bp_row0, bp_sdx, BLOCK_N, BLOCK_D)
            tl.store(
                dx_ptrs, dx_out, mask=_bp_mask(dx_offs, bp_n, bp_d, BLOCK_N, BLOCK_D)
            )
            partial_dw = tl.sum(dy_block * x_block * xhat * sigmoid_deriv, axis=0)
            partial_db = tl.sum(dy_block * x_block * sigmoid_deriv, axis=0)
        else:
            c1 = tl.sum(xhat * wdy, axis=1) / D
            c2 = tl.sum(wdy, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (wdy - (xhat * c1 + c2)) * rstd
            if ADD_RESIDUAL:
                dx_rt = dx.to(DX.dtype.element_ty).to(tl.float32)
                r_ptrs, r_offs = _bp_ptrs(RESIDUAL, bp_row0, bp_sr, BLOCK_N, BLOCK_D)
                dx = dx_rt + tl.load(
                    r_ptrs,
                    mask=_bp_mask(r_offs, bp_n, bp_d, BLOCK_N, BLOCK_D),
                    other=0.0,
                ).to(tl.float32)
            # Write dx
            dx_out = dx.to(DX.dtype.element_ty)
            dx_ptrs, dx_offs = _bp_ptrs(DX, bp_row0, bp_sdx, BLOCK_N, BLOCK_D)
            tl.store(
                dx_ptrs, dx_out, mask=_bp_mask(dx_offs, bp_n, bp_d, BLOCK_N, BLOCK_D)
            )
            partial_dw = tl.sum(dy_block * xhat, axis=0)
            partial_db = tl.sum(dy_block, axis=0)

        # Accumulate partial sums in shared memory
        acc_dw += partial_dw
        acc_db += partial_db
        if SUM_DX:
            # dx is defined by both arms of the IS_SWISH branch above and is
            # still resident. Rows >= N need no mask for the same reason
            # partial_db does not: every term feeding them is either padded to
            # zero on load or explicitly row-masked, so dx is zero there.
            acc_dxsum += tl.sum(dx, axis=0)

    # Store accumulated sums back to global memory
    dw_ptrs = DW + pid.to(tl.int64) * D + cols
    db_ptrs = DB + pid.to(tl.int64) * D + cols
    tl.store(dw_ptrs, acc_dw, mask=col_mask)
    tl.store(db_ptrs, acc_db, mask=col_mask)
    if SUM_DX:
        dxsum_ptrs = DXSUM + pid.to(tl.int64) * D + cols
        tl.store(dxsum_ptrs, acc_dxsum, mask=col_mask)


@triton.jit
def _mask_bits_to_cols(kb, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """DLRM_HSTU_FP8_DROPOUT_MASK_BITS: packed mask bytes [BLOCK_N, BLOCK_D // 8]
    (int32) -> 0/1 [BLOCK_N, BLOCK_D], column 8 * j + k <- bit k of byte j.
    Each join adds an in-register minor dim (flattened index 4 * s0 + 2 * s1 +
    s2 over the inner, middle, outer join), so a byte's 8 columns stay in the
    thread that loaded it: one byte load per 8 columns, no layout change."""
    b0 = kb & 1
    b1 = (kb >> 1) & 1
    b2 = (kb >> 2) & 1
    b3 = (kb >> 3) & 1
    b4 = (kb >> 4) & 1
    b5 = (kb >> 5) & 1
    b6 = (kb >> 6) & 1
    b7 = (kb >> 7) & 1
    return tl.reshape(
        tl.join(
            tl.join(tl.join(b0, b4), tl.join(b2, b6)),
            tl.join(tl.join(b1, b5), tl.join(b3, b7)),
        ),
        (BLOCK_N, BLOCK_D),
    )


@triton.jit
def _weighted_layer_norm_bwd_dx_v2(
    DX,
    DY,
    RESIDUAL,
    DW,
    DB,
    DXSUM,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_residual,
    stride_x,
    D,
    eps,
    IS_SWISH: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    SUM_DX: tl.constexpr,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CONTIG: tl.constexpr,
    EVEN_D: tl.constexpr,
    FAST_SIGMOID: tl.constexpr = False,
    DMASK=None,
    stride_mask=0,
    mask_scale=1.0,
    MASK_SCALE: tl.constexpr = False,
    IS_POW2_D: tl.constexpr = False,
    MASK_BITS: tl.constexpr = False,
):
    """Pipelined variant of _weighted_layer_norm_bwd_dx (same math per element).

    Differences from the v1 kernel, none of which change the arithmetic:
      * raw-pointer loads/stores instead of tl.make_block_ptr (v1 also uses
        raw pointers since the Triton 3.9 port), and
        tl.range(num_stages=NUM_STAGES) so the loads of block i+1 are issued
        while block i is still being reduced (v1's plain range() loop is not
        software-pipelined on gfx950);
      * B is loaded once per program rather than once per row block;
      * CONTIG selects contiguous row-block ranges per program instead of the
        v1 strided walk; it only changes which program's dw/db/dxsum partial
        a row lands in (fp32 reduction order), never dx;
      * EVEN_D (BLOCK_D == D) drops the column masks, which are all-true.
    FAST_SIGMOID (swish only) is the one arithmetic option: sigmoid via
    fast_dividef(1, 1 + exp(-z)) instead of tl.sigmoid; see
    _wln_bwd_v2_fast_sigmoid for its numerics.
    The [tile_num, D] dw/db/dxsum partial-buffer contract is unchanged, so the
    same _layer_norm_bwd_dwdb / _layer_norm_bwd_dwdbdx finalize consumes it.
    Gated by DLRM_HSTU_FP8_WLN_BWD_V2 (gin default on, yambda_5b.gin item (14); absent or "0" = v1).
    MASK_SCALE folds the following dropout's backward into the epilogue:
    dx <- (float(DMASK) * float(bf16(dx))) * mask_scale, the exact fp32 op
    sequence of aten's masked_scale_kernel (see ops/input_dropout_fold.py).
    MASK_BITS (DLRM_HSTU_FP8_DROPOUT_MASK_BITS): DMASK is the packed mask,
    uint8 [N, D / 8] (D % 8 == 0), bit k of byte j of a row = column 8 * j + k
    (_mask_bits_to_cols); the bit is the same 0.0 / 1.0 factor, so dx, dw, db
    are bitwise the bool-mask fold's wherever the compiler keeps the reduction
    layouts (_MASK_BITS_FOLD_CFG; wln_bwd_mscale_supported).
    IS_POW2_D (DLRM_HSTU_FP8_WLN_BWD_POW2, only True when EVEN_D): c1/c2 are
    sum * (1.0 / BLOCK_D) instead of sum / D -- bitwise equal (exact 2^-k
    reciprocal), two precise fp32 divides per row become multiplies; see
    _wln_bwd_pow2_enabled.
    """
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    extra = num_blocks % tile_num
    if CONTIG:
        start_block = pid * blocks_per_tile + tl.minimum(pid, extra)
        step = 1
    else:
        start_block = pid
        step = tile_num
    if pid < extra:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    if IS_SWISH:
        b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum = tl.zeros([BLOCK_D], dtype=tl.float32)

    for idx in tl.range(0, blocks_per_tile, num_stages=NUM_STAGES):
        current_block = start_block + idx * step
        start_row = current_block * BLOCK_N
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        rows64 = rows.to(tl.int64)
        if EVEN_D:
            m2 = row_mask[:, None] & (cols < BLOCK_D)[None, :]
        else:
            m2 = row_mask[:, None] & col_mask[None, :]
        x_block = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        dy_block = tl.load(
            DY + rows64[:, None] * stride_dy + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)

        # Masked rows/cols load x=dy=mean=rstd=0, so every term below is
        # already zero there; the tl.where's are kept on the !EVEN_D path to
        # mirror v1 exactly where padding columns exist.
        xhat = (x_block - mean) * rstd
        xhat = xhat if EVEN_D else tl.where(m2, xhat, 0.0)
        wdy = w[None, :] * dy_block
        wdy = wdy if EVEN_D else tl.where(m2, wdy, 0.0)

        if IS_SWISH:
            if FAST_SIGMOID:
                sigmoid_layer_norm = fast_dividef(
                    1.0, 1.0 + tl.exp(-(xhat * w[None, :] + b[None, :]))
                )
            else:
                sigmoid_layer_norm = tl.sigmoid(xhat * w[None, :] + b[None, :])
            sigmoid_layer_norm = (
                sigmoid_layer_norm
                if EVEN_D
                else tl.where(m2, sigmoid_layer_norm, 0.0)
            )
            sigmoid_deriv = sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = wdy * x_block * sigmoid_deriv
            x_ = x_ if EVEN_D else tl.where(m2, x_, 0.0)
            if IS_POW2_D:
                c1 = tl.sum(xhat * x_, axis=1) * (1.0 / BLOCK_D)
                c2 = tl.sum(x_, axis=1) * (1.0 / BLOCK_D)
            else:
                c1 = tl.sum(xhat * x_, axis=1) / D
                c2 = tl.sum(x_, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (x_ - (xhat * c1 + c2)) * rstd
            dx = dy_block * sigmoid_layer_norm + dx
            partial_dw = tl.sum(dy_block * x_block * xhat * sigmoid_deriv, axis=0)
            partial_db = tl.sum(dy_block * x_block * sigmoid_deriv, axis=0)
        else:
            if IS_POW2_D:
                c1 = tl.sum(xhat * wdy, axis=1) * (1.0 / BLOCK_D)
                c2 = tl.sum(wdy, axis=1) * (1.0 / BLOCK_D)
            else:
                c1 = tl.sum(xhat * wdy, axis=1) / D
                c2 = tl.sum(wdy, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (wdy - (xhat * c1 + c2)) * rstd
            partial_dw = tl.sum(dy_block * xhat, axis=0)
            partial_db = tl.sum(dy_block, axis=0)
        if ADD_RESIDUAL:
            dx = dx.to(DX.dtype.element_ty).to(tl.float32) + tl.load(
                RESIDUAL + rows64[:, None] * stride_residual + cols[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        if MASK_SCALE:
            # aten masked_scale: (float)mask * grad * scale in fp32, where grad
            # is the bf16 tensor the unfolded path would have stored.
            if MASK_BITS:
                bcols = tl.arange(0, BLOCK_D // 8)
                if EVEN_D:
                    mb = row_mask[:, None] & (bcols < BLOCK_D // 8)[None, :]
                else:
                    mb = row_mask[:, None] & (bcols * 8 < D)[None, :]
                keep = _mask_bits_to_cols(
                    tl.load(
                        DMASK + rows64[:, None] * stride_mask + bcols[None, :],
                        mask=mb,
                        other=0,
                    ).to(tl.int32),
                    BLOCK_N,
                    BLOCK_D,
                ).to(tl.float32)
            else:
                keep = tl.load(
                    DMASK + rows64[:, None] * stride_mask + cols[None, :],
                    mask=m2,
                    other=0,
                ).to(tl.float32)
            dx = (keep * dx.to(DX.dtype.element_ty).to(tl.float32)) * mask_scale
        tl.store(
            DX + rows64[:, None] * stride_dx + cols[None, :],
            dx.to(DX.dtype.element_ty),
            mask=m2,
        )
        acc_dw += partial_dw
        acc_db += partial_db
        if SUM_DX:
            acc_dxsum += tl.sum(dx, axis=0)

    tl.store(DW + pid.to(tl.int64) * D + cols, acc_dw, mask=col_mask)
    tl.store(DB + pid.to(tl.int64) * D + cols, acc_db, mask=col_mask)
    if SUM_DX:
        tl.store(DXSUM + pid.to(tl.int64) * D + cols, acc_dxsum, mask=col_mask)


@triton.jit
def _weighted_layer_norm_fwd_v3(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    D: tl.constexpr,
    eps,
    stride_x,
    stride_y,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EMU: tl.constexpr,
):
    """_weighted_layer_norm_fwd_v2 (IS_SWISH, BLOCK_D == D) with D a
    constexpr: the two per-row `/ D` become multiplies by 2^-8, which is
    bit-identical to the IEEE division for every input (exact power-of-two
    scaling, same single rounding), and drops the v_div_scale / div_fmas /
    div_fixup sequences from the loop. Everything else is v2 verbatim.
    Gate: DLRM_HSTU_FP8_D256_V3 (gin default on in yambda_5b.gin; absent
    or =0 launches v2)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    cols = tl.arange(0, D)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    for blk in tl.range(pid, num_blocks, nprog, num_stages=NUM_STAGES):
        rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mask_2d = row_mask[:, None]
        r64 = rows.to(tl.int64)
        x_block = tl.load(
            X + r64[:, None] * stride_x + cols[None, :], mask=mask_2d, other=0.0
        ).to(tl.float32)
        mean, x_mean, var = _wln_row_moments(x_block, mask_2d, D, BLOCK_N, D, EMU)
        tl.store(Mean + rows, mean, row_mask)
        rstd = libdevice_rsqrt(var + eps)
        tl.store(Rstd + rows, rstd, row_mask)
        y = x_mean * tl.expand_dims(rstd, 1)
        y = tl.fma(y, w[None, :], b[None, :])
        s, d_sig = _wln_sigmoid_ieee_fast(y)
        if (
            tl.max(tl.max(d_sig.to(tl.uint32, bitcast=True), axis=1), axis=0)
            > 0x7E800000
        ):
            s = tl.sigmoid(y)
        y = s * x_block
        tl.store(
            Y + r64[:, None] * stride_y + cols[None, :],
            y.to(Y.dtype.element_ty),
            mask=mask_2d,
        )


@triton.jit
def _weighted_layer_norm_bwd_dx_v3(
    DX,
    DY,
    DW,
    DB,
    DXSUM,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    N,
    D: tl.constexpr,
    SUM_DX: tl.constexpr,
    FAST_SIGMOID: tl.constexpr,
):
    """_weighted_layer_norm_bwd_dx_v2 specialised to the config v2 runs for
    swish at D == BLOCK_D == 256 (BLOCK_N=1, NUM_STAGES=1, CONTIG=False,
    EVEN_D, no residual / dropout fold), with D a constexpr so the two
    per-row `/ D` are exact multiplies by 2^-8 (bit-identical to the IEEE
    division). Same row -> program map, same per-program accumulation order,
    same [tile_num, D] partials: dx, dw, db and dxsum are bit-identical to v2.
    Gate: DLRM_HSTU_FP8_D256_V3 (gin default on in yambda_5b.gin; absent
    or =0 launches v2)."""
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    blocks_per_tile = N // tile_num
    extra = N % tile_num
    if pid < extra:
        blocks_per_tile += 1

    cols = tl.arange(0, D)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)

    acc_dw = tl.zeros([D], dtype=tl.float32)
    acc_db = tl.zeros([D], dtype=tl.float32)
    acc_dxsum = tl.zeros([D], dtype=tl.float32)

    for idx in tl.range(0, blocks_per_tile, num_stages=1):
        rows = pid + idx * tile_num + tl.arange(0, 1)
        row_mask = rows < N
        rows64 = rows.to(tl.int64)
        m2 = row_mask[:, None] & (cols < D)[None, :]
        x_block = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        dy_block = tl.load(
            DY + rows64[:, None] * stride_dy + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)

        xhat = (x_block - mean) * rstd
        wdy = w[None, :] * dy_block
        if FAST_SIGMOID:
            sigmoid_layer_norm = fast_dividef(
                1.0, 1.0 + tl.exp(-(xhat * w[None, :] + b[None, :]))
            )
        else:
            sigmoid_layer_norm = tl.sigmoid(xhat * w[None, :] + b[None, :])
        sigmoid_deriv = sigmoid_layer_norm * (1 - sigmoid_layer_norm)
        x_ = wdy * x_block * sigmoid_deriv
        c1 = tl.sum(xhat * x_, axis=1) / D
        c2 = tl.sum(x_, axis=1) / D
        c1 = tl.expand_dims(c1, 1)
        c2 = tl.expand_dims(c2, 1)
        dx = (x_ - (xhat * c1 + c2)) * rstd
        dx = dy_block * sigmoid_layer_norm + dx
        partial_dw = tl.sum(dy_block * x_block * xhat * sigmoid_deriv, axis=0)
        partial_db = tl.sum(dy_block * x_block * sigmoid_deriv, axis=0)
        tl.store(
            DX + rows64[:, None] * stride_dx + cols[None, :],
            dx.to(DX.dtype.element_ty),
            mask=m2,
        )
        acc_dw += partial_dw
        acc_db += partial_db
        if SUM_DX:
            acc_dxsum += tl.sum(dx, axis=0)

    tl.store(DW + pid.to(tl.int64) * D + cols, acc_dw)
    tl.store(DB + pid.to(tl.int64) * D + cols, acc_db)
    if SUM_DX:
        tl.store(DXSUM + pid.to(tl.int64) * D + cols, acc_dxsum)


@triton.jit
def _layer_norm_bwd_dwdbdx_p1(
    DW,
    DB,
    DXSUM,
    ACC,
    N,
    D,
    NB: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Phase 1 of a bit-exact split of _layer_norm_bwd_dwdbdx at its pinned
    config (BLOCK_N = NB = 128, num_warps = 8). That kernel keeps an
    [NB, BLOCK_D] fp32 accumulator whose row j is the sequential chain
    ((0 + P[j]) + P[NB + j]) + P[2 NB + j] + ... over the [N, D] partials
    (masked rows add +0.0), then tl.sum's it over axis 0. At D=256 it gets
    BLOCK_D=4: 64 programs, each walking all N=8192 partial rows 16 B at a
    time (0.072 ms/call in trace for 25 MB). Here each chain is one lane
    doing the same adds in the same order, over NB x cdiv(D, BLOCK_C)
    programs; ACC[a, j, :] is accumulator row j of array a."""
    j = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < D
    a0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    a1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    a2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    for i in range(0, N, NB):
        m = cm & (i + j < N)
        offs = (i + j) * D + c
        a0 += tl.load(DW + offs, mask=m, other=0.0)
        a1 += tl.load(DB + offs, mask=m, other=0.0)
        a2 += tl.load(DXSUM + offs, mask=m, other=0.0)
    o = j * D + c
    tl.store(ACC + o, a0, mask=cm)
    tl.store(ACC + NB * D + o, a1, mask=cm)
    tl.store(ACC + 2 * NB * D + o, a2, mask=cm)


@triton.jit
def _layer_norm_bwd_dwdbdx_p2(
    ACC,
    FINAL_DW,
    FINAL_DB,
    FINAL_DXSUM,
    D,
    NB: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Phase 2: the [NB, BLOCK_D] accumulator tile of _layer_norm_bwd_dwdbdx
    loaded with the same address pattern (row stride D, BLOCK_D contiguous
    columns), hence the same blocked layout and tl.sum tree when launched at
    the pinned num_warps=8."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    rows = tl.arange(0, NB)
    mask = (rows[:, None] < NB) & (cols[None, :] < D)
    offs = rows[:, None] * D + cols[None, :]
    dw = tl.zeros((NB, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((NB, BLOCK_D), dtype=tl.float32)
    dxsum = tl.zeros((NB, BLOCK_D), dtype=tl.float32)
    dw += tl.load(ACC + offs, mask=mask, other=0.0)
    db += tl.load(ACC + NB * D + offs, mask=mask, other=0.0)
    dxsum += tl.load(ACC + 2 * NB * D + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    sum_dxsum = tl.sum(dxsum, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)
    tl.store(
        FINAL_DXSUM + cols,
        sum_dxsum.to(FINAL_DXSUM.dtype.element_ty),
        mask=cols < D,
    )


@triton.jit
def _weighted_layer_norm_bwd_dx_dxs512(
    DX,
    DY,
    DW,
    DB,
    DXSUM,
    X,
    W,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PF2: tl.constexpr = False,
):
    """Non-swish LayerNorm backward with the dx column-sum (SUM_DX) and no
    residual: the LinearNorm D=512 call of triton_weighted_layer_norm_bwd_dxsum.

    Same per-row arithmetic, expression for expression, as the non-swish
    SUM_DX branch of _weighted_layer_norm_bwd_dx_v2 (EVEN_D, strided row-block
    walk, same [tile_num, D] partial contract). Two differences:
      * software pipelining is written out by hand: the loads of the next row
        block are issued (into loop-carried registers) before the current
        block is reduced, and tl.range runs with num_stages=1. There is
        therefore ONE copy of the arithmetic, so every row gets the
        steady-state code. v2's num_stages=2 pipeliner peels the last
        iteration and the peeled copy contracts differently (a handful of
        rows per call differ from the steady-state bits by one bf16 ulp);
      * the launcher runs it on a larger grid (more waves per SIMD), which
        is what removes the latency/placement sensitivity of v2's sms*4
        one-wave programs at this site.
    Gated by DLRM_HSTU_FP8_WLN_DXSUM_V3 (gin default on in yambda_5b.gin; absent or "0" = v2).
    """
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    extra = num_blocks % tile_num
    start_block = pid
    step = tile_num
    if pid < extra:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    w = tl.load(W + cols).to(tl.float32)

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum = tl.zeros([BLOCK_D], dtype=tl.float32)

    # prologue: block 0 of this program (blocks_per_tile >= 1 by the grid cap)
    rows_n = start_block * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask_n = rows_n < N
    rows64_n = rows_n.to(tl.int64)
    m2_n = rmask_n[:, None] & (cols < BLOCK_D)[None, :]
    x_n = tl.load(X + rows64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
    dy_n = tl.load(DY + rows64_n[:, None] * stride_dy + cols[None, :], mask=m2_n, other=0.0)
    mean_n = tl.load(Mean + rows_n, rmask_n, other=0.0)
    rstd_n = tl.load(Rstd + rows_n, rmask_n, other=0.0)
    if PF2:
        rows_q = (start_block + step) * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask_q = (rows_q < N) & (1 < blocks_per_tile)
        rows64_q = rows_q.to(tl.int64)
        m2_q = rmask_q[:, None] & (cols < BLOCK_D)[None, :]
        x_q = tl.load(X + rows64_q[:, None] * stride_x + cols[None, :], mask=m2_q, other=0.0)
        dy_q = tl.load(DY + rows64_q[:, None] * stride_dy + cols[None, :], mask=m2_q, other=0.0)
        mean_q = tl.load(Mean + rows_q, rmask_q, other=0.0)
        rstd_q = tl.load(Rstd + rows_q, rmask_q, other=0.0)

    for idx in tl.range(0, blocks_per_tile, num_stages=1):
        rows64 = rows64_n
        m2 = m2_n
        x_block = x_n.to(tl.float32)
        dy_block = dy_n.to(tl.float32)
        mean = tl.expand_dims(mean_n, 1)
        rstd = tl.expand_dims(rstd_n, 1)
        if PF2:
            rows64_n = rows64_q
            m2_n = m2_q
            x_n = x_q
            dy_n = dy_q
            mean_n = mean_q
            rstd_n = rstd_q
            rows_q = (start_block + (idx + 2) * step) * BLOCK_N + tl.arange(0, BLOCK_N)
            rmask_q = (rows_q < N) & (idx + 2 < blocks_per_tile)
            rows64_q = rows_q.to(tl.int64)
            m2_q = rmask_q[:, None] & (cols < BLOCK_D)[None, :]
            x_q = tl.load(
                X + rows64_q[:, None] * stride_x + cols[None, :], mask=m2_q, other=0.0
            )
            dy_q = tl.load(
                DY + rows64_q[:, None] * stride_dy + cols[None, :], mask=m2_q, other=0.0
            )
            mean_q = tl.load(Mean + rows_q, rmask_q, other=0.0)
            rstd_q = tl.load(Rstd + rows_q, rmask_q, other=0.0)
        else:
            # issue the next block's loads before this block's math
            rows_n = (start_block + (idx + 1) * step) * BLOCK_N + tl.arange(0, BLOCK_N)
            rmask_n = (rows_n < N) & (idx + 1 < blocks_per_tile)
            rows64_n = rows_n.to(tl.int64)
            m2_n = rmask_n[:, None] & (cols < BLOCK_D)[None, :]
            x_n = tl.load(
                X + rows64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0
            )
            dy_n = tl.load(
                DY + rows64_n[:, None] * stride_dy + cols[None, :], mask=m2_n, other=0.0
            )
            mean_n = tl.load(Mean + rows_n, rmask_n, other=0.0)
            rstd_n = tl.load(Rstd + rows_n, rmask_n, other=0.0)

        xhat = (x_block - mean) * rstd
        wdy = w[None, :] * dy_block
        c1 = tl.sum(xhat * wdy, axis=1) / D
        c2 = tl.sum(wdy, axis=1) / D
        c1 = tl.expand_dims(c1, 1)
        c2 = tl.expand_dims(c2, 1)
        dx = (wdy - (xhat * c1 + c2)) * rstd
        partial_dw = tl.sum(dy_block * xhat, axis=0)
        partial_db = tl.sum(dy_block, axis=0)
        tl.store(
            DX + rows64[:, None] * stride_dx + cols[None, :],
            dx.to(DX.dtype.element_ty),
            mask=m2,
        )
        acc_dw += partial_dw
        acc_db += partial_db
        acc_dxsum += tl.sum(dx, axis=0)

    tl.store(DW + pid.to(tl.int64) * D + cols, acc_dw)
    tl.store(DB + pid.to(tl.int64) * D + cols, acc_db)
    tl.store(DXSUM + pid.to(tl.int64) * D + cols, acc_dxsum)


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    BLOCK_N_CHOICES = [32, 64, 128, 256]
    if is_sm100_plus():
        BLOCK_N_CHOICES = [128, 256, 512, 1024]
    for BLOCK_N in BLOCK_N_CHOICES:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=pinned_or_full(
        [triton.Config({"BLOCK_N": 128}, num_warps=8)],
        _get_bwd_dwdb_configs,
    ),
    key=["D"],
)
@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


@triton_autotune(
    configs=pinned_or_full(
        [triton.Config({"BLOCK_N": 128}, num_warps=8)],
        _get_bwd_dwdb_configs,
    ),
    key=["D"],
)
@triton.jit
def _layer_norm_bwd_dwdbdx(
    DW,
    DB,
    DXSUM,
    FINAL_DW,
    FINAL_DB,
    FINAL_DXSUM,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """F1 finalize: _layer_norm_bwd_dwdb plus the dx column-sum.

    This is a SEPARATE kernel rather than a flag on _layer_norm_bwd_dwdb, for
    two reasons. The control path stays byte-identical -- the shipped kernel is
    not recompiled, not re-autotuned, and not touched -- so an A/B against it is
    honest. And the distinct name is self-proving in a trace: seeing
    _layer_norm_bwd_dwdbdx in the kernel list IS the arming proof, with no env
    echo to trust. The rule here is that an arm must be shown armed at kernel
    level before any delta is quoted, and a renamed kernel satisfies it directly.

    The reduction is the same [tile_num, D] -> [D] shape the dw/db partials
    already use, so DXSUM costs one more strided read of an array the bwd_dx
    kernel has just written, not a new pass over the [N, D] activations.
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dxsum = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
        dxsum += tl.load(DXSUM + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    sum_dxsum = tl.sum(dxsum, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)
    tl.store(
        FINAL_DXSUM + cols,
        sum_dxsum.to(FINAL_DXSUM.dtype.element_ty),
        mask=cols < D,
    )


# DLRM_HSTU_FP8_WLN_FWD_V2 (gin default on, yambda_5b.gin item (20); absent or
# "0" = the old launch): the swish LinearNorm forward at
# D == 256 (ops/fused_linear_norm._norm_fwd, the preprocessor MLPs) runs
# _weighted_layer_norm_fwd_v2 instead of _weighted_layer_norm_fwd. "0"
# restores the old launch, its grid and its autotune exactly. Read once.
_WLN_FWD_V2_ENABLED: Optional[bool] = None
# (BLOCK_N, num_warps, grid multiplier of the CU count, NUM_STAGES), from an
# interleaved L2-flushed MI350X sweep at N=2.32M (probes/iter12/d256).
_WLN_FWD_V2_CFG: Tuple[int, int, int, int] = (16, 4, 64, 2)
# Old autotune pick (BLOCK_N, num_warps) -> the v2 EMU mode reproducing its
# bits (y, mean, rstd), verified on MI350X at N=2.32M/2.32M+1/1.73M. Picks
# not listed (e.g. (4,1), (8,1), (8,2), (1,2)) keep the old kernel.
_WLN_FWD_V2_EMU_TRITON36: Dict[Tuple[int, int], int] = {
    (1, 1): 0,
    (2, 2): 0,
    (4, 4): 0,
    (8, 8): 0,
    (2, 1): 1,
    (4, 2): 2,
    (8, 4): 2,
}
# ROCm 10.2 port (Triton 3.9, with TRITON_DEFAULT_FP_FUSION=1): the same picks
# and the same two lane classes, in 3.9's in-lane order (EMU 3 / 4, see
# _wln_row_moments; read off the 3.9 gfx950 ISA of the old kernel: packed
# v_pk_add_f32 / v_pk_fma_f32 trees inside a lane, the DPP lane trees as on
# 3.6). Verified bitwise (y, mean, rstd) on MI355X, Triton 3.9.0, against the
# ported old kernel at N = 2.32M / 2.32M+1 / 1.73M (plus edge-value rows) for
# every pick, and at N = 4.19M (> 2 GB, no buffer ops) for every pick but
# (1,1) (the 3.6 table, likewise, misses (4,2) / (8,4) above 2 GB).
_WLN_FWD_V2_EMU_TRITON39: Dict[Tuple[int, int], int] = {
    (1, 1): 3,
    (2, 2): 3,
    (4, 4): 3,
    (8, 8): 3,
    (2, 1): 4,
    (4, 2): 4,
    (8, 4): 4,
}


def _triton_major_minor() -> Tuple[int, int]:
    try:
        major, minor = triton.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (0, 0)


def _triton_fp_fusion_default() -> bool:
    """Whether Triton contracts a * b + c into an fma for kernels launched
    without enable_fp_fusion (the old kernel). Both EMU tables assume it."""
    lang = triton.knobs.language
    fusion = getattr(lang, "fp_fusion_enabled", None)  # Triton >= 3.9
    if fusion is not None:
        return bool(fusion(None))
    return bool(getattr(lang, "default_fp_fusion", True))


_WLN_FWD_V2_EMU: Dict[Tuple[int, int], int] = (
    _WLN_FWD_V2_EMU_TRITON39
    if _triton_major_minor() >= (3, 9)
    else _WLN_FWD_V2_EMU_TRITON36
)
# Per (x, w, b dtype) key: the EMU mode for the old autotuner's pick, or -1
# if v2 cannot reproduce it. v2 runs only once this is known, so it never
# changes the bits relative to the old path in the same process. The first
# call per key always takes the old path (and runs its autotune as before).
_WLN_FWD_V2_MODE: Dict[Tuple[torch.dtype, torch.dtype, torch.dtype], int] = {}


# DLRM_HSTU_FP8_D256_V3 (gin default on in yambda_5b.gin; absent = off; HIP only): the D=256 swish LinearNorm
# kernels (preprocessor MLPs, ops/fused_linear_norm.py) launch
#   _weighted_layer_norm_fwd_v3     instead of _weighted_layer_norm_fwd_v2,
#   _weighted_layer_norm_bwd_dx_v3  instead of _weighted_layer_norm_bwd_dx_v2,
#   _layer_norm_bwd_dwdbdx_p1 + _p2 instead of _layer_norm_bwd_dwdbdx
# (the last only in triton_weighted_layer_norm_bwd_dxsum at D=256 with the
# pinned finalize config). All three are bit-identical to what they replace
# (probes/iter16/d256). "0" restores the v2 launches exactly. Read once.
_D256_V3_ENABLED: Optional[bool] = None
_D256_V3_REPORTED: Dict[str, bool] = {}


def _d256_v3_enabled() -> bool:
    global _D256_V3_ENABLED
    if _D256_V3_ENABLED is None:
        _D256_V3_ENABLED = (
            torch.version.hip is not None
            and os.environ.get("DLRM_HSTU_FP8_D256_V3", "0") != "0"
        )
    return _D256_V3_ENABLED


def _d256_v3_note(what: str) -> None:
    """One-time arming line per launch site (fwd / bwd / fin), printed on every
    rank and tagged [rank N] (see ops.utils.arm_print_all_ranks)."""
    if what in _D256_V3_REPORTED:
        return
    _D256_V3_REPORTED[what] = True
    arm_print_all_ranks(
        f"armed OK: D256 swish LN v3 ACTIVE ({what}) -- "
        "_weighted_layer_norm_fwd_v3 / _weighted_layer_norm_bwd_dx_v3 / "
        "_layer_norm_bwd_dwdbdx_p1+p2 (DLRM_HSTU_FP8_D256_V3)"
    )


def _wln_fwd_v2_enabled() -> bool:
    global _WLN_FWD_V2_ENABLED
    if _WLN_FWD_V2_ENABLED is None:
        _WLN_FWD_V2_ENABLED = (
            torch.version.hip is not None
            and os.environ.get("DLRM_HSTU_FP8_WLN_FWD_V2", "0") != "0"
        )
    return _WLN_FWD_V2_ENABLED


def _wln_fwd_v2_key(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> Optional[Tuple[torch.dtype, torch.dtype, torch.dtype]]:
    if (
        not _wln_fwd_v2_enabled()
        or x.dim() != 2
        or x.shape[1] != 256
        or x.shape[0] < (1 << 16)
        or x.stride(1) != 1
        or not weight.is_contiguous()
        or not bias.is_contiguous()
        or torch.compiler.is_compiling()
    ):
        return None
    return (x.dtype, weight.dtype, bias.dtype)


def maybe_weighted_layer_norm_swish_fwd_v2(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(y, mean, rstd) of the swish weighted LayerNorm via the v2 kernel, or
    None when the caller must launch the old kernel (gate off, shape not
    covered, or the old autotune pick for this key not yet known / not in the
    class v2 reproduces). Callers report each old launch back through
    note_weighted_layer_norm_swish_fwd_old()."""
    key = _wln_fwd_v2_key(x, weight, bias)
    emu = -1 if key is None else _WLN_FWD_V2_MODE.get(key, -1)
    if emu < 0:
        return None
    N, D = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    block_n, num_warps, mult, num_stages = _WLN_FWD_V2_CFG
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = max(1, min(sms * mult, triton.cdiv(N, block_n)))
    if _d256_v3_enabled():
        # DLRM_HSTU_FP8_D256_V3: v2 with D constexpr (bit-identical).
        _weighted_layer_norm_fwd_v3[(grid,)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            N,
            256,
            eps,
            x.stride(0),
            y.stride(0),
            BLOCK_N=block_n,
            NUM_STAGES=num_stages,
            EMU=emu,
            num_warps=num_warps,
        )
        _d256_v3_note("fwd")
        return y, mean, rstd
    _weighted_layer_norm_fwd_v2[(grid,)](
        x,
        y,
        weight,
        bias,
        mean,
        rstd,
        N,
        D,
        eps,
        x.stride(0),
        y.stride(0),
        IS_SWISH=True,
        BLOCK_D=256,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        EMU=emu,
        num_warps=num_warps,
    )
    return y, mean, rstd


def note_weighted_layer_norm_swish_fwd_old(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> None:
    """Call right after an old swish _weighted_layer_norm_fwd launch: records
    which v2 EMU mode (if any) reproduces the config the autotuner used."""
    key = _wln_fwd_v2_key(x, weight, bias)
    if key is None or key in _WLN_FWD_V2_MODE:
        return
    cfg = getattr(_weighted_layer_norm_fwd, "best_config", None)
    if cfg is None:
        return
    pick = (cfg.kwargs.get("BLOCK_N"), cfg.num_warps)
    emu = _WLN_FWD_V2_EMU.get(pick, -1)
    if emu >= 0 and not _triton_fp_fusion_default():
        # ROCm 10.2 port: Triton 3.9 turns FP contraction off by default (the
        # T13 knob TRITON_DEFAULT_FP_FUSION=1 turns it back on). Unfused, the
        # old kernel's y * w + b and sum of squares round differently and no
        # EMU mode reproduces its bits: keep the old kernel for this key.
        arm_print_all_ranks(
            f"[wln-fwd-v2] DECLINED for {tuple(str(t) for t in key)}: Triton "
            f"{triton.__version__} FP contraction is off (TRITON_DEFAULT_FP_FUSION"
            f"=1 not set), so no v2 EMU mode reproduces the old kernel's "
            f"{pick[0]}x{pick[1]} pick; the old kernel keeps running"
        )
        emu = -1
    _WLN_FWD_V2_MODE[key] = emu


def compute_BLOCK_D(x: torch.Tensor) -> int:
    """Compute the BLOCK_D parameter for layer norm kernels."""
    D = x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    return min(MAX_FUSED_SIZE, triton.next_power_of_2(D))


@maybe_register_custom_op(
    "generative_recommenders::triton_weighted_layer_norm_fwd", mutates_args=()
)
def triton_weighted_layer_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    output_fp8: bool = False,
    inv_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    learnable = weight is not None
    if learnable:
        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

    if output_fp8:
        assert learnable, "FP8 LayerNorm output requires learnable affine parameters"
        y = torch.empty(
            (N, D),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
    else:
        y = torch.empty_like(x)
    compute_mean_and_rstd = mean is None or rstd is None
    # Always allocate new tensors to avoid aliasing inputs with outputs
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if not compute_mean_and_rstd:
        assert mean is not None and rstd is not None
        out_mean.copy_(mean)
        out_rstd.copy_(rstd)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if N == 0:
        return y, out_mean, out_rstd

    # pyre-ignore[28]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    if learnable:
        _weighted_layer_norm_fwd[grid](
            x,
            y,
            y,
            weight,
            bias,
            out_mean,
            out_rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            y.stride(0),
            float(inv_scale),
            None,  # RESIDUAL
            0,  # stride_residual
            IS_SWISH=False,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
            OUTPUT_FP8=output_fp8,
            OUTPUT_BOTH=False,
            ADD_RESIDUAL=False,
            # Saved mean/rstd (the backward recompute): Triton 3.9's LLVM packs
            # x - Mean[row] / * Rstd[row] of two rows into v_pk_add/mul_f32 with
            # the gfx950-invalid op_sel:[0,1]; compile without packed FP32 there
            # ({} on Triton 3.6 / other targets). See gfx950_pk_opsel.
            **({} if compute_mean_and_rstd else no_packed_fp32_launch_opts()),
        )
    else:
        _layer_norm_fwd[grid](
            x,
            y,
            out_mean,
            out_rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
        )

    return y, out_mean, out_rstd


def triton_weighted_layer_norm_fp8_save_bf16_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    inv_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward LayerNorm emitting scaled E4M3, BF16, mean and rstd in one pass.

    Store-instead-of-recompute (DLRM_HSTU_FP8_SAVE_NORMED_X): the FP8 forward
    used to write only the E4M3 copy, so backward re-ran this kernel with the
    saved mean/rstd (COMPUTE_MEAN_AND_RSTD=False) to rebuild the BF16 copy for
    the projection wgrad. Here the forward also writes the BF16 copy. Both
    outputs derive from the same fp32 ``y = (x - mean) * rstd * w + b`` using
    the very fp32 mean/rstd that are stored and that the backward recompute
    would load, so the BF16 tensor equals that recompute bit for bit and the
    E4M3 tensor equals the OUTPUT_FP8 store (``y * inv_scale`` then cast).
    Returns (y_bf16, y_fp8, mean, rstd).
    """
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and weight.numel() == D
    assert bias.dim() == 1 and bias.numel() == D
    y = torch.empty_like(x)
    y_fp8 = torch.empty((N, D), dtype=torch.float8_e4m3fn, device=x.device)
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if N == 0:
        return y, y_fp8, out_mean, out_rstd
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    _weighted_layer_norm_fwd[grid](
        x,
        y,
        y_fp8,
        weight,
        bias,
        out_mean,
        out_rstd,
        N,
        D,
        eps,
        x.stride(0),
        y.stride(0),
        y_fp8.stride(0),
        float(inv_scale),
        None,  # RESIDUAL
        0,  # stride_residual
        IS_SWISH=False,
        TRAINING=True,
        BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=True,
        OUTPUT_FP8=False,
        OUTPUT_BOTH=True,
        ADD_RESIDUAL=False,
    )
    return y, y_fp8, out_mean, out_rstd


def triton_weighted_layer_norm_dual_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    inv_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute affine LayerNorm once and store BF16 plus scaled E4M3."""
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and weight.numel() == D
    assert bias.dim() == 1 and bias.numel() == D
    y = torch.empty_like(x)
    y_fp8 = torch.empty((N, D), dtype=torch.float8_e4m3fn, device=x.device)
    if N == 0:
        return y, y_fp8

    BLOCK_D = compute_BLOCK_D(x)
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
    _weighted_layer_norm_fwd[grid](
        x,
        y,
        y_fp8,
        weight,
        bias,
        mean,
        rstd,
        N,
        D,
        eps,
        x.stride(0),
        y.stride(0),
        y_fp8.stride(0),
        float(inv_scale),
        None,  # RESIDUAL
        0,  # stride_residual
        IS_SWISH=False,
        TRAINING=False,
        BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=False,
        OUTPUT_FP8=False,
        OUTPUT_BOTH=True,
        ADD_RESIDUAL=False,
        # Saved mean/rstd: no packed FP32 on gfx950 + Triton 3.9 (as in
        # triton_weighted_layer_norm_fwd; see gfx950_pk_opsel).
        **no_packed_fp32_launch_opts(),
    )
    return y, y_fp8


@triton.jit
def _weighted_layer_norm_recompute_fp8t(
    X,
    YT,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N,
    Mp,
    D,
    stride_x,
    stride_yt,
    stride_y,
    Y_INV_SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    ALSO_BF16: tl.constexpr,
    YT_TILED: tl.constexpr = False,
):
    """Affine LN recompute from saved mean/rstd -> token-major E4M3 [D, Mp].

    Program (pid_d, pid_m) owns BLOCK_M token rows x BLOCK_DC features. The
    fp32 arithmetic is the COMPUTE_MEAN_AND_RSTD=False / OUTPUT_BOTH path of
    _weighted_layer_norm_fwd op for op, so the E4M3 bytes (and the optional
    BF16 y) are bitwise identical to triton_weighted_layer_norm_dual_fwd. The
    tile is transposed on chip and stored as BLOCK_DC rows of BLOCK_M
    contiguous bytes. Token columns in [N, Mp) are written as zero.
    YT_TILED (DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED): the same bytes step-tiled
    [Mp / 128, D, 128], (token t, feature c) at (t // 128) * stride_yt +
    c * 128 + t % 128 (stride_yt = D * 128); only the YT address changes.
    """
    # 1-D grid, feature chunk fastest: the D/BLOCK_DC programs of one token
    # block run adjacently and share its x rows' DRAM pages.
    pid = tl.program_id(0)
    n_dc = tl.cdiv(D, BLOCK_DC)
    pid_d = pid % n_dc
    pid_m = pid // n_dc
    rows = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    cols = pid_d * BLOCK_DC + tl.arange(0, BLOCK_DC)
    row_mask = rows < N
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    x_block = tl.load(
        X + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    mean = tl.load(Mean + rows, row_mask, other=0.0)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    x_mean = tl.where(mask, x_mean, 0.0)
    rstd = tl.load(Rstd + rows, row_mask, other=0.0)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]
    if ALSO_BF16:
        tl.store(
            Y + rows[:, None] * stride_y + cols[None, :],
            y.to(Y.dtype.element_ty),
            mask=mask,
        )
    y_fp8 = y * Y_INV_SCALE
    y_fp8 = tl.where(row_mask[:, None], y_fp8, 0.0)
    q = y_fp8.to(YT.dtype.element_ty).to(tl.uint8, bitcast=True)
    qt = tl.trans(q)
    if YT_TILED:
        tl.store(
            YT.to(tl.pointer_type(tl.uint8))
            + (rows[None, :] // 128) * stride_yt
            + cols[:, None].to(tl.int64) * 128
            + (rows[None, :] % 128),
            qt,
            mask=col_mask[:, None] & (rows < Mp)[None, :],
        )
    else:
        tl.store(
            YT.to(tl.pointer_type(tl.uint8))
            + cols[:, None].to(tl.int64) * stride_yt
            + rows[None, :],
            qt,
            mask=col_mask[:, None] & (rows < Mp)[None, :],
        )


# Fixed launch configs (no runtime autotune), (BLOCK_M, BLOCK_DC, num_warps)
# keyed by also_bf16; picked by a config sweep at the production token counts,
# D = 512 on MI350X (Arbor R14 testbed).
_LN_FP8T_CONFIG = {False: (256, 128, 4), True: (256, 64, 4)}


def triton_weighted_layer_norm_recompute_fp8t(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    inv_scale: float,
    pad_to: int = 128,
    also_bf16: bool = False,
    block_m: Optional[int] = None,
    block_dc: Optional[int] = None,
    num_warps: Optional[int] = None,
    tiled_xt: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Recompute affine LN from saved mean/rstd -> (y_bf16 | None, yT_fp8).

    yT_fp8 is float8_e4m3fn [D, Mp], Mp = roundup(M, pad_to), token-major
    (contiguous along tokens, strides (Mp, 1)), holding e4m3(LN(x) * inv_scale);
    columns >= M are zero. yT_fp8.t()[:M] is bitwise equal to the E4M3 output
    of triton_weighted_layer_norm_dual_fwd; y_bf16 (also_bf16=True) is bitwise
    equal to its BF16 output. `eps` is unused (rstd is supplied) and kept for
    signature parity. block_m/block_dc/num_warps override the fixed config
    (benchmarking only).
    tiled_xt=True (the fp8t call sites pass wgrad_ck.tiled_active(device)):
    yT_fp8 is the same bytes step-tiled, a 3-D [Mp / 128, D, 128] tensor
    (wgrad_ck "Step-tiled xT"); D != 512 or Mp % 128 != 0 declines (row-major,
    logged once).
    """
    del eps
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and weight.numel() == D
    assert bias.dim() == 1 and bias.numel() == D
    assert pad_to >= 1
    Mp = (N + pad_to - 1) // pad_to * pad_to
    y = torch.empty_like(x) if also_bf16 else None
    xt_shape = _xt_tiled_shape(tiled_xt, "recompute_fp8t", True, "", D, Mp, x.device)
    # DLRM_HSTU_FP8_TM_NONEXP: token-major wgrad operand from a non-expandable pool
    # (DLRM_HSTU_FP8_TM_NONEXP_XT=0: plain torch.empty; default 1 = the private pool).
    yT = fp8_tm_alloc.empty_xt(xt_shape or (D, Mp), torch.float8_e4m3fn, x.device)
    if Mp == 0:
        return y, yT
    cfg_m, cfg_dc, cfg_w = _LN_FP8T_CONFIG[bool(also_bf16)]
    BLOCK_M = block_m or cfg_m
    BLOCK_DC = block_dc or min(cfg_dc, triton.next_power_of_2(D))
    grid = (triton.cdiv(D, BLOCK_DC) * triton.cdiv(Mp, BLOCK_M),)
    _weighted_layer_norm_recompute_fp8t[grid](
        x,
        yT,
        y if y is not None else x,
        weight,
        bias,
        mean,
        rstd,
        N,
        Mp,
        D,
        x.stride(0),
        yT.stride(0),
        y.stride(0) if y is not None else 0,
        float(inv_scale),
        BLOCK_M=BLOCK_M,
        BLOCK_DC=BLOCK_DC,
        ALSO_BF16=also_bf16,
        YT_TILED=xt_shape is not None,
        num_warps=num_warps or cfg_w,
    )
    if xt_shape is not None:
        wgrad_ck.note_tiled_producer(
            "recompute_fp8t", x.device, True, f"M={N} D={D} Mp={Mp}"
        )
    return y, yT


def _xt_tiled_shape(
    tiled_xt: bool,
    producer: str,
    eligible: bool,
    why: str,
    d: int,
    mp: int,
    device: torch.device,
) -> Optional[Tuple[int, int, int]]:
    """The step-tiled yT shape when `tiled_xt` asks for it and the producer
    (`eligible`, else `why`) and the geometry allow it; None: row-major [D, Mp]
    (a refusal of a request is logged once per producer and device)."""
    if not tiled_xt:
        return None
    shape = wgrad_ck.tiled_shape(d, mp) if eligible else None
    if shape is None:
        wgrad_ck.note_tiled_producer(
            producer, device, False, why if not eligible else f"D={d} Mp={mp}"
        )
    return shape


@triton.jit
def _wln_hex(v, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """Split [BLOCK_N, BLOCK_D] into the sixteen column phases e0..e15 of
    [BLOCK_N, BLOCK_D // 16]: element 16*l + j of a row is e_j[l]."""
    v16 = tl.reshape(v, [BLOCK_N, BLOCK_D // 16, 2, 2, 2, 2])  # j bits b3 b2 b1 b0
    a, bb = tl.split(v16)  # b0: a = even j, bb = odd j
    a0, a1 = tl.split(a)  # b1: a0 = j % 4 == 0, a1 = j % 4 == 2
    b0, b1 = tl.split(bb)  # b0 = j % 4 == 1, b1 = j % 4 == 3
    a00, a01 = tl.split(a0)  # b2: (0, 8) vs (4, 12)
    a10, a11 = tl.split(a1)  # (2, 10) vs (6, 14)
    b00, b01 = tl.split(b0)  # (1, 9) vs (5, 13)
    b10, b11 = tl.split(b1)  # (3, 11) vs (7, 15)
    e0, e8 = tl.split(a00)  # b3
    e4, e12 = tl.split(a01)
    e2, e10 = tl.split(a10)
    e6, e14 = tl.split(a11)
    e1, e9 = tl.split(b00)
    e5, e13 = tl.split(b01)
    e3, e11 = tl.split(b10)
    e7, e15 = tl.split(b11)
    return e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15


@triton.jit
def _store_t_moments_512(
    x_block, mask, D, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """(mean, masked x - mean, var) of the store_t kernels at BLOCK_D = 512, in
    the exact fp32 operation order of their Triton 3.6 build (tile layout
    [1, 16] x [2, 32] lanes: 16 consecutive features per lane, 32 lanes per
    row), which is also the order the MXFP6 LayerNorm producer
    (ops/hip/ln_store_mx6.cu) computes and its producer check
    (ops/hip/mx6_fwd.py) compares bitwise:
      sum  = c0 + c1 + ... + c15 per lane, sequential, then the 32-lane xor
             tree (lane bits 4, 3, 2, 1, 0); mean = sum / D;
      q    = fma(xm0, xm0, xm1 * xm1), q = fma(xm_j, xm_j, q) for
             j = 2 .. 15 per lane, the same lane tree; var = q / D.
    Triton 3.9 (ROCm 10.2 port) lowers the per-lane part of tl.sum as a
    pairwise tree of 2-wide adds and contracts the squares differently, which
    changed mean in ~0.01% and rstd in ~13-15% of the rows (and the e4m3 bytes
    that depend on them). Written out here, the order does not depend on the
    compiler's reduce lowering or contraction choices: every add has two
    explicit operands (each lane-tree step sums one size-2 axis), and every
    product that feeds an add is an explicit fma."""
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 = _wln_hex(
        x_block, BLOCK_N, BLOCK_D
    )
    s = e0 + e1
    s = s + e2
    s = s + e3
    s = s + e4
    s = s + e5
    s = s + e6
    s = s + e7
    s = s + e8
    s = s + e9
    s = s + e10
    s = s + e11
    s = s + e12
    s = s + e13
    s = s + e14
    s = s + e15
    mean = _wln_lane_tree32(s, BLOCK_N) / D
    x_mean = x_block - tl.expand_dims(mean, 1)
    x_mean = tl.where(mask, x_mean, 0.0)
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15 = _wln_hex(
        x_mean, BLOCK_N, BLOCK_D
    )
    p = tl.fma(q0, q0, q1 * q1)
    p = tl.fma(q2, q2, p)
    p = tl.fma(q3, q3, p)
    p = tl.fma(q4, q4, p)
    p = tl.fma(q5, q5, p)
    p = tl.fma(q6, q6, p)
    p = tl.fma(q7, q7, p)
    p = tl.fma(q8, q8, p)
    p = tl.fma(q9, q9, p)
    p = tl.fma(q10, q10, p)
    p = tl.fma(q11, q11, p)
    p = tl.fma(q12, q12, p)
    p = tl.fma(q13, q13, p)
    p = tl.fma(q14, q14, p)
    p = tl.fma(q15, q15, p)
    var = _wln_lane_tree32(p, BLOCK_N) / D
    return mean, x_mean, var


@triton.jit
def _weighted_layer_norm_fwd_fp8_store_t(
    X,
    Y,
    YT,
    W,
    B,
    Mean,
    Rstd,
    N,
    Mp,
    D,
    eps,
    stride_x,
    stride_y,
    stride_yt,
    Y_INV_SCALE,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Training forward LN -> row-major E4M3 Y [N, D] + token-major E4M3 YT [D, Mp].

    DLRM_HSTU_FP8_BWD_UVQK_STORE_T (default off): one pass over x computes the
    statistics and E4M3 Y of _weighted_layer_norm_fwd(COMPUTE_MEAN_AND_RSTD,
    TRAINING, OUTPUT_FP8) -- at BLOCK_D 512 in the fixed Triton 3.6 order of
    _store_t_moments_512, so not always that kernel's bits -- and transposes
    the E4M3 tile on chip into the token-major layout the FP8 projection wgrad
    consumes. Backward then skips _weighted_layer_norm_recompute_fp8t (a full
    re-read of x). YT is byte-identical to the recompute kernel's output
    (test-checked); Y / mean / rstd can differ from the generic forward's in
    some rows. Token columns [N, Mp) of YT are zero.
    """
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < N
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    x_block = tl.load(
        X + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)

    if BLOCK_D == 512:
        # Statistics in the fixed order of the Triton 3.6 build (see
        # _store_t_moments_512); y = fma(x_mean * rstd, w, b) * Y_INV_SCALE.
        mean, x_mean, var = _store_t_moments_512(x_block, mask, D, BLOCK_N, BLOCK_D)
        tl.store(Mean + rows, mean, row_mask)
        rstd = libdevice_rsqrt(var + eps)
        tl.store(Rstd + rows, rstd, row_mask)
        y = tl.fma(x_mean * tl.expand_dims(rstd, 1), w[None, :], b[None, :])
    else:
        mean = tl.sum(x_block, axis=1) / D
        tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
        x_mean = x_block - mean
        x_mean = tl.where(mask, x_mean, 0.0)
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = libdevice_rsqrt(var + eps)
        tl.store(Rstd + rows, rstd, row_mask)
        rstd = tl.expand_dims(rstd, 1)
        y = x_mean * rstd
        y = y * w[None, :] + b[None, :]
    y = y * Y_INV_SCALE
    y8 = y.to(Y.dtype.element_ty)
    tl.store(Y + rows[:, None] * stride_y + cols[None, :], y8, mask=mask)
    q = tl.where(row_mask[:, None], y8.to(tl.uint8, bitcast=True), 0)
    qt = tl.trans(q)
    tl.store(
        YT.to(tl.pointer_type(tl.uint8))
        + cols[:, None].to(tl.int64) * stride_yt
        + rows[None, :],
        qt,
        mask=col_mask[:, None] & (rows < Mp)[None, :],
    )


# (BLOCK_N, num_warps) for D = 512: best of a config sweep at the production
# token counts (Arbor R14 testbed).
_LN_FWD_STORE_T_CONFIG = (128, 16)


@triton.jit
def _weighted_layer_norm_fwd_fp8_store_t_v2_tile(
    X,
    Y,
    YT,
    W,
    B,
    Mean,
    Rstd,
    N,
    Mp,
    D,
    eps,
    stride_x,
    stride_y,
    stride_yt,
    Y_INV_SCALE,
    pid,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    YT_TILED: tl.constexpr,
):
    # The body of _weighted_layer_norm_fwd_fp8_store_t for block `pid`, op for
    # op (keep in sync with it). YT_TILED: step-tiled YT [Mp / 128, D, 128]
    # (see _weighted_layer_norm_recompute_fp8t); only the YT address changes.
    rows = pid.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    row_mask = rows < N
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    x_block = tl.load(
        X + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)

    if BLOCK_D == 512:
        # Statistics in the fixed order of the Triton 3.6 build (see
        # _store_t_moments_512); y = fma(x_mean * rstd, w, b) * Y_INV_SCALE.
        mean, x_mean, var = _store_t_moments_512(x_block, mask, D, BLOCK_N, BLOCK_D)
        tl.store(Mean + rows, mean, row_mask)
        rstd = libdevice_rsqrt(var + eps)
        tl.store(Rstd + rows, rstd, row_mask)
        y = tl.fma(x_mean * tl.expand_dims(rstd, 1), w[None, :], b[None, :])
    else:
        mean = tl.sum(x_block, axis=1) / D
        tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
        x_mean = x_block - mean
        x_mean = tl.where(mask, x_mean, 0.0)
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = libdevice_rsqrt(var + eps)
        tl.store(Rstd + rows, rstd, row_mask)
        rstd = tl.expand_dims(rstd, 1)
        y = x_mean * rstd
        y = y * w[None, :] + b[None, :]
    y = y * Y_INV_SCALE
    y8 = y.to(Y.dtype.element_ty)
    tl.store(Y + rows[:, None] * stride_y + cols[None, :], y8, mask=mask)
    q = tl.where(row_mask[:, None], y8.to(tl.uint8, bitcast=True), 0)
    qt = tl.trans(q)
    if YT_TILED:
        tl.store(
            YT.to(tl.pointer_type(tl.uint8))
            + (rows[None, :] // 128) * stride_yt
            + cols[:, None].to(tl.int64) * 128
            + (rows[None, :] % 128),
            qt,
            mask=col_mask[:, None] & (rows < Mp)[None, :],
        )
    else:
        tl.store(
            YT.to(tl.pointer_type(tl.uint8))
            + cols[:, None].to(tl.int64) * stride_yt
            + rows[None, :],
            qt,
            mask=col_mask[:, None] & (rows < Mp)[None, :],
        )


# N and n_blocks must not be specialized: with N % 16 == 0 specialization the
# BLOCK_N=32 build contracted the variance arithmetic differently (rstd
# differed from the shipped BLOCK_N=128 kernel in ~2.6% of rows at
# M=2365056); the unspecialized build is bit-identical (iter32b castdiet).
# (Since the ROCm 10.2 port the D = 512 statistics are written out in a fixed
# order, _store_t_moments_512, so they no longer depend on such choices; the
# unspecialized build is kept.)
@triton.jit(do_not_specialize=["N", "n_blocks"])
def _weighted_layer_norm_fwd_fp8_store_t_v2(
    X,
    Y,
    YT,
    W,
    B,
    Mean,
    Rstd,
    N,
    Mp,
    D,
    eps,
    stride_x,
    stride_y,
    stride_yt,
    Y_INV_SCALE,
    n_blocks,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    XCD_GROUP: tl.constexpr,
    YT_TILED: tl.constexpr = False,
):
    """`_weighted_layer_norm_fwd_fp8_store_t` with small blocks, XCD-grouped.

    DLRM_HSTU_FP8_BWD_STORE_T_V2=1 (default off; Arbor R14 iter-33 M1). Y,
    YT (zero tail included), mean and rstd are bit-identical to the shipped
    kernel at D = 512. A block's YT row segment is BLOCK_N bytes, less than a
    128-byte line when BLOCK_N < 128; workgroups are dispatched round-robin
    over the 8 XCDs, so programs k, k+8, ..., k+8*(XCD_GROUP-1) (one XCD, one
    L2) take XCD_GROUP consecutive blocks and complete each YT line in the
    same L2 instead of as partial lines from different XCDs. Programs past
    the last block do nothing.
    """
    k = tl.program_id(0)
    xcd = k % 8
    r = k // 8
    blk = ((r // XCD_GROUP) * 8 + xcd) * XCD_GROUP + (r % XCD_GROUP)
    if blk < n_blocks:
        _weighted_layer_norm_fwd_fp8_store_t_v2_tile(
            X,
            Y,
            YT,
            W,
            B,
            Mean,
            Rstd,
            N,
            Mp,
            D,
            eps,
            stride_x,
            stride_y,
            stride_yt,
            Y_INV_SCALE,
            blk,
            BLOCK_N,
            BLOCK_D,
            YT_TILED,
        )


# DLRM_HSTU_FP8_BWD_STORE_T_V2=1 (default off): read at import, checked by
# `triton_weighted_layer_norm_fwd_fp8_store_t` at call time. Applies to the
# validated geometry only (BLOCK_D = 512, no block_n/num_warps override);
# anything else keeps the shipped kernel.
_STORE_T_V2: bool = os.environ.get("DLRM_HSTU_FP8_BWD_STORE_T_V2", "0") == "1"
# (BLOCK_N, num_warps) of the v2 kernel: bit-identical to the shipped (128, 16)
# at D = 512 (both compute the statistics with _store_t_moments_512, whose
# operation order does not depend on BLOCK_N or the tile layout).
# 64 tokens per block (ROCm 10.2 port; v26.3 / Triton 3.6 shipped (32, 4) with
# XCD_GROUP 8): a block writes 64 bytes of each of its 512 step-tiled YT lines,
# i.e. whole 64-byte DRAM write granules. With 32-token blocks each block wrote
# a 32-byte quarter line and relied on L2 merging the four quarters before the
# write-back; the Triton 3.9 build keeps all X loads of a wave in flight, the
# streaming X / y lines evict partial YT lines first, and every unmerged
# quarter became a sub-granule (partial) DRAM write: ~5.3 extra <= 32-byte
# write requests per row (rocprofv3 TCC_EA0_WRREQ - WRREQ_64B; 0.6..2 on the
# Triton 3.6 build) and up to 1.3x the v26.3 kernel time at X > 2 GiB. 64-token
# blocks: 0 partial write requests, <= the v26.3 kernel time
# (docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md T21). Same per-lane work (4 rows x 16 features), 32 KiB LDS.
_LN_FWD_STORE_T_V2_CONFIG = (64, 8)
# Programs per XCD taking consecutive BLOCK_N-token blocks (4 x 64 = 256 tokens
# = two full 128-byte YT lines per group).
_LN_FWD_STORE_T_V2_XCD_GROUP = 4


def _log_store_t_v2_armed(device: torch.device, n: int, d: int, mp: int) -> None:
    # Imported here, not at module level: only the v2 path needs bwd_fp8.
    from generative_recommenders.ops.triton import bwd_fp8

    bwd_fp8.log_armed_v2(
        "store_t_v2",
        device,
        lambda: (
            f"ARMED store_t_v2 on {device}: "
            f"_weighted_layer_norm_fwd_fp8_store_t_v2 (BLOCK_N="
            f"{_LN_FWD_STORE_T_V2_CONFIG[0]}, {_LN_FWD_STORE_T_V2_CONFIG[1]} "
            f"warps, XCD_GROUP={_LN_FWD_STORE_T_V2_XCD_GROUP}, N unspecialized; "
            f"y/yT/mean/rstd bitwise vs _weighted_layer_norm_fwd_fp8_store_t), "
            f"M={n} D={d} Mp={mp}"
        ),
    )


# Largest storage (bytes) of the Triton AMD backend's "within 2 GiB" pointer
# class (get_tensor_specialization: storage size <= 2**31 - 1).
_TRITON_2GIB_CLASS_MAX = 2**31 - 1
# B15T(L2) skipped-y scratch: row stride (bytes) of the discarded y rows. Any
# multiple of 16 keeps stride_y's specialization ("divisible by 16", as the
# production y's D = 512); 16 spreads the discarded rows over 16 * M + D bytes
# (about 38 MB at M = 2.38M, 1/32 of y) instead of M * D (1.2 GB), and keeps
# them off a single cache line (a stride of 0 would send every row to the same
# D bytes). The kernel still issues every y store; they land in this scratch,
# so the saving is fewer DRAM write-backs, not fewer store instructions.
_SKIP_Y_ROW_STRIDE = 16


def _store_t_v2_takes(
    x: torch.Tensor, block_n: Optional[int], num_warps: Optional[int]
) -> bool:
    """Whether triton_weighted_layer_norm_fwd_fp8_store_t runs the v2 kernel."""
    return (
        _STORE_T_V2
        and compute_BLOCK_D(x) == 512
        and block_n is None
        and num_warps is None
    )


def store_t_skip_y_decline_reason(
    x: torch.Tensor, pad_to: int, tiled_xt: bool
) -> Optional[Tuple[str, str]]:
    """B15T(L2) (yambda_5b.gin item B15T(L2)), on the host before the LayerNorm:
    why triton_weighted_layer_norm_fwd_fp8_store_t(x, ..., pad_to=pad_to,
    tiled_xt=tiled_xt, store_y=False) would NOT send y's bytes to the 1/32-size
    discarded scratch with a step-tiled yT (the wrapper then writes and returns
    the [M, D] y), or None when it will (the v2 kernel takes the call and yT is
    [Mp / 128, 512, 128]). Mirrors the wrapper's own decisions; reads only
    shapes and module flags (no tensor data, no sync). The details are constant
    per condition (the decline line is printed once per kind and detail)."""
    if x.dim() != 2:
        return ("geometry", f"x is {x.dim()}-D (the store_t takes 2-D x)")
    n, d = x.shape
    if not _store_t_v2_takes(x, None, None):
        return (
            "store_t_v2_off",
            "DLRM_HSTU_FP8_BWD_STORE_T_V2 off or D != 512 (only the v2 store_t "
            "kernel call can write y into the discarded scratch)",
        )
    if n < 1:
        return ("rows", "M=0")
    if n * d > _TRITON_2GIB_CLASS_MAX:
        # y [M, D] e4m3 would be a > 2 GiB storage: the skipped-y scratch (a
        # smaller tensor) would then select another Triton binary.
        return (
            "y_class",
            "M * D > 2**31 - 1 (y would be a > 2 GiB storage; the skipped-y "
            "scratch must keep the production y's pointer class)",
        )
    if not tiled_xt:
        return (
            "layout",
            "the store_t yT is not step-tiled (wgrad_ck.tiled_active is False)",
        )
    mp = (n + pad_to - 1) // pad_to * pad_to
    if wgrad_ck.tiled_shape(d, mp) is None:
        return (
            "layout",
            f"pad_to={pad_to} D={d}: no step-tiled yT for this geometry",
        )
    return None


def triton_weighted_layer_norm_fwd_fp8_store_t(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    inv_scale: float,
    pad_to: int = 128,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    tiled_xt: bool = False,
    store_y: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training forward LN -> (y_fp8 [N, D], mean, rstd, yT_fp8 [D, Mp]).

    y_fp8 / mean / rstd: a training LN forward in E4M3 (at D=512 in the fixed
    order ln_store_mx6 checks; not always triton_weighted_layer_norm_fwd(...,
    output_fp8=True)'s bits); yT_fp8 equals triton_weighted_layer_norm_recompute_fp8t(
    x, ..., mean, rstd, inv_scale, pad_to)[1] (token-major, strides (Mp, 1), zero tail).
    tiled_xt=True (the fp8t call sites pass wgrad_ck.tiled_active(device)):
    on the v2 kernel (DLRM_HSTU_FP8_BWD_STORE_T_V2), yT_fp8 is the same bytes
    step-tiled, a 3-D [Mp / 128, D, 128] tensor (wgrad_ck "Step-tiled xT");
    y_fp8 / mean / rstd are unchanged. The shipped (v1) kernel, D != 512 or
    Mp % 128 != 0 decline it (row-major, logged once).
    store_y=False (B15T(L2), gin item B15T(L2); its one caller is the
    targets-only forward after store_t_skip_y_decline_reason returned None):
    on the v2 kernel with 1 <= M and M * D <= 2**31 - 1 no [M, D] y is
    allocated and None is returned in its place. y's bytes are still WRITTEN:
    the UNMODIFIED v2 kernel runs (the same compiled binary as a store_y=True
    call: same pointer classes and stride specializations) with Y = a
    dedicated 1/32-size scratch of 16 * M + D bytes and row stride
    _SKIP_Y_ROW_STRIDE, so every y store lands in overlapping scratch rows
    that nothing reads (fewer DRAM write-backs, not fewer stores). The scratch
    is its own caching-allocator storage, never a view into yT, x or a pool
    block. yT, mean and rstd are the same bytes (same binary, same inputs).
    Elsewhere (v1 kernel, M = 0, or M * D > 2**31 - 1, where y's storage
    would be in the other Triton pointer class) store_y is forced True: y is
    written and returned.
    """
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and weight.numel() == D
    assert bias.dim() == 1 and bias.numel() == D
    assert pad_to >= 1
    BLOCK_D = compute_BLOCK_D(x)
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    cfg_n, cfg_w = _LN_FWD_STORE_T_CONFIG
    # One program holds a BLOCK_N x BLOCK_D fp32 tile (and its uint8
    # transpose): keep it at the D = 512 tuned size for wider features.
    BLOCK_N = max(16, min(block_n or cfg_n, 65536 // BLOCK_D))
    Mp = (N + pad_to - 1) // pad_to * pad_to
    use_v2 = _store_t_v2_takes(x, block_n, num_warps)
    xt_shape = _xt_tiled_shape(
        tiled_xt, "store_t_v2" if use_v2 else "store_t", use_v2,
        "v1 kernel: STORE_T_V2 off or a non-v2 geometry", D, Mp, x.device,
    )
    # K-1 (r14): store_y is forced True when y [N, D] would be a > 2 GiB
    # storage (the scratch would select the other is_within_2gb binary).
    skip_y = (
        not store_y and use_v2 and Mp > 0 and N * D <= _TRITON_2GIB_CLASS_MAX
    )
    # skip_y: y's bytes are still written, into this dedicated 1/32-size
    # scratch (16-byte aligned, <= 2 GiB: the production y's pointer class).
    # K-2 (r14): a fresh torch.empty per call, never a view into yT, x or a
    # pool block (overlapping row writes are harmless only where nothing reads).
    y = torch.empty(
        (_SKIP_Y_ROW_STRIDE * N + D,) if skip_y else (N, D),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    # K-1 (r14): the scratch is its own caching-allocator storage of exactly
    # its size, not a B11 pool view (those carry the 2**31-byte block storage).
    assert not skip_y or (
        y.storage_offset() == 0
        and y.untyped_storage().nbytes() == y.numel() <= _TRITON_2GIB_CLASS_MAX
    ), "B15T(L2) scratch is not a dedicated <= 2 GiB storage"
    # DLRM_HSTU_FP8_TM_NONEXP: token-major wgrad operand from a non-expandable pool
    # (DLRM_HSTU_FP8_TM_NONEXP_XT=0: plain torch.empty; default 1 = the private pool).
    yT = fp8_tm_alloc.empty_xt(xt_shape or (D, Mp), torch.float8_e4m3fn, x.device)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if Mp == 0:
        return y, mean, rstd, yT
    if use_v2:
        v2_n, v2_w = _LN_FWD_STORE_T_V2_CONFIG
        n_blocks = triton.cdiv(Mp, v2_n)
        group = 8 * _LN_FWD_STORE_T_V2_XCD_GROUP
        _weighted_layer_norm_fwd_fp8_store_t_v2[
            (triton.cdiv(n_blocks, group) * group,)
        ](
            x,
            y,
            yT,
            weight,
            bias,
            mean,
            rstd,
            N,
            Mp,
            D,
            eps,
            x.stride(0),
            _SKIP_Y_ROW_STRIDE if skip_y else y.stride(0),
            yT.stride(0),
            float(inv_scale),
            n_blocks,
            BLOCK_N=v2_n,
            BLOCK_D=BLOCK_D,
            XCD_GROUP=_LN_FWD_STORE_T_V2_XCD_GROUP,
            YT_TILED=xt_shape is not None,
            num_warps=v2_w,
        )
        _log_store_t_v2_armed(x.device, N, D, Mp)
        if xt_shape is not None:
            wgrad_ck.note_tiled_producer(
                "store_t_v2", x.device, True, f"M={N} D={D} Mp={Mp}"
            )
        return (None if skip_y else y), mean, rstd, yT
    grid = (triton.cdiv(Mp, BLOCK_N),)
    _weighted_layer_norm_fwd_fp8_store_t[grid](
        x,
        y,
        yT,
        weight,
        bias,
        mean,
        rstd,
        N,
        Mp,
        D,
        eps,
        x.stride(0),
        y.stride(0),
        yT.stride(0),
        float(inv_scale),
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps or cfg_w,
    )
    return y, mean, rstd, yT


@triton_weighted_layer_norm_fwd.register_fake
def _(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    output_fp8: bool = False,
    inv_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = x.shape[0]
    y = torch.empty(
        x.shape,
        dtype=torch.float8_e4m3fn if output_fp8 else x.dtype,
        device=x.device,
    )
    # Always allocate new tensors to avoid aliasing inputs with outputs
    out_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    out_rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    return y, out_mean, out_rstd


_LN_BWD_SWISH_SMALL_D_TILE_MULT = 32


def _ln_bwd_tile_num(N: int, D: int, sms: int, is_swish: bool) -> int:
    """Persistent-grid size for _weighted_layer_norm_bwd_dx.

    The kernel runs num_warps=1 programs that each walk rows serially, so the
    number of in-flight programs per CU is what hides HBM latency. At
    BLOCK_D<=256 a row is only 512 B of bf16 per tensor, so sms*8 single-warp
    programs leave the CU starved: on MI350X at the production shape
    (N=2.28M, D=256, SUM_DX, IS_SWISH) the dx kernel ran at 2.17 TB/s
    (1.61 ms), versus 1.26 / 1.04 / 1.01 ms at sms*16 / 32 / 64
    (iter_22/lnbwd/sweep_d256.log). 32 is used: end-to-end with the
    dw/db/dxsum finalize (which reads [tile_num, D] partials) 32 and 64 tie at
    1.24 ms vs 1.32-1.45 ms at 8, and 32 needs half the partial buffers.
    It is gated on IS_SWISH: the swish epilogue (sigmoid recompute) makes each row latency-heavier, whereas plain
    LN at D=256 already reaches ~4 TB/s at sms*8 and gets slower with more
    tiles (0.88 -> 1.06 ms end-to-end). At D=512 both are ~4.1-4.3 TB/s and the
    multiplier is neutral, so it stays at 8. Production (hidden_dim=256,
    output_dim=512) only hits the swish/D=256 case.

    For a fixed BLOCK_N, dx does not depend on the grid (rows are independent)
    and is bit-identical. The autotuner (key BLOCK_D) re-picks BLOCK_N on the
    new grid, however: BLOCK_N=1 wins by a wide margin at sms*32 (1.04 vs
    1.37 ms), and BLOCK_N=1 vs 8 changes the intra-row reduction order of dx
    at the bf16-ulp level. dw/db/dxsum are fp32 sums over [tile_num, D] partials, so
    their reduction order changes (measured rel diff ~4e-7).
    """
    small_swish = is_swish and triton.next_power_of_2(D) <= 256
    mult = _LN_BWD_SWISH_SMALL_D_TILE_MULT if small_swish else 8
    return max(1, min(sms * mult, N // 4))


# DLRM_HSTU_FP8_WLN_BWD_V2 (gin default on, item (14); absent = v1): route every launch of the weighted
# LayerNorm backward dx kernel through _weighted_layer_norm_bwd_dx_v2. "0"
# restores the v1 kernel, its grid and its autotune exactly. Read once, cached.
_WLN_BWD_V2_ENABLED: Optional[bool] = None


def _wln_bwd_v2_enabled() -> bool:
    global _WLN_BWD_V2_ENABLED
    if _WLN_BWD_V2_ENABLED is None:
        _WLN_BWD_V2_ENABLED = os.environ.get("DLRM_HSTU_FP8_WLN_BWD_V2", "0") != "0"
    return _WLN_BWD_V2_ENABLED


# Sub-gate (only consulted when v2 is on): DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID
# (gin default on, item (14); absent = tl.sigmoid) computes the swish sigmoid as fast_dividef(1, 1 + exp(-z)).
# At the production swish shape (N=2.36M, D=256) dx error vs an fp64
# reference is unchanged (max abs 1.5625e-2, rel L2 1.65968667e-3 for both);
# dx differs from tl.sigmoid in ~2.5e-5 of elements by one bf16 ulp.
_WLN_BWD_V2_FAST_SIGMOID: Optional[bool] = None


def _wln_bwd_v2_fast_sigmoid() -> bool:
    global _WLN_BWD_V2_FAST_SIGMOID
    if _WLN_BWD_V2_FAST_SIGMOID is None:
        _WLN_BWD_V2_FAST_SIGMOID = (
            os.environ.get("DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID", "0") != "0"
        )
    return _WLN_BWD_V2_FAST_SIGMOID


# DLRM_HSTU_FP8_WLN_BWD_POW2 (default OFF in code; the gin binding
# apply_env_bootstrap.DLRM_HSTU_FP8_WLN_BWD_POW2 turns it on; "0" is the kill
# switch and wins over gin): when D ==
# BLOCK_D (D a power of two: 256 and 512 at every production call), the v2
# weighted-LN backward dx kernel (_weighted_layer_norm_bwd_dx_v2, item (14),
# every production launch) forms c1/c2 as row_sum * (1.0 / BLOCK_D) instead of
# row_sum / D. 2^-k is exact in fp32, and x * 2^-k and x / 2^k are the same
# real number, so the two correctly rounded results are equal bit for bit
# (subnormal results included); the change only drops the two precise fp32
# divides per row (D is a runtime int, so `/ D` compiles to the
# v_div_scale/v_rcp/v_fma x5/v_div_fmas/v_div_fixup sequence). Same row sums,
# same per-element formula, same config/grid: BITWISE (dx, dw, db, dxsum) on
# gfx950 / Triton 3.6 -- validated at every production call site and 90
# kernel configs (probes/iter63/lnport/validate.py); in the ISA the divide
# sequences become plain v_mul/v_pk_mul by 2^-k and the per-element FMA count
# is unchanged. Re-run that check after a Triton/LLVM change: the multiply is
# an FMA-contraction candidate where the divide was not. The v1 kernel
# (DLRM_HSTU_FP8_WLN_BWD_V2=0 fallback) is deliberately NOT changed: the same
# edit there was not bitwise at its pinned D=512 SUM_DX config (BLOCK_N=1,
# num_stages=3; 506 of 134M dx elements differ). Non-power-of-two D keeps
# `/ D`. Read once, cached. Arming line (rank log): "armed OK:
# wln_bwd_pow2=1" (train_ranker.py).
_WLN_BWD_POW2_ENABLED: Optional[bool] = None


def _wln_bwd_pow2_enabled() -> bool:
    global _WLN_BWD_POW2_ENABLED
    if _WLN_BWD_POW2_ENABLED is None:
        _WLN_BWD_POW2_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_WLN_BWD_POW2", "").strip().lower()
            in ("1", "true", "on", "yes")
        )
    return _WLN_BWD_POW2_ENABLED


def _wln_bwd_is_pow2_d(D: int, BLOCK_D: int) -> bool:
    """IS_POW2_D for a _weighted_layer_norm_bwd_dx_v2 launch (see above)."""
    return D == BLOCK_D and _wln_bwd_pow2_enabled()


def _wln_bwd_v2_config(
    BLOCK_D: int, is_swish: bool, add_residual: bool, sum_dx: bool
) -> Tuple[int, int, int, bool, int, bool]:
    """(tile multiplier, BLOCK_N, NUM_STAGES, CONTIG, num_warps, FAST_SIGMOID)
    for v2; the grid is min(sms * multiplier, cdiv(N, 4 * BLOCK_N)).

    Picked from interleaved MI350X sweeps at the three production shapes
    (N~2.36M; probes/iter9/wln/sweep*_{A,B,C}.json):
      D=512 residual (no swish):  sms*8,  BLOCK_N=1, 2 stages
      D=512 SUM_DX (no swish):    sms*4,  BLOCK_N=4, 2 stages
      D=256 swish SUM_DX:         sms*32, BLOCK_N=1, 1 stage
    The non-swish non-SUM_DX config is shared by the residual and plain
    calls so a fused residual never changes dw/db. CONTIG=True and
    num_warps>1 lost everywhere at these shapes. Rows wider than 512 keep
    BLOCK_N=1 (register pressure) and add warps only past 2048 columns.
    """
    if BLOCK_D > 512:
        return (8, 1, 1, False, max(1, BLOCK_D // 2048), False)
    if is_swish:
        return (32, 1, 1, False, 1, _wln_bwd_v2_fast_sigmoid())
    if sum_dx:
        return (4, 4, 2, False, 1, False)
    return (8, 1, 2, False, 1, False)


# DLRM_HSTU_FP8_WLN_DXSUM_V3 (gin default on in yambda_5b.gin; absent = off; only consulted when v2 is on): the
# non-swish D=512 call of triton_weighted_layer_norm_bwd_dxsum (the LinearNorm
# D512 backward) runs _weighted_layer_norm_bwd_dx_dxs512 on sms*8 one-wave
# programs, BLOCK_N=4, instead of v2's sms*4 programs with a num_stages=2
# pipelined loop. v2's grid is one wave per SIMD, which made the call latency-
# and placement-sensitive (1.6-1.8 ms at N=2.32M vs ~1.39 ms for v3; L2-flushed
# MI350X microbench, probes/iter16/d512). "0" restores the v2 launch exactly.
_WLN_DXSUM_V3_ENABLED: Optional[bool] = None
_WLN_DXSUM_V3_REPORTED = False
# (tile multiplier of the CU count, BLOCK_N)
_WLN_DXSUM_V3_CFG: Tuple[int, int] = (8, 4)


def _wln_dxsum_v3_enabled() -> bool:
    global _WLN_DXSUM_V3_ENABLED
    if _WLN_DXSUM_V3_ENABLED is None:
        _WLN_DXSUM_V3_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_WLN_DXSUM_V3", "0") != "0"
        )
    return _WLN_DXSUM_V3_ENABLED


def _wln_dxsum_v3_partials(
    dx: torch.Tensor,
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    sms: int,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """v3 launch (non-swish, no residual, SUM_DX, D == BLOCK_D == 512): writes
    dx, returns (tile_num, dw, db, dxsum) [tile_num, D] fp32 partials for the
    unchanged _layer_norm_bwd_dwdbdx finalize. N > 0."""
    global _WLN_DXSUM_V3_REPORTED
    N, D = x.shape
    mult, block_n = _WLN_DXSUM_V3_CFG
    # >= 4 row blocks per program, as in _wln_bwd_dx_partials.
    tile_num = max(1, min(sms * mult, triton.cdiv(N, 4 * block_n)))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dxsum = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _weighted_layer_norm_bwd_dx_dxs512[(tile_num,)](
        dx,
        dy,
        _dweight,
        _dbias,
        _dxsum,
        x,
        weight,
        mean,
        rstd,
        dx.stride(0),
        dy.stride(0),
        x.stride(0),
        D,
        N,
        BLOCK_D=D,
        BLOCK_N=block_n,
        num_warps=1,
    )
    if not _WLN_DXSUM_V3_REPORTED:
        _WLN_DXSUM_V3_REPORTED = True
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(
            "armed OK: wln dxsum v3 ACTIVE -- LinearNorm D512 backward runs "
            "_weighted_layer_norm_bwd_dx_dxs512 (DLRM_HSTU_FP8_WLN_DXSUM_V3)"
        )
    return tile_num, _dweight, _dbias, _dxsum


def _wln_bwd_dx_partials(
    dx: torch.Tensor,
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    is_swish: bool,
    add_residual: bool,
    sum_dx: bool,
    BLOCK_D: int,
    sms: int,
    dropout_mask: Optional[torch.Tensor] = None,
    dropout_scale: float = 1.0,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """v2 launch: writes dx, returns (tile_num, dw, db, dxsum) [tile_num, D]
    fp32 partials for the unchanged finalize kernels. When not sum_dx, the
    dxsum slot aliases the db partial (as the v1 call sites do). N > 0.
    dropout_mask (bool [N, D], or uint8 [N, D / 8] packed bits) applies the
    folded dropout backward to dx."""
    N, D = x.shape
    mult, block_n, num_stages, contig, num_warps, fast_sigmoid = _wln_bwd_v2_config(
        BLOCK_D, is_swish, add_residual, sum_dx
    )
    # >= ~4 row blocks per program, as v1's N // 4 cap does for BLOCK_N=1.
    tile_num = max(1, min(sms * mult, triton.cdiv(N, 4 * block_n)))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    if sum_dx:
        _dxsum = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    else:
        _dxsum = _dbias
    # The fold args are only passed when folding, so the unfolded launch is
    # the same specialization it always was (MASK_BITS: only for a packed mask).
    fold_kwargs = (
        {}
        if dropout_mask is None
        else dict(
            DMASK=dropout_mask.view(torch.uint8),
            stride_mask=dropout_mask.stride(0),
            mask_scale=dropout_scale,
            MASK_SCALE=True,
        )
    )
    if dropout_mask is not None and dropout_mask.dtype == torch.uint8:
        fold_kwargs["MASK_BITS"] = True
    if (
        is_swish
        and D == 256
        and BLOCK_D == 256
        and not add_residual
        and dropout_mask is None
        and (block_n, num_stages, contig, num_warps) == (1, 1, False, 1)
        and dx.stride(1) == 1
        and dy.stride(1) == 1
        and x.stride(1) == 1
        and _d256_v3_enabled()
    ):
        # DLRM_HSTU_FP8_D256_V3: the same launch with D constexpr
        # (bit-identical dx and partials).
        _weighted_layer_norm_bwd_dx_v3[(tile_num,)](
            dx,
            dy,
            _dweight,
            _dbias,
            _dxsum,
            x,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            N,
            256,
            SUM_DX=sum_dx,
            FAST_SIGMOID=fast_sigmoid,
            num_warps=1,
        )
        _d256_v3_note("bwd")
        return tile_num, _dweight, _dbias, _dxsum
    _weighted_layer_norm_bwd_dx_v2[(tile_num,)](
        dx,
        dy,
        residual,
        _dweight,
        _dbias,
        _dxsum,
        x,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        dy.stride(0),
        residual.stride(0) if add_residual else 0,
        x.stride(0),
        D,
        eps,
        IS_SWISH=is_swish,
        ADD_RESIDUAL=add_residual,
        SUM_DX=sum_dx,
        N=N,
        BLOCK_D=BLOCK_D,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        CONTIG=contig,
        EVEN_D=BLOCK_D == D,
        FAST_SIGMOID=fast_sigmoid,
        IS_POW2_D=_wln_bwd_is_pow2_d(D, BLOCK_D),
        num_warps=num_warps,
        **fold_kwargs,
    )
    return tile_num, _dweight, _dbias, _dxsum


# DLRM_HSTU_FP8_WLN_BWD_FLYDSL (GitHub PR #7; absent or "0" = off): the
# weighted LayerNorm backward of the four variants the step runs -- the STU
# norms (D=512, residual), the targets-only STU norm of item (20e) (D=512, no
# residual) and the preprocessor MLP norms (D=512, and the D=256
# SwishLayerNorm, both with the dx sum) -- runs as the FlyDSL kernel in
# ops/flydsl/ln_bwd.py, ahead of the Triton kernel (v1 or (14)'s v2), so (14)'s
# v2 and fast-sigmoid switches do not reach these calls. It covers bf16 x, dy
# and residual, fp32 mean and rstd, D of 256 or 512, consistent shapes and
# tensors under 4 GiB, on gfx950; every other call stays on Triton. Each
# (variant, weight dtype, path) logs once, so a run's log shows which path each
# variant took. A FlyDSL import, compile or launch error latches the path off
# for the process (_WLN_BWD_FLYDSL_BROKEN, one "[ln-bwd] flydsl DISABLED" line)
# and every later call takes Triton. Not an FP8 knob; the prefix is what
# launch_slurm.sh forwards.
_WLN_BWD_FLYDSL_ENABLED: Optional[bool] = None
_WLN_BWD_FLYDSL_BROKEN: Optional[str] = None
_WLN_BWD_FLYDSL_LOGGED: set = set()


def _wln_bwd_flydsl_enabled() -> bool:
    global _WLN_BWD_FLYDSL_ENABLED
    if _WLN_BWD_FLYDSL_ENABLED is None:
        _WLN_BWD_FLYDSL_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_WLN_BWD_FLYDSL", "0") != "0"
        )
    return _WLN_BWD_FLYDSL_ENABLED


# DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE (absent or "0" = off; needs
# DLRM_HSTU_FP8_WLN_BWD_FLYDSL on as well): the layer-0 input-LayerNorm
# backward that item (20)(d) DROPOUT_MSCALE_FOLD folds the input dropout's
# backward into (_triton_weighted_layer_norm_bwd_mscale_impl: residual, bool
# mask * scale epilogue) runs as the FlyDSL residual kernel with the same
# epilogue (ops/flydsl/ln_bwd.py, mscale) instead of the Triton v2 mscale
# kernel; see _wln_bwd_flydsl_mscale. That covers both mask layouts: aten's
# bool [N, D] mask and the bit-packed uint8 [N, D / 8] mask the fused
# posenc+dropout forward writes under DLRM_HSTU_FP8_DROPOUT_MASK_BITS (R-C7;
# FlyDSL kernel ..._mscaleb_..., bitwise the bool call). The packed mask is
# only ever written where the Triton bits fold can take it
# (wln_bwd_mscale_bits_ok), so every FlyDSL decline still has a fold. Not an
# FP8 knob; the prefix is what launch_slurm.sh forwards.
_WLN_BWD_FLYDSL_MSCALE_ENABLED: Optional[bool] = None


def _wln_bwd_flydsl_mscale_enabled() -> bool:
    global _WLN_BWD_FLYDSL_MSCALE_ENABLED
    if _WLN_BWD_FLYDSL_MSCALE_ENABLED is None:
        _WLN_BWD_FLYDSL_MSCALE_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE", "0") != "0"
        )
    return _WLN_BWD_FLYDSL_MSCALE_ENABLED


def _wln_bwd_flydsl_disable(where: str, error: BaseException) -> None:
    """Latch the FlyDSL LayerNorm backward off for the rest of the process
    (the first error wins) and say so once, loudly: one WARNING record whose
    message holds the first line of the error and whose traceback holds the
    rest. Called on a FlyDSL import, compile or launch error by the dispatch
    below and by the startup prebuild (dlrm_v4/train/blas_warmup.py); every
    later call takes Triton."""
    global _WLN_BWD_FLYDSL_BROKEN
    if _WLN_BWD_FLYDSL_BROKEN is not None:
        return
    lines = str(error).strip().splitlines()
    _WLN_BWD_FLYDSL_BROKEN = (
        f"{where}: {type(error).__name__}: {lines[0][:200] if lines else ''}"
    )
    import logging

    logging.getLogger(__name__).warning(
        "[ln-bwd] flydsl DISABLED (%s); all calls fall back to Triton",
        _WLN_BWD_FLYDSL_BROKEN,
        exc_info=error,
    )


_WLN_BWD_FLYDSL_UNAVAILABLE = "flydsl unavailable"


def _wln_bwd_flydsl_reason(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: Optional[torch.Tensor],
    is_swish: bool,
    sum_dx: bool,
    latch: bool = True,
) -> Optional[str]:
    """None when the FlyDSL kernel covers this call, else the stable fallback
    reason the dispatch logs (never N). _WLN_BWD_FLYDSL_UNAVAILABLE means the
    FlyDSL module could not answer (import or offsets_fit error); with latch
    (the dispatch) that error also latches the path off, without (the
    side-effect-free predicates below) it does not. Assumes the switch is on
    and not latched off."""
    N, D = x.shape
    arch = getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "")
    # Every reason below is stable text (never N), so each fallback logs once.
    why = None
    if not arch.startswith("gfx950"):
        why = f"arch {arch or 'unknown'}"
    elif D not in (256, 512):
        why = f"D={D}"
    elif x.dtype != torch.bfloat16 or dy.dtype != torch.bfloat16:
        why = f"dtype {x.dtype}/{dy.dtype}"
    elif residual is not None and residual.dtype != torch.bfloat16:
        why = f"residual {residual.dtype}"
    elif (
        weight.dtype not in (torch.bfloat16, torch.float32)
        or bias is None
        or bias.dtype != weight.dtype
        or weight.numel() != D
        or bias.numel() != D
    ):
        why = f"weight {weight.dtype}"
    elif mean.dtype != torch.float32 or rstd.dtype != torch.float32:
        why = f"stats {mean.dtype}"
    elif dy.shape != x.shape:
        why = "dy shape"
    elif residual is not None and residual.shape != x.shape:
        why = "residual shape"
    elif mean.numel() != N or rstd.numel() != N:
        why = "stats shape"
    elif not (
        x.is_contiguous()
        and dy.is_contiguous()
        and (residual is None or residual.is_contiguous())
        and mean.is_contiguous()
        and rstd.is_contiguous()
    ):
        why = "layout"
    elif N == 0:
        why = "N=0"
    elif N * D * 2 >= 2**32:
        why = "tensor >= 4 GiB"
    if why is None:
        # The persistent grid addresses ceil(N / (G*rows)) * G*rows rows; bound
        # THAT, not N, by 32-bit offsets (ops/flydsl/ln_bwd.py offsets_fit).
        try:
            from generative_recommenders.ops.flydsl.ln_bwd import offsets_fit

            fits = offsets_fit(N, D, x.device, is_swish, residual is not None, sum_dx)
        except Exception as error:
            if latch:
                _wln_bwd_flydsl_disable("dispatch", error)
            return _WLN_BWD_FLYDSL_UNAVAILABLE
        if not fits:
            why = "padded rows past 32-bit offsets"
    return why


def flydsl_ln_bwd_would_take(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    is_swish: bool = False,
    sum_dx: bool = False,
) -> bool:
    """True when the weighted-LayerNorm backward dispatch (PR #7 path, switch
    DLRM_HSTU_FP8_WLN_BWD_FLYDSL) would run this call on FlyDSL. For callers that
    batch several calls (e.g. actckpt's grouped preprocessor-tail backward) and
    must defer to the FlyDSL path when it owns the call. Launches nothing, logs
    nothing and never latches; same checks as the dispatch."""
    return (
        flydsl_ln_bwd_decline_reason(
            x, dy, weight, bias, mean, rstd, residual, is_swish=is_swish, sum_dx=sum_dx
        )
        is None
    )


def flydsl_ln_bwd_decline_reason(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    is_swish: bool = False,
    sum_dx: bool = False,
) -> Optional[str]:
    """flydsl_ln_bwd_would_take with the reason: None when the dispatch would
    run this call on FlyDSL, else why not (stable text, never N). Launches
    nothing, logs nothing and never latches."""
    if not _wln_bwd_flydsl_enabled():
        return "DLRM_HSTU_FP8_WLN_BWD_FLYDSL is off"
    if _WLN_BWD_FLYDSL_BROKEN is not None:
        return "the FlyDSL LayerNorm backward is latched off"
    return _wln_bwd_flydsl_reason(
        x, dy, weight, bias, mean, rstd, residual, is_swish, sum_dx, latch=False
    )


def _wln_bwd_flydsl(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: Optional[torch.Tensor],
    is_swish: bool,
    sum_dx: bool,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """The FlyDSL path's outputs -- (dx, dweight, dbias) or, with sum_dx,
    (dx, dweight, dbias, dxsum) -- or None when the switch is off, the path is
    latched off, or the call is outside what the kernel covers; the caller then
    runs Triton."""
    if not _wln_bwd_flydsl_enabled() or _WLN_BWD_FLYDSL_BROKEN is not None:
        return None
    why = _wln_bwd_flydsl_reason(
        x, dy, weight, bias, mean, rstd, residual, is_swish, sum_dx
    )
    if why is _WLN_BWD_FLYDSL_UNAVAILABLE:
        return None
    N, D = x.shape
    key = (D, bool(is_swish), residual is not None, bool(sum_dx), weight.dtype, why)
    if key not in _WLN_BWD_FLYDSL_LOGGED:
        _WLN_BWD_FLYDSL_LOGGED.add(key)
        import logging

        opts = ""
        if why is None:
            # the kernel's bitwise-neutral options (ops/flydsl/ln_bwd.py
            # kernel_options: XCD block-order run length, load/store cache
            # policy; DLRM_HSTU_FP8_WLN_BWD_FLYDSL_XCD_REMAP / _STREAM_CACHE)
            try:
                from generative_recommenders.ops.flydsl.ln_bwd import kernel_options

                remap, c_ld, c_st = kernel_options(
                    D, bool(is_swish), residual is not None, bool(sum_dx)
                )
            except Exception as error:
                _wln_bwd_flydsl_disable("dispatch", error)
                return None
            opts = f" remap={remap} cache={c_ld}/{c_st}"
        logging.getLogger(__name__).warning(
            "[ln-bwd] %s: D=%d swish=%d residual=%d dxsum=%d%s weight=%s%s",
            "triton" if why else "flydsl",
            D,
            int(is_swish),
            int(residual is not None),
            int(sum_dx),
            f" ({why})" if why else "",
            weight.dtype,
            opts,
        )
    if why:
        return None
    try:
        from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd

        return ln_bwd(
            dy, x, weight, bias, mean, rstd, residual,
            is_swish=bool(is_swish), sum_dx=bool(sum_dx),
        )
    except Exception as error:
        # A host-side import, JIT-compile or launch error is raised before or
        # between this call's launches, and Triton writes fresh outputs, so the
        # fallback is safe. (An asynchronous device fault is not raised here;
        # it surfaces at the next synchronize.)
        _wln_bwd_flydsl_disable("dispatch", error)
        return None


@maybe_register_custom_op(
    "generative_recommenders::triton_weighted_layer_norm_bwd", mutates_args=()
)
def _triton_weighted_layer_norm_bwd_impl(
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    add_residual: bool,
    eps: float,
    BLOCK_D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    if learnable:
        fly = _wln_bwd_flydsl(
            x, dy, weight, bias, mean, rstd,
            residual if add_residual else None, False, False,
        )
        if fly is not None:
            return fly
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        use_v2 = _wln_bwd_v2_enabled()
        if not use_v2:
            tile_num = _ln_bwd_tile_num(N, D, sms, is_swish=False)
            _dweight = torch.empty(
                (tile_num, D), dtype=torch.float32, device=x.device
            )
            _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        # F1 (bias-grad fold) is not plumbed through this entry point yet; the
        # kernel gate stays off so the compiled control path is unchanged.
        sum_dx = False
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias
        if use_v2:
            tile_num, _dweight, _dbias, _ = _wln_bwd_dx_partials(
                dx,
                dy,
                residual,
                x,
                weight,
                bias,
                mean,
                rstd,
                eps,
                is_swish=False,
                add_residual=add_residual,
                sum_dx=sum_dx,
                BLOCK_D=BLOCK_D,
                sms=sms,
            )
        else:
            _dxsum = _dbias
            # pyre-ignore[28]
            _weighted_layer_norm_bwd_dx[(tile_num,)](
                dx,
                dy,
                residual,
                _dweight,
                _dbias,
                _dxsum,
                x,
                weight,
                bias,
                mean,
                rstd,
                dx.stride(0),
                dy.stride(0),
                residual.stride(0) if add_residual else 0,
                x.stride(0),
                D,
                eps,
                IS_SWISH=False,
                ADD_RESIDUAL=add_residual,
                SUM_DX=sum_dx,
                N=N,
                BLOCK_D=BLOCK_D,
            )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](
            _dweight,
            _dbias,
            dweight,
            dbias,
            tile_num,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, dbias
    else:
        N, D = x.shape
        dx = torch.empty_like(x)
        # Return empty tensors as sentinels for None
        dweight = torch.empty(0, dtype=x.dtype, device=x.device)
        dbias = torch.empty(0, dtype=x.dtype, device=x.device)
        if N == 0:
            return dx, dweight, dbias
        # pyre-ignore[28]
        _layer_norm_bwd_dx[(N,)](
            dx,
            dy,
            x,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            eps,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
        return dx, dweight, dbias


@_triton_weighted_layer_norm_bwd_impl.register_fake
def _(
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    add_residual: bool,
    eps: float,
    BLOCK_D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    if learnable:
        D = x.shape[-1]
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    else:
        dweight = torch.empty(0, dtype=x.dtype, device=x.device)
        dbias = torch.empty(0, dtype=x.dtype, device=x.device)
    return dx, dweight, dbias


def _wln_bwd_flydsl_mscale(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: torch.Tensor,
    dropout_mask: torch.Tensor,
    dropout_scale: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The FlyDSL path of the residual LN backward with the dropout mask*scale
    epilogue (DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE) -- (dx, dweight, dbias) --
    or None when a switch is off, the FlyDSL path is latched off, or the call
    is outside what the kernel covers; the caller then runs Triton. dx equals
    native_dropout_backward of the FlyDSL residual call's dx bit for bit and
    dW/dB equal that call's (so vs Triton: the residual variant's rare 1-ulp
    dx differences, masked, and its fp32 dW/dB order). The checks are
    _wln_bwd_flydsl's for the residual variant plus the mask layout; the same
    stable reasons, the same latch, and the same once-per-key log line with
    " mscale=1". dropout_mask is aten's bool [N, D] mask or the bit-packed
    uint8 [N, D / 8] one (DLRM_HSTU_FP8_DROPOUT_MASK_BITS; " mscale=1 bits=1",
    kernel ..._mscaleb_...): the outputs are bitwise the bool call's with the
    unpacked mask. Any decline (e.g. a strided packed mask) leaves the call
    to the Triton fold, which takes either layout."""
    if (
        not _wln_bwd_flydsl_enabled()
        or not _wln_bwd_flydsl_mscale_enabled()
        or _WLN_BWD_FLYDSL_BROKEN is not None
    ):
        return None
    N, D = x.shape
    bits = dropout_mask.dtype == torch.uint8
    arch = getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "")
    why = None
    if not arch.startswith("gfx950"):
        why = f"arch {arch or 'unknown'}"
    elif D not in (256, 512):
        why = f"D={D}"
    elif x.dtype != torch.bfloat16 or dy.dtype != torch.bfloat16:
        why = f"dtype {x.dtype}/{dy.dtype}"
    elif residual.dtype != torch.bfloat16:
        why = f"residual {residual.dtype}"
    elif (
        weight.dtype not in (torch.bfloat16, torch.float32)
        or bias is None
        or bias.dtype != weight.dtype
        or weight.numel() != D
        or bias.numel() != D
    ):
        why = f"weight {weight.dtype}"
    elif mean.dtype != torch.float32 or rstd.dtype != torch.float32:
        why = f"stats {mean.dtype}"
    elif dy.shape != x.shape:
        why = "dy shape"
    elif residual.shape != x.shape:
        why = "residual shape"
    elif mean.numel() != N or rstd.numel() != N:
        why = "stats shape"
    elif not (
        x.is_contiguous()
        and dy.is_contiguous()
        and residual.is_contiguous()
        and mean.is_contiguous()
        and rstd.is_contiguous()
    ):
        why = "layout"
    elif (
        (
            D % 8 != 0 or dropout_mask.shape != (N, D // 8)
            if bits
            else dropout_mask.dtype != torch.bool or dropout_mask.shape != x.shape
        )
        or not dropout_mask.is_contiguous()
        or dropout_mask.device != x.device
    ):
        why = "mask layout"
    elif N == 0:
        why = "N=0"
    elif N * D * 2 >= 2**32:
        why = "tensor >= 4 GiB"
    if why is None:
        try:
            from generative_recommenders.ops.flydsl.ln_bwd import offsets_fit

            fits = offsets_fit(N, D, x.device, False, True, False)
        except Exception as error:
            _wln_bwd_flydsl_disable("dispatch", error)
            return None
        if not fits:
            why = "padded rows past 32-bit offsets"
    key = (D, False, True, False, "mscale_bits" if bits else "mscale", weight.dtype, why)
    if key not in _WLN_BWD_FLYDSL_LOGGED:
        _WLN_BWD_FLYDSL_LOGGED.add(key)
        import logging

        opts = ""
        if why is None:
            try:
                from generative_recommenders.ops.flydsl.ln_bwd import kernel_options

                remap, c_ld, c_st = kernel_options(D, False, True, False)
            except Exception as error:
                _wln_bwd_flydsl_disable("dispatch", error)
                return None
            opts = f" remap={remap} cache={c_ld}/{c_st}"
        logging.getLogger(__name__).warning(
            "[ln-bwd] %s: D=%d swish=0 residual=1 dxsum=0 mscale=1%s%s weight=%s%s",
            "triton" if why else "flydsl",
            D,
            " bits=1" if bits else "",
            f" ({why})" if why else "",
            weight.dtype,
            opts,
        )
    if why:
        return None
    try:
        from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd

        # a uint8 mask selects the packed-bits kernel (ln_bwd's mscale bits)
        return ln_bwd(
            dy, x, weight, bias, mean, rstd, residual,
            dropout_mask=dropout_mask, dropout_scale=float(dropout_scale),
        )
    except Exception as error:
        # as _wln_bwd_flydsl: a host-side error is raised before or between
        # this call's launches and Triton writes fresh outputs
        _wln_bwd_flydsl_disable("dispatch", error)
        return None


@maybe_register_custom_op(
    "generative_recommenders::triton_weighted_layer_norm_bwd_mscale",
    mutates_args=(),
)
def _triton_weighted_layer_norm_bwd_mscale_impl(
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dropout_mask: torch.Tensor,
    dropout_scale: float,
    eps: float,
    BLOCK_D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Learnable, residual, v2-kernel weighted LN backward whose dx is then
    passed through the following dropout's backward (mask * scale) in the
    kernel epilogue. Identical to _triton_weighted_layer_norm_bwd_impl's
    learnable/v2/residual path otherwise (same config, grid and finalize).
    With DLRM_HSTU_FP8_WLN_BWD_FLYDSL and DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE
    on, the FlyDSL residual kernel with the same epilogue runs first, for the
    bool and the bit-packed (DLRM_HSTU_FP8_DROPOUT_MASK_BITS) mask alike."""
    fly = _wln_bwd_flydsl_mscale(
        x, dy, weight, bias, mean, rstd, residual, dropout_mask, dropout_scale
    )
    if fly is not None:
        return fly
    N, D = x.shape
    dx = torch.empty_like(x)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    if N == 0:
        dweight.zero_()
        dbias.zero_()
        return dx, dweight, dbias
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num, _dweight, _dbias, _ = _wln_bwd_dx_partials(
        dx,
        dy,
        residual,
        x,
        weight,
        bias,
        mean,
        rstd,
        eps,
        is_swish=False,
        add_residual=True,
        sum_dx=False,
        BLOCK_D=BLOCK_D,
        sms=sms,
        dropout_mask=dropout_mask,
        dropout_scale=dropout_scale,
    )

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    FIN_BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    FIN_BLOCK_D = min(max(FIN_BLOCK_D, 4), 128)
    _layer_norm_bwd_dwdb[grid](
        _dweight,
        _dbias,
        dweight,
        dbias,
        tile_num,
        D,
        BLOCK_D=FIN_BLOCK_D,
    )
    return dx, dweight, dbias


@_triton_weighted_layer_norm_bwd_mscale_impl.register_fake
def _(
    dy: torch.Tensor,
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dropout_mask: torch.Tensor,
    dropout_scale: float,
    eps: float,
    BLOCK_D: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D = x.shape[-1]
    return (
        torch.empty_like(x),
        torch.empty((D,), dtype=weight.dtype, device=x.device),
        torch.empty((D,), dtype=weight.dtype, device=x.device),
    )


# DLRM_HSTU_FP8_DROPOUT_MASK_BITS: the packed-mask fold runs only where its
# epilogue was validated bitwise against the bool one: BLOCK_D <= 512 with the
# (BLOCK_N, NUM_STAGES, CONTIG, num_warps) the mscale launch uses there. In that
# config the compiler keeps the row / column reduction layouts of the bool
# kernel (TTGIR checked; tripwire: posenc_concat_dropout_test.py
# test_mask_bits_ln_bwd_reduce_layouts). Elsewhere it may not (BLOCK_D 1024,
# 1 stage: about 1 dx element in 10^6 differs), so the producer keeps the bool
# mask there (wln_bwd_mscale_bits_ok). Re-validate after a Triton or config
# change (probes/iter67/rc7: validate_rc7.py kmat).
_MASK_BITS_FOLD_MAX_BLOCK_D = 512
_MASK_BITS_FOLD_CFG = (1, 2, False, 1)


def wln_bwd_mscale_bits_why_not(D: int) -> str:
    """Empty when the mask/scale fold takes the packed mask (uint8 [N, D / 8],
    DLRM_HSTU_FP8_DROPOUT_MASK_BITS) for a bf16 [N, D] input, else the reason.
    Tensor-free, so the producer (the fused posenc+dropout forward) decides
    the mask layout before this backward runs: it packs only when this holds,
    and a fold that declines never has to unpack. wln_bwd_mscale_supported's
    packed branch applies the same predicate."""
    if not _wln_bwd_v2_enabled():
        return "the Triton v2 LN backward is off (DLRM_HSTU_FP8_WLN_BWD_V2)"
    if D % 8 != 0:
        return f"D={D} is not a multiple of 8"
    BLOCK_D = min(65536 // 2, triton.next_power_of_2(D))  # compute_BLOCK_D, bf16
    if not 8 <= BLOCK_D <= _MASK_BITS_FOLD_MAX_BLOCK_D:
        return f"BLOCK_D={BLOCK_D} is outside the validated 8..512"
    cfg = _wln_bwd_v2_config(BLOCK_D, False, True, False)[1:5]
    if cfg != _MASK_BITS_FOLD_CFG:
        return f"launch config {cfg} != the validated {_MASK_BITS_FOLD_CFG}"
    return ""


def wln_bwd_mscale_bits_ok(D: int) -> bool:
    return not wln_bwd_mscale_bits_why_not(D)


def wln_bwd_mscale_supported(
    x: torch.Tensor, dropout_mask: torch.Tensor
) -> bool:
    """Whether the dropout mask/scale fold can run for this LN backward.
    dropout_mask: bool [N, D], or (DLRM_HSTU_FP8_DROPOUT_MASK_BITS) uint8
    [N, D / 8] packed bits, bit k of byte j of a row = column 8 * j + k
    (input_dropout_fold.pack_dropout_mask_bits), only where
    wln_bwd_mscale_bits_ok(D) holds (D % 8 == 0, the validated launch
    config)."""
    if dropout_mask.dtype == torch.uint8:
        mask_ok = (
            x.dim() == 2
            and wln_bwd_mscale_bits_ok(x.shape[1])
            and dropout_mask.shape == (x.shape[0], x.shape[1] // 8)
        )
    else:
        mask_ok = dropout_mask.dtype == torch.bool and dropout_mask.shape == x.shape
    return (
        _wln_bwd_v2_enabled()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and mask_ok
        and dropout_mask.stride(1) == 1
        and x.stride(1) == 1
        and dropout_mask.device == x.device
    )


def triton_weighted_layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    eps: float,
    BLOCK_D: int,
    residual: Optional[torch.Tensor] = None,
    dropout_mask: Optional[torch.Tensor] = None,
    dropout_scale: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if dropout_mask is not None:
        # Caller checked wln_bwd_mscale_supported(); the fold is only defined
        # for the learnable residual call.
        assert learnable and residual is not None
        assert weight is not None and bias is not None
        return _triton_weighted_layer_norm_bwd_mscale_impl(
            dy=dy,
            residual=residual,
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            dropout_mask=dropout_mask,
            dropout_scale=dropout_scale,
            eps=eps,
            BLOCK_D=BLOCK_D,
        )
    # Use sentinel tensors for custom_op compatibility (can't return Optional[Tensor])
    _weight = (
        weight if weight is not None else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    _bias = bias if bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)
    _residual = (
        residual
        if residual is not None
        else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    dx, dweight, dbias = _triton_weighted_layer_norm_bwd_impl(
        dy=dy,
        residual=_residual,
        x=x,
        weight=_weight,
        bias=_bias,
        mean=mean,
        rstd=rstd,
        learnable=learnable,
        add_residual=residual is not None,
        eps=eps,
        BLOCK_D=BLOCK_D,
    )
    if not learnable:
        return dx, None, None
    return dx, dweight, dbias


def _dwdbdx_pinned_128x8() -> bool:
    """True iff _layer_norm_bwd_dwdbdx can only run its pinned config
    (BLOCK_N=128, num_warps=8), the one _layer_norm_bwd_dwdbdx_p1/_p2
    reproduce bit for bit (not under TRITON_FULL_AUTOTUNE=1)."""
    cfgs = getattr(_layer_norm_bwd_dwdbdx, "configs", None)
    return (
        cfgs is not None
        and len(cfgs) == 1
        and cfgs[0].kwargs.get("BLOCK_N") == 128
        and cfgs[0].num_warps == 8
    )


def _wln_dxsum_triton_path(D: int, BLOCK_D: int, is_swish: bool) -> str:
    """The Triton dx kernel triton_weighted_layer_norm_bwd_dxsum launches for
    N > 0 once PR #7 (item (29)) has declined the call: "v3" (item (28)(b)
    DLRM_HSTU_FP8_WLN_DXSUM_V3, non-swish D == BLOCK_D == 512), "v2" (item
    (14) DLRM_HSTU_FP8_WLN_BWD_V2) or "v1". The dispatch and
    wln_bwd_dxsum_path() both use this, so they cannot disagree."""
    if not _wln_bwd_v2_enabled():
        return "v1"
    if not is_swish and D == 512 and BLOCK_D == D and _wln_dxsum_v3_enabled():
        return "v3"
    return "v2"


def wln_bwd_dxsum_path(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    is_swish: bool,
) -> str:
    """The path triton_weighted_layer_norm_bwd_dxsum(dy, x, weight, bias, mean,
    rstd, eps, BLOCK_D, is_swish) would take: "flydsl" (PR #7, item (29)),
    "empty" (N == 0), else _wln_dxsum_triton_path's "v3" / "v2" / "v1".
    Launches nothing, logs nothing and never latches (flydsl_ln_bwd_decline_
    reason). For callers that batch several calls into one kernel that is
    bit-identical to one of these paths only (actckpt E-M2's Triton grouped
    kernel: v2). A host-side FlyDSL error inside the real call would still
    move that call to Triton; a batching caller must handle that itself."""
    if (
        flydsl_ln_bwd_decline_reason(
            x, dy, weight, bias, mean, rstd, None, is_swish=is_swish, sum_dx=True
        )
        is None
    ):
        return "flydsl"
    N, D = x.shape
    if N == 0:
        return "empty"
    return _wln_dxsum_triton_path(D, BLOCK_D, is_swish)


def triton_weighted_layer_norm_bwd_dxsum(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float,
    BLOCK_D: int,
    is_swish: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """F1: the norm backward, additionally returning sum(dx, axis=0).

    A SEPARATE entry point from triton_weighted_layer_norm_bwd, deliberately.
    That function returns a 3-tuple and is unpacked at six call sites
    (layer_norm_test.py:56,67; triton_hstu_preprocess_and_attention.py:855,1384;
    and twice in this file). Widening its arity would touch all six to buy
    nothing for the five that do not want the extra value, so the fused path
    gets its own 4-tuple and the shipped path is left exactly as it was.

    WHY THIS EXISTS. The Linear feeding this norm has its bias gradient
    computed by ATen as a standalone aten::sum over dx -- 2.3425 ms/rank/step at
    D=512 plus 1.6183 at D=256, 3.9608 total, measured over 8 ranks x 5 steps.
    That sum re-reads the whole [N, D] dx from HBM immediately after the norm
    backward wrote it. The column sum is available for free while dx is still
    in registers, so the fused path computes it there and deletes the reread.
    Both are HBM-bound, which is why the removal is worth more than the add:
    the standalone sums run at 3.07 TB/s (D=512) and 2.22 TB/s (D=256), well
    under the 5.84 TB/s the part can do, and a reduction that never touches
    HBM beats one that does at any bandwidth.

    NUMERICS: dxsum reduces the fp32 dx still in registers, whereas aten::sum
    reduces the bf16 dx after it was stored. This is the MORE accurate of the
    two, but it is not bit-identical -- measured relative difference 1.6-2.1e-3,
    the bf16 rounding scale. That is why this arm owes a convergence gate and
    not merely a timing gate.
    """
    fly = _wln_bwd_flydsl(x, dy, weight, bias, mean, rstd, None, is_swish, True)
    if fly is not None:
        return fly
    N, D = x.shape
    dx = torch.empty_like(x)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    # dxsum is the preceding Linear's bias gradient, so it is fp32 to match what
    # autograd would otherwise hand that parameter from aten::sum.
    dxsum = torch.empty((D,), dtype=torch.float32, device=x.device)
    if N == 0:
        dweight.zero_()
        dbias.zero_()
        dxsum.zero_()
        return dx, dweight, dbias, dxsum

    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    # The Triton dx kernel; wln_bwd_dxsum_path() asks the same helper.
    triton_path = _wln_dxsum_triton_path(D, BLOCK_D, is_swish)
    if triton_path == "v3":
        tile_num, _dweight, _dbias, _dxsum = _wln_dxsum_v3_partials(
            dx, dy, x, weight, mean, rstd, sms
        )
    elif triton_path == "v2":
        tile_num, _dweight, _dbias, _dxsum = _wln_bwd_dx_partials(
            dx,
            dy,
            dx,
            x,
            weight,
            bias,
            mean,
            rstd,
            eps,
            is_swish=is_swish,
            add_residual=False,
            sum_dx=True,
            BLOCK_D=BLOCK_D,
            sms=sms,
        )
    else:
        tile_num = _ln_bwd_tile_num(N, D, sms, is_swish)
        _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        _dxsum = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)

        # pyre-ignore[28]
        _weighted_layer_norm_bwd_dx[(tile_num,)](
            dx,
            dy,
            dx,
            _dweight,
            _dbias,
            _dxsum,
            x,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            0,
            x.stride(0),
            D,
            eps,
            IS_SWISH=is_swish,
            ADD_RESIDUAL=False,
            SUM_DX=True,
            N=N,
            BLOCK_D=BLOCK_D,
        )

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    FINAL_BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    FINAL_BLOCK_D = min(max(FINAL_BLOCK_D, 4), 128)
    if (
        D == 256
        and is_swish
        and _wln_bwd_v2_enabled()
        and _d256_v3_enabled()
        and _dwdbdx_pinned_128x8()
    ):
        # DLRM_HSTU_FP8_D256_V3: bit-exact two-phase finalize. Only with the
        # v2 partials (item (14)), so DLRM_HSTU_FP8_WLN_BWD_V2=0 restores the
        # trunk kernel list.
        nb = 128
        acc = torch.empty((3, nb, D), dtype=torch.float32, device=x.device)
        _layer_norm_bwd_dwdbdx_p1[(nb, triton.cdiv(D, 64))](
            _dweight, _dbias, _dxsum, acc, tile_num, D, NB=nb, BLOCK_C=64,
            num_warps=1,
        )
        _layer_norm_bwd_dwdbdx_p2[(triton.cdiv(D, FINAL_BLOCK_D),)](
            acc, dweight, dbias, dxsum, D, NB=nb, BLOCK_D=FINAL_BLOCK_D,
            num_warps=8,
        )
        _d256_v3_note("fin")
        return dx, dweight, dbias, dxsum
    _layer_norm_bwd_dwdbdx[grid](
        _dweight,
        _dbias,
        _dxsum,
        dweight,
        dbias,
        dxsum,
        tile_num,
        D,
        BLOCK_D=FINAL_BLOCK_D,
    )
    return dx, dweight, dbias, dxsum


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        y, mean, rstd = triton_weighted_layer_norm_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        BLOCK_D = compute_BLOCK_D(x)
        learnable = weight is not None
        if learnable:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
        else:
            ctx.save_for_backward(x, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.eps = eps
        ctx.learnable = learnable
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None]:
        if ctx.learnable:
            x, weight, bias, mean, rstd = ctx.saved_tensors
        else:
            x, mean, rstd = ctx.saved_tensors
            weight, bias = None, None
        dx, dweight, dbias = triton_weighted_layer_norm_bwd(
            dy=dy,
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            learnable=ctx.learnable,
            eps=ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
        )
        return dx, dweight, dbias, None


def _get_rms_norm_fwd_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row RMSNorm kernels."""
    configs = []
    for BLOCK_N in [1, 4, 16]:
        for num_warps in [2, 4]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton.autotune(
    configs=_get_rms_norm_fwd_configs(),
    key=["BLOCK_D", "SILU"],
)
@triton.jit
def _weighted_rms_norm_fwd(
    X,
    Y,
    W,
    Rstd,
    N,
    D: tl.constexpr,
    eps,
    stride_x,
    stride_y,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Load weight once (shared across all rows in this block)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    # X / Y block pointers (shape (N, D), strides (stride, 1), offsets
    # (start_row, 0)) as pointer + mask, see _bp_ptrs (ROCm 10.2 port, T7).
    bp_n = _bp_i64(N)
    bp_d = _bp_i64(D)
    bp_sx = _bp_i64(stride_x)
    bp_row0 = _bp_i64(start_row)
    bp_sy = _bp_i64(stride_y)

    x_ptrs, x_offs = _bp_ptrs(X, bp_row0, bp_sx, BLOCK_N, BLOCK_D)
    x_block = tl.load(
        x_ptrs, mask=_bp_mask(x_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
    ).to(tl.float32)

    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    # Compute variance (RMS norm uses x directly, not x - mean)
    x_masked = tl.where(row_mask[:, None] & col_mask[None, :], x_block, 0.0)
    _var = x_masked * x_masked
    var = tl.sum(_var, axis=1) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, row_mask)

    # Normalize and apply linear transformation
    rstd = tl.expand_dims(rstd, 1)
    y = x_block * rstd
    y = y * w[None, :]

    if SILU:
        # pyre-ignore[16]: Module `triton.language.math` has no attribute `fast_dividef`
        y = fast_dividef(y, 1.0 + tl.exp(-y))

    y_out = y.to(Y.dtype.element_ty)
    y_ptrs, y_offs = _bp_ptrs(Y, bp_row0, bp_sy, BLOCK_N, BLOCK_D)
    tl.store(y_ptrs, y_out, mask=_bp_mask(y_offs, bp_n, bp_d, BLOCK_N, BLOCK_D))


@triton.jit
def _weighted_rms_norm_bwd_dx(
    DX,
    DY,
    DW,
    X,
    W,
    Rstd,
    Lock,
    stride_dx,
    stride_dy,
    stride_x,
    D: tl.constexpr,
    eps,
    GROUP_N,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    wdy = w * dy

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / D
    dx = (wdy - (xhat * c1)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    Lock += lock_id
    Count = Lock + GROUP_N
    DW = DW + lock_id * D + cols
    # Accumulate partial sums for dw/db
    partial_dw = dy * xhat
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton_autotune(
    configs=_get_norm_bwd_configs(),
    key=["BLOCK_D", "SILU"],
    reset_to_zero=["DW"],
)
@triton.jit
def _weighted_rms_norm_bwd(
    DX,
    DY,
    DW,
    X,
    W,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    N,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARDS_PER_SM: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    if pid < num_blocks % tile_num:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    start_block = pid

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)

    for idx in range(blocks_per_tile):
        current_block = start_block + idx * tile_num
        start_row = current_block * BLOCK_N

        # X / DX / DY block pointers (shape (N, D), strides (stride, 1),
        # offsets (start_row, 0)) as pointer + mask, see _bp_ptrs (ROCm 10.2
        # port, T7).
        bp_n = _bp_i64(N)
        bp_d = _bp_i64(D)
        bp_sx = _bp_i64(stride_x)
        bp_row0 = _bp_i64(start_row)
        bp_sdx = _bp_i64(stride_dx)
        bp_sdy = _bp_i64(stride_dy)

        # Load data blocks
        x_ptrs, x_offs = _bp_ptrs(X, bp_row0, bp_sx, BLOCK_N, BLOCK_D)
        x_block = tl.load(
            x_ptrs, mask=_bp_mask(x_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
        ).to(tl.float32)
        dy_ptrs, dy_offs = _bp_ptrs(DY, bp_row0, bp_sdy, BLOCK_N, BLOCK_D)
        dy_block = tl.load(
            dy_ptrs, mask=_bp_mask(dy_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
        ).to(tl.float32)

        # Load rstd for all rows in this block
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)

        # Expand dimensions for broadcasting
        rstd = tl.expand_dims(rstd, 1)

        # Compute dx
        xhat = x_block * rstd

        # Apply SILU backward if enabled
        if SILU:
            y_before_silu = xhat * w[None, :]
            # pyre-fixme[16]
            sig_y = fast_dividef(1.0, 1.0 + tl.exp(-y_before_silu))
            # SILU derivative: sigmoid(y) + y * sigmoid(y) * (1 - sigmoid(y))
            dy_block = dy_block * (sig_y + y_before_silu * sig_y * (1.0 - sig_y))

        wdy = w[None, :] * dy_block

        c1 = tl.sum(xhat * wdy, axis=1) / D
        c1 = tl.expand_dims(c1, 1)
        dx = (wdy - (xhat * c1)) * rstd

        # Write dx
        dx_out = dx.to(DX.dtype.element_ty)
        dx_ptrs, dx_offs = _bp_ptrs(DX, bp_row0, bp_sdx, BLOCK_N, BLOCK_D)
        tl.store(dx_ptrs, dx_out, mask=_bp_mask(dx_offs, bp_n, bp_d, BLOCK_N, BLOCK_D))

        # Accumulate partial sums for dw
        # Compute dw for all rows, then sum locally before atomic operation
        partial_dw_block = dy_block * xhat
        # Local reduction: sum across all rows in this block
        partial_dw = tl.sum(partial_dw_block, axis=0)
        acc_dw += partial_dw

    DW_ptr = DW + cols
    tl.atomic_add(DW_ptr, acc_dw, col_mask)


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _rms_norm_bwd_dwdb(
    DW,
    FINAL_DW,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        silu: bool,
    ) -> torch.Tensor:
        assert x.dim() == 2
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape
        assert weight.dim() == 1
        assert weight.numel() == D

        y = torch.empty_like(x)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
        if D > BLOCK_D:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        ctx.save_for_backward(x, weight, rstd)
        ctx.silu = silu
        if N == 0:
            return y

        # pyre-ignore[28]
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
        _weighted_rms_norm_fwd[grid](
            x,
            y,
            weight,
            rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            SILU=silu,
            BLOCK_D=BLOCK_D,
        )

        ctx.BLOCK_D = BLOCK_D
        ctx.eps = eps
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None, None]:
        x, weight, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        dweight = torch.zeros((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            return dx, dweight, None, None

        sms = torch.cuda.get_device_properties(x.device).multi_processor_count

        # pyre-ignore[28]
        grid = lambda meta: (  # noqa E731
            max(1, min(sms * meta["SHARDS_PER_SM"], N // 4)),
        )
        _weighted_rms_norm_bwd[grid](
            dx,
            dy,
            dweight,
            x,
            weight,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            ctx.eps,
            N=N,
            SILU=ctx.silu,
            BLOCK_D=ctx.BLOCK_D,
        )

        return dx, dweight, None, None


class SwishLayerNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape

        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

        y = torch.empty_like(x)
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        num_warps = min(max(BLOCK_D // 256, 1), 8)

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        if N == 0:
            return y

        # pyre-ignore[28]
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa E731
        _weighted_layer_norm_fwd[grid](
            x,
            y,
            y,
            weight,
            bias,
            mean,
            rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            y.stride(0),
            1.0,
            None,  # RESIDUAL
            0,  # stride_residual
            IS_SWISH=True,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=True,
            OUTPUT_FP8=False,
            OUTPUT_BOTH=False,
            ADD_RESIDUAL=False,
        )

        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None]:
        x, weight, bias, mean, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        use_v2 = _wln_bwd_v2_enabled()
        if not use_v2:
            tile_num = _ln_bwd_tile_num(N, D, sms, is_swish=True)
            _dweight = torch.empty(
                (tile_num, D), dtype=torch.float32, device=x.device
            )
            _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        # F1 (bias-grad fold) is not plumbed through this entry point yet; the
        # kernel gate stays off so the compiled control path is unchanged.
        sum_dx = False
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias, None
        if use_v2:
            tile_num, _dweight, _dbias, _ = _wln_bwd_dx_partials(
                dx,
                dy,
                dx,
                x,
                weight,
                bias,
                mean,
                rstd,
                ctx.eps,
                is_swish=True,
                add_residual=False,
                sum_dx=sum_dx,
                BLOCK_D=ctx.BLOCK_D,
                sms=sms,
            )
        else:
            _dxsum = _dbias
            # pyre-ignore[28]
            _weighted_layer_norm_bwd_dx[(tile_num,)](
                dx,
                dy,
                dx,
                _dweight,
                _dbias,
                _dxsum,
                x,
                weight,
                bias,
                mean,
                rstd,
                dx.stride(0),
                dy.stride(0),
                0,
                x.stride(0),
                D,
                ctx.eps,
                IS_SWISH=True,
                ADD_RESIDUAL=False,
                SUM_DX=sum_dx,
                N=N,
                BLOCK_D=ctx.BLOCK_D,
            )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](
            _dweight,
            _dbias,
            dweight,
            dbias,
            tile_num,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, dbias, None


@torch.jit.unused
@torch.fx.wrap
def triton_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return LayerNormFunction.apply(x, weight, bias, eps)


@torch.jit.unused
@torch.fx.wrap
def triton_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    silu: bool = False,
) -> torch.Tensor:
    return RMSNormFunction.apply(x, weight, eps, silu)


@torch.jit.unused
@torch.fx.wrap
def triton_swish_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return SwishLayerNormFunction.apply(x, weight, bias, eps)
