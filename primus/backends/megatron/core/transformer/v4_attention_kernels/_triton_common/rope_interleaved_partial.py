###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused interleaved partial RoPE FWD/BWD (plan-6 P35).

The eager body in :func:`apply_interleaved_partial_rope` is a 9-op chain:

.. code-block:: python

    x_nope = x[..., :nope]                           # slice  (1)
    x_rope = x[..., nope:]                           # slice  (2)
    x_pairs = x_rope.reshape(..., rd // 2, 2)        # reshape (3)
    even, odd = x_pairs[..., 0], x_pairs[..., 1]
    cos = cos.unsqueeze(-2).to(orig_dtype)           # unsqueeze + cast (4)
    sin = sin.unsqueeze(-2).to(orig_dtype)           # unsqueeze + cast (5)
    rot_even = even * cos - odd * sin                # 2 muls + 1 sub (6)
    rot_odd  = even * sin + odd * cos                # 2 muls + 1 add (7)
    rotated  = torch.stack([rot_even, rot_odd], -1).reshape(..., rd)  # stack + reshape (8)
    return torch.cat([x_nope, rotated], -1)          # cat (9)

The plan-5 P32 final EP=8 proxy trace attributes:

* ``CatArrayBatchedCopy_contig`` ≈ **10.0 ms / 24 calls** to the closing
  ``torch.cat`` (the nope-prefix copy + the rotated suffix into one
  contiguous tensor), and
* a non-trivial share of ``elementwise_kernel_manual_unroll<128, 8>``
  (~61 ms / 693 calls) to the four broadcast muls.

At 16 invocations per iter (q + k per ``DualRoPE`` call × 8 layers) the
per-call cost is **~3-5 ms** at the V4-Flash widths.

This module collapses the 9-op chain into one Triton kernel that:

1. Flattens the input to ``[N, H, head_dim]`` where ``N`` is the product
   of all leading axes (caller does the ``.reshape(-1, H, head_dim)``
   plumbing; the kernel is shape-agnostic).
2. Per program processes a ``[BLOCK_H]`` slice of one position's
   ``H * head_dim`` row.  cos/sin are loaded **once per position**
   (shared across the ``BLOCK_H`` heads in the program) so the per-call
   HBM traffic for cos/sin is exactly ``N * rd / 2`` reads, not
   ``N * H * rd / 2`` (which the broadcast-muls in the eager body would
   imply if they hit memory).
3. Writes ``out [N, H, head_dim]`` in one pass: nope channels copied
   verbatim; trailing ``rotary_dim`` channels rotated using the
   interleaved (2k, 2k+1) pairing.  No ``torch.cat``-style second
   memcpy; the kernel writes the full output in a single contiguous
   pass.

The BWD kernel is the transpose of the FWD rotation matrix
(``cos, sin / -sin, cos -> cos, -sin / sin, cos``) — analytically:

.. code-block:: python

    dx[..., 2k]     = dout[..., 2k] * cos + dout[..., 2k+1] * sin
    dx[..., 2k+1]   = -dout[..., 2k] * sin + dout[..., 2k+1] * cos
    dx[..., :nope]  = dout[..., :nope]                       # straight copy

Cos / sin are buffers (not Parameters) so they have no gradient.

Gating: routed through :func:`apply_interleaved_partial_rope` when
``PRIMUS_ROPE_TRITON != "0"`` (default-on).  Set to ``"0"`` to fall back
to the eager body (kept in tree for A/B and as the reference path for
G38).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton dtype mapping
# ---------------------------------------------------------------------------

_TORCH_TO_TL_DTYPE = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(t: torch.dtype):
    try:
        return _TORCH_TO_TL_DTYPE[t]
    except KeyError as exc:
        raise TypeError(
            f"rope_interleaved_partial: unsupported dtype {t}; " f"expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _apply_rope_fwd_kernel(
    X_PTR,  # [N, H, head_dim] contiguous
    COS_PTR,  # [N, rd_half] contiguous (broadcast over H)
    SIN_PTR,  # [N, rd_half] contiguous (broadcast over H)
    OUT_PTR,  # [N, H, head_dim] contiguous
    N,
    H,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,  # next_pow2(NOPE) — block stride for nope copy
    BLOCK_RD_HALF: tl.constexpr,  # next_pow2(RD_HALF)
    DTYPE: tl.constexpr,
):
    """Apply interleaved partial RoPE FWD over a tile of ``BLOCK_H`` heads
    for one position ``pid_n``.

    Layout: x / out are ``[N, H, head_dim]``; the program processes
    ``x[pid_n, pid_h*BLOCK_H : (pid_h+1)*BLOCK_H, :]`` and writes the
    rotated result to the same slice of ``out``.

    The nope prefix (``head_dim - rotary_dim`` channels) is copied
    verbatim — kept inside the kernel so the eager-body
    ``torch.cat([x_nope, rotated], -1)`` second-pass memcpy goes away.
    """

    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # cos / sin for this position — load once, broadcast across BLOCK_H heads.
    rd_half_offs = tl.arange(0, BLOCK_RD_HALF)
    rd_half_mask = rd_half_offs < RD_HALF
    cos = tl.load(
        COS_PTR + pid_n * RD_HALF + rd_half_offs,
        mask=rd_half_mask,
        other=0.0,
    )
    sin = tl.load(
        SIN_PTR + pid_n * RD_HALF + rd_half_offs,
        mask=rd_half_mask,
        other=0.0,
    )
    cos = cos.to(DTYPE)
    sin = sin.to(DTYPE)

    # Base pointers for this position × head block.
    row_base = pid_n * H * HEAD_DIM + h_offs[:, None] * HEAD_DIM  # [BLOCK_H, 1]

    # 1) Copy the nope channels verbatim.
    if NOPE > 0:
        nope_offs = tl.arange(0, BLOCK_NOPE)
        nope_mask = (nope_offs < NOPE)[None, :] & h_mask[:, None]
        x_nope = tl.load(
            X_PTR + row_base + nope_offs[None, :],
            mask=nope_mask,
            other=0.0,
        )
        tl.store(
            OUT_PTR + row_base + nope_offs[None, :],
            x_nope,
            mask=nope_mask,
        )

    # 2) Rotate the trailing rotary_dim channels (interleaved pairs).
    even_offs = NOPE + 2 * rd_half_offs
    odd_offs = NOPE + 2 * rd_half_offs + 1
    pair_mask = h_mask[:, None] & rd_half_mask[None, :]
    even = tl.load(
        X_PTR + row_base + even_offs[None, :],
        mask=pair_mask,
        other=0.0,
    )
    odd = tl.load(
        X_PTR + row_base + odd_offs[None, :],
        mask=pair_mask,
        other=0.0,
    )
    rot_even = even * cos[None, :] - odd * sin[None, :]
    rot_odd = even * sin[None, :] + odd * cos[None, :]
    tl.store(
        OUT_PTR + row_base + even_offs[None, :],
        rot_even,
        mask=pair_mask,
    )
    tl.store(
        OUT_PTR + row_base + odd_offs[None, :],
        rot_odd,
        mask=pair_mask,
    )


@triton.jit
def _apply_rope_bwd_kernel(
    DOUT_PTR,  # [N, H, head_dim] contiguous
    COS_PTR,  # [N, rd_half] contiguous
    SIN_PTR,  # [N, rd_half] contiguous
    DX_PTR,  # [N, H, head_dim] contiguous
    N,
    H,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Apply the transpose rotation for the BWD pass.

    .. code-block:: python

        dx[..., 2k]     = dout[..., 2k] * cos + dout[..., 2k+1] * sin
        dx[..., 2k+1]   = -dout[..., 2k] * sin + dout[..., 2k+1] * cos
        dx[..., :nope]  = dout[..., :nope]
    """

    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    rd_half_offs = tl.arange(0, BLOCK_RD_HALF)
    rd_half_mask = rd_half_offs < RD_HALF
    cos = tl.load(
        COS_PTR + pid_n * RD_HALF + rd_half_offs,
        mask=rd_half_mask,
        other=0.0,
    )
    sin = tl.load(
        SIN_PTR + pid_n * RD_HALF + rd_half_offs,
        mask=rd_half_mask,
        other=0.0,
    )
    cos = cos.to(DTYPE)
    sin = sin.to(DTYPE)

    row_base = pid_n * H * HEAD_DIM + h_offs[:, None] * HEAD_DIM

    # 1) Straight copy of the nope-prefix gradient.
    if NOPE > 0:
        nope_offs = tl.arange(0, BLOCK_NOPE)
        nope_mask = (nope_offs < NOPE)[None, :] & h_mask[:, None]
        dout_nope = tl.load(
            DOUT_PTR + row_base + nope_offs[None, :],
            mask=nope_mask,
            other=0.0,
        )
        tl.store(
            DX_PTR + row_base + nope_offs[None, :],
            dout_nope,
            mask=nope_mask,
        )

    # 2) Transposed rotation for the rotary suffix.
    even_offs = NOPE + 2 * rd_half_offs
    odd_offs = NOPE + 2 * rd_half_offs + 1
    pair_mask = h_mask[:, None] & rd_half_mask[None, :]
    dout_even = tl.load(
        DOUT_PTR + row_base + even_offs[None, :],
        mask=pair_mask,
        other=0.0,
    )
    dout_odd = tl.load(
        DOUT_PTR + row_base + odd_offs[None, :],
        mask=pair_mask,
        other=0.0,
    )
    dx_even = dout_even * cos[None, :] + dout_odd * sin[None, :]
    dx_odd = -dout_even * sin[None, :] + dout_odd * cos[None, :]
    tl.store(
        DX_PTR + row_base + even_offs[None, :],
        dx_even,
        mask=pair_mask,
    )
    tl.store(
        DX_PTR + row_base + odd_offs[None, :],
        dx_odd,
        mask=pair_mask,
    )


# ---------------------------------------------------------------------------
# Block-size heuristic
# ---------------------------------------------------------------------------


def _pick_block_h(h: int) -> int:
    """Pick BLOCK_H tiling the heads axis.

    V4-Flash Q has ``H = 64`` heads, K has ``H = 1``.  For H=1 the block
    must be 1; for H=64 a BLOCK_H of 8 keeps the per-program live tile
    under ~16 KiB at head_dim=512 / bf16 while amortising the cos/sin
    load across 8 heads.
    """

    if h <= 1:
        return 1
    if h <= 4:
        return min(h, 4)
    if h <= 16:
        return 8
    return 8  # default for V4-Flash Q (H=64) and K (H=1 handled above)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class RoPEInterleavedPartialFn(torch.autograd.Function):
    """Autograd-aware wrapper around the FWD/BWD Triton kernels.

    The wrapper:

    1. Flattens ``x`` to ``[N, H, head_dim]`` and ``cos / sin`` to
       ``[N, rd_half]`` (callers may pass any leading shape; the wrapper
       does the reshape).
    2. Casts cos / sin to ``x.dtype`` if needed (matches the plan-5 P32
       RoPE bf16 cast contract in :func:`apply_interleaved_partial_rope`).
    3. Saves only ``cos, sin, rotary_dim, head_dim, nope`` for the
       backward — the input ``x`` is not needed, and the saved tensors
       are buffers (cos / sin) so they have no autograd graph node.

    Layout assumptions (validated up front):

    * ``x.is_contiguous()``  — call ``.contiguous()`` in the caller if needed.
    * ``cos.shape == sin.shape == leading_shape + (rotary_dim // 2,)`` where
      ``leading_shape == x.shape[:-2]`` (i.e. one cos/sin row per position).
    * ``rotary_dim <= x.shape[-1]`` and ``rotary_dim % 2 == 0``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_dim: int,
    ) -> torch.Tensor:
        if rotary_dim == 0:
            ctx.save_for_backward(cos, sin)
            ctx.rotary_dim = 0
            ctx.head_dim = x.shape[-1]
            return x.contiguous()

        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        head_dim = x.shape[-1]
        if rotary_dim > head_dim:
            raise ValueError(f"rotary_dim ({rotary_dim}) must be <= head_dim ({head_dim})")

        x = x.contiguous()
        leading_shape = x.shape[:-2]
        H = x.shape[-2]
        N = 1
        for s in leading_shape:
            N *= s

        rd_half = rotary_dim // 2
        # cos / sin should have shape == leading_shape + (rd_half,)
        if cos.shape[-1] != rd_half or sin.shape[-1] != rd_half:
            raise ValueError(
                f"cos/sin last dim must be rotary_dim // 2 ({rd_half}); "
                f"got cos.shape={tuple(cos.shape)}, sin.shape={tuple(sin.shape)}"
            )
        cos_flat = cos.contiguous().reshape(N, rd_half).to(x.dtype)
        sin_flat = sin.contiguous().reshape(N, rd_half).to(x.dtype)

        out = torch.empty_like(x)

        nope = head_dim - rotary_dim
        block_h = _pick_block_h(H)
        block_nope = max(triton.next_power_of_2(max(nope, 1)), 1)
        block_rd_half = triton.next_power_of_2(rd_half)

        grid = (N, triton.cdiv(H, block_h))
        _apply_rope_fwd_kernel[grid](
            x,
            cos_flat,
            sin_flat,
            out,
            N,
            H,
            HEAD_DIM=head_dim,
            NOPE=nope,
            RD_HALF=rd_half,
            BLOCK_H=block_h,
            BLOCK_NOPE=block_nope,
            BLOCK_RD_HALF=block_rd_half,
            DTYPE=_triton_dtype(x.dtype),
        )

        ctx.save_for_backward(cos_flat, sin_flat)
        ctx.rotary_dim = rotary_dim
        ctx.head_dim = head_dim
        ctx.leading_shape = tuple(leading_shape)
        ctx.H = H
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):  # type: ignore[override]
        rotary_dim = ctx.rotary_dim
        cos_flat, sin_flat = ctx.saved_tensors

        if rotary_dim == 0:
            return dout.contiguous(), None, None, None

        dout = dout.contiguous()
        head_dim = ctx.head_dim
        H = ctx.H
        leading_shape = ctx.leading_shape
        N = 1
        for s in leading_shape:
            N *= s

        rd_half = rotary_dim // 2
        nope = head_dim - rotary_dim
        block_h = _pick_block_h(H)
        block_nope = max(triton.next_power_of_2(max(nope, 1)), 1)
        block_rd_half = triton.next_power_of_2(rd_half)

        dx = torch.empty_like(dout)

        grid = (N, triton.cdiv(H, block_h))
        _apply_rope_bwd_kernel[grid](
            dout,
            cos_flat,
            sin_flat,
            dx,
            N,
            H,
            HEAD_DIM=head_dim,
            NOPE=nope,
            RD_HALF=rd_half,
            BLOCK_H=block_h,
            BLOCK_NOPE=block_nope,
            BLOCK_RD_HALF=block_rd_half,
            DTYPE=_triton_dtype(dout.dtype),
        )
        return dx, None, None, None


# ---------------------------------------------------------------------------
# Fused variant: compute cos/sin IN-KERNEL from (position_ids, inv_freq).
#
# The plain RoPE kernel above consumes precomputed cos/sin tensors, which
# means the caller still pays 3 small kernels per call
# (``position_ids.float() * inv_freq`` -> ``cos`` -> ``sin``) plus a
# ``.to(x.dtype)`` cast, and (in ``DeepseekV4Attention._apply_rope_q_k``)
# recomputes them identically for Q and K.  This variant folds the cos/sin
# generation into the rotation kernel: it loads the row's scalar position +
# the ``[rd_half]`` inv_freq vector and computes ``cos``/``sin`` in registers,
# so no cos/sin HBM tensor is ever materialised.  YaRN is already baked into
# ``inv_freq`` (a buffer), so the compress base is handled transparently.
# ---------------------------------------------------------------------------


@triton.jit
def _rope_gen_fwd_kernel(
    X_PTR,  # [N, H, head_dim], last dim contiguous, rows strided by (IN_SN, IN_SH)
    POS_PTR,  # [N] (position per row; fp32)
    INVFREQ_PTR,  # [rd_half] fp32
    OUT_PTR,  # [N, H, head_dim] contiguous
    N,
    H,
    IN_SN,
    IN_SH,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
    DTYPE: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    rd_half_offs = tl.arange(0, BLOCK_RD_HALF)
    rd_half_mask = rd_half_offs < RD_HALF

    # cos/sin for this position, computed once, broadcast across BLOCK_H heads.
    pos = tl.load(POS_PTR + pid_n).to(tl.float32)
    inv_freq = tl.load(INVFREQ_PTR + rd_half_offs, mask=rd_half_mask, other=0.0).to(tl.float32)
    angle = pos * inv_freq
    # ``R(-pos) == R(pos)^T``, i.e. the same rotation with a negated sine, so
    # the inverse costs a sign flip rather than a second kernel.
    if INVERSE:
        angle = -angle
    cos = tl.cos(angle).to(DTYPE)
    sin = tl.sin(angle).to(DTYPE)

    row_base = pid_n * H * HEAD_DIM + h_offs[:, None] * HEAD_DIM
    in_base = pid_n.to(tl.int64) * IN_SN + h_offs[:, None].to(tl.int64) * IN_SH

    if NOPE > 0:
        nope_offs = tl.arange(0, BLOCK_NOPE)
        nope_mask = (nope_offs < NOPE)[None, :] & h_mask[:, None]
        x_nope = tl.load(X_PTR + in_base + nope_offs[None, :], mask=nope_mask, other=0.0)
        tl.store(OUT_PTR + row_base + nope_offs[None, :], x_nope, mask=nope_mask)

    even_offs = NOPE + 2 * rd_half_offs
    odd_offs = NOPE + 2 * rd_half_offs + 1
    pair_mask = h_mask[:, None] & rd_half_mask[None, :]
    even = tl.load(X_PTR + in_base + even_offs[None, :], mask=pair_mask, other=0.0)
    odd = tl.load(X_PTR + in_base + odd_offs[None, :], mask=pair_mask, other=0.0)
    rot_even = even * cos[None, :] - odd * sin[None, :]
    rot_odd = even * sin[None, :] + odd * cos[None, :]
    tl.store(OUT_PTR + row_base + even_offs[None, :], rot_even, mask=pair_mask)
    tl.store(OUT_PTR + row_base + odd_offs[None, :], rot_odd, mask=pair_mask)


@triton.jit
def _rope_gen_bwd_kernel(
    DOUT_PTR,  # [N, H, head_dim], last dim contiguous, rows strided by (IN_SN, IN_SH)
    POS_PTR,  # [N] fp32
    INVFREQ_PTR,  # [rd_half] fp32
    DX_PTR,  # [N, H, head_dim] contiguous
    N,
    H,
    IN_SN,
    IN_SH,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_RD_HALF: tl.constexpr,
    DTYPE: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    rd_half_offs = tl.arange(0, BLOCK_RD_HALF)
    rd_half_mask = rd_half_offs < RD_HALF

    pos = tl.load(POS_PTR + pid_n).to(tl.float32)
    inv_freq = tl.load(INVFREQ_PTR + rd_half_offs, mask=rd_half_mask, other=0.0).to(tl.float32)
    angle = pos * inv_freq
    # Must mirror the forward's sign so the transposed rotation below is the
    # actual adjoint of what was applied.
    if INVERSE:
        angle = -angle
    cos = tl.cos(angle).to(DTYPE)
    sin = tl.sin(angle).to(DTYPE)

    row_base = pid_n * H * HEAD_DIM + h_offs[:, None] * HEAD_DIM
    in_base = pid_n.to(tl.int64) * IN_SN + h_offs[:, None].to(tl.int64) * IN_SH

    if NOPE > 0:
        nope_offs = tl.arange(0, BLOCK_NOPE)
        nope_mask = (nope_offs < NOPE)[None, :] & h_mask[:, None]
        dout_nope = tl.load(DOUT_PTR + in_base + nope_offs[None, :], mask=nope_mask, other=0.0)
        tl.store(DX_PTR + row_base + nope_offs[None, :], dout_nope, mask=nope_mask)

    even_offs = NOPE + 2 * rd_half_offs
    odd_offs = NOPE + 2 * rd_half_offs + 1
    pair_mask = h_mask[:, None] & rd_half_mask[None, :]
    dout_even = tl.load(DOUT_PTR + in_base + even_offs[None, :], mask=pair_mask, other=0.0)
    dout_odd = tl.load(DOUT_PTR + in_base + odd_offs[None, :], mask=pair_mask, other=0.0)
    dx_even = dout_even * cos[None, :] + dout_odd * sin[None, :]
    dx_odd = -dout_even * sin[None, :] + dout_odd * cos[None, :]
    tl.store(DX_PTR + row_base + even_offs[None, :], dx_even, mask=pair_mask)
    tl.store(DX_PTR + row_base + odd_offs[None, :], dx_odd, mask=pair_mask)


@triton.jit
def _rope_gen_row_kernel(
    X_PTR,  # [N, H, head_dim], last dim contiguous, rows strided by (IN_SN, IN_SH)
    POS_PTR,
    INVFREQ_PTR,
    OUT_PTR,  # [N, H, head_dim] contiguous
    N,
    H,
    IN_SN,
    IN_SH,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    INVERSE: tl.constexpr,
):
    """Whole-row variant (g90mem): one contiguous load / store per head row, pairs split in registers
    (``tl.split``), rotation in fp32 and one rounding. ~2x faster than the even/odd strided kernels on
    gfx1250 ([4096, 64, 512]: 70.6 -> 36.5 us, 87% of the HBM floor) and closer to the fp32 reference."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = HEAD_DIM // 2
    p = tl.arange(0, HALF)
    fi = p - NOPE // 2
    is_rope = fi >= 0
    ang = tl.load(POS_PTR + pid_n).to(tl.float32) * tl.load(INVFREQ_PTR + fi, mask=is_rope, other=0.0).to(tl.float32)
    if INVERSE:
        ang = -ang
    c = tl.where(is_rope, tl.cos(ang), 1.0)
    s = tl.where(is_rope, tl.sin(ang), 0.0)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, HEAD_DIM)
    xv = tl.load(
        X_PTR + pid_n.to(tl.int64) * IN_SN + h[:, None].to(tl.int64) * IN_SH + d[None, :], mask=hm[:, None], other=0.0
    ).to(tl.float32)
    a, b = tl.split(tl.reshape(xv, (BLOCK_H, HALF, 2)))
    o = tl.reshape(tl.join(a * c[None, :] - b * s[None, :], a * s[None, :] + b * c[None, :]), (BLOCK_H, HEAD_DIM))
    tl.store(
        OUT_PTR + (pid_n.to(tl.int64) * H + h[:, None]) * HEAD_DIM + d[None, :],
        o.to(OUT_PTR.dtype.element_ty),
        mask=hm[:, None],
    )


def _rope_row_ok(head_dim: int) -> bool:
    return os.environ.get("PRIMUS_V4_ROPE_ROW", "1") != "0" and head_dim >= 2 and (head_dim & (head_dim - 1)) == 0


def _launch_rope_row(x3, pos_c, inv_c, out, N, H, head_dim, nope, inverse):
    bh = min(16, triton.next_power_of_2(max(H, 1)))
    _rope_gen_row_kernel[(N, triton.cdiv(H, bh))](
        x3, pos_c, inv_c, out, N, H, x3.stride(0), x3.stride(1),
        HEAD_DIM=head_dim, NOPE=nope, BLOCK_H=bh, INVERSE=bool(inverse), num_warps=8 if bh >= 16 else 4,
    )


@triton.jit
def _rope_tail_inplace_kernel(
    X_PTR,  # [N, H, head_dim] rows strided by (SN, SH), last dim contiguous; rotated in place on [NOPE, head_dim)
    POS_PTR,
    INVFREQ_PTR,
    N,
    H,
    SN,
    SH,
    NOPE: tl.constexpr,
    RD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    INVERSE: tl.constexpr,
):
    """In-place variant of ``_rope_gen_row_kernel`` touching only the rotary tail: the same fp32 angle, cos / sin,
    pair rotation and single rounding as the row kernel, so the tail bytes are identical; the NoPE columns are not
    read or written (the row kernel copies them through x * 1 - y * 0 / x * 0 + y * 1, which is the identity for
    finite values up to the sign of a zero)."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = RD // 2
    fi = tl.arange(0, HALF)
    ang = tl.load(POS_PTR + pid_n).to(tl.float32) * tl.load(INVFREQ_PTR + fi).to(tl.float32)
    if INVERSE:
        ang = -ang
    c = tl.cos(ang)
    s = tl.sin(ang)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = NOPE + tl.arange(0, RD)
    ptr = X_PTR + pid_n.to(tl.int64) * SN + h[:, None].to(tl.int64) * SH + d[None, :]
    xv = tl.load(ptr, mask=hm[:, None], other=0.0).to(tl.float32)
    a, b = tl.split(tl.reshape(xv, (BLOCK_H, HALF, 2)))
    o = tl.reshape(tl.join(a * c[None, :] - b * s[None, :], a * s[None, :] + b * c[None, :]), (BLOCK_H, RD))
    tl.store(ptr, o.to(X_PTR.dtype.element_ty), mask=hm[:, None])


def _rope_bwd_inplace_ok(head_dim: int, rotary_dim: int) -> bool:
    return (
        os.environ.get("PRIMUS_V4_ROPE_BWD_INPLACE", "0") == "1"
        and _rope_row_ok(head_dim)
        and rotary_dim >= 2
        and (rotary_dim & (rotary_dim - 1)) == 0
        and (head_dim - rotary_dim) % 2 == 0
    )


def _launch_rope_tail_inplace(x3, pos_c, inv_c, N, H, nope, rd, inverse):
    bh = min(64, triton.next_power_of_2(max(H, 1)))
    _rope_tail_inplace_kernel[(N, triton.cdiv(H, bh))](
        x3, pos_c, inv_c, N, H, x3.stride(0), x3.stride(1),
        NOPE=nope, RD=rd, BLOCK_H=bh, INVERSE=bool(inverse), num_warps=4,
    )


def _rows_view(t: torch.Tensor, N: int, H: int, D: int) -> torch.Tensor:
    """``t`` as an ``[N, H, D]`` view whose last dim is unit-stride, copying only if needed.

    The gen kernels address rows through ``(stride(0), stride(1))``, so any layout whose
    leading dims flatten into one stride (e.g. a ``permute``d BHSD view of BSHD storage, or
    a slice of a padded buffer) is consumed in place.
    """
    try:
        v = t.view(N, H, D)
    except RuntimeError:
        v = None
    if v is None or (D > 1 and v.stride(2) != 1):
        v = t.contiguous().view(N, H, D)
    return v


class RoPEFromPositionsFn(torch.autograd.Function):
    """RoPE that computes cos/sin in-kernel from ``(position_ids, inv_freq)``.

    Eliminates the separate ``cos``/``sin`` generation kernels and HBM
    tensors.  ``position_ids`` / ``inv_freq`` are non-differentiable
    (positions are integer, inv_freq is a buffer), so the backward returns
    ``None`` for both and recomputes cos/sin in-kernel.

    ``inverse=True`` rotates by ``-position`` instead, which is what the V4
    attention output needs before the O projection. It is the same kernel with
    a negated angle, so the de-rotation gets the same single-pass, no-cos/sin-in-
    HBM treatment as Q / KV instead of paying for a separate cos/sin build.
    """

    @staticmethod
    def forward(ctx, x, pos_flat, inv_freq, rotary_dim, inverse=False):  # type: ignore[override]
        head_dim = x.shape[-1]
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        if rotary_dim > head_dim:
            raise ValueError(f"rotary_dim ({rotary_dim}) must be <= head_dim ({head_dim})")

        leading = x.shape[:-2]
        H = x.shape[-2]
        N = 1
        for s in leading:
            N *= s
        x3 = _rows_view(x, N, H, head_dim)
        rd_half = rotary_dim // 2
        if inv_freq.shape[-1] != rd_half:
            raise ValueError(
                f"inv_freq last dim must be rotary_dim // 2 ({rd_half}); got {tuple(inv_freq.shape)}"
            )

        pos_c = pos_flat.contiguous().to(torch.float32)
        inv_c = inv_freq.contiguous().to(torch.float32)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

        nope = head_dim - rotary_dim
        block_h = _pick_block_h(H)
        block_nope = max(triton.next_power_of_2(max(nope, 1)), 1)
        block_rd_half = triton.next_power_of_2(rd_half)
        grid = (N, triton.cdiv(H, block_h))
        if _rope_row_ok(head_dim):
            _launch_rope_row(x3, pos_c, inv_c, out, N, H, head_dim, nope, inverse)
        else:
          _rope_gen_fwd_kernel[grid](
            x3,
            pos_c,
            inv_c,
            out,
            N,
            H,
            x3.stride(0),
            x3.stride(1),
            HEAD_DIM=head_dim,
            NOPE=nope,
            RD_HALF=rd_half,
            BLOCK_H=block_h,
            BLOCK_NOPE=block_nope,
            BLOCK_RD_HALF=block_rd_half,
            DTYPE=_triton_dtype(x.dtype),
            INVERSE=bool(inverse),
        )

        ctx.save_for_backward(pos_c, inv_c)
        ctx.rotary_dim = rotary_dim
        ctx.head_dim = head_dim
        ctx.H = H
        ctx.N = N
        ctx.inverse = bool(inverse)
        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        pos_c, inv_c = ctx.saved_tensors
        rotary_dim = ctx.rotary_dim
        head_dim = ctx.head_dim
        H = ctx.H
        N = ctx.N
        # The gradient often arrives as a row-strided view (e.g. the sparse-MLA adapter's
        # dq is a slice of its rope-padded [B*S, H, D+pad] buffer): read it through its
        # strides instead of materialising a contiguous copy first.
        dout3 = _rows_view(dout, N, H, head_dim)
        rd_half = rotary_dim // 2
        nope = head_dim - rotary_dim
        if ctx.inverse and _rope_bwd_inplace_ok(head_dim, rotary_dim):
            # PRIMUS_V4_ROPE_BWD_INPLACE=1 (output de-rotation only): its incoming gradient is the fresh dX of the
            # o_a projection, read by nobody else, so rotate just its rope tail in place -- the NoPE columns pass
            # through unchanged. Same fp32 math as the row kernel; 1/8 of the traffic at head_dim 512 / rope 64.
            _launch_rope_tail_inplace(dout3, pos_c, inv_c, N, H, nope, rotary_dim, not ctx.inverse)
            dx = dout if dout3.data_ptr() == dout.data_ptr() else dout3.view(dout.shape)  # rotated in place
            return dx, None, None, None, None
        block_h = _pick_block_h(H)
        block_nope = max(triton.next_power_of_2(max(nope, 1)), 1)
        block_rd_half = triton.next_power_of_2(rd_half)
        dx = torch.empty(dout.shape, dtype=dout.dtype, device=dout.device)
        grid = (N, triton.cdiv(H, block_h))
        if _rope_row_ok(head_dim):  # adjoint of R(+-theta) is R(-+theta)
            _launch_rope_row(dout3, pos_c, inv_c, dx, N, H, head_dim, nope, not ctx.inverse)
        else:
          _rope_gen_bwd_kernel[grid](
            dout3,
            pos_c,
            inv_c,
            dx,
            N,
            H,
            dout3.stride(0),
            dout3.stride(1),
            HEAD_DIM=head_dim,
            NOPE=nope,
            RD_HALF=rd_half,
            BLOCK_H=block_h,
            BLOCK_NOPE=block_nope,
            BLOCK_RD_HALF=block_rd_half,
            DTYPE=_triton_dtype(dout.dtype),
            INVERSE=ctx.inverse,
        )
        return dx, None, None, None, None


def apply_rope_from_positions(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    rotary_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Fused interleaved partial RoPE that generates cos/sin in-kernel.

    ``x``: ``[..., H, head_dim]``; ``position_ids`` broadcastable to
    ``x.shape[:-2]``; ``inv_freq``: ``[rotary_dim // 2]`` (YaRN pre-applied).
    Dispatches to the Triton kernel when enabled + CUDA, else falls back to
    the eager ``cos = pos*inv_freq -> cos/sin -> rotate`` path.

    ``inverse=True`` applies ``R(-position)``, which the V4 attention output
    needs before the O projection. Both paths are out-of-place.
    """
    if rotary_dim == 0:
        return x
    if is_triton_path_enabled() and x.is_cuda:
        leading = x.shape[:-2]
        pos_flat = position_ids.broadcast_to(leading).reshape(-1)
        return RoPEFromPositionsFn.apply(x, pos_flat, inv_freq, rotary_dim, inverse)
    # Eager fallback: build cos/sin then apply.
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    if inverse:
        freqs = -freqs
    return eager_apply_interleaved_partial_rope(x, freqs.cos(), freqs.sin(), rotary_dim=rotary_dim)


# ---------------------------------------------------------------------------
# Public Python entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Return True iff the ``PRIMUS_ROPE_TRITON`` env knob is not ``"0"``.

    Default-on, A/B toggle via ``PRIMUS_ROPE_TRITON=0``.
    """

    return os.environ.get("PRIMUS_ROPE_TRITON", "1") != "0"


def eager_apply_interleaved_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
) -> torch.Tensor:
    """Reference eager implementation matching
    :func:`primus.backends.megatron.core.transformer.dual_rope.apply_interleaved_partial_rope`.

    Kept here so the test / bench code can A/B against the exact eager
    body without depending on the consumer module.  Bit-for-bit
    equivalent to the original eager body (same op order, same dtype
    cast).
    """

    head_dim = x.shape[-1]
    if rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even and <= head_dim ({head_dim}), got {rotary_dim}")
    if rotary_dim == 0:
        return x

    orig_dtype = x.dtype
    nope = head_dim - rotary_dim
    x_nope = x[..., :nope]
    x_rope = x[..., nope:]

    x_pairs = x_rope.reshape(*x_rope.shape[:-1], rotary_dim // 2, 2)
    even = x_pairs[..., 0]
    odd = x_pairs[..., 1]

    cos = cos.unsqueeze(-2).to(orig_dtype)
    sin = sin.unsqueeze(-2).to(orig_dtype)

    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos

    rotated = torch.stack([rot_even, rot_odd], dim=-1).reshape(*x_rope.shape[:-1], rotary_dim)
    return torch.cat([x_nope, rotated], dim=-1)


def apply_rope_interleaved_partial(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
) -> torch.Tensor:
    """Dispatch: Triton path when ``PRIMUS_ROPE_TRITON != "0"`` (default),
    else eager fallback.

    Shape contract matches the dual-RoPE caller:

    * ``x``: ``[..., H, head_dim]`` (any leading shape; flattened
      internally).
    * ``cos, sin``: ``[..., rotary_dim // 2]`` where the leading shape
      is ``x.shape[:-2]`` (i.e. one cos/sin row per position).
    * ``rotary_dim``: even and ``<= head_dim``.

    Output shape matches ``x``.
    """

    if rotary_dim == 0:
        return x

    if is_triton_path_enabled() and x.is_cuda:
        return RoPEInterleavedPartialFn.apply(x, cos, sin, rotary_dim)
    return eager_apply_interleaved_partial_rope(x, cos, sin, rotary_dim=rotary_dim)


__all__ = [
    "RoPEInterleavedPartialFn",
    "RoPEFromPositionsFn",
    "apply_rope_interleaved_partial",
    "apply_rope_from_positions",
    "eager_apply_interleaved_partial_rope",
    "is_triton_path_enabled",
]
