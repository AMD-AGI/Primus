###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused (optionally prob-weighted) SwiGLU forward / backward, one pinned Triton config.

Replaces Megatron's eager SwiGLU chains (what runs when ``jit_fuser`` / torch.compile is disabled):

  routed experts  WeightedSwiGLUFunction  y [M, 2H], w [M, 1] fp32 -> out [M, H]
                  eager: fwd 4 kernels, bwd ~16 kernels with fp32 temporaries and an fp32 ``cat``
  shared expert   SwiGLUFunction          same without w            (2 fwd / ~11 bwd kernels)

Fused: one kernel per direction, fp32 math, one rounding to the output dtype:

  fwd  out[r, :]  = silu(gate) * up * w[r]                     gate = y[r, :H], up = y[r, H:]
  bwd  dy[r, :H]  = g * w[r] * up * sig * (1 + gate * (1 - sig)),  sig = sigmoid(gate)
       dy[r, H:]  = g * w[r] * silu(gate)
       dw[r]      = sum_c silu(gate) * up * g                  (fp32 accumulation over the row)

The eager chain rounds intermediates (silu, silu * up, sigmoid, ...) to the activation dtype; the fused
kernel does not, so it is at least as close to an fp32 reference.

Launch: one program per row (grid = (M,)), the row's H columns in NUM_TILES = cdiv(H, BLOCK_N) unrolled
tiles. Pinned: BLOCK_N 1024, num_warps 4, num_stages 1. No autotune, no in-process benchmarking; the only
run-time choice is NUM_TILES, a constexpr of H (one binary per H).

Bounds: row r < M (grid is exactly M), columns c < H masked, offsets in int64; y / dy rows have stride >= 2H,
out / g rows stride >= H, w / dw stride >= 1.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_N = 1024
NUM_WARPS = 4
NUM_STAGES = 1


@triton.jit
def _sigmoid(x):
    # 1 / (1 + exp(-x)) with the fast (rcp-based, ~1 ulp) divide instead of the IEEE div_scale/fmas/fixup
    # sequence: x -> -inf gives 1 / inf = 0, x -> +inf gives 1 / 1 = 1
    return libdevice.fast_dividef(tl.full(x.shape, 1.0, tl.float32), 1.0 + tl.exp(-x))


@triton.jit
def _swiglu_fwd_kernel(
    y_ptr,
    w_ptr,
    out_ptr,
    H,
    stride_y,
    stride_w,
    stride_o,
    HAS_W: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    y_row = y_ptr + row * stride_y
    o_row = out_ptr + row * stride_o
    if HAS_W:
        w = tl.load(w_ptr + row * stride_w).to(tl.float32)
    for t in tl.static_range(NUM_TILES):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        m = cols < H
        gate = tl.load(y_row + cols, mask=m, other=0.0).to(tl.float32)
        up = tl.load(y_row + H + cols, mask=m, other=0.0).to(tl.float32)
        out = gate * _sigmoid(gate) * up
        if HAS_W:
            out = out * w
        tl.store(o_row + cols, out.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_bwd_kernel(
    g_ptr,
    y_ptr,
    w_ptr,
    dy_ptr,
    dw_ptr,
    H,
    stride_g,
    stride_y,
    stride_w,
    stride_dy,
    stride_dw,
    HAS_W: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g_row = g_ptr + row * stride_g
    y_row = y_ptr + row * stride_y
    dy_row = dy_ptr + row * stride_dy
    if HAS_W:
        w = tl.load(w_ptr + row * stride_w).to(tl.float32)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for t in tl.static_range(NUM_TILES):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        m = cols < H
        g = tl.load(g_row + cols, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(y_row + cols, mask=m, other=0.0).to(tl.float32)
        up = tl.load(y_row + H + cols, mask=m, other=0.0).to(tl.float32)
        sig = _sigmoid(gate)
        silu = gate * sig
        if HAS_W:
            acc += silu * up * g  # masked lanes: g = 0
            gw = g * w
        else:
            gw = g
        d_gate = gw * up * (sig * (1.0 + gate * (1.0 - sig)))
        d_up = gw * silu
        tl.store(dy_row + cols, d_gate.to(dy_ptr.dtype.element_ty), mask=m)
        tl.store(dy_row + H + cols, d_up.to(dy_ptr.dtype.element_ty), mask=m)
    if HAS_W:
        tl.store(dw_ptr + row * stride_dw, tl.sum(acc, axis=0).to(dw_ptr.dtype.element_ty))


def num_tiles(H):
    return triton.cdiv(H, BLOCK_N)


def fwd(y, w, out):
    """out[M, H] <- silu(y[:, :H]) * y[:, H:] (* w[:, None]). y, out: 2-D, last dim contiguous; w: [M] view or None."""
    M, K = y.shape
    H = K // 2
    assert K == 2 * H and tuple(out.shape) == (M, H) and y.stride(1) == 1 and out.stride(1) == 1
    if M == 0:
        return out
    has_w = w is not None
    _swiglu_fwd_kernel[(M,)](
        y,
        w if has_w else y,
        out,
        H,
        y.stride(0),
        w.stride(0) if has_w else 0,
        out.stride(0),
        HAS_W=has_w,
        NUM_TILES=num_tiles(H),
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def bwd(g, y, w, dy, dw):
    """dy[M, 2H] (and dw[M] when w is given) from g[M, H], y[M, 2H]. Last dims contiguous."""
    M, K = y.shape
    H = K // 2
    assert tuple(g.shape) == (M, H) and tuple(dy.shape) == (M, K)
    assert g.stride(1) == 1 and y.stride(1) == 1 and dy.stride(1) == 1
    if M == 0:
        return dy, dw
    has_w = w is not None
    _swiglu_bwd_kernel[(M,)](
        g,
        y,
        w if has_w else y,
        dy,
        dw if has_w else dy,
        H,
        g.stride(0),
        y.stride(0),
        w.stride(0) if has_w else 0,
        dy.stride(0),
        dw.stride(0) if has_w else 0,
        HAS_W=has_w,
        NUM_TILES=num_tiles(H),
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return dy, dw
