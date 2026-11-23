###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MSA indexer's input norm in one launch, bit for bit.

``MiniMaxM3Attention._indexer_input`` recomputes the input norm that the TE specs fuse into
``linear_qkv`` as ``rms_norm(x.float(), w.float() [+ 1], eps).to(x.dtype)``: five launches per
layer and micro-batch (weight widening, ``+ 1``, a BF16 -> FP32 copy of ``[sq, b, h]``, ATen's
``vectorized_layer_norm_kernel`` and an FP32 -> BF16 copy). The Triton kernel here reads the
BF16 rows once and writes the BF16 result.

The sum of squares follows ATen's ``vectorized_layer_norm_kernel`` (rms path, one row per
block of ``warp_size x threads_y`` = 32 x 8 threads): thread ``t`` adds the squares of the
4-wide vectors ``t, t + 256, ...`` of the row in order, the 32 lanes of a wave fold by
``shfl_down`` (16, 8, 4, 2, 1), the 8 waves fold pairwise (4, 2, 1); then
``rstd = rsqrt(sum / N + eps)`` and ``y = gamma * (rstd * x)``. This is a transcription of
another library's kernel: a torch release that changes the block shape or the order breaks it
silently, which is what the ``bitwise_check`` self-check (first call per row width, eps and
gamma kind) is for; on a mismatch the eager ops stay.

ATen's non-vectorised layer-norm path is never the reference here: it is taken only for rows
that are not 16-byte aligned or whose width is not a multiple of 4, and the original code
normalises ``x.float()``, a fresh (aligned) allocation, with ``N % 4 == 0`` required below.
The block shape rule (``_aten_block_is_32x8``) is transcribed from ATen as well.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check

_THREADS = 256  # ATen's block: warp_size 32 x threads_y 8
_ROW_BLOCK = 1024
# widest row the kernel is built for (its loops are unrolled: N / 1024 passes each); unit-tested up to here
_MAX_WIDTH = 8192


@triton.jit
def _indexer_input_norm_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    eps,
    N: tl.constexpr,
    NV: tl.constexpr,
    TRIPS: tl.constexpr,
    BLOCKS: tl.constexpr,
    RB: tl.constexpr,
    ZC: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = X_ptr + row * N
    t = tl.arange(0, 256)
    # thread t of ATen's 32 x 8 block: its 4-wide vectors t, t + 256, ... in order
    s = tl.zeros([256], dtype=tl.float32)
    for i in tl.static_range(TRIPS):
        vec = i * 256 + t
        m = vec < NV
        for k in tl.static_range(4):
            v = tl.load(x_row + vec * 4 + k, mask=m, other=0.0).to(tl.float32)
            s = s + v * v
    # lanes of a wave: shfl_down by 16, 8, 4, 2, 1 (lane l takes l + off); then waves by 4, 2, 1
    s = tl.sum(tl.reshape(s, [8, 2, 16]), axis=1)
    s = tl.sum(tl.reshape(s, [8, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [8, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [8, 2, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [8, 2, 1]), axis=1)
    s = tl.sum(tl.reshape(s, [2, 4]), axis=0)
    s = tl.sum(tl.reshape(s, [2, 2]), axis=0)
    total = tl.sum(tl.reshape(s, [2]), axis=0)
    rstd = tl.rsqrt(total / N + eps)
    c = tl.arange(0, RB)
    for j in tl.static_range(BLOCKS):
        col = j * RB + c
        m = col < N
        x = tl.load(x_row + col, mask=m, other=0.0).to(tl.float32)
        g = tl.load(W_ptr + col, mask=m, other=0.0).to(tl.float32)
        if ZC:
            g = g + 1.0
        tl.store(Y_ptr + row * N + col, (g * (rstd * x)).to(tl.bfloat16), mask=m)


def _launch(x: torch.Tensor, w: torch.Tensor, eps: float, zero_centered: bool) -> torch.Tensor:
    N = x.shape[-1]
    rows = x.numel() // N
    y = torch.empty_like(x)
    nv = N // 4
    _indexer_input_norm_kernel[(rows,)](
        x,
        w,
        y,
        eps,
        N=N,
        NV=nv,
        TRIPS=triton.cdiv(nv, _THREADS),
        BLOCKS=triton.cdiv(N, _ROW_BLOCK),
        RB=_ROW_BLOCK,
        ZC=bool(zero_centered),
        num_warps=8,
    )
    return y


def _aten_block_is_32x8(N: int) -> bool:
    """ATen's ``launch_vectorized_layer_norm_kernel`` picks 32 x 8 threads for this row width."""
    if getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "warp_size", None) != 32:
        return False
    ty = 8
    while (ty // 2) * 32 >= N // 4 and ty > 1:
        ty //= 2
    return ty == 8


def supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Shapes and dtypes the kernel transcribes: BF16 rows on the GPU whose FP32 copy ATen
    normalises with its vectorised kernel in 32 x 8 blocks."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.numel() > 0):
        return False
    N = x.shape[-1]
    if not (x.is_contiguous() and w.is_contiguous() and w.dim() == 1 and w.numel() == N):
        return False
    return N % 4 == 0 and N <= _MAX_WIDTH and _aten_block_is_32x8(N)


_PROBE_ROWS = 2096


def _probe(shape_tail, w: torch.Tensor):
    """Rows of the training width. The sum, ``/ N``, ``+ eps`` and ``rsqrt`` are evaluated once
    per row, so the probe has many rows: 1024 plain, 512 large, 512 tiny, then rows with outlier
    channels (a residual stream) and with exact zeros, an all-zero row and a row holding an inf.
    The weight is moved off its init value (a zero-centered gamma is exactly 0 at init and hides
    the ``+ 1``)."""
    N = shape_tail[-1]
    dev = w.device
    x = bitwise_check.randn((_PROBE_ROWS, N), dev, seed=11)
    x[1024:1536] *= 300.0
    x[1536:2048] *= 1e-3
    x[2048:2064] *= bitwise_check.randn((N,), dev, seed=12).abs() * 20.0
    x[2064:2080] *= (bitwise_check.randn((16, N), dev, seed=13) > 0).float()
    x[2080] = 0.0
    x[2081, 5] = float("inf")
    noise = bitwise_check.randn((N,), dev, seed=14)
    return x.to(torch.bfloat16), (w.detach().float() + 0.1 * noise).to(torch.bfloat16)


def self_check(x: torch.Tensor, w: torch.Tensor, eps: float, zero_centered: bool, reference) -> bool:
    """``reference(x, w)`` runs the original ops (``MinimaxSparseAttention._indexer_input_eager``)."""
    N = x.shape[-1]
    return bitwise_check.passed(
        "msa_indexer_input_norm",
        lambda: [_launch(*_probe(x.shape, w), eps, zero_centered)],
        lambda: [reference(*_probe(x.shape, w))],
        # the row width fixes ATen's block shape and trip counts; eps and the gamma kind are
        # the kernel's scalars
        key=(N, float(eps), bool(zero_centered)),
    )


def indexer_input_norm(x: torch.Tensor, w: torch.Tensor, eps: float, zero_centered: bool, reference):
    """``reference(x, w)`` -- the original ``rms_norm(x.float(), w.float() [+ 1], eps).to(x.dtype)``
    -- bit for bit in one launch, or None where the kernel does not apply or has not passed its
    self-check (the caller then runs the original ops). No autograd: both inputs are detached."""
    if supported(x, w) and self_check(x, w, eps, zero_centered, reference):
        return _launch(x, w, eps, zero_centered)
    return None
