# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""FP8 weight gradient over the token axis, from row-major (token-major rows)
operands: ``C[N1, N2] = sa * sb * sum_t A[t, :]^T B[t, :]``.

Two implementations. ``fp8_wgrad_tn`` (used by default by the model path) makes
token-major copies with a copy-speed Triton byte transpose and runs the library
FP8 TN GEMM. ``fp8_wgrad`` (below) reads row-major tiles directly; at the out-proj
shape it was slower than both the TN route and the pinned BF16 wgrad in the
microbench, so it is kept only as an alternative
(DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL=triton). The model's default sub-mode under
the knob is the hipBLASLt NN GEMM (ops/hipblaslt_fp8_nn.py).

Why a row-major kernel was tried: a wgrad contracts over the tokens. Both operands are
stored one token per row, so for a library FP8 GEMM both would have to be
re-laid-out with tokens contiguous (two transposed copies per call). Measured on
this stack, those copies cost more than the FP8 GEMM saves, and the library's NT
FP8 kernels are slower than the pinned BF16 wgrad. This kernel reads
the row-major tiles directly and lets the compiler transpose A in LDS.

Split-K over tokens with a deterministic second pass: each program writes an fp32
partial tile to a workspace ``[SPLIT, N1, N2]``; a reduce kernel sums the splits in
a fixed order and applies the two per-tensor scales. No atomics, so the result is
bitwise reproducible run to run.

Numerics: FP8 operands (e4m3 A, e5m2 or e4m3 B), fp32 accumulation. A declared
precision change relative to the BF16 wgrad; default-off at every call site.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_wgrad_partial_kernel(
    a_ptr,
    b_ptr,
    ws_ptr,
    M,
    N1: tl.constexpr,
    N2: tl.constexpr,
    stride_am,
    stride_bm,
    tokens_per_split,
    BLOCK_K: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_split = tl.program_id(1)
    tiles_n2: tl.constexpr = N2 // BLOCK_N2
    tile_n1 = pid_tile // tiles_n2
    tile_n2 = pid_tile % tiles_n2
    rn1 = tile_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    rn2 = tile_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    rk = tl.arange(0, BLOCK_K)
    start = pid_split.to(tl.int64) * tokens_per_split
    end = tl.minimum(start + tokens_per_split, M)
    acc = tl.zeros((BLOCK_N1, BLOCK_N2), dtype=tl.float32)
    for k in range(start, end, BLOCK_K):
        rows = k + rk
        row_mask = rows < end
        a = tl.load(
            a_ptr + rows[:, None] * stride_am + rn1[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        b = tl.load(
            b_ptr + rows[:, None] * stride_bm + rn2[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(tl.trans(a), b, acc)
    ws = (
        ws_ptr
        + pid_split.to(tl.int64) * (N1 * N2)
        + rn1[:, None] * N2
        + rn2[None, :]
    )
    tl.store(ws, acc)


@triton.jit
def _fp8_wgrad_reduce_kernel(
    ws_ptr,
    out_ptr,
    scale_a_ptr,
    scale_b_ptr,
    SPLIT: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT):  # fixed order: deterministic
        acc += tl.load(ws_ptr + s * NUMEL + offs, mask=mask, other=0.0)
    scale = tl.load(scale_a_ptr).to(tl.float32) * tl.load(scale_b_ptr).to(tl.float32)
    tl.store(out_ptr + offs, (acc * scale).to(out_ptr.dtype.element_ty), mask=mask)


# (BLOCK_K, BLOCK_N1, BLOCK_N2, num_warps, num_stages). Default picked by the
# iter-22 microbench; override with fp8_wgrad(..., config=...).
DEFAULT_CONFIG: Tuple[int, int, int, int, int] = (128, 256, 256, 8, 2)


def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def fp8_wgrad(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    config: Optional[Tuple[int, int, int, int, int]] = None,
    split: Optional[int] = None,
) -> torch.Tensor:
    """Return ``(a * scale_a)^T @ (b * scale_b)`` of shape [N1, N2].

    a_fp8: [M, N1] row-major FP8. b_fp8: [M, N2] row-major FP8. scale_*: one-element
    fp32 device tensors (dequant multipliers). Never syncs the host.
    """
    assert a_fp8.dim() == 2 and b_fp8.dim() == 2 and a_fp8.shape[0] == b_fp8.shape[0]
    assert a_fp8.stride(1) == 1 and b_fp8.stride(1) == 1
    m, n1 = a_fp8.shape
    n2 = b_fp8.shape[1]
    if m == 0:
        return torch.zeros((n1, n2), device=a_fp8.device, dtype=out_dtype)
    bk, bn1, bn2, num_warps, num_stages = config or DEFAULT_CONFIG
    bn1 = min(bn1, n1)
    bn2 = min(bn2, n2)
    assert n1 % bn1 == 0 and n2 % bn2 == 0, (n1, n2, bn1, bn2)
    tiles = (n1 // bn1) * (n2 // bn2)
    if split is None:
        # About one wave of programs over the CUs, but never a split shorter
        # than a few K blocks.
        split = max(1, min(_num_cus(a_fp8.device) // tiles, triton.cdiv(m, 4 * bk)))
    tokens_per_split = triton.cdiv(triton.cdiv(m, split), bk) * bk
    split = triton.cdiv(m, tokens_per_split)
    ws = torch.empty((split, n1, n2), device=a_fp8.device, dtype=torch.float32)
    _fp8_wgrad_partial_kernel[(tiles, split)](
        a_fp8,
        b_fp8,
        ws,
        m,
        n1,
        n2,
        a_fp8.stride(0),
        b_fp8.stride(0),
        tokens_per_split,
        BLOCK_K=bk,
        BLOCK_N1=bn1,
        BLOCK_N2=bn2,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    out = torch.empty((n1, n2), device=a_fp8.device, dtype=out_dtype)
    numel = n1 * n2
    block = 1024
    _fp8_wgrad_reduce_kernel[(triton.cdiv(numel, block),)](
        ws,
        out,
        scale_a.reshape(-1),
        scale_b.reshape(-1),
        SPLIT=split,
        NUMEL=numel,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _transpose_bytes_kernel(
    src_ptr,
    dst_ptr,
    M,
    M_PAD,
    N: tl.constexpr,
    stride_src,
    stride_dst,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(
        src_ptr + rm[:, None] * stride_src + rn[None, :],
        mask=(rm < M)[:, None],
        other=0,
    )
    tl.store(
        dst_ptr + rn[:, None].to(tl.int64) * stride_dst + rm[None, :],
        tl.trans(x),
        mask=(rm < M_PAD)[None, :],
    )


# (BLOCK_M tokens, BLOCK_N columns, num_warps). Picked by a block sweep: this
# config runs at plain-copy speed at the out-proj shape; smaller tiles are slower.
TRANSPOSE_CONFIG: Tuple[int, int, int] = (256, 128, 8)


def transpose_fp8_token_major(
    x_fp8: torch.Tensor,
    pad_to: int = 16,
    config: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """[M, N] row-major FP8 -> [N, M_pad] (tokens contiguous), zero-padded so the
    token count is a multiple of ``pad_to`` (torch._scaled_mm needs K % 16)."""
    assert x_fp8.dim() == 2 and x_fp8.stride(1) == 1
    m, n = x_fp8.shape
    m_pad = triton.cdiv(m, pad_to) * pad_to
    bm, bn, num_warps = config or TRANSPOSE_CONFIG
    bn = min(bn, n)
    assert n % bn == 0
    out = torch.empty((n, m_pad), device=x_fp8.device, dtype=x_fp8.dtype)
    src = x_fp8.view(torch.uint8)
    dst = out.view(torch.uint8)
    _transpose_bytes_kernel[(triton.cdiv(m_pad, bm), n // bn)](
        src,
        dst,
        m,
        m_pad,
        n,
        x_fp8.stride(0),
        m_pad,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=num_warps,
    )
    return out


def fp8_wgrad_tn(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Same result as fp8_wgrad, via token-major copies (Triton transpose) and the
    library FP8 TN GEMM. hipBLASLt accumulates in fp32."""
    a_t = transpose_fp8_token_major(a_fp8)
    b_t = transpose_fp8_token_major(b_fp8)
    return torch._scaled_mm(
        a_t,
        b_t.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )
