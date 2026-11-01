###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton GEMMs for an FP32 MoE router: the gating GEMM and its backward.

The router GEMM is [tokens x hidden] x [hidden x experts] with few experts, an output too narrow for
hipBLASLt to spread over the GPU. These kernels give each workgroup a block of tokens and the whole
expert dimension:

- forward: FP32 logits = BF16 x @ BF16 W^T with FP32 accumulation, the math of TE's BF16 -> FP32 GEMM;
- backward: BF16 dX = dlogits @ W, and dW from FP32 partial sums over token splits, summed and cast to
  W's dtype. The FP32 dlogits enter the MMAs as a BF16 hi + lo pair, which keeps the backward about as
  accurate as Megatron's FP32 GEMMs without FP32 copies of x and W.

:func:`supported` says which operands they take. On gfx1250 the forward has its own launch config and the
backward forms the hi + lo pair once, in a kernel both GEMMs read; both give the bits of the default
kernels, as each output sums the same products in the same order.
"""

import torch
import triton
import triton.language as tl

from primus.core.utils.gpu_arch import is_gfx1250

_FWD_BLOCK_M, _FWD_BLOCK_K = 32, 64
# dX with four warps: eight are slower with many experts. dW's tile (``_dw_config``) depends on the
# expert count: the narrow block is faster from 128 experts up.
_DX_BLOCK_M, _DX_BLOCK_H, _DX_WARPS = 128, 128, 4

# gfx1250 forward: a deep, three-stage k-loop keeps W's tile in flight. Every output still sums the same
# 32-deep k-steps in the same order, so the logits are those of the default config bit for bit; the
# stages must fit in LDS, so the k-step shrinks as the expert count grows.
_FWD_STAGES, _FWD_LDS_BUDGET = 3, 256 * 1024


def _fwd_launch(k: int, n: int, gfx1250: bool):
    """(BLOCK_K, launch kwargs) of the forward for K = ``k`` and ``n`` experts."""
    if gfx1250:
        for block_k in (256, 128):
            if k % block_k == 0 and _FWD_STAGES * (_FWD_BLOCK_M + n) * block_k * 2 <= _FWD_LDS_BUDGET:
                return block_k, {"num_warps": 4, "num_stages": _FWD_STAGES}
    return _FWD_BLOCK_K, {"num_warps": 4}


def _dw_config(num_experts: int):
    return (32, 64, 4, 4) if num_experts >= 128 else (64, 64, 4, 8)


# gfx1250 backward on a pre-split hi + lo pair (``_split``), formed once instead of in every workgroup of
# both GEMMs. The row block and the split count of dW stay those of ``_dw_config``: they fix each
# output's summation order.
_SPLIT_BLOCK = 4096
_DX_SPLIT_LAUNCH = (128, 256, {"num_warps": 8, "num_stages": 1})
_DW_SPLIT_BLOCK_H, _DW_SPLIT_STAGES = 64, 3


def _dw_split_warps(num_experts: int) -> int:
    return 4 if num_experts >= 256 else 8


@triton.jit
def _router_fwd_kernel(
    X, W, O, M, sxm, swn, som, K: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    """O[M, N] (fp32) = X[M, K] @ W[N, K]^T; N is the whole expert dimension."""
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    mmask = rm < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(X + rm[:, None] * sxm + (k0 + rk)[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(W + rn[None, :] * swn + (k0 + rk)[:, None])
        acc += tl.dot(x, w)
    tl.store(O + rm[:, None] * som + rn[None, :], acc, mask=mmask[:, None])


@triton.jit
def _router_dx_kernel(
    G, W, DX, M, sgm, swn, sdm, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    """DX[M, H] (bf16) = G[M, N] (fp32, as bf16 hi + lo) @ W[N, H] (bf16)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rn = tl.arange(0, N)
    mmask = rm < M
    g = tl.load(G + rm[:, None] * sgm + rn[None, :], mask=mmask[:, None], other=0.0)
    w = tl.load(W + rn[:, None] * swn + rh[None, :])
    g_hi = g.to(tl.bfloat16)
    acc = tl.dot(g_hi, w)
    acc += tl.dot((g - g_hi.to(tl.float32)).to(tl.bfloat16), w)
    tl.store(DX + rm[:, None] * sdm + rh[None, :], acc.to(tl.bfloat16), mask=mmask[:, None])


@triton.jit
def _router_dw_kernel(
    G, X, P, M, sgm, sxm, sps, spn, N: tl.constexpr, M_PER_SPLIT, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    """P[s, N, H] (fp32) = G[rows of split s, N]^T @ X[rows of split s, H] (G as bf16 hi + lo)."""
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    rn = tl.arange(0, N)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((N, BLOCK_H), dtype=tl.float32)
    m_start = pid_s * M_PER_SPLIT
    for m0 in range(m_start, m_start + M_PER_SPLIT, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M)
        mmask = rm < M
        g = tl.load(G + rm[None, :] * sgm + rn[:, None], mask=mmask[None, :], other=0.0)
        x = tl.load(X + rm[:, None] * sxm + rh[None, :], mask=mmask[:, None], other=0.0)
        g_hi = g.to(tl.bfloat16)
        acc += tl.dot(g_hi, x)
        acc += tl.dot((g - g_hi.to(tl.float32)).to(tl.bfloat16), x)
    tl.store(P + pid_s * sps + rn[:, None] * spn + rh[None, :], acc)


@triton.jit
def _router_split_kernel(G, HI, LO, numel, BLOCK: tl.constexpr):
    """HI, LO (bf16) = the hi + lo pair of G (fp32), formed as the kernels above form it."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.load(G + offs, mask=mask)
    hi = g.to(tl.bfloat16)
    tl.store(HI + offs, hi, mask=mask)
    tl.store(LO + offs, (g - hi.to(tl.float32)).to(tl.bfloat16), mask=mask)


@triton.jit
def _router_dx_split_kernel(
    GH, GL, W, DX, M, sgm, swn, sdm, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    """DX[M, H] (bf16) = (GH + GL)[M, N] @ W[N, H]."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rn = tl.arange(0, N)
    mmask = rm < M
    gh = tl.load(GH + rm[:, None] * sgm + rn[None, :], mask=mmask[:, None], other=0.0)
    gl = tl.load(GL + rm[:, None] * sgm + rn[None, :], mask=mmask[:, None], other=0.0)
    w = tl.load(W + rn[:, None] * swn + rh[None, :])
    acc = tl.dot(gh, w)
    acc += tl.dot(gl, w)
    tl.store(DX + rm[:, None] * sdm + rh[None, :], acc.to(tl.bfloat16), mask=mmask[:, None])


@triton.jit
def _router_dw_split_kernel(
    GH,
    GL,
    X,
    P,
    M,
    sgm,
    sxm,
    sps,
    spn,
    N: tl.constexpr,
    M_PER_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """P[s, N, H] (fp32) = (GH + GL)[rows of split s, N]^T @ X[rows of split s, H], the products summed
    in the order of ``_router_dw_kernel``."""
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    rn = tl.arange(0, N)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((N, BLOCK_H), dtype=tl.float32)
    m_start = pid_s * M_PER_SPLIT
    for m0 in range(m_start, m_start + M_PER_SPLIT, BLOCK_M):
        rm = m0 + tl.arange(0, BLOCK_M)
        mmask = rm < M
        gh = tl.load(GH + rm[None, :] * sgm + rn[:, None], mask=mmask[None, :], other=0.0)
        gl = tl.load(GL + rm[None, :] * sgm + rn[:, None], mask=mmask[None, :], other=0.0)
        x = tl.load(X + rm[:, None] * sxm + rh[None, :], mask=mmask[:, None], other=0.0)
        acc += tl.dot(gh, x)
        acc += tl.dot(gl, x)
    tl.store(P + pid_s * sps + rn[:, None] * spn + rh[None, :], acc)


def supported(inp: torch.Tensor, weight: torch.Tensor, bias, router_dtype) -> bool:
    """Whether these kernels take the router GEMM of ``inp`` [..., hidden] and ``weight`` [N, hidden]."""
    n, k = weight.shape
    return (
        router_dtype == torch.float32
        and inp.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and bias is None
        and inp.is_cuda
        and weight.is_contiguous()
        and inp.shape[-1] == k
        and 16 <= n <= 256
        and n & (n - 1) == 0
        # the tiles along hidden are not masked: hidden is a multiple of the widest (gfx1250's dX, 256)
        and k % 256 == 0
        # the kernels index with int32 offsets
        and inp.numel() < 2**31
    )


def forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP32 logits [M, N] = ``x`` [M, K] @ ``w`` [N, K]^T, for operands :func:`supported` takes."""
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    block_k, launch = _fwd_launch(x.shape[1], w.shape[0], is_gfx1250())
    _router_fwd_kernel[(triton.cdiv(x.shape[0], _FWD_BLOCK_M),)](
        x,
        w,
        out,
        x.shape[0],
        x.stride(0),
        w.stride(0),
        out.stride(0),
        K=x.shape[1],
        N=w.shape[0],
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_K=block_k,
        **launch,
    )
    return out


def backward(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """(dX [M, K] bf16, dW [N, K] in ``w``'s dtype) for FP32 dlogits ``g`` [M, N] of :func:`forward`."""
    if is_gfx1250():
        hi, lo = _split(g)
        return _dx_split(hi, lo, w), _dw_split(hi, lo, x).to(w.dtype)
    return _dx(g, w), _dw(g, x).to(w.dtype)


def _dx(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    dx = torch.empty(g.shape[0], w.shape[1], dtype=torch.bfloat16, device=g.device)
    _router_dx_kernel[(triton.cdiv(g.shape[0], _DX_BLOCK_M), w.shape[1] // _DX_BLOCK_H)](
        g,
        w,
        dx,
        g.shape[0],
        g.stride(0),
        w.stride(0),
        dx.stride(0),
        N=w.shape[0],
        BLOCK_M=_DX_BLOCK_M,
        BLOCK_H=_DX_BLOCK_H,
        num_warps=_DX_WARPS,
    )
    return dx


def _dw(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = g.shape[0]
    block_m, block_h, splits, warps = _dw_config(g.shape[1])
    per = triton.cdiv(triton.cdiv(m, splits), block_m) * block_m
    p = torch.empty(splits, g.shape[1], x.shape[1], dtype=torch.float32, device=g.device)
    _router_dw_kernel[(x.shape[1] // block_h, splits)](
        g,
        x,
        p,
        m,
        g.stride(0),
        x.stride(0),
        p.stride(0),
        p.stride(1),
        N=g.shape[1],
        M_PER_SPLIT=per,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=warps,
    )
    return p.sum(0)


def _split(g: torch.Tensor):
    hi = torch.empty(g.shape, dtype=torch.bfloat16, device=g.device)
    lo = torch.empty_like(hi)
    _router_split_kernel[(triton.cdiv(g.numel(), _SPLIT_BLOCK),)](g, hi, lo, g.numel(), BLOCK=_SPLIT_BLOCK)
    return hi, lo


def _dx_split(hi: torch.Tensor, lo: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    dx = torch.empty(hi.shape[0], w.shape[1], dtype=torch.bfloat16, device=hi.device)
    block_m, block_h, launch = _DX_SPLIT_LAUNCH
    _router_dx_split_kernel[(triton.cdiv(hi.shape[0], block_m), w.shape[1] // block_h)](
        hi,
        lo,
        w,
        dx,
        hi.shape[0],
        hi.stride(0),
        w.stride(0),
        dx.stride(0),
        N=w.shape[0],
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        **launch,
    )
    return dx


def _dw_split(hi: torch.Tensor, lo: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = hi.shape[0]
    block_m, _, splits, _ = _dw_config(hi.shape[1])
    per = triton.cdiv(triton.cdiv(m, splits), block_m) * block_m
    p = torch.empty(splits, hi.shape[1], x.shape[1], dtype=torch.float32, device=hi.device)
    _router_dw_split_kernel[(x.shape[1] // _DW_SPLIT_BLOCK_H, splits)](
        hi,
        lo,
        x,
        p,
        m,
        hi.stride(0),
        x.stride(0),
        p.stride(0),
        p.stride(1),
        N=hi.shape[1],
        M_PER_SPLIT=per,
        BLOCK_M=block_m,
        BLOCK_H=_DW_SPLIT_BLOCK_H,
        num_warps=_dw_split_warps(hi.shape[1]),
        num_stages=_DW_SPLIT_STAGES,
    )
    return p.sum(0)
