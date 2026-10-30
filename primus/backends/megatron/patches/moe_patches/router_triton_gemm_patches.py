###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton GEMMs for an FP32 MoE router (opt-in, ``moe_router_triton_gemm: true``).

The router GEMMs are [tokens x hidden] x [hidden x experts] with experts = 128: the output is too
narrow for hipBLASLt on gfx1250 (the forward BF16->FP32 GEMM gets ~32 workgroups, the backward runs
FP32 GEMMs with 32x32 tiles on FP32 copies of the input and weight). These kernels give each
workgroup a block of tokens and the whole expert dimension:

- forward: FP32 logits = BF16 x @ BF16 W^T, FP32 accumulation (same math as TE's BF16->FP32 GEMM);
- dX: BF16 = FP32 dlogits @ BF16 W, and dW: FP32 partial sums over 4 token splits, summed and cast to
  the weight dtype. dlogits enter the MMAs as a BF16 hi + lo pair, so the backward keeps FP32
  accuracy (measured equal to Megatron's FP32 backward against an FP64 reference) without FP32
  copies of the input and weight.

Router dtype FP32 with BF16 input and weight, no bias, and shapes the kernels tile exactly; anything
else keeps Megatron's path. Measured on MI455X for [8192 x 7168] x [7168 x 128]: fwd 346 -> 56 us,
dX 282 -> 84 us, dW 324 -> 100 us per call.
"""

import torch
import triton
import triton.language as tl

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_FWD_BLOCK_M, _FWD_BLOCK_K = 32, 64
# Measured on MI455X at 8192 tokens, hidden 7168, 16-256 experts. Four warps for dX: eight cost
# 2.8x at 256 experts (338 -> 120 us). dW (BLOCK_M, BLOCK_H, splits, warps) depends on the expert
# count: the narrow block wins from 128 experts up (256: 217 -> 130 us), the shipped one below.
_DX_BLOCK_M, _DX_BLOCK_H, _DX_WARPS = 128, 128, 4


def _dw_config(num_experts: int):
    return (32, 64, 4, 4) if num_experts >= 128 else (64, 64, 4, 8)


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


def _supported(inp: torch.Tensor, weight: torch.Tensor, bias, router_dtype) -> bool:
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
        and k % _FWD_BLOCK_K == 0
        and k % _DX_BLOCK_H == 0
        and k % _dw_config(n)[1] == 0
        # the kernels index with int32 offsets
        and inp.numel() < 2**31
    )


def router_gemm_on_triton(inp: torch.Tensor, weight: torch.Tensor, bias, router_dtype) -> bool:
    """Whether the router gating GEMM for these operands runs on these Triton kernels: the patch is
    installed and the shapes are ones it takes (so the GEMM uses no TE workspace)."""
    from megatron.core.transformer.moe import moe_utils

    return bool(getattr(moe_utils.RouterGatingLinearFunction, "_primus_router_triton_gemm_patched", False)) and (
        _supported(inp.reshape(-1, inp.shape[-1]), weight, bias, router_dtype)
    )


def _fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
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
        BLOCK_K=_FWD_BLOCK_K,
        num_warps=4,
    )
    return out


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


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_triton_gemm", False))


@register_patch(
    "megatron.moe.router_triton_gemm",
    backend="megatron",
    phase="before_train",
    description="Triton forward/backward GEMMs for the FP32 MoE router (FP32-accurate backward)",
    condition=_enabled,
)
def patch_router_triton_gemm(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import moe_utils

    fn = moe_utils.RouterGatingLinearFunction
    if getattr(fn, "_primus_router_triton_gemm_patched", False):
        return
    if getattr(get_args(ctx), "moe_router_bwd_bf16", False):
        log_rank_0(
            "[Patch:megatron.moe.router_triton_gemm] not installed: moe_router_bwd_bf16 is set "
            "(the two options are mutually exclusive)"
        )
        return
    orig_forward, orig_backward = fn.forward, fn.backward

    def forward(ctx_, inp, weight, bias, router_dtype):
        ctx_.primus_triton = _supported(inp, weight, bias, router_dtype)
        if not ctx_.primus_triton:
            return orig_forward(ctx_, inp, weight, bias, router_dtype)
        ctx_.save_for_backward(inp, weight, bias)
        ctx_.router_dtype = router_dtype
        ctx_.input_dtype = inp.dtype
        ctx_.weight_dtype = weight.dtype
        shape = inp.shape
        out = _fwd(inp.reshape(-1, shape[-1]).contiguous(), weight)
        return out.view(*shape[:-1], -1)

    def backward(ctx_, grad_output):
        if not getattr(ctx_, "primus_triton", False):
            return orig_backward(ctx_, grad_output)
        inp, weight, _ = ctx_.saved_tensors
        shape = inp.shape
        x = inp.reshape(-1, shape[-1]).contiguous()
        g = grad_output.reshape(-1, grad_output.shape[-1]).float().contiguous()
        grad_input = _dx(g, weight).view(*shape)
        grad_weight = _dw(g, x).to(ctx_.weight_dtype)
        return grad_input, grad_weight, None, None

    fn.forward = staticmethod(forward)
    fn.backward = staticmethod(backward)
    fn._primus_router_triton_gemm_patched = True
    log_rank_0("[Patch:megatron.moe.router_triton_gemm] FP32 router GEMMs on Triton kernels")
