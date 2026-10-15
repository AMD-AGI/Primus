###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton kernels for the MoE router gating linear (Megatron ``router_gating_linear``), one pinned config each.

With ``moe_router_dtype: fp32`` Megatron computes, per micro-batch (M tokens, K hidden, E experts):
  fwd    logits[M,E] fp32 = x[M,K] bf16 @ W[E,K]^T bf16            TE general_gemm TN, fp32 out
  dgrad  dx[M,K] bf16     = bf16(g[M,E] fp32 @ fp32(W)[E,K])        fp32 GEMM NN + 2 casts
  wgrad  dW[E,K] bf16     = bf16(g^T[E,M] fp32 @ fp32(x)[M,K])      fp32 GEMM NT + 2 casts (fp32 copy of x)
For E = 32 these skinny shapes land on large generic hipBLASLt tiles.

Here:
  fwd    tl.dot on the bf16 operands with fp32 accumulation (the same math as the TN GEMM, other summation order)
  dgrad / wgrad  the fp32 gradient is split exactly into three bf16 terms g = g1 + g2 + g3 (each split removes
         8 significant bits of the 24; the residual is exact in fp32), W and x are bf16 already, so every
         product g_i * w is exact in fp32 and the fp32 accumulation is the only rounding, as in the fp32 GEMM.
  wgrad reduces over M in SPLIT_M row slices into an fp32 partial buffer [SPLIT_M, E, K]; torch.sum over the
         split dim (fixed order, deterministic) and one cast give dW. No atomics anywhere.

One pinned config per kernel (constants below), no autotune, num_warps 4. No masks are needed: every launch is
guarded by ``eligible()`` (M % 1024 == 0, K % 128 == 0, E == 32, contiguous operands), which keeps every
program's tile inside its tensors (the unit tests check this bound for every admitted launch).

The autograd function and the drop-in for ``router_gating_linear`` are at the end of this file.
"""

import torch
import triton
import triton.language as tl

# ---- pinned configs (do not tune in process) ----
FWD_BM, FWD_BK, FWD_WARPS, FWD_STAGES = 32, 128, 4, 2
DG_BM, DG_BN, DG_WARPS, DG_STAGES = 64, 128, 4, 1
WG_BM, WG_BN, WG_SPLIT_M, WG_WARPS, WG_STAGES = 64, 128, 16, 4, 2
E_SUPPORTED = (32,)
M_MULTIPLE = 1024  # lcm(FWD_BM, DG_BM, WG_SPLIT_M * WG_BM)
K_MULTIPLE = 128  # lcm(FWD_BK, DG_BN, WG_BN)


@triton.jit
def _router_fwd_kernel(x_ptr, w_ptr, y_ptr, M, K, k_len, E: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    # y[pid_k, m, e] (fp32) = sum_{k in split pid_k} x[m, k] * w[e, k]
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_e = tl.arange(0, E)
    offs_k = tl.arange(0, BK)
    k0 = pid_k * k_len
    x_ptrs = x_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :]
    w_ptrs = w_ptr + offs_e[:, None] * K + (k0 + offs_k)[None, :]
    acc = tl.zeros((BM, E), dtype=tl.float32)
    for _ in range(0, k_len, BK):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BK
        w_ptrs += BK
    y_ptrs = y_ptr + pid_k.to(tl.int64) * M * E + offs_m[:, None] * E + offs_e[None, :]
    tl.store(y_ptrs, acc)


@triton.jit
def _split3(v):
    # exact: v == f32(v1) + f32(v2) + f32(v3) for every finite fp32 v (barring subnormal underflow)
    v1 = v.to(tl.bfloat16)
    r1 = v - v1.to(tl.float32)
    v2 = r1.to(tl.bfloat16)
    r2 = r1 - v2.to(tl.float32)
    v3 = r2.to(tl.bfloat16)
    return v1, v2, v3


@triton.jit
def _router_dgrad_kernel(g_ptr, w_ptr, dx_ptr, M, K, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # dx[m, n] (bf16) = sum_e g[m, e] (fp32) * w[e, n] (bf16)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_e = tl.arange(0, E)
    g = tl.load(g_ptr + offs_m[:, None] * E + offs_e[None, :])
    w = tl.load(w_ptr + offs_e[:, None] * K + offs_n[None, :])
    g1, g2, g3 = _split3(g)
    acc = tl.dot(g1, w)
    acc = tl.dot(g2, w, acc)
    acc = tl.dot(g3, w, acc)
    tl.store(dx_ptr + offs_m[:, None] * K + offs_n[None, :], acc.to(tl.bfloat16))


@triton.jit
def _router_wgrad_kernel(
    g_ptr, x_ptr, part_ptr, M, K, rows_per_split, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr
):
    # part[s, e, n] (fp32) = sum_{m in split s} g[m, e] (fp32) * x[m, n] (bf16)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_e = tl.arange(0, E)
    offs_bm = tl.arange(0, BM)
    m_start = pid_s * rows_per_split
    acc = tl.zeros((E, BN), dtype=tl.float32)
    for m0 in range(0, rows_per_split, BM):
        offs_m = (m_start + m0 + offs_bm).to(tl.int64)
        gt = tl.load(g_ptr + offs_m[None, :] * E + offs_e[:, None])  # [E, BM] = g^T slice
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_n[None, :])  # [BM, BN]
        g1, g2, g3 = _split3(gt)
        acc = tl.dot(g1, x, acc)
        acc = tl.dot(g2, x, acc)
        acc = tl.dot(g3, x, acc)
    p_ptrs = part_ptr + pid_s.to(tl.int64) * E * K + offs_e[:, None] * K + offs_n[None, :]
    tl.store(p_ptrs, acc)


def eligible(x2d, weight):
    """True when the pinned kernels cover the call exactly (else the caller keeps Megatron's path)."""
    if not (x2d.is_cuda and x2d.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16):
        return False
    if x2d.dim() != 2 or weight.dim() != 2 or not x2d.is_contiguous() or not weight.is_contiguous():
        return False
    M, K = x2d.shape
    E, K2 = weight.shape
    return K2 == K and E in E_SUPPORTED and M > 0 and M % M_MULTIPLE == 0 and K % K_MULTIPLE == 0


def router_fwd(x2d, weight):
    """logits fp32 [M, E] = x2d @ weight^T (x2d [M, K] bf16, weight [E, K] bf16)."""
    M, K = x2d.shape
    E = weight.shape[0]
    assert K % FWD_BK == 0 and M % FWD_BM == 0
    y = torch.empty((M, E), device=x2d.device, dtype=torch.float32)
    # One K split (grid dim 1 == 1): the kernel's partial-sum layout [split, M, E] is then y itself.
    _router_fwd_kernel[(M // FWD_BM, 1)](
        x2d, weight, y, M, K, K, E=E, BM=FWD_BM, BK=FWD_BK, num_warps=FWD_WARPS, num_stages=FWD_STAGES
    )
    return y


def router_dgrad(g2d, weight):
    """dx bf16 [M, K] = g2d (fp32 [M, E]) @ weight (bf16 [E, K])."""
    M, E = g2d.shape
    K = weight.shape[1]
    assert M % DG_BM == 0 and K % DG_BN == 0
    dx = torch.empty((M, K), device=g2d.device, dtype=torch.bfloat16)
    _router_dgrad_kernel[(M // DG_BM, K // DG_BN)](
        g2d, weight, dx, M, K, E=E, BM=DG_BM, BN=DG_BN, num_warps=DG_WARPS, num_stages=DG_STAGES
    )
    return dx


def router_wgrad(g2d, x2d, out_dtype=torch.bfloat16):
    """dW [E, K] = g2d^T (fp32) @ x2d (bf16), fp32 accumulation, cast to out_dtype."""
    M, E = g2d.shape
    K = x2d.shape[1]
    assert M % (WG_SPLIT_M * WG_BM) == 0 and K % WG_BN == 0
    part = torch.empty((WG_SPLIT_M, E, K), device=g2d.device, dtype=torch.float32)
    _router_wgrad_kernel[(K // WG_BN, WG_SPLIT_M)](
        g2d,
        x2d,
        part,
        M,
        K,
        M // WG_SPLIT_M,
        E=E,
        BM=WG_BM,
        BN=WG_BN,
        num_warps=WG_WARPS,
        num_stages=WG_STAGES,
    )
    return part.sum(dim=0).to(out_dtype)


# ---------------------------------------------------------------- autograd + drop-in

ORIGINAL_ATTR = "__primus_triton_router_gemm_orig__"


class RouterGatingLinearTriton(torch.autograd.Function):
    """logits = inp @ weight^T in fp32 (bias None, router dtype fp32); backward through the Triton kernels."""

    @staticmethod
    def forward(ctx, inp, weight):
        ctx.save_for_backward(inp, weight)
        shape = inp.shape
        y = router_fwd(inp.view(-1, shape[-1]), weight)
        return y.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        shape = inp.shape
        x2d = inp.view(-1, shape[-1])
        g = grad_output.reshape(-1, grad_output.shape[-1])
        if g.dtype != torch.float32:
            g = g.float()
        g = g.contiguous()
        dx = router_dgrad(g, weight).view(*shape)
        dw = router_wgrad(g, x2d, out_dtype=weight.dtype)
        return dx, dw


def make_router_gating_linear(orig):
    """Drop-in for megatron.core.transformer.moe.moe_utils.router_gating_linear(inp, weight, bias, router_dtype).

    Calls the Triton path when bias is None, router_dtype is fp32 and ``eligible()`` accepts the flattened input
    and the weight; every other call goes to ``orig`` unchanged.
    """

    def router_gating_linear(inp, weight, bias, router_dtype):
        if (
            bias is None
            and router_dtype == torch.float32
            and inp.is_contiguous()
            and eligible(inp.view(-1, inp.shape[-1]), weight)
        ):
            return RouterGatingLinearTriton.apply(inp, weight)
        return orig(inp, weight, bias, router_dtype)

    router_gating_linear.__wrapped__ = orig
    setattr(router_gating_linear, ORIGINAL_ATTR, orig)
    return router_gating_linear
