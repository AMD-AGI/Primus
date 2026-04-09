###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Triton BF16 Fused Grouped GEMM for MoE Expert computation on AMD MI355X.

All E experts are computed in a **single kernel launch** using a 2-D grid
``(E, max_tiles_per_expert)`` — no CPU-side tile map needed.  Each program
computes its own (expert, m_tile, n_tile) from its grid coordinates.

Three kernels:
  1. Forward / dA:  ``_bf16_fused_fwd_kernel``  — C_e = A_e @ B_e (or B_e^T)
  2. Weight grad:   ``_bf16_fused_dw_kernel``   — dW_e = A_e^T @ grad_e

Public entry: :func:`fused_gmm` (drop-in for ``grouped_gemm.ops.gmm``).
"""

import math
import os
from typing import List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


# ---------------------------------------------------------------------------
# Tile-size selection
# ---------------------------------------------------------------------------

_BN = 256
_BK = 64
_WARPS = 8
_STAGES = 2

_DW_BK = 128
_DW_BN = 128
_DW_BM_RED = 64
_DW_WARPS = 4
_DW_STAGES = 2


def _select_block_m(max_tokens: int) -> int:
    if max_tokens <= 64:
        return 64
    return 128


# ---------------------------------------------------------------------------
# Triton kernels  (2-D grid, no pre-built tile map)
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _bf16_fused_fwd_kernel(
        A_ptr, B_ptr, C_ptr,
        expert_starts_ptr,
        N_OUT: tl.constexpr,
        K_IN: tl.constexpr,
        stride_am, stride_ak,
        stride_be, stride_b1, stride_b2,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TRANS_B: tl.constexpr,
        NUM_N_TILES: tl.constexpr,
    ):
        """Grid = (E, max_tiles_per_expert).
        Program (eid, local_tid) computes one [BM, BN] output tile.
        """
        eid = tl.program_id(0)
        local_tid = tl.program_id(1)

        e_start = tl.load(expert_starts_ptr + eid)
        e_end   = tl.load(expert_starts_ptr + eid + 1)
        e_M     = e_end - e_start

        num_m_tiles = tl.cdiv(e_M, BLOCK_M)
        total_tiles = num_m_tiles * NUM_N_TILES
        if local_tid >= total_tiles:
            return

        pid_m = local_tid // NUM_N_TILES
        pid_n = local_tid % NUM_N_TILES

        m0  = pid_m * BLOCK_M
        n0  = pid_n * BLOCK_N
        gm0 = e_start + m0

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        A_base = A_ptr + gm0 * stride_am
        B_base = B_ptr + eid * stride_be

        for k0 in range(0, K_IN, BLOCK_K):
            m_offs = tl.arange(0, BLOCK_M)[:, None]
            k_offs = k0 + tl.arange(0, BLOCK_K)[None, :]
            a_mask = (m_offs < (e_M - m0)) & (k_offs < K_IN)
            a_ptrs = A_base + m_offs * stride_am + k_offs * stride_ak
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

            kb_offs = k0 + tl.arange(0, BLOCK_K)[:, None]
            nb_offs = n0 + tl.arange(0, BLOCK_N)[None, :]
            b_mask = (kb_offs < K_IN) & (nb_offs < N_OUT)
            if TRANS_B:
                b_ptrs = B_base + nb_offs * stride_b1 + kb_offs * stride_b2
            else:
                b_ptrs = B_base + kb_offs * stride_b1 + nb_offs * stride_b2
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc = tl.dot(
                a_tile.to(tl.bfloat16),
                b_tile.to(tl.bfloat16),
                acc,
                out_dtype=tl.float32,
            )

        m_offs = tl.arange(0, BLOCK_M)[:, None]
        n_offs = n0 + tl.arange(0, BLOCK_N)[None, :]
        c_mask = ((gm0 + m_offs) < e_end) & (n_offs < N_OUT)
        c_ptrs = C_ptr + (gm0 + m_offs) * stride_cm + n_offs * stride_cn
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

    @triton.jit
    def _bf16_fused_dw_kernel(
        A_ptr, grad_ptr, dW_ptr,
        expert_starts_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_am, stride_ak,
        stride_gm, stride_gn,
        stride_dwe, stride_dwk, stride_dwn,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        NUM_N_TILES: tl.constexpr,
    ):
        """Grid = (E, num_k_tiles * num_n_tiles).
        dW[e] = A_e^T @ grad_e, reduced over the token dimension.
        """
        eid = tl.program_id(0)
        local_tid = tl.program_id(1)

        e_start = tl.load(expert_starts_ptr + eid)
        e_end   = tl.load(expert_starts_ptr + eid + 1)
        e_M     = e_end - e_start
        if e_M == 0:
            return

        num_k_tiles = tl.cdiv(K, BLOCK_K)
        total_tiles = num_k_tiles * NUM_N_TILES
        if local_tid >= total_tiles:
            return

        pid_k = local_tid // NUM_N_TILES
        pid_n = local_tid % NUM_N_TILES

        k0 = pid_k * BLOCK_K
        n0 = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

        for m0 in range(0, e_M, BLOCK_M):
            gm = e_start + m0
            m_len = e_M - m0

            m_offs = tl.arange(0, BLOCK_M)[:, None]
            k_offs = k0 + tl.arange(0, BLOCK_K)[None, :]
            a_mask = (m_offs < m_len) & (k_offs < K)
            a_ptrs = A_ptr + (gm + m_offs) * stride_am + k_offs * stride_ak
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

            gm_offs = tl.arange(0, BLOCK_M)[:, None]
            gn_offs = n0 + tl.arange(0, BLOCK_N)[None, :]
            g_mask = (gm_offs < m_len) & (gn_offs < N)
            g_ptrs = grad_ptr + (gm + gm_offs) * stride_gm + gn_offs * stride_gn
            g_tile = tl.load(g_ptrs, mask=g_mask, other=0.0)

            acc = tl.dot(
                tl.trans(a_tile).to(tl.bfloat16),
                g_tile.to(tl.bfloat16),
                acc,
                out_dtype=tl.float32,
            )

        k_offs = k0 + tl.arange(0, BLOCK_K)[:, None]
        n_offs = n0 + tl.arange(0, BLOCK_N)[None, :]
        dw_mask = (k_offs < K) & (n_offs < N)
        dw_ptrs = dW_ptr + eid * stride_dwe + k_offs * stride_dwk + n_offs * stride_dwn
        tl.store(dw_ptrs, acc.to(tl.bfloat16), mask=dw_mask)


# ---------------------------------------------------------------------------
# Python wrappers  (minimal CPU overhead: only expert_starts + grid dims)
# ---------------------------------------------------------------------------

def _make_expert_starts(tpe_list: List[int], device) -> torch.Tensor:
    """Build [E+1] cumsum tensor on GPU from Python list."""
    tpe_t = torch.tensor(tpe_list, dtype=torch.int32, device=device)
    return torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        tpe_t.cumsum(0),
    ])


def fused_bf16_fwd(
    A: torch.Tensor,
    B: torch.Tensor,
    tpe_list: List[int],
    trans_b: bool = False,
    BLOCK_M: Optional[int] = None,
    BLOCK_N: int = _BN,
    BLOCK_K: int = _BK,
    num_warps: int = _WARPS,
    num_stages: int = _STAGES,
) -> torch.Tensor:
    """BF16 fused grouped GEMM forward (single launch, all experts)."""
    assert HAVE_TRITON, "Triton required"
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    E = B.shape[0]
    if trans_b:
        K_in = B.shape[2]
        N_out = B.shape[1]
    else:
        K_in = B.shape[1]
        N_out = B.shape[2]

    total_M = A.shape[0]
    max_m = max(tpe_list) if tpe_list else 0

    if BLOCK_M is None:
        BLOCK_M = _select_block_m(max_m)

    num_n_tiles = math.ceil(N_out / BLOCK_N)
    num_m_tiles_max = math.ceil(max_m / BLOCK_M) if max_m > 0 else 0
    max_tiles = num_m_tiles_max * num_n_tiles

    if max_tiles == 0:
        return torch.zeros(total_M, N_out, dtype=torch.bfloat16, device=A.device)

    es = _make_expert_starts(tpe_list, A.device)
    C = torch.zeros(total_M, N_out, dtype=torch.bfloat16, device=A.device)

    _bf16_fused_fwd_kernel[(E, max_tiles)](
        A, B, C,
        es,
        N_OUT=N_out,
        K_IN=K_in,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_be=B.stride(0), stride_b1=B.stride(1), stride_b2=B.stride(2),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TRANS_B=trans_b,
        NUM_N_TILES=num_n_tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


def fused_bf16_dw(
    A: torch.Tensor,
    grad: torch.Tensor,
    tpe_list: List[int],
    E: int,
    BLOCK_K: int = _DW_BK,
    BLOCK_N: int = _DW_BN,
    BLOCK_M: int = _DW_BM_RED,
    num_warps: int = _DW_WARPS,
    num_stages: int = _DW_STAGES,
) -> torch.Tensor:
    """Weight gradient: dW[e] = A_e^T @ grad_e (single launch, all experts)."""
    assert HAVE_TRITON, "Triton required"
    if not A.is_contiguous():
        A = A.contiguous()
    if not grad.is_contiguous():
        grad = grad.contiguous()

    K = A.shape[1]
    N = grad.shape[1]

    num_k_tiles = math.ceil(K / BLOCK_K)
    num_n_tiles = math.ceil(N / BLOCK_N)
    max_tiles = num_k_tiles * num_n_tiles

    es = _make_expert_starts(tpe_list, A.device)
    dW = torch.zeros(E, K, N, dtype=torch.bfloat16, device=A.device)

    if max_tiles == 0:
        return dW

    _bf16_fused_dw_kernel[(E, max_tiles)](
        A, grad, dW,
        es,
        K=K, N=N,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_gm=grad.stride(0), stride_gn=grad.stride(1),
        stride_dwe=dW.stride(0), stride_dwk=dW.stride(1), stride_dwn=dW.stride(2),
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        NUM_N_TILES=num_n_tiles,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dW


# ---------------------------------------------------------------------------
# Drop-in replacement for grouped_gemm.backend.gmm
# ---------------------------------------------------------------------------

def fused_bf16_gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """Drop-in replacement for ``grouped_gemm.backend.gmm``."""
    tpe_list = batch_sizes.tolist()
    E = len(tpe_list)

    if trans_a:
        return fused_bf16_dw(a, b, tpe_list, E)
    else:
        return fused_bf16_fwd(a, b, tpe_list, trans_b=trans_b)


# ---------------------------------------------------------------------------
# Autograd Function (Triton FWD+dA, hipBLASLt dW)
# ---------------------------------------------------------------------------

try:
    from grouped_gemm import backend as _gg_backend
    _HAVE_GG_BACKEND = True
except ImportError:
    _HAVE_GG_BACKEND = False


class FusedBF16GroupedGemm(torch.autograd.Function):
    """Autograd wrapper: FWD+dA via fused Triton, dW via hipBLASLt."""

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        assert torch.count_nonzero(batch_sizes) != 0, (
            "Input batch_sizes should not be all zeros!"
        )
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        tpe_list = batch_sizes.tolist()
        return fused_bf16_fwd(a, b, tpe_list, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        bgrad = None

        if ctx.needs_input_grad[0]:
            tpe_list = batch_sizes.tolist()
            agrad = fused_bf16_fwd(grad, b, tpe_list, trans_b=not trans_b)

        if ctx.needs_input_grad[1]:
            if _HAVE_GG_BACKEND:
                lhs, rhs = (grad, a) if trans_b else (a, grad)
                batch_sizes_cpu = batch_sizes.cpu() if batch_sizes.is_cuda else batch_sizes
                bgrad = _gg_backend.gmm(
                    lhs, rhs, batch_sizes_cpu, trans_a=True, trans_b=False,
                )
            else:
                tpe_list = batch_sizes.tolist()
                E = b.shape[0]
                lhs, rhs = (grad, a) if trans_b else (a, grad)
                bgrad = fused_bf16_dw(lhs, rhs, tpe_list, E)

        return agrad, bgrad, None, None


def fused_gmm(a, b, batch_sizes, trans_b=False):
    """Public API: drop-in replacement for ``grouped_gemm.ops.gmm``."""
    return FusedBF16GroupedGemm.apply(a, b, batch_sizes, trans_b)
