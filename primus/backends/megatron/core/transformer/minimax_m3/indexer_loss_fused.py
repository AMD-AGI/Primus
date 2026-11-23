###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bitwise fused form of :func:`indexer_loss.compute_sparse_indexer_loss`.

The eager loss is ~25 tiny kernels in the forward and ~35 in the backward of every MSA layer
(rows of ``topk`` ≈ 18 elements). This module keeps every rounding of the eager graph:

* the order-sensitive reductions -- the softmax, its backward, the target's row sum and the
  broadcast-gradient sum of ``torch.minimum`` -- still run on the same ATen kernels;
* the scalar chain ``kl.sum(-1).mean() * coeff`` stays ordinary autograd;
* everything else is elementwise (gather, fills, compare/select, ``min``, ``log``, IEEE divides)
  or an exact row max/count, and is fused into Triton kernels; the logs use ATen's log kernel
  (one in-place launch), whose bits Triton's ``log`` does not reproduce.

``fold_scatter`` (config switch ``msa_fused_indexer_loss_scatter``) also writes the gradient of
``block_scores`` row by row from the last backward kernel instead of
``zeros(...).scatter_add_(-1, idx.clamp_min(0), g)``: three launches fewer per call. Exact because
the valid block indices of a row are distinct and every invalid slot adds +0 to block 0, so each
element is ``0 + (its one g, or nothing)`` in any order. The premise is structural, not a property
of the data: ``block_indices`` is the output of ``block_scores.topk`` (``indexer.py``, invalid
slots set to -1) or of ``msa_fused_select_plan``, itself checked bit-identical to it, and
``scatter_add_`` on a GPU is order-free only under the same premise. The first-call check runs on
the call's own inputs. Rows wider than ``_SCATTER_MAX_BLOCKS`` = 64 blocks (sequence 8192 at
``sparse_block_size`` 128) and scores whose leading shape is not the indices' keep ``scatter_add_``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check
from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
    compute_sparse_indexer_loss,
)


@triton.jit
def _fwd_scores_kernel(
    bs_ptr,
    idx_ptr,
    logits_ptr,
    scores_ptr,
    rowinfo_ptr,
    R,
    NB: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    idx = tl.load(idx_ptr + rows[:, None] * K + cols[None, :], mask=m, other=-1)
    invalid = idx < 0
    idxc = tl.where(invalid, 0, idx)
    s = tl.load(bs_ptr + rows[:, None] * NB + idxc, mask=m, other=0.0)
    ninf = float("-inf")
    s = tl.where(invalid, ninf, s)
    finite = (s == s) & (tl.abs(s) != float("inf"))
    w = tl.where(finite & m, s, ninf)
    fm = tl.max(w, axis=1)
    no_free = fm == ninf
    cnt = tl.sum(((w == fm[:, None]) & m).to(tl.int32), axis=1)
    # torch.minimum: NaN propagates
    lg = tl.where((s < fm[:, None]) | (s != s), s, fm[:, None])
    lg = tl.where(no_free[:, None], 0.0, lg)
    off = rows[:, None] * K + cols[None, :]
    tl.store(logits_ptr + off, lg, mask=m)
    tl.store(scores_ptr + off, s, mask=m)
    tl.store(rowinfo_ptr + rows * 3 + 0, fm, mask=rmask)
    tl.store(rowinfo_ptr + rows * 3 + 1, no_free.to(tl.float32), mask=rmask)
    tl.store(rowinfo_ptr + rows * 3 + 2, cnt.to(tl.float32), mask=rmask)


@triton.jit
def _fwd_target_kernel(
    t0_ptr,
    tsum_ptr,
    p_ptr,
    t_ptr,
    q_ptr,
    N,
    R,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    t0 = tl.load(t0_ptr + off, mask=m, other=1.0)
    ts = tl.load(tsum_ptr + rows, mask=rmask, other=1.0)
    ts = tl.where(ts < 1e-10, 1e-10, ts)  # clamp_min (NaN passes through, as in ATen)
    t = t0 / ts[:, None]
    p = tl.load(p_ptr + off, mask=m, other=1.0)
    tl.store(t_ptr + off, t, mask=m)
    # The logs are taken by ATen's own log kernel (one in-place launch over both halves):
    # Triton's log differs from it in the last bit for about a third of the inputs.
    tl.store(q_ptr + off, t + 1e-10, mask=m)
    tl.store(q_ptr + N + off, p + 1e-10, mask=m)


@triton.jit
def _fwd_kl_kernel(
    t_ptr,
    q_ptr,
    idx_ptr,
    rowinfo_ptr,
    kl_ptr,
    N,
    R,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    t = tl.load(t_ptr + off, mask=m, other=0.0)
    lt = tl.load(q_ptr + off, mask=m, other=0.0)
    lp = tl.load(q_ptr + N + off, mask=m, other=0.0)
    kl = t * (lt - lp)
    idx = tl.load(idx_ptr + off, mask=m, other=-1)
    nf = tl.load(rowinfo_ptr + rows * 3 + 1, mask=rmask, other=0.0) != 0.0
    kl = tl.where((idx < 0) | nf[:, None], 0.0, kl)
    tl.store(kl_ptr + off, kl, mask=m)


@triton.jit
def _bwd_p_kernel(
    g_ptr,
    g_s0,
    g_s1,
    t_ptr,
    p_ptr,
    idx_ptr,
    rowinfo_ptr,
    gp_ptr,
    R,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    g = tl.load(g_ptr + rows[:, None] * g_s0 + cols[None, :] * g_s1, mask=m, other=0.0)
    idx = tl.load(idx_ptr + off, mask=m, other=-1)
    nf = tl.load(rowinfo_ptr + rows * 3 + 1, mask=rmask, other=0.0) != 0.0
    g = tl.where((idx < 0) | nf[:, None], 0.0, g)  # MaskedFillBackward
    t = tl.load(t_ptr + off, mask=m, other=0.0)
    gd = g * t  # MulBackward (other = log diff): grad * t
    p = tl.load(p_ptr + off, mask=m, other=1.0)
    gq = (-gd) / (p + 1e-10)  # SubBackward (neg) then LogBackward (grad / self)
    tl.store(gp_ptr + off, gq, mask=m)


@triton.jit
def _bwd_min_kernel(
    gl_ptr,
    scores_ptr,
    rowinfo_ptr,
    gself_ptr,
    goth_ptr,
    R,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    g = tl.load(gl_ptr + off, mask=m, other=0.0)
    nf = tl.load(rowinfo_ptr + rows * 3 + 1, mask=rmask, other=0.0) != 0.0
    g = tl.where(nf[:, None], 0.0, g)  # MaskedFillBackward (no_free)
    s = tl.load(scores_ptr + off, mask=m, other=0.0)
    fm = tl.load(rowinfo_ptr + rows * 3 + 0, mask=rmask, other=0.0)[:, None]
    half = tl.where(s == fm, g / 2, g)
    gs = tl.where(s > fm, 0.0, half)
    go = tl.where(s < fm, 0.0, half)
    tl.store(gself_ptr + off, gs, mask=m)
    tl.store(goth_ptr + off, go, mask=m)


@triton.jit
def _bwd_scores_kernel(
    gself_ptr,
    gothsum_ptr,
    scores_ptr,
    idx_ptr,
    rowinfo_ptr,
    gs_ptr,
    R,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
    SCATTER: tl.constexpr = False,
    NB: tl.constexpr = 1,
    NBP: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    s = tl.load(scores_ptr + off, mask=m, other=0.0)
    fm = tl.load(rowinfo_ptr + rows * 3 + 0, mask=rmask, other=0.0)[:, None]
    cnt = tl.load(rowinfo_ptr + rows * 3 + 2, mask=rmask, other=1.0)[:, None]
    go = tl.load(gothsum_ptr + rows, mask=rmask, other=0.0)[:, None]
    finite = (s == s) & (tl.abs(s) != float("inf"))
    w = tl.where(finite, s, float("-inf"))
    # AmaxBackward: (grad / count) * mask ; WhereBackward: where(finite, ., 0)
    ga = (go / cnt) * (w == fm).to(tl.float32)
    ga = tl.where(finite, ga, 0.0)
    gsf = tl.load(gself_ptr + off, mask=m, other=0.0)
    g = gsf + ga
    idx = tl.load(idx_ptr + off, mask=m, other=-1)
    g = tl.where(idx < 0, 0.0, g)  # MaskedFillBackward (invalid)
    if SCATTER:
        # zeros(NB).scatter_add_(-1, idx.clamp_min(0), g) written as whole rows: a row's valid
        # blocks are distinct and every other contribution (an invalid slot -> block 0, g = +0)
        # is +0, so each block is 0 + (its one g, or nothing) in any order, which is what the
        # scatter's atomic adds produce; "+ 0.0" turns a -0 into +0 as 0 + (-0) does.
        nb = tl.arange(0, NBP)
        hit = (idx[:, :, None] == nb[None, None, :]) & m[:, :, None]
        contrib = tl.where(hit, g[:, :, None], 0.0)
        out = tl.sum(contrib, axis=1) + 0.0
        om = rmask[:, None] & (nb[None, :] < NB)
        tl.store(gs_ptr + rows[:, None] * NB + nb[None, :], out, mask=om)
    else:
        tl.store(gs_ptr + off, g, mask=m)


_BR = 64
# fold_scatter: rows per program and the widest block_scores row the folded kernel takes
_SCATTER_BR = 16
_SCATTER_MAX_BLOCKS = 64


def _can_fold(block_scores, block_indices) -> bool:
    """Whether the folded kernel may write the gradient of ``block_scores``: it writes one row of
    blocks per row of indices into an uninitialised tensor, so the two must have the same rows."""
    return (
        block_scores.dtype == torch.float32
        and block_scores.shape[-1] <= _SCATTER_MAX_BLOCKS
        and block_scores.shape[:-1] == block_indices.shape[:-1]
    )


class _SparseIndexerKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block_scores, block_indices, slot_mass, fold_scatter=False):
        shape = block_indices.shape
        K = shape[-1]
        NB = block_scores.shape[-1]
        R = block_indices.numel() // K
        KP = triton.next_power_of_2(K)
        bs = block_scores.contiguous()
        idx = block_indices.contiguous()
        dev = bs.device
        logits = torch.empty(shape, device=dev, dtype=torch.float32)
        scores = torch.empty(shape, device=dev, dtype=torch.float32)
        rowinfo = torch.empty((R, 3), device=dev, dtype=torch.float32)
        grid = (triton.cdiv(R, _BR),)
        _fwd_scores_kernel[grid](bs, idx, logits, scores, rowinfo, R, NB, K, KP, _BR)
        p = torch.softmax(logits, dim=-1)
        target = slot_mass.detach().float()
        tsum = target.sum(dim=-1, keepdim=True).contiguous()  # on the eager layout: same reduction
        t0 = target.contiguous()
        t = torch.empty_like(t0)
        N = R * K
        q = torch.empty((2,) + tuple(shape), device=dev, dtype=torch.float32)
        _fwd_target_kernel[grid](t0, tsum, p, t, q, N, R, K, KP, _BR)
        q.log_()
        kl = torch.empty(shape, device=dev, dtype=torch.float32)
        _fwd_kl_kernel[grid](t, q, idx, rowinfo, kl, N, R, K, KP, _BR)
        ctx.save_for_backward(idx, scores, rowinfo, p, t)
        # Eligibility of the folded scatter, decided here before any backward work.
        fold = bool(fold_scatter) and _can_fold(block_scores, block_indices)
        ctx.meta = (block_scores.shape, block_scores.dtype, R, K, KP, fold)
        return kl

    @staticmethod
    def backward(ctx, grad_kl):
        idx, scores, rowinfo, p, t = ctx.saved_tensors
        bs_shape, bs_dtype, R, K, KP, fold = ctx.meta
        grid = (triton.cdiv(R, _BR),)
        g2 = grad_kl.reshape(R, K)
        gp = torch.empty_like(p)
        _bwd_p_kernel[grid](g2, g2.stride(0), g2.stride(1), t, p, idx, rowinfo, gp, R, K, KP, _BR)
        gl = torch._softmax_backward_data(gp, p, -1, torch.float32)
        gself = torch.empty_like(gl)
        goth = torch.empty_like(gl)
        _bwd_min_kernel[grid](gl, scores, rowinfo, gself, goth, R, K, KP, _BR)
        gothsum = goth.sum(dim=-1, keepdim=True)
        if fold:
            # the scatter folded into the kernel: no zero fill, clamp or scatter_add_ launches
            NB = bs_shape[-1]
            grad_bs = torch.empty(bs_shape, device=gl.device, dtype=bs_dtype)
            _bwd_scores_kernel[(triton.cdiv(R, _SCATTER_BR),)](
                gself, gothsum, scores, idx, rowinfo, grad_bs, R, K, KP, _SCATTER_BR,
                SCATTER=True, NB=NB, NBP=triton.next_power_of_2(NB),
            )  # fmt: skip
            return grad_bs, None, None, None
        gs = torch.empty_like(gl)
        _bwd_scores_kernel[grid](gself, gothsum, scores, idx, rowinfo, gs, R, K, KP, _BR)
        grad_bs = torch.zeros(bs_shape, device=gs.device, dtype=bs_dtype).scatter_add_(
            -1, idx.clamp_min(0), gs
        )
        return grad_bs, None, None, None


def fused_sparse_indexer_loss(slot_mass, block_scores, block_indices, loss_coeff, fold_scatter=False):
    """Same value and gradients as ``compute_sparse_indexer_loss`` (bit for bit)."""
    kl = _SparseIndexerKL.apply(block_scores, block_indices, slot_mass, fold_scatter)
    return kl.sum(dim=-1).mean() * loss_coeff


def _checked(name, slot_mass, block_scores, block_indices, loss_coeff, fold_scatter) -> bool:
    inputs = (slot_mass, block_scores, block_indices)
    return bitwise_check.passed(
        name,
        lambda: bitwise_check.outputs_and_grads(
            lambda m, s, i: fused_sparse_indexer_loss(m, s, i, loss_coeff, fold_scatter), *inputs
        ),
        lambda: bitwise_check.outputs_and_grads(
            lambda m, s, i: compute_sparse_indexer_loss(m, s, i, loss_coeff), *inputs
        ),
        key=tuple(tuple(t.shape) for t in inputs),
    )


def sparse_indexer_loss(slot_mass, block_scores, block_indices, loss_coeff, fold_scatter=False):
    """:func:`fused_sparse_indexer_loss` if the first call with these shapes finds it
    bit-identical to ``compute_sparse_indexer_loss`` on its inputs, else the eager loss for
    every call with these shapes. ``fold_scatter`` (see the module docstring) is checked the same
    way under its own name and falls back to the fused loss with ``scatter_add_``."""
    args = (slot_mass, block_scores, block_indices, loss_coeff)
    if not _checked("msa_fused_indexer_loss", *args, False):
        return compute_sparse_indexer_loss(*args)
    fold = (
        bool(fold_scatter)
        and _can_fold(block_scores, block_indices)
        and _checked("msa_fused_indexer_loss_scatter", *args, True)
    )
    return fused_sparse_indexer_loss(*args, fold_scatter=fold)
