###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_indexer_loss_one_pass``: the flydsl backend's slot mass and sparse indexer KL in one
Triton kernel, and their backward in one. The same terms summed in another order.

What it replaces (``attention.msa_core`` with ``want_loss``), per MSA layer call:

* ``slot_mass_from_slot_lse`` (subtract, exp, mean over the GQA group's heads: three ATen
  launches over the ``[sq, b, heads, topk]`` slot_lse) and, with ``msa_aligned_block_walk``, the
  scatter that puts the mass back into the indexer's slot order;
* ``indexer_loss_fused.sparse_indexer_loss`` (three Triton kernels, ATen softmax, row sum, log,
  ``kl.sum(-1)``) and in the backward three more kernels, ATen's softmax backward and a row sum.

That path is bit-identical to ``compute_sparse_indexer_loss`` and pays for it with ATen launches
for every order-sensitive reduction and for ``log``. Here one forward kernel reads slot_lse, lse,
the block table and the block scores and writes each row's KL (a row = one (batch, index head,
token)); ``klrow.mean() * coeff`` stays ordinary autograd. One backward kernel takes the row
gradient to the gradient of ``block_scores``.

Same computation: the KL is block-level over the selected blocks, the
target is the attention mass of each selected block (mean over the group's heads of
``exp(slot_lse - lse)``, L1-normalised with the 1e-10 clamp), the prediction is the indexer's
softmax over the selected blocks with the forced ``+inf`` clamped to the row's largest finite
score, rows without a free block give 0, ``-1`` slots give 0. What differs is rounding only:

* the sums of a row (group mean, target sum, softmax, KL row sum, softmax backward, the
  ``minimum`` broadcast sum) are Triton reductions over <= 32 elements instead of ATen's;
* ``exp`` and ``log`` are Triton's (they differ from ATen's in the last bit on some inputs);
* with ``msa_aligned_block_walk`` a row is processed in walk order (the forward kernel's slot_lse
  order, with the walk table as the row's block list) instead of being permuted back first. Every
  step is either elementwise or a sum over the row, so only the order of the sums changes.

Deterministic: no atomics; every output element is written by exactly one program (the folded
scatter writes whole rows; the direct scatter for wide rows writes ``out[row, block]`` for the
row's valid blocks, which are distinct, into a zero-filled tensor). Cannot hang: the kernels are
straight-line Triton launched with one wave per program (``_WARPS``), so the compiled code has no
workgroup barrier, no wait on another wave and no loop; their grids come from the row count.

The first call for a shape runs this path twice and the path it replaces once, on the call's own
inputs, and keeps the replaced path unless both runs give the same bits and the loss and the
block-score gradient are within ``_LOSS_RTOL`` / ``_GRAD_RTOL`` of it (:func:`_checked`).
A gfx1250 path of ``attention._MSA_GFX1250_PATHS``, taken with ``msa_fused_indexer_loss``.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Optional

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check
from primus.core.utils.module_utils import log_rank_0, warning_rank_all

# First-call tolerances against the replaced (bit-exact) path. The MSA gate allows the loss a
# relative error of 1e-6 against its FP32 reference (the starting tree's is 0), so the loss bound
# sits below that; the gradient bound is far below the ~2e-3 the gate allows d index q / k.
_LOSS_RTOL = 5.0e-7
_GRAD_RTOL = 2.0e-5
# widest block_scores row the folded scatter takes (as indexer_loss_fused._SCATTER_MAX_BLOCKS)
_FOLD_MAX_BLOCKS = 64
# One wave per program (num_warps 1): a Triton reduction across the waves of a program goes through
# LDS and workgroup barriers (30 barrier pairs in the forward at 4 waves); with one wave the
# compiled kernels contain no barrier at all (checked on the gfx1250 listings, see the unit tests'
# notes), and nothing in them waits for another wave.
_WARPS = 1
_FOLD_BR = 4
_BWD_BR = 8
# elements of the forward's [rows, heads per group, slots] block per program, and its most rows
_FWD_ELEMS = 1024
_FWD_MAX_ROWS = 8

_decided: Dict[Hashable, bool] = {}


@triton.jit
def _fwd_kernel(
    bs_ptr,
    tbl_ptr,
    slse_ptr,
    lse_ptr,
    p_ptr,
    t_ptr,
    scores_ptr,
    rowinfo_ptr,
    klrow_ptr,
    R,
    S,
    B,
    G,
    NB: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    NREP: tl.constexpr,
    NREPP: tl.constexpr,
    BR: tl.constexpr,
):
    # one program: BR rows; row r = (b * G + g) * S + s of the [b, n_index, sq] layout
    rows = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    cols = tl.arange(0, KP)
    reps = tl.arange(0, NREPP)
    rmask = rows < R
    cm = cols < K
    m = rmask[:, None] & cm[None, :]
    ninf = float("-inf")

    # ---- target: mean over the group's heads of exp(slot_lse - lse), then L1-normalised
    s = rows % S
    bg = rows // S
    g = bg % G
    b = bg // G
    qrow = (s * B + b) * (G * NREP) + g * NREP  # first head of the row's group in [sq, b, heads]
    hm = rmask[:, None] & (reps[None, :] < NREP)
    m3 = hm[:, :, None] & cm[None, None, :]
    head = qrow[:, None] + reps[None, :]
    sl = tl.load(slse_ptr + head[:, :, None] * K + cols[None, None, :], mask=m3, other=0.0)
    ls = tl.load(lse_ptr + head, mask=hm, other=0.0)
    mass = tl.where(m3, tl.exp(sl - ls[:, :, None]), 0.0)
    t0 = tl.sum(mass, axis=1) / NREP
    ts = tl.sum(t0, axis=1)
    ts = tl.where(ts < 1e-10, 1e-10, ts)  # clamp_min (NaN passes through, as in ATen)
    t = t0 / ts[:, None]

    # ---- prediction: softmax of the selected blocks' scores, +inf clamped to the finite max
    off = rows[:, None] * K + cols[None, :]
    idx = tl.load(tbl_ptr + off, mask=m, other=-1)
    invalid = idx < 0
    idxc = tl.where(invalid, 0, idx)
    sc = tl.load(bs_ptr + rows[:, None] * NB + idxc, mask=m, other=0.0)
    sc = tl.where(invalid, ninf, sc)
    finite = (sc == sc) & (tl.abs(sc) != float("inf"))
    w = tl.where(finite & m, sc, ninf)
    fm = tl.max(w, axis=1)
    no_free = fm == ninf
    cnt = tl.sum(((w == fm[:, None]) & m).to(tl.int32), axis=1)
    # torch.minimum: NaN propagates
    lg = tl.where((sc < fm[:, None]) | (sc != sc), sc, fm[:, None])
    lg = tl.where(no_free[:, None], 0.0, lg)
    lg = tl.where(m, lg, ninf)  # padding columns take no probability
    e = tl.exp(lg - tl.max(lg, axis=1)[:, None])
    e = tl.where(m, e, 0.0)
    p = e / tl.sum(e, axis=1)[:, None]

    # ---- KL per row
    kl = t * (tl.log(t + 1e-10) - tl.log(p + 1e-10))
    kl = tl.where(invalid | no_free[:, None], 0.0, kl)
    kl = tl.where(m, kl, 0.0)
    tl.store(klrow_ptr + rows, tl.sum(kl, axis=1), mask=rmask)

    # ---- saved for the backward
    tl.store(p_ptr + off, p, mask=m)
    tl.store(t_ptr + off, t, mask=m)
    tl.store(scores_ptr + off, sc, mask=m)
    tl.store(rowinfo_ptr + rows * 3 + 0, fm, mask=rmask)
    tl.store(rowinfo_ptr + rows * 3 + 1, no_free.to(tl.float32), mask=rmask)
    tl.store(rowinfo_ptr + rows * 3 + 2, cnt.to(tl.float32), mask=rmask)


@triton.jit
def _bwd_kernel(
    g_ptr,
    t_ptr,
    p_ptr,
    tbl_ptr,
    scores_ptr,
    rowinfo_ptr,
    out_ptr,
    R,
    NB: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
    FOLD: tl.constexpr,
    NBP: tl.constexpr,
):
    rows = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    cols = tl.arange(0, KP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    idx = tl.load(tbl_ptr + off, mask=m, other=-1)
    invalid = idx < 0
    nf = tl.load(rowinfo_ptr + rows * 3 + 1, mask=rmask, other=0.0) != 0.0
    fm = tl.load(rowinfo_ptr + rows * 3 + 0, mask=rmask, other=0.0)[:, None]
    cnt = tl.load(rowinfo_ptr + rows * 3 + 2, mask=rmask, other=1.0)
    # row sum backward (the row's gradient on every slot), then the KL's masked fill
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0)[:, None] + tl.zeros([BR, KP], tl.float32)
    g = tl.where(invalid | nf[:, None], 0.0, g)
    g = tl.where(m, g, 0.0)
    t = tl.load(t_ptr + off, mask=m, other=0.0)
    p = tl.load(p_ptr + off, mask=m, other=1.0)
    gq = (-(g * t)) / (p + 1e-10)  # mul (other = log difference), sub (neg), log (grad / self)
    # softmax backward: (grad - sum(grad * p)) * p
    dot = tl.sum(tl.where(m, gq * p, 0.0), axis=1)
    gl = (gq - dot[:, None]) * p
    gl = tl.where(nf[:, None], 0.0, gl)  # masked fill (no_free)
    # minimum(scores, finite_max): ties split in half, as torch
    s = tl.load(scores_ptr + off, mask=m, other=0.0)
    half = tl.where(s == fm, gl / 2, gl)
    gs = tl.where(s > fm, 0.0, half)
    go = tl.where(s < fm, 0.0, half)
    gosum = tl.sum(tl.where(m, go, 0.0), axis=1)
    # amax backward: (grad / count) on the maxima; where(finite, ., -inf) backward
    finite = (s == s) & (tl.abs(s) != float("inf"))
    w = tl.where(finite, s, float("-inf"))
    ga = (gosum / cnt)[:, None] * (w == fm).to(tl.float32)
    ga = tl.where(finite, ga, 0.0)
    gg = gs + ga
    gg = tl.where(invalid, 0.0, gg)  # masked fill (invalid) of the gathered scores
    if FOLD:
        # zeros(NB).scatter_add_(-1, idx.clamp_min(0), gg) written as whole rows; see
        # indexer_loss_fused._bwd_scores_kernel (a row's valid blocks are distinct)
        nb = tl.arange(0, NBP)
        hit = (idx[:, :, None] == nb[None, None, :]) & m[:, :, None]
        out = tl.sum(tl.where(hit, gg[:, :, None], 0.0), axis=1) + 0.0
        om = rmask[:, None] & (nb[None, :] < NB)
        tl.store(out_ptr + rows[:, None] * NB + nb[None, :], out, mask=om)
    else:
        # wide rows: out is zero-filled; a row's valid blocks are distinct, so every element is
        # written at most once ("+ 0.0" turns a -0 into +0, as 0 + (-0) does)
        idxc = tl.where(invalid, 0, idx)
        tl.store(out_ptr + rows[:, None] * NB + idxc, gg + 0.0, mask=m & (idx >= 0))


def _fwd_rows(nrepp: int, kp: int) -> int:
    """Rows per forward program: a power of two, at most ``_FWD_MAX_ROWS``, with the program's
    [rows, heads, slots] block at most ``_FWD_ELEMS`` elements. From the head and slot counts only
    (never from S or B)."""
    br = _FWD_MAX_ROWS
    while br > 1 and br * nrepp * kp > _FWD_ELEMS:
        br //= 2
    return br


class _OnePassKL(torch.autograd.Function):
    """``klrow [R]`` from ``(block_scores, table, slot_lse, lse)``; differentiable in
    ``block_scores`` only (the target is detached in the loss it replaces)."""

    @staticmethod
    def forward(ctx, block_scores, table, slot_lse, lse):
        B, G, S, NB = block_scores.shape
        K = table.shape[-1]
        NREP = slot_lse.shape[2] // G
        R = B * G * S
        KP = triton.next_power_of_2(K)
        NREPP = triton.next_power_of_2(NREP)
        BR = _fwd_rows(NREPP, KP)
        bs = block_scores.contiguous()
        tbl = table.contiguous()
        dev = bs.device
        p = torch.empty((R, K), device=dev, dtype=torch.float32)
        t = torch.empty((R, K), device=dev, dtype=torch.float32)
        scores = torch.empty((R, K), device=dev, dtype=torch.float32)
        rowinfo = torch.empty((R, 3), device=dev, dtype=torch.float32)
        klrow = torch.empty((R,), device=dev, dtype=torch.float32)
        _fwd_kernel[(triton.cdiv(R, BR),)](
            bs, tbl, slot_lse.contiguous(), lse.contiguous(), p, t, scores, rowinfo, klrow,
            R, S, B, G, NB=NB, K=K, KP=KP, NREP=NREP, NREPP=NREPP, BR=BR, num_warps=_WARPS,
        )  # fmt: skip
        ctx.save_for_backward(tbl, scores, rowinfo, p, t)
        ctx.set_materialize_grads(False)
        # which scatter the backward uses is decided here, before any backward work
        ctx.meta = (bs.shape, R, K, KP, NB, NB <= _FOLD_MAX_BLOCKS)
        return klrow

    @staticmethod
    def backward(ctx, grad_klrow):
        if grad_klrow is None:
            return None, None, None, None
        tbl, scores, rowinfo, p, t = ctx.saved_tensors
        bs_shape, R, K, KP, NB, fold = ctx.meta
        g = grad_klrow.contiguous()
        if fold:
            out = torch.empty(bs_shape, device=g.device, dtype=torch.float32)
            br = _FOLD_BR
        else:
            out = torch.zeros(bs_shape, device=g.device, dtype=torch.float32)
            br = _BWD_BR
        _bwd_kernel[(triton.cdiv(R, br),)](
            g, t, p, tbl, scores, rowinfo, out, R,
            NB=NB, K=K, KP=KP, BR=br, FOLD=fold, NBP=triton.next_power_of_2(NB) if fold else 1,
            num_warps=_WARPS,
        )  # fmt: skip
        return out, None, None, None


def eligible(slot_lse, lse, block_scores, table) -> bool:
    """Shapes and dtypes the kernels serve, decided before any work: FP32 slot_lse
    ``[sq, b, heads, topk]``, lse ``[sq, b, heads]`` and block scores ``[b, n_index, sq, n_blocks]``
    on the GPU, an int32 / int64 ``[b, n_index, sq, topk]`` table, heads a multiple of n_index,
    at most 32 slots and 64 heads per group (the kernels' row blocks)."""
    if not (block_scores.is_cuda and block_scores.dim() == 4 and slot_lse.dim() == 4 and lse.dim() == 3):
        return False
    B, G, S, NB = block_scores.shape
    K = table.shape[-1]
    heads = slot_lse.shape[2]
    return (
        block_scores.dtype == torch.float32
        and slot_lse.dtype == torch.float32
        and lse.dtype == torch.float32
        and table.dtype in (torch.int32, torch.int64)
        and tuple(table.shape) == (B, G, S, K)
        and tuple(slot_lse.shape) == (S, B, heads, K)
        and tuple(lse.shape) == (S, B, heads)
        and heads % G == 0
        and 1 <= K <= 32
        and heads // G <= 64
        and B * G * S > 0
    )


def one_pass_loss(slot_lse, lse, block_scores, table, loss_coeff):
    """The loss on the one-pass kernels, unchecked. ``table`` is the block list in the order of
    slot_lse's last axis: the aligned walk's table when the forward walked it, else the indexer's
    ``block_indices``."""
    klrow = _OnePassKL.apply(block_scores, table, slot_lse.detach(), lse.detach())
    return klrow.mean() * loss_coeff


def _loss_and_grad(fn: Callable, block_scores):
    leaf = block_scores.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = fn(leaf)
    (grad,) = torch.autograd.grad(loss, leaf)
    return loss.detach(), grad


def _checked(key: Hashable, run: Callable, reference: Callable, block_scores) -> bool:
    """First-call decision for ``key``: the one-pass path twice (same bits required) and the
    replaced path once, on the call's inputs; relative loss error <= ``_LOSS_RTOL``, relative
    gradient error (2-norm) <= ``_GRAD_RTOL``. Like ``bitwise_check.passed``: nothing is decided
    under inference mode or graph capture, a GPU fault raises, any other error keeps the replaced
    path, and each rank decides alone (no collective on either path)."""
    if key in _decided:
        return _decided[key]
    if torch.is_inference_mode_enabled() or (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    ):
        return False
    try:
        loss1, grad1 = _loss_and_grad(run, block_scores)
        loss2, grad2 = _loss_and_grad(run, block_scores)
        ref_loss, ref_grad = _loss_and_grad(reference, block_scores)
        same = bitwise_check.same_bits(loss1, loss2) and bitwise_check.same_bits(grad1, grad2)
        finite = bool(torch.isfinite(ref_loss)) and bool(torch.isfinite(ref_grad).all())
        loss_err = float(
            (loss1.double() - ref_loss.double()).abs() / ref_loss.double().abs().clamp_min(1e-30)
        )
        grad_err = float(
            (grad1.double() - ref_grad.double()).norm() / ref_grad.double().norm().clamp_min(1e-30)
        )
        ok = same and finite and loss_err <= _LOSS_RTOL and grad_err <= _GRAD_RTOL
        reason = (
            f"loss rel err {loss_err:.3e} (limit {_LOSS_RTOL:.1e}), block-score gradient rel err "
            f"{grad_err:.3e} (limit {_GRAD_RTOL:.1e}), two runs {'identical' if same else 'DIFFER'}"
            f"{'' if finite else ', reference not finite'}"
        )
    except Exception as e:  # noqa: BLE001 - any failure but a GPU fault keeps the replaced path
        if bitwise_check._device_error(e):
            raise
        ok, reason = False, f"{type(e).__name__}: {e}"
    _decided[key] = ok
    if ok:
        log_rank_0(f"[msa_indexer_loss_one_pass {key}] checked: {reason}; in use")
    else:
        warning_rank_all(
            f"[msa_indexer_loss_one_pass {key}] rejected: {reason}; keeping the path it replaces"
        )
    return ok


def one_pass_indexer_loss(
    slot_lse, lse, block_scores, block_indices, walk, loss_coeff, reference: Callable
) -> Optional[torch.Tensor]:
    """The sparse indexer loss of ``msa_core`` on the one-pass kernels, or None when the shapes are
    not :func:`eligible` or the first call for these shapes rejected the path (the caller then runs
    the path it replaces).

    Args:
        slot_lse, lse: the flydsl forward's per-slot and total logsumexp (walk order if ``walk``).
        block_scores, block_indices: the indexer's outputs.
        walk: ``(walk_table, slots)`` of ``block_walk.aligned_block_walk`` or None.
        reference: ``block_scores -> loss`` on the replaced path (slot mass, walk scatter and
            ``sparse_indexer_loss``), used by the first-call check only.
    """
    table = block_indices if walk is None else walk[0]
    if not eligible(slot_lse, lse, block_scores, table):
        return None
    key = (
        tuple(block_scores.shape),
        tuple(slot_lse.shape),
        str(table.dtype),
        float(loss_coeff),
        walk is not None,
    )
    if not _checked(
        key, lambda bs: one_pass_loss(slot_lse, lse, bs, table, loss_coeff), reference, block_scores
    ):
        return None
    return one_pass_loss(slot_lse, lse, block_scores, table, loss_coeff)


def emulate(slot_lse, lse, block_scores, table, loss_coeff):
    """The kernels' arithmetic in plain PyTorch, differentiable by autograd; runs on any device.
    The unit tests' oracle for the walk-order formulation and (through
    :func:`emulate_backward`) for the hand-written backward chain; not used at run time."""
    B, G, S, NB = block_scores.shape
    K = table.shape[-1]
    heads = slot_lse.shape[2]
    mass = torch.exp(slot_lse.detach() - lse.detach().unsqueeze(-1))
    t0 = mass.view(S, B, G, heads // G, K).sum(dim=3) / (heads // G)
    t0 = t0.permute(1, 2, 0, 3)
    t = t0 / t0.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    idx = table.long()
    invalid = idx < 0
    sc = block_scores.gather(-1, idx.clamp_min(0)).masked_fill(invalid, -torch.inf)
    finite = torch.isfinite(sc)
    fm = torch.where(finite, sc, -torch.inf).amax(dim=-1, keepdim=True)
    no_free = ~finite.any(dim=-1, keepdim=True)
    lg = torch.minimum(sc, fm).masked_fill(no_free, 0.0)
    e = torch.exp(lg - lg.amax(dim=-1, keepdim=True).detach())
    p = e / e.sum(dim=-1, keepdim=True)
    kl = t * (torch.log(t + 1e-10) - torch.log(p + 1e-10))
    kl = kl.masked_fill(invalid | no_free, 0.0)
    return kl.sum(dim=-1).reshape(-1).mean() * loss_coeff


def emulate_backward(slot_lse, lse, block_scores, table, grad_klrow):
    """``_bwd_kernel``'s chain in plain PyTorch (float64 inside): the gradient of ``block_scores``
    for a row gradient ``grad_klrow [R]``. Test oracle only."""
    B, G, S, NB = block_scores.shape
    K = table.shape[-1]
    heads = slot_lse.shape[2]
    dd = torch.float64
    mass = torch.exp(slot_lse.to(dd) - lse.to(dd).unsqueeze(-1))
    t0 = (mass.view(S, B, G, heads // G, K).sum(dim=3) / (heads // G)).permute(1, 2, 0, 3)
    t = (t0 / t0.sum(dim=-1, keepdim=True).clamp_min(1e-10)).reshape(-1, K)
    idx = table.long().reshape(-1, K)
    invalid = idx < 0
    bs = block_scores.to(dd).reshape(-1, NB)
    s = bs.gather(-1, idx.clamp_min(0)).masked_fill(invalid, -torch.inf)
    finite = torch.isfinite(s)
    w = torch.where(finite, s, -torch.inf)
    fm = w.amax(dim=-1, keepdim=True)
    nf = fm == -torch.inf
    cnt = (w == fm).sum(dim=-1, keepdim=True).to(dd)
    lg = torch.where((s < fm) | s.isnan(), s, fm.expand_as(s))
    lg = torch.where(nf, torch.zeros_like(lg), lg)
    e = torch.exp(lg - lg.amax(dim=-1, keepdim=True))
    p = e / e.sum(dim=-1, keepdim=True)
    g = grad_klrow.to(dd).reshape(-1, 1).expand(-1, K)
    g = torch.where(invalid | nf, torch.zeros_like(g), g)
    gq = (-(g * t)) / (p + 1e-10)
    gl = (gq - (gq * p).sum(dim=-1, keepdim=True)) * p
    gl = torch.where(nf, torch.zeros_like(gl), gl)
    half = torch.where(s == fm, gl / 2, gl)
    gs = torch.where(s > fm, torch.zeros_like(half), half)
    go = torch.where(s < fm, torch.zeros_like(half), half)
    ga = (go.sum(dim=-1, keepdim=True) / cnt) * (w == fm).to(dd)
    ga = torch.where(finite, ga, torch.zeros_like(ga))
    gg = torch.where(invalid, torch.zeros_like(gs), gs + ga)
    out = torch.zeros_like(bs).scatter_add_(-1, idx.clamp_min(0), gg)
    return out.reshape(block_scores.shape).to(block_scores.dtype)
