###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Differentiable stages the FlyDSL KDA chunk assembly is built from.

Two stages live here.

:class:`DecayScores`
    The intra-chunk score matrices, forward by the FlyDSL kernel in
    :mod:`.kda_decay_scores_kernel` and backward by the one in
    :mod:`.kda_decay_scores_bwd_kernel` — the *same* blocked decay-weighted
    contraction with the contraction axis swapped.

    An earlier version took the adjoint by recomputing :func:`decay_scores_torch`
    under ``enable_grad``, which was correct by construction but measured
    **12871 µs of a 23164 µs backward, 56 %**: ~40 elementwise
    ops and batched GEMMs on 100 MB tensors, doubled by autograd, for ~8 GFLOP
    of real arithmetic. :func:`decay_scores_bwd_torch` is the blocked torch twin
    of the new kernel and remains both the unit-test oracle and the fallback for
    geometries the kernel cannot take.

:func:`ut_inverse`
    ``(I − L)^{-1}`` for strictly-lower-triangular ``L``, forward by the FlyDSL
    kernel in :mod:`.kda_ut_inverse_kernel` and backward by the analytic adjoint
    ``dL = tril(Pᵀ dP Pᵀ, −1)``.

    Passes 1–3 spelled the forward as Neumann doubling —  ``L`` is nilpotent
    with ``L^C = 0`` and ``Σ_{k<2n} L^k = (Σ_{k<n} L^k)(I + L^n)``, so
    ``log2(C)`` doublings, 10 batched GEMMs at ``C = 64``. Pass 4 profiled that
    at **576 µs of a 4227 µs on-device forward**, the largest item outside the
    two kernels, and the reason is traffic and not arithmetic: ten round trips
    of a 100 MB tensor, ~3 GB, to invert a matrix that is 16 KB. Forward
    substitution on-chip is one launch, 200 MB and ~68 µs, and it is also
    *more* accurate (3.8e-05 against a fp64 oracle where doubling is 1.1e-04).
    :func:`_ut_inverse_doubling` stays as the fallback for fp64 and for widths
    the kernel is not built for, and the note above it records the blocked
    alternative that was tried and measured slower still.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._compile import (
    compiled,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_decay_scores_bwd_kernel import (
    build_kda_decay_scores_bwd,
    supports_bwd_geometry,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_decay_scores_kernel import (
    SUB_BLOCK,
    SUPPORTED_K,
    build_kda_decay_scores,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_ut_inverse_kernel import (
    build_kda_ut_inverse,
    supports_ut_geometry,
)

__all__ = [
    "SUB_BLOCK",
    "SUPPORTED_K",
    "decay_scores",
    "decay_scores_torch",
    "decay_scores_bwd",
    "decay_scores_bwd_torch",
    "ut_inverse",
    "ut_inverse_beta",
    "supports_ut_beta",
    "supports_geometry",
    "supports_bwd_geometry",
    "supports_ut_geometry",
]

_KERNEL_CACHE: Dict[Tuple[int, int], object] = {}
_BWD_KERNEL_CACHE: Dict[Tuple[int, int], object] = {}
_UT_KERNEL_CACHE: Dict[int, object] = {}
_KERNEL_LOCK = threading.Lock()


def supports_geometry(chunk_size: int, k_dim: int) -> Optional[str]:
    """``None`` when the kernel can run this geometry, else why it cannot."""
    if chunk_size % SUB_BLOCK != 0:
        return f"chunk_size={chunk_size} is not a multiple of the {SUB_BLOCK}-row sub-block"
    if k_dim not in SUPPORTED_K:
        return f"head_dim={k_dim} is not one of {list(SUPPORTED_K)}"
    return None


def _get_kernel(chunk_size: int, k_dim: int):
    key = (int(chunk_size), int(k_dim))
    with _KERNEL_LOCK:
        launch = _KERNEL_CACHE.get(key)
        if launch is None:
            launch = build_kda_decay_scores(chunk_size=key[0], k_dim=key[1])
            _KERNEL_CACHE[key] = launch
        return launch


def _get_bwd_kernel(chunk_size: int, k_dim: int):
    key = (int(chunk_size), int(k_dim))
    with _KERNEL_LOCK:
        launch = _BWD_KERNEL_CACHE.get(key)
        if launch is None:
            launch = build_kda_decay_scores_bwd(chunk_size=key[0], k_dim=key[1])
            _BWD_KERNEL_CACHE[key] = launch
        return launch


# ---------------------------------------------------------------------------
# intra-chunk decay-weighted score matrices
# ---------------------------------------------------------------------------


def _decay_scores_flydsl(q: torch.Tensor, k: torch.Tensor, cg: torch.Tensor):
    """Launch the kernel. ``q, k, cg: [NB, C, K]`` fp32 contiguous."""
    nb, chunk_size, k_dim = q.shape
    launch = _get_kernel(chunk_size, k_dim)
    aqk = torch.empty(nb, chunk_size, chunk_size, dtype=torch.float32, device=q.device)
    akk = torch.empty_like(aqk)
    launch(
        q.reshape(-1),
        k.reshape(-1),
        cg.reshape(-1),
        aqk.reshape(-1),
        akk.reshape(-1),
        int(nb),
    )
    return aqk, akk


def decay_scores_torch(
    q: torch.Tensor, k: torch.Tensor, cg: torch.Tensor, sub_block: int = SUB_BLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch twin of the kernel: same blocking, same reference rows.

    ``q, k, cg: [NB, C, K]``. Returns ``(Aqk, Akk)``, ``[NB, C, C]``, with
    ``Aqk`` masked to ``c ≤ r`` and ``Akk`` to ``c < r``.

    Differentiable, and the adjoint of the kernel's forward by construction.
    Every exponent it evaluates is the one the kernel evaluates, so it is also
    the natural unit-test oracle for the kernel.
    """
    nb, C, _ = q.shape
    SB = sub_block
    NSB = C // SB
    dev, dt = q.device, q.dtype
    idx = torch.arange(SB, device=dev)
    mask_le = (idx.unsqueeze(-1) >= idx.unsqueeze(0)).to(dt)  # c <= r
    mask_lt = (idx.unsqueeze(-1) > idx.unsqueeze(0)).to(dt)  # c <  r

    rows_qk, rows_kk = [], []
    for i in range(NSB):
        r0, r1 = i * SB, (i + 1) * SB
        cols_qk, cols_kk = [], []
        if i > 0:
            # strictly earlier sub-blocks: reference the first row of block i,
            # which makes both exponents non-positive.
            ref = cg[:, r0 : r0 + 1]
            lf = (cg[:, r0:r1] - ref).exp()
            rf = (ref - cg[:, :r0]).exp()
            kt = (k[:, :r0] * rf).transpose(-1, -2)
            cols_qk.append((q[:, r0:r1] * lf) @ kt)
            cols_kk.append((k[:, r0:r1] * lf) @ kt)
        # diagonal sub-block: reference the midpoint, bounding |exponent| by
        # (SB/2)*|g|_max so nothing overflows before the mask is applied.
        mid = r0 + SB // 2
        ref = cg[:, mid : mid + 1]
        lf = (cg[:, r0:r1] - ref).exp()
        rf = (ref - cg[:, r0:r1]).exp()
        kt = (k[:, r0:r1] * rf).transpose(-1, -2)
        cols_qk.append(((q[:, r0:r1] * lf) @ kt) * mask_le)
        cols_kk.append(((k[:, r0:r1] * lf) @ kt) * mask_lt)
        if i < NSB - 1:
            pad = q.new_zeros(nb, SB, C - r1)
            cols_qk.append(pad)
            cols_kk.append(pad)
        rows_qk.append(torch.cat(cols_qk, dim=-1))
        rows_kk.append(torch.cat(cols_kk, dim=-1))
    return torch.cat(rows_qk, dim=-2), torch.cat(rows_kk, dim=-2)


def _decay_scores_bwd_flydsl(q, k, cg, d_aqk, d_akk):
    """Launch the adjoint kernel. All operands ``[NB, *]`` fp32 contiguous."""
    nb, chunk_size, k_dim = q.shape
    launch = _get_bwd_kernel(chunk_size, k_dim)
    dq = torch.empty_like(q)
    dk = torch.empty_like(q)
    dcg = torch.empty_like(q)
    launch(
        q.reshape(-1),
        k.reshape(-1),
        cg.reshape(-1),
        d_aqk.reshape(-1),
        d_akk.reshape(-1),
        dq.reshape(-1),
        dk.reshape(-1),
        dcg.reshape(-1),
        int(nb),
    )
    return dq, dk, dcg


def decay_scores_bwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cg: torch.Tensor,
    d_aqk: torch.Tensor,
    d_akk: torch.Tensor,
    sub_block: int = SUB_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch twin of :func:`_decay_scores_bwd_flydsl`.

    Same blocking, same reference rows, same masks, so it is both the unit-test
    oracle for the kernel and the fallback for geometries it cannot take. See
    :mod:`.kda_decay_scores_bwd_kernel` for the derivation; in short, with
    ``E[r,c,d] = exp(cg[r,d] − cg[c,d])``,

        dq  = Σ_c dAqk[r,c]·E·k[c,d]        A2 = Σ_c dAkk[r,c]·E·k[c,d]
        A1  = Σ_r dAqk[r,c]·E·q[r,d]        A3 = Σ_r dAkk[r,c]·E·k[r,d]
        dk  = A1 + A2 + A3                  dcg = q·dq + k·(A2 − A1 − A3)

    The three reference rows differ per direction because that is what keeps
    both decay factors in ``(0, 1]``: the ``Σ_c`` blocks reference the first row
    of the owning row-block and the ``Σ_r`` blocks the **last** row of the
    owning column-block.
    """
    nb, C, _ = q.shape
    SB = sub_block
    NSB = C // SB
    dev, dt = q.device, q.dtype
    idx = torch.arange(SB, device=dev)
    mask_le = (idx.unsqueeze(-1) >= idx.unsqueeze(0)).to(dt)  # c <= r
    mask_lt = (idx.unsqueeze(-1) > idx.unsqueeze(0)).to(dt)  # c <  r

    dq = torch.zeros_like(q)
    a1 = torch.zeros_like(q)
    a2 = torch.zeros_like(q)
    a3 = torch.zeros_like(q)
    for b in range(NSB):
        r0, r1 = b * SB, (b + 1) * SB
        if b > 0:
            # row owner, every earlier column block at once, ref = first row of b
            ref = cg[:, r0 : r0 + 1]
            lf = (cg[:, r0:r1] - ref).exp()
            krf = k[:, :r0] * (ref - cg[:, :r0]).exp()
            dq[:, r0:r1] += lf * (d_aqk[:, r0:r1, :r0] @ krf)
            a2[:, r0:r1] += lf * (d_akk[:, r0:r1, :r0] @ krf)
        # the diagonal block, ref = midpoint, both gradients masked
        mid = r0 + SB // 2
        ref = cg[:, mid : mid + 1]
        lf = (cg[:, r0:r1] - ref).exp()
        rf = (ref - cg[:, r0:r1]).exp()
        gq = d_aqk[:, r0:r1, r0:r1] * mask_le
        gk = d_akk[:, r0:r1, r0:r1] * mask_lt
        krf = k[:, r0:r1] * rf
        dq[:, r0:r1] += lf * (gq @ krf)
        a2[:, r0:r1] += lf * (gk @ krf)
        a1[:, r0:r1] += rf * (gq.transpose(-1, -2) @ (q[:, r0:r1] * lf))
        a3[:, r0:r1] += rf * (gk.transpose(-1, -2) @ (k[:, r0:r1] * lf))
        if b < NSB - 1:
            # column owner, every later row block at once, ref = LAST row of b
            ref = cg[:, r1 - 1 : r1]
            rf = (ref - cg[:, r0:r1]).exp()
            lf = (cg[:, r1:] - ref).exp()
            a1[:, r0:r1] += rf * (d_aqk[:, r1:, r0:r1].transpose(-1, -2) @ (q[:, r1:] * lf))
            a3[:, r0:r1] += rf * (d_akk[:, r1:, r0:r1].transpose(-1, -2) @ (k[:, r1:] * lf))
    return dq, a1 + a2 + a3, q * dq + k * (a2 - a1 - a3)


def decay_scores_bwd(q, k, cg, d_aqk, d_akk):
    """``(dq, dk, dcg)``; the kernel where it is available, else the twin."""
    if supports_bwd_geometry(q.shape[-2], q.shape[-1]) is None:
        return _decay_scores_bwd_flydsl(q, k, cg, d_aqk, d_akk)
    return decay_scores_bwd_torch(q, k, cg, d_aqk, d_akk)


class _DecayScores(torch.autograd.Function):
    """FlyDSL forward, FlyDSL backward — one kernel launch each."""

    @staticmethod
    def forward(ctx, q, k, cg):  # type: ignore[override]
        ctx.save_for_backward(q, k, cg)
        return _decay_scores_flydsl(q, k, cg)

    @staticmethod
    def backward(ctx, d_aqk, d_akk):  # type: ignore[override]
        q, k, cg = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if not any(needs):
            return None, None, None
        grads = decay_scores_bwd(q, k, cg, d_aqk.contiguous(), d_akk.contiguous())
        return tuple(g if n else None for g, n in zip(grads, needs))


def decay_scores(q: torch.Tensor, k: torch.Tensor, cg: torch.Tensor):
    """``(Aqk, Akk)`` for one chunk batch. ``q, k, cg: [NB, C, K]`` fp32."""
    return _DecayScores.apply(q.contiguous(), k.contiguous(), cg.contiguous())


# ---------------------------------------------------------------------------
# UT transform
# ---------------------------------------------------------------------------


def _ut_inverse_doubling(low: torch.Tensor) -> torch.Tensor:
    """``(I − L)^{-1}`` by Neumann doubling. ``low``: ``[..., C, C]``, strictly lower.

    ``P(I + L^n) = P + P L^n`` is written as an accumulating GEMM rather than as
    ``P @ (eye + power)``: the latter materialises ``eye + power`` on every
    doubling, which at production shape is five extra full passes over a 100 MB
    tensor for no arithmetic.
    """
    C = low.shape[-1]
    eye = torch.eye(C, dtype=low.dtype, device=low.device)
    flat = low.reshape(-1, C, C)
    partial = eye + flat  # Σ_{k<2} L^k
    power = flat  # L^n
    n = 2
    while n < C:
        power = power @ power  # L^(2n)
        partial = torch.baddbmm(partial, partial, power)  # Σ_{k<4n} L^k
        n *= 2
    return partial.reshape(low.shape)


def _get_ut_kernel(chunk_size: int, fuse_beta: bool = False, emit_p: bool = False):
    key = (int(chunk_size), bool(fuse_beta), bool(emit_p))
    with _KERNEL_LOCK:
        launch = _UT_KERNEL_CACHE.get(key)
        if launch is None:
            launch = build_kda_ut_inverse(chunk_size=key[0], fuse_beta=key[1], emit_p=key[2])
            _UT_KERNEL_CACHE[key] = launch
        return launch


def _ut_inverse_flydsl(low: torch.Tensor) -> torch.Tensor:
    """``(I − L)^{-1}`` in one launch, the matrix resident on-chip.

    Pass 4 measured the doubling below at 576 µs of a 4227 µs on-device forward
    at production geometry — 3 GB of HBM traffic for 1.1 GFLOP, because it
    round-trips a 100 MB tensor ten times to invert a matrix that is 16 KB. See
    :mod:`.kda_ut_inverse_kernel`.
    """
    c = low.shape[-1]
    flat = low.reshape(-1, c, c).contiguous()
    out = torch.empty_like(flat)
    dummy = torch.empty(1, dtype=torch.float32, device=low.device)
    _get_ut_kernel(c)(flat.reshape(-1), dummy, out.reshape(-1), dummy, int(flat.shape[0]))
    return out.reshape(low.shape)


def _use_ut_kernel(low: torch.Tensor) -> bool:
    """The kernel is fp32-only and square-width-gated; everything else doubles.

    In particular the fp64 gradient test and any non-production chunk width stay
    on :func:`_ut_inverse_doubling`, which remains the oracle they compare to.
    """
    return (
        low.is_cuda
        and low.dtype is torch.float32
        and low.shape[-1] == low.shape[-2]
        and supports_ut_geometry(low.shape[-1]) is None
    )


# Measured and rejected: block forward substitution over 16x16 blocks, i.e.
# `P_ij = D_i (delta_ij I + sum_k L_ik P_kj)` with `D_i = (I - L_ii)^-1` from one
# batched doubling over the diagonal blocks. It does a *sixteenth* of the tile
# area — the doubling above multiplies the above-diagonal blocks it already knows
# are zero — but it needs ~25 launches on `[6144, 16, 16]` tensors where doubling
# needs 10 on `[6144, 64, 64]`, and at this size a batched GEMM is bound by launch
# overhead and by tiles too small to fill a CU, not by arithmetic. Measured 961 µs
# against doubling's 817 µs at production shape, and much worse on the unit shapes
# (forward 703 -> 982 µs at `[2, 128, 4, 64, 64]`). Fewer, larger GEMMs win here.


class _UTInverse(torch.autograd.Function):
    """``P = (I − L)^{-1}``, with the analytic adjoint ``dL = tril(Pᵀ dP Pᵀ, −1)``."""

    @staticmethod
    def forward(ctx, low):  # type: ignore[override]
        p = _ut_inverse_flydsl(low) if _use_ut_kernel(low) else _ut_inverse_doubling(low)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, d_p):  # type: ignore[override]
        (p,) = ctx.saved_tensors
        pt = p.transpose(-1, -2)
        return torch.tril(pt @ d_p @ pt, diagonal=-1)


def ut_inverse(low: torch.Tensor) -> torch.Tensor:
    """``(I − L)^{-1}`` for strictly-lower-triangular ``L`` of shape ``[..., C, C]``."""
    return _UTInverse.apply(low)


# ---------------------------------------------------------------------------
# UT transform with both beta products folded in
# ---------------------------------------------------------------------------


def _ut_inverse_beta_flydsl(akk: torch.Tensor, betaf: torch.Tensor, want_p: bool):
    """``M = (I − Akk·(−β_r))^{-1} · Diag(β_c)`` in one launch, plus ``P`` if asked."""
    nb, c, _ = akk.shape
    out = torch.empty_like(akk)
    dummy = torch.empty(1, dtype=torch.float32, device=akk.device)
    pu = torch.empty_like(akk) if want_p else dummy
    _get_ut_kernel(c, True, want_p)(
        akk.reshape(-1), betaf.reshape(-1), out.reshape(-1), pu.reshape(-1), int(nb)
    )
    return out, (pu if want_p else None)


@compiled
def _ut_beta_bwd(p, akk, betaf, d_m):
    """The adjoint of :func:`ut_inverse_beta`, as one Inductor region.

    With ``L = Akk ⊙ (−β_r)``, ``P = (I − L)^{-1}`` and ``M = P ⊙ β_c``::

        dP   = dM ⊙ β_c                    dβ_c = Σ_r (dM ⊙ P)
        dL   = tril(Pᵀ dP Pᵀ, −1)          dAkk = dL ⊙ (−β_r)
                                           dβ_r = −Σ_c (dL ⊙ Akk)

    the first line by the column scale, the second by the same
    ``dL = tril(Pᵀ dP Pᵀ, −1)`` the unfused Function uses. ``β`` collects both
    contributions because it enters on the rows *and* the columns.
    """
    d_p = d_m * betaf.unsqueeze(-2)
    d_beta_c = (d_m * p).sum(-2)
    pt = p.transpose(-1, -2)
    d_low = torch.tril(pt @ d_p @ pt, diagonal=-1)
    d_akk = d_low * (-betaf).unsqueeze(-1)
    d_beta_r = -(d_low * akk).sum(-1)
    return d_akk, d_beta_r + d_beta_c


class _UTInverseBeta(torch.autograd.Function):
    """``(I − Akk⊙(−β_r))^{-1} ⊙ β_c``, and its adjoint."""

    @staticmethod
    def forward(ctx, akk, betaf):  # type: ignore[override]
        want_grad = any(ctx.needs_input_grad[:2])
        m, p = _ut_inverse_beta_flydsl(akk, betaf, want_grad)
        if want_grad:
            ctx.save_for_backward(p, akk, betaf)
        return m

    @staticmethod
    def backward(ctx, d_m):  # type: ignore[override]
        p, akk, betaf = ctx.saved_tensors
        needs = ctx.needs_input_grad
        d_akk, d_beta = _ut_beta_bwd(p, akk, betaf, d_m.contiguous())
        return (d_akk if needs[0] else None), (d_beta if needs[1] else None)


def supports_ut_beta(chunk_size: int, dtype: torch.dtype) -> Optional[str]:
    """``None`` when the fused UT-plus-beta kernel can run this configuration."""
    if dtype is not torch.float32:
        return f"dtype {dtype} is not fp32"
    return supports_ut_geometry(int(chunk_size))


def ut_inverse_beta(akk: torch.Tensor, betaf: torch.Tensor) -> torch.Tensor:
    """``M = (I − L)^{-1} Diag(β)`` with ``L = Akk ⊙ (−β_r)``, in one launch."""
    return _UTInverseBeta.apply(akk.contiguous(), betaf.contiguous())
