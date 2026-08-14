###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Chunkwise-parallel KDA assembled around the FlyDSL kernel.

Same algorithm as :func:`..._eager.reference.eager_chunk_kda` — read that
docstring for the derivation — with the two stages the eager form is
deliberately slow at replaced:

===========================================  ==========================================
eager reference                              here
===========================================  ==========================================
``[C, C]`` scores built one column at a       one FlyDSL kernel launch
time (``C`` launches, and a                  (:mod:`.kda_decay_scores_kernel`)
``[B, H, NC, C, C, K]`` backward
intermediate)
``(I − L)^{-1}`` by ``C``-step Python         one FlyDSL kernel launch, the
forward substitution                         matrix resident in LDS
                                             (:mod:`.kda_ut_inverse_kernel`)
the inter-chunk state sweep, ``NC``           one FlyDSL kernel launch, state
serialised steps of small torch GEMMs         resident in LDS (:mod:`.sweep`)
===========================================  ==========================================

The third row is the second optimisation pass. Profiling measured the sweep at 95 % of
the forward (7902 µs against a 340 µs score kernel) purely because it was 64
serialised launches of GEMMs far too small to fill an MI355X, with the state
round-tripping through HBM every step. :func:`.sweep.fused_chunk_sweep` keeps
only the genuinely sequential ``[K, V]`` recurrence inside a kernel and hands
every per-chunk term back to a single batched GEMM.

The ``W``/``U`` projections are left to batched torch GEMMs: they were never the
bottleneck and hipBLAS already runs them at library speed.

Backward
--------
:func:`flydsl_chunk_kda_bwd` recomputes the assembly under ``enable_grad`` and
differentiates it, rather than hand-deriving the chunked adjoint. So the
gradient is correct by construction from the forward it is paired with, only
the five input tensors are kept between the passes, and the FlyDSL kernel runs
in the backward too. The cost is one extra forward evaluation, which is the
same trade ``fla`` makes by default (its ``disable_recompute=False``). The sweep
is the one stage that cannot work that way — autograd cannot see through a
custom kernel — so :class:`.sweep._FusedSweep` carries the analytic adjoint of
its own four lines, and runs the *same* kernel in reverse to get it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._compile import (
    compiled,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
    SUB_BLOCK,
    decay_scores,
    supports_geometry,
    ut_inverse,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.prep import (
    chunk_prep,
    supports_prep,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.sweep import (
    fused_chunk_sweep,
    supports_sweep,
    sweep_operand_mode,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels.autograd_adapter import (
    make_kda_chunk,
)

__all__ = ["flydsl_chunk_kda", "flydsl_chunk_kda_fwd", "flydsl_chunk_kda_bwd"]

_COMPUTE_DTYPE = torch.float32


def _lay_out(x: torch.Tensor, chunk_size: int, pad: int) -> torch.Tensor:
    """``[B, T, H, D]`` -> ``[B*H*NC, C, D]`` fp32, in **one** pass.

    The obvious spelling — ``x.to(fp32).transpose(1, 2).reshape(...)
    .contiguous()`` — is two full passes, because ``to()`` preserves the input
    layout and the transpose then has to copy again. ``copy_`` into a fresh
    contiguous fp32 buffer converts and transposes at the same time. At
    production geometry each input is 100 MB in and 200 MB out, so the pass this
    saves is ~400 MB of HBM traffic per tensor, and one launch of 52.
    """
    batch, seq_len, num_heads, dim = x.shape
    padded_len = seq_len + pad
    shape = (batch, num_heads, padded_len, dim)
    if pad:
        # padded steps carry g = 0 (no decay) and beta = 0 (no write), so they
        # perturb neither the outputs nor the carried state
        out = torch.zeros(shape, dtype=_COMPUTE_DTYPE, device=x.device)
        out[:, :, :seq_len].copy_(x.transpose(1, 2))
    else:
        out = torch.empty(shape, dtype=_COMPUTE_DTYPE, device=x.device)
        out.copy_(x.transpose(1, 2))
    return out.view(batch * num_heads * (padded_len // chunk_size), chunk_size, dim)


def _lay_out_beta(beta: torch.Tensor, chunk_size: int, pad: int) -> torch.Tensor:
    """``[B, T, H]`` -> ``[B*H*NC, C]`` fp32; :func:`_lay_out` for the 3-D one."""
    batch, seq_len, num_heads = beta.shape
    padded_len = seq_len + pad
    shape = (batch, num_heads, padded_len)
    if pad:
        out = torch.zeros(shape, dtype=_COMPUTE_DTYPE, device=beta.device)
        out[:, :, :seq_len].copy_(beta.transpose(1, 2))
    else:
        out = torch.empty(shape, dtype=_COMPUTE_DTYPE, device=beta.device)
        out.copy_(beta.transpose(1, 2))
    return out.view(batch * num_heads * (padded_len // chunk_size), chunk_size)


@compiled
def _prepare_operands(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[B, T, H, *]`` -> ``[B*H*NC, C, *]`` fp32 for all five inputs, plus ``cg``.

    One Inductor region instead of five transposed casts and a ``cumsum``. Eager
    that is 6 launches and ~1.6 GB of HBM traffic at production geometry, of
    which the whole of ``gf`` — 200 MB written and 200 MB read — exists only to
    be scanned into ``cg`` and is never looked at again; fused, ``gf`` never
    reaches memory.

    The body is :func:`_lay_out` verbatim rather than a functional rewrite, so
    that **a fallback to eager is exactly the pre-existing code and never a
    regression**. That is not hypothetical: the first version spelled this as
    ``.transpose().to().reshape().contiguous()``, which Inductor fuses to one
    kernel but ATen runs as two passes per tensor — and when Dynamo fell back in
    a long multi-shape run the forward went from 21 launches to 26 and got
    *slower than the baseline*. Inductor traces ``empty`` + ``copy_`` perfectly
    well (it functionalises it), so the one-pass eager spelling costs the fusion
    nothing.

    Only the unpadded case comes here; :func:`_assemble` keeps the padded path on
    plain :func:`_lay_out`.
    """
    cg = _lay_out(g, chunk_size, 0).cumsum(dim=-2)
    return (
        _lay_out(q, chunk_size, 0),
        _lay_out(k, chunk_size, 0),
        _lay_out(v, chunk_size, 0),
        cg,
        _lay_out_beta(beta, chunk_size, 0),
    )


@compiled
def _low_from_scores(akk: torch.Tensor, betaf: torch.Tensor) -> torch.Tensor:
    """``L[r, c] = -beta_r * Akk[r, c]``, strictly lower because ``Akk`` is.

    The sign goes on ``beta``, which is ``[NB, C]`` where the product is
    ``[NB, C, C]``: 1.6 MB of traffic instead of 200 MB, and the adjoint's
    negation lands on the small tensor too.
    """
    return akk * (-betaf).unsqueeze(-1)


@compiled
def _scale_ut(p: torch.Tensor, betaf: torch.Tensor) -> torch.Tensor:
    """``M = (I - L)^{-1} Diag(beta)`` — the UT transform's columns scaled by beta."""
    return p * betaf.unsqueeze(-2)


@compiled
def _store_w(qw: torch.Tensor, w: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """``qw[:, C:] = W`` at ``qw``'s dtype, returning ``qw``.

    ``_ChunkPrep`` hands back the whole ``[NB, 2C, K]`` MFMA operand with only
    the ``QG`` half written, so the ``W`` GEMM's result lands straight into the
    other half and the sweep needs no concatenate. The store itself is a
    fp32 -> bf16 narrowing of a 200 MB tensor, which ATen's strided ``copy_``
    ran at 2.7 TB/s; Inductor tiles it. One launch either way.
    """
    qw[:, chunk_size:].copy_(w)
    return qw


@compiled
def _lay_back(o: torch.Tensor, batch: int, num_heads: int, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """``[NB, C, V]`` fp32 -> ``[B, T, H, V]`` at ``dtype``, in **one** pass.

    Mirror of :func:`_lay_out`: the transpose back and the cast to the caller's
    dtype are the same copy. Compiled so that its *adjoint* — a transposed cast
    in the other direction — fuses with whatever Inductor puts next to it.
    """
    v_dim = o.shape[-1]
    src = o.view(batch, num_heads, -1, v_dim)[:, :, :seq_len].transpose(1, 2)
    out = torch.empty((batch, seq_len, num_heads, v_dim), dtype=dtype, device=o.device)
    out.copy_(src)
    return out


def _assemble(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    scale: float,
    chunk_size: int,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The chunked recurrence. Returns ``(o, final_state)``.

    ``o`` is ``[B, T, H, V]`` at ``out_dtype`` and ``final_state`` is
    ``[B, H, K, V]`` at fp32; the caller drops what it does not need.
    """
    batch, seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]

    # The sweep kernel's arithmetic follows the caller's dtype: bf16 operands
    # with fp32 accumulate for bf16 in (what `fla` does), otherwise fp32
    # throughout, so the fp32 parity test exercises the kernel rather than a
    # different code path.
    _, op_dtype = sweep_operand_mode(v.dtype)
    use_kernel = supports_sweep(chunk_size, k_dim, v_dim) is None
    prep_ok = supports_prep(chunk_size, k_dim, op_dtype) is None

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    padded_len = seq_len + pad
    num_chunks = padded_len // chunk_size

    # [B, T, H, *] -> [NB, C, *] with NB = B * H * NC: every chunk is an
    # independent workgroup as far as the kernel is concerned. `cg` comes back
    # from the same region, so the intermediate `gf` never reaches memory.
    if pad:
        qf, kf, vf, gf = (_lay_out(x, chunk_size, pad) for x in (q, k, v, g))
        betaf = _lay_out_beta(beta, chunk_size, pad)
        cg = gf.cumsum(dim=-2)
    else:
        qf, kf, vf, cg, betaf = _prepare_operands(q, k, v, g, beta, chunk_size)

    # `q` is deliberately *not* scaled here. `softmax_scale` multiplies both
    # terms of the chunk output -- `Aqk @ T` and `QG @ S` -- and nothing else
    # (the state carries no q), so it rides on the output GEMM's alpha/beta
    # inside :class:`.sweep._FusedSweep` for free, instead of costing a full
    # 400 MB pass over `q` here and another one in the backward's recompute.
    aqk, akk = decay_scores(qf, kf, cg)

    # L[r, c] = -beta_r * <k_r . Gamma, k_c>, strictly lower (akk already is)
    low = _low_from_scores(akk, betaf)
    # M = (I - L)^{-1} @ Diag(beta): the UT transform, columns scaled by beta
    ut = _scale_ut(ut_inverse(low), betaf)

    # Everything the sweep needs that does not depend on the running state, in
    # one kernel: `Gamma = exp(cg)`, `QGamma`, `KGamma`, `KG = K exp(ct - cg)`
    # and `dec = exp(ct)`. See :mod:`.prep` -- as six torch ops this was ~3.3 GB
    # of HBM and six launches for one `exp` of arithmetic per element.
    # `qw` comes back as the whole [NB, 2C, K] operand with only its top half
    # written; `W` is the GEMM below and goes straight into the bottom half.
    qw, kgam, kg, dec = chunk_prep(qf, kf, cg, op_dtype, use_kernel=prep_ok)
    # `ut` and `kgam` stay fp32 even though the sweep reads `qw` at bf16.
    # Running this GEMM at the operand dtype was measured 1.5x faster *and*
    # pushed the bf16 output error from 2.6e-3 to 5.7e-3: `ut` is
    # (I-L)^-1 Diag(beta), whose entries span a wide range, so rounding it
    # before the GEMM rather than after costs real accuracy. That margin is
    # what keeps this backend's output error at 2.7e-3 against `fla`'s 6.7e-3,
    # so it is not for sale; the fp32->bf16 store that pays for it goes through
    # Inductor instead, where it runs at bandwidth rather than ATen's 2.7 TB/s.
    qw = _store_w(qw, ut @ kgam, chunk_size)
    u = ut @ vf

    # The sequential half goes to one fused kernel launch; every per-chunk term
    # around it stays a batched GEMM. See :mod:`.sweep`.
    s0 = None
    if initial_state is not None:
        s0 = initial_state.reshape(batch * num_heads, k_dim, v_dim)
    o, state = fused_chunk_sweep(
        qw,
        u,
        aqk,
        kg,
        dec,
        s0,
        num_chunks=num_chunks,
        op_dtype=op_dtype,
        use_kernel=use_kernel,
        scale=scale,
    )

    return (
        _lay_back(o, batch, num_heads, seq_len, out_dtype),
        state.reshape(batch, num_heads, k_dim, v_dim),
    )


def _check_geometry(chunk_size: int, k_dim: int) -> None:
    """Gate on the *score* kernel, which is the one with no fallback.

    The sweep kernel is stricter — it needs a whole 16-row MFMA tile per wave, so
    ``head_dim = 32`` and ``chunk_size = 32`` clear the score kernel but not the
    sweep — and those geometries degrade to :func:`.sweep.state_sweep_torch`
    rather than raising, because a correct-but-slower sweep is a fair trade where
    a missing score kernel is not.
    """
    reason = supports_geometry(chunk_size, k_dim)
    if reason is not None:
        raise ValueError(
            f"the FlyDSL KDA kernel cannot run this geometry: {reason}. Its intra-chunk "
            f"tile is {SUB_BLOCK}x{SUB_BLOCK} and its 256-thread fill phase needs "
            "256 % head_dim == 0. Select a different kda_backend "
            "(eager | eager_recurrent | fla)."
        )


def flydsl_chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    chunk_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[Any, ...]]:
    """Forward half of the kernel pair consumed by :func:`...make_kda_chunk`."""
    _check_geometry(chunk_size, q.shape[-1])
    o, final_state = _assemble(q, k, v, g, beta, initial_state, scale, chunk_size, v.dtype)
    meta: Dict[str, Any] = {"scale": float(scale), "chunk_size": int(chunk_size)}
    saved = (q, k, v, g, beta, initial_state, meta)
    return o, (final_state if output_final_state else None), saved


def flydsl_chunk_kda_bwd(saved: Tuple[Any, ...], do: torch.Tensor, dht: Optional[torch.Tensor]):
    """Backward half: recompute the assembly and differentiate it."""
    q, k, v, g, beta, initial_state, meta = saved
    with torch.enable_grad():
        inputs = [q, k, v, g, beta]
        leaves = [t.detach().requires_grad_(True) for t in inputs]
        h0 = None
        if initial_state is not None:
            h0 = initial_state.detach().requires_grad_(True)
            leaves.append(h0)
        o, final_state = _assemble(
            leaves[0],
            leaves[1],
            leaves[2],
            leaves[3],
            leaves[4],
            h0,
            meta["scale"],
            meta["chunk_size"],
            v.dtype,
        )
    outputs, grad_outputs = [o], [do.to(o.dtype)]
    if dht is not None:
        outputs.append(final_state)
        grad_outputs.append(dht.to(final_state.dtype))
    grads = torch.autograd.grad(outputs, leaves, grad_outputs=grad_outputs, allow_unused=True)
    dq, dk, dv, dg, dbeta = (gr.to(t.dtype) if gr is not None else None for gr, t in zip(grads[:5], inputs))
    dh0 = None
    if h0 is not None and grads[5] is not None:
        dh0 = grads[5].to(initial_state.dtype)
    return dq, dk, dv, dg, dbeta, dh0


flydsl_chunk_kda = make_kda_chunk(flydsl_chunk_kda_fwd, flydsl_chunk_kda_bwd)
