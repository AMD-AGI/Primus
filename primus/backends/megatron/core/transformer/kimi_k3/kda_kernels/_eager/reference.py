###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager-PyTorch references for Kimi Delta Attention (KDA).

These functions are the **numerical ground truth** for every other KDA
backend (``fla``'s ``chunk_kda`` today, a FlyDSL kernel in WP9). They are
written for readability and for auditability against the published math,
not for speed.

Lineage
-------
KDA is Gated DeltaNet with a **per-channel** forget gate. With state
``S ∈ R^{K×V}``, per-channel retention ``α_t = exp(g_t) ∈ R^K`` and
per-head write strength ``β_t ∈ R``::

    S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ
    o_t = S_tᵀ q_t

Two properties of that recurrence drive everything below:

* **Decay first, then the delta correction.** The DeltaNet correction is
  taken against the *already decayed* state, so the transition matrix is
  ``T_t = Diag(α_t) − β_t k_t (α_t ⊙ k_t)ᵀ`` — a diagonal-plus-rank-1
  form in which *both* low-rank vectors are tied to ``k``.
* **``o_t`` reads the POST-update state ``S_t``**, not ``S_{t−1}``. The
  intra-chunk attention matrix in :func:`eager_chunk_kda` therefore
  **retains its diagonal** (``triu(diagonal=1)`` is masked, not
  ``triu(diagonal=0)``).

Collapsing ``g`` from ``[B, T, H, K]`` to a channel-constant value
reduces KDA exactly to Gated DeltaNet, so
:func:`eager_chunk_kda` must reproduce Megatron's
``megatron.core.ssm.gated_delta_net.torch_chunk_gated_delta_rule``. That
equivalence is the project's primary numerical acceptance gate.

Tensor contract
---------------
Chosen to match ``fla.ops.kda.chunk_kda`` verbatim so backends are
drop-in interchangeable:

===============  ==========================================
``q``, ``k``     ``[B, T, H, K]``
``v``            ``[B, T, H, V]``
``g``            ``[B, T, H, K]`` — **log** decay, ``g ≤ 0``
``beta``         ``[B, T, H]`` — already sigmoid-activated
state            ``[B, H, K, V]``, accumulation dtype
output           ``[B, T, H, V]``, ``v.dtype``
===============  ==========================================

``scale`` defaults to ``K ** -0.5`` and is applied to ``q``. The
recurrence accumulates in at least fp32 (see :func:`_compute_dtype`).

Memory note
-----------
:func:`eager_chunk_kda` builds its two ``[B, H, NC, C, C]`` score
matrices one column at a time. Each column materialises a
``[B, H, NC, C, K]`` decay tensor, so a *backward* pass keeps ``C`` of
them alive — i.e. the ``[B, H, NC, C, C, K]`` intermediate the real
kernel exists to avoid. That is acceptable for an oracle at unit-test
shapes and is precisely the cost WP9's FlyDSL kernel removes. The
column-wise form is used (rather than the algebraically equivalent
two-matmul ``(K ⊙ Γ)(K / Γ)ᵀ``) because every exponent it evaluates is
``≤ 0``: with ``g ≥ -5`` and ``C = 64`` the ``1 / Γ`` factor of the
two-matmul form reaches ``exp(320)``, which overflows fp32.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "kda_gate",
    "kda_l2norm",
    "eager_recurrent_kda",
    "eager_chunk_kda",
]


# ---------------------------------------------------------------------------
# Precision policy
# ---------------------------------------------------------------------------


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """The dtype the recurrence accumulates in: at least fp32.

    ``fla``'s kernels and Megatron's eager Gated DeltaNet both accumulate
    in fp32 regardless of the activation dtype, so bf16 and fp16 inputs
    are promoted. float64 inputs are *not* demoted — that is what makes
    ``torch.autograd.gradcheck`` in float64 meaningful, since a float32
    interior would put the analytical Jacobian ~5 digits away from the
    float64 numerical one.
    """
    dtype = torch.float32
    for t in tensors:
        dtype = torch.promote_types(dtype, t.dtype)
    return dtype


# ---------------------------------------------------------------------------
# Input transforms that the fused kernels fold in via `use_*_in_kernel`
# ---------------------------------------------------------------------------


def kda_gate(
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Per-channel **log**-decay gate.

    Transcribes ``fla.ops.kda.gate.naive_kda_gate`` (``lower_bound is
    None``) and ``naive_kda_lowerbound_gate`` (otherwise). Kimi K3 sets
    ``gate_lower_bound = -5.0`` and so takes the bounded branch::

        g = lower_bound * sigmoid(exp(A_log) * (z + dt_bias))

    which confines the retention ``α = exp(g)`` to
    ``(exp(-5), 1) ≈ (6.7e-3, 1)``. The bound is not only a stability
    device: it keeps the chunked formulation's ``1 / Γ`` factor
    representable in bf16, which is what lets every causal tile stay on
    the MFMA units. Do not change the bound without revisiting the
    chunked form.

    Args:
        z: raw gate pre-activation, ``[..., H, K]``.
        A_log: per-**head** log-scale, ``[H]``.
        dt_bias: per-**channel** bias, ``[H * K]``, or ``None``.
        lower_bound: negative bound for the bounded form, or ``None`` for
            the plain Gated-DeltaNet ``-exp(A_log) * softplus(z)``.

    Returns:
        Log-decay of shape ``[..., H, K]`` at :func:`_compute_dtype`.
    """
    num_heads, head_dim = z.shape[-2:]
    dtype = _compute_dtype(z, A_log) if dt_bias is None else _compute_dtype(z, A_log, dt_bias)
    z = z.to(dtype)
    if dt_bias is not None:
        z = z + dt_bias.view(num_heads, head_dim).to(dtype)
    a = A_log.to(dtype).view(num_heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * z)
    return -a * F.softplus(z)


def kda_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise L2 normalisation over the last axis, matching ``fla.modules.l2norm``.

    ``use_qk_l2norm_in_kernel=True`` makes the fused kernel apply this to
    ``q`` and ``k``; the eager path must replicate it. The reduction runs
    at ``_compute_dtype`` precision and the result is cast back to
    ``x.dtype``.
    """
    x32 = x.to(_compute_dtype(x))
    return (x32 * torch.rsqrt(x32.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float],
    use_qk_l2norm_in_kernel: bool,
) -> Tuple[torch.Tensor, ...]:
    """Validate shapes, optionally L2-norm ``q``/``k``, upcast, and scale ``q``."""
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(
            f"q, k, g must share a shape; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(g.shape)}"
        )
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(f"v must agree with q on [B, T, H]; got {tuple(v.shape)} vs {tuple(q.shape)}")
    if beta.shape != q.shape[:3]:
        raise ValueError(f"beta must be [B, T, H]; got {tuple(beta.shape)}")

    if use_qk_l2norm_in_kernel:
        q = kda_l2norm(q)
        k = kda_l2norm(k)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    dtype = _compute_dtype(q, k, v, g, beta)
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    return q * scale, k, v, g, beta


def _init_state(
    ref: torch.Tensor,
    batch: int,
    num_heads: int,
    k_dim: int,
    v_dim: int,
    initial_state: Optional[torch.Tensor],
) -> torch.Tensor:
    state = ref.new_zeros(batch, num_heads, k_dim, v_dim)
    if initial_state is not None:
        state = state + initial_state.to(state.dtype)
    return state


# ---------------------------------------------------------------------------
# (a) O(T) sequential recurrence — the unambiguous ground truth
# ---------------------------------------------------------------------------


def eager_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Literal ``O(T)`` transcription of the KDA state recurrence.

    Obviously correct by inspection against the recurrence in this
    module's docstring, and far too slow for training — its only job is
    to be the oracle that :func:`eager_chunk_kda` and every fused kernel
    are validated against.

    ``chunk_size`` is accepted (and ignored) so this function is a
    drop-in for the chunked backends on the shared dispatch signature.

    Returns ``(o, final_state)`` with ``o`` of shape ``[B, T, H, V]`` in
    ``v.dtype`` and ``final_state`` of shape ``[B, H, K, V]`` at the
    accumulation dtype (``None`` when ``output_final_state`` is False).
    """
    del chunk_size
    out_dtype = v.dtype
    batch, seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    q, k, v, g, beta = _prepare(q, k, v, g, beta, scale, use_qk_l2norm_in_kernel)

    state = _init_state(q, batch, num_heads, k_dim, v_dim, initial_state)
    outputs: List[torch.Tensor] = []
    for t in range(seq_len):
        q_t, k_t, v_t, g_t, beta_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        # 1. decay: S <- Diag(alpha_t) S
        state = state * g_t.exp().unsqueeze(-1)
        # 2. delta correction against the ALREADY DECAYED state
        pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # S^T k_t -> [B, H, V]
        delta = beta_t.unsqueeze(-1) * (v_t - pred)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # 3. read the POST-update state
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

    o = torch.stack(outputs, dim=1).to(out_dtype)
    return o, (state if output_final_state else None)


# ---------------------------------------------------------------------------
# (b) Chunkwise-parallel form — what the training path actually runs
# ---------------------------------------------------------------------------


def _decay_weighted_scores(
    left: torch.Tensor, keys: torch.Tensor, cum_g: torch.Tensor, chunk_size: int, keep: torch.Tensor
) -> torch.Tensor:
    """``out[..., r, c] = Σ_d left[r, d] · exp(cum_g[r, d] − cum_g[c, d]) · keys[c, d]``.

    Built column by column so no ``[..., C, C, K]`` tensor is ever
    materialised.

    ``keep`` is a ``[C, C]`` 0/1 matrix marking the entries the caller
    will actually use. The exponent is zeroed everywhere else *before*
    ``exp``, which is not cosmetic: ``cum_g`` is a cumsum of non-positive
    log-decays and therefore non-increasing in ``r``, so a discarded
    ``r < c`` entry has a large **positive** exponent — up to ``5·C``, i.e.
    ``exp(320)`` at ``C = 64`` with the bounded gate. Left alone that
    overflows to ``inf``, which the forward would hide (the caller masks
    those entries to zero) but the backward would not: ``inf`` times the
    zero upstream gradient is ``nan``. Megatron's per-head reference
    achieves the same by applying ``.tril()`` both before and after
    ``.exp()`` (``gated_delta_net.py:638``).
    """
    columns: List[torch.Tensor] = []
    for c in range(chunk_size):
        exponent = (cum_g - cum_g[..., c : c + 1, :]) * keep[:, c].reshape(chunk_size, 1)
        columns.append(((left * exponent.exp()) * keys[..., c : c + 1, :]).sum(-1))
    return torch.stack(columns, dim=-1)


def eager_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunkwise-parallel KDA via the WY / UT-transform factorisation.

    Structurally identical to Megatron's
    ``torch_chunk_gated_delta_rule`` and to ``fla``'s
    ``naive_chunk_kda``, with the per-head scalar ``decay_mask`` promoted
    to per-channel ``Γ`` factors. Sequences whose length is not a
    multiple of ``chunk_size`` are zero-padded on the right; padded
    positions carry ``g = 0`` (no decay) and ``β = 0`` (no write), so
    they perturb neither the outputs nor the carried state.

    The algorithm, per chunk of ``C`` steps, with
    ``Γ^{c→r} = exp(cum_g_r − cum_g_c)``:

    1. ``L[r, c] = −β_r ⟨k_r ⊙ Γ^{c→r}, k_c⟩`` for ``c < r`` (strictly
       lower triangular).
    2. ``M = (I − L)^{-1} Diag(β)`` by forward substitution — the UT
       transform, replacing the sequential WY recurrence.
    3. ``W = M (Γ^{1→r} ⊙ K)``, ``U = M V``.
    4. ``Ṽ = U − W S_n``;
       ``A[r, c] = ⟨q_r ⊙ Γ^{c→r}, k_c⟩`` for ``c ≤ r`` (**diagonal
       retained** — ``o_t`` reads the post-update state);
       ``O = (Γ ⊙ Q) S_n + A Ṽ``.
    5. ``S_{n+1} = Diag(Γ^{1→C}) S_n + (Γ^{r→C} ⊙ K)ᵀ Ṽ``.

    Returns ``(o, final_state)`` exactly as :func:`eager_recurrent_kda`.
    """
    out_dtype = v.dtype
    batch, seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    q, k, v, g, beta = _prepare(q, k, v, g, beta, scale, use_qk_l2norm_in_kernel)

    # [B, T, H, *] -> [B, H, T, *]
    q, k, v, g = (x.transpose(1, 2) for x in (q, k, v, g))
    beta = beta.transpose(1, 2)

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, pad))
    padded_len = seq_len + pad
    num_chunks = padded_len // chunk_size

    # [B, H, T, *] -> [B, H, NC, C, *]
    q, k, v, g = (x.reshape(batch, num_heads, num_chunks, chunk_size, -1) for x in (q, k, v, g))
    beta = beta.reshape(batch, num_heads, num_chunks, chunk_size)
    # within-chunk cumulative log-decay; cum_g[..., r, :] = Σ_{i<=r} g_i
    cum_g = g.cumsum(dim=-2)

    ones = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device)
    strictly_upper = torch.triu(ones, diagonal=1)
    upper_with_diagonal = torch.triu(ones, diagonal=0)
    # 0/1 companions of the masks above, consumed by _decay_weighted_scores
    keep_lower = (~upper_with_diagonal).to(q.dtype)
    keep_lower_with_diagonal = (~strictly_upper).to(q.dtype)
    identity = torch.eye(chunk_size, dtype=q.dtype, device=q.device)

    # --- 1./2. UT transform: M = (I - L)^{-1} Diag(beta) -----------------
    attn = _decay_weighted_scores(k, k, cum_g, chunk_size, keep_lower)
    attn = -(attn * beta.unsqueeze(-1)).masked_fill(upper_with_diagonal, 0)
    for r in range(1, chunk_size):
        row = attn[..., r, :r].clone()
        sub = attn[..., :r, :r].clone()
        attn[..., r, :r] = row + (row.unsqueeze(-1) * sub).sum(-2)
    # fold beta into the columns: (I - L)^{-1} @ Diag(beta)
    ut_transform = (attn + identity) * beta.unsqueeze(-2)

    # --- 3. WY representation --------------------------------------------
    w = ut_transform @ (cum_g.exp() * k)
    u = ut_transform @ v

    # --- 4./5. sweep the chunks ------------------------------------------
    state = _init_state(q, batch, num_heads, k_dim, v_dim, initial_state)
    chunk_outputs: List[torch.Tensor] = []
    for n in range(num_chunks):
        q_n, k_n, cum_g_n, u_n, w_n = q[:, :, n], k[:, :, n], cum_g[:, :, n], u[:, :, n], w[:, :, n]

        v_tilde = u_n - w_n @ state
        # diagonal RETAINED: o_t reads the post-update state S_t
        a_intra = _decay_weighted_scores(q_n, k_n, cum_g_n, chunk_size, keep_lower_with_diagonal).masked_fill(
            strictly_upper, 0
        )
        chunk_outputs.append((q_n * cum_g_n.exp()) @ state + a_intra @ v_tilde)

        chunk_total = cum_g_n[..., -1, :]
        state = state * chunk_total.unsqueeze(-1).exp()
        state = state + (k_n * (chunk_total.unsqueeze(-2) - cum_g_n).exp()).transpose(-1, -2) @ v_tilde

    o = torch.stack(chunk_outputs, dim=2).reshape(batch, num_heads, padded_len, v_dim)
    o = o[:, :, :seq_len].transpose(1, 2).contiguous().to(out_dtype)
    return o, (state if output_final_state else None)
