###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Eager-Python references for the V4 attention kernels (plan-4 P24).

The math here is **bit-identical** to what previously lived inline in
:meth:`DeepseekV4Attention._attention_forward` and
:meth:`DeepseekV4Attention._csa_forward`. Plan-4 extracts it into pure
functions so that:

* ``DeepseekV4Attention`` itself, the plan-4 Triton kernels (P25 / P26),
  and the plan-4 unit-test harness share exactly one definition;
* the function signatures match the kernel signatures
  (``v4_attention_v1`` / ``v4_csa_attention_v0``) so the test harness can
  plug reference ↔ candidate interchangeably;
* the compute kernel of V4 attention is decoupled from the
  ``DeepseekV4Attention`` class (no ``self``-bound state) — the
  per-call inputs are explicit.

The two functions:

* :func:`eager_v4_attention` — single-key-axis attention with optional
  per-head learned softmax sink, optional sliding window, and optional
  ``[Sq, Sk]`` additive bias. Covers ``compress_ratio == 0`` (dense +
  SWA + sink, no bias) and ``compress_ratio == 128`` (HCA — caller
  pre-concatenates the compressed pool to the local keys and supplies
  the joint-softmax additive bias).
* :func:`eager_v4_csa_attention` — fused local-SWA + per-query top-K
  sparse attention with shared per-head sink and joint softmax.
  Covers ``compress_ratio == 4`` (CSA). The caller is responsible for
  the per-query top-K gather; the function takes the gathered
  ``[B, Sq, K, head_dim]`` tensor directly.

Both functions:

* keep every matmul / einsum on tensor cores in the input dtype (bf16
  in production); the matmul accumulator inside is fp32. The *only*
  fp32 step is the softmax block (max-subtract + ``exp`` + sum +
  divide), which :func:`_softmax_with_sink` upcasts internally and
  returns in fp32; the caller-side ``probs.to(v.dtype)`` puts probs
  back on the bf16 path before the V-matmul. This matches the
  FlashAttention / Megatron de-facto layout (bf16 ``Q @ K^T``, fp32
  softmax, bf16 ``probs @ V``);
* honor ``attn_dropout`` only when ``training`` is also ``True`` (so
  the eval / inference path is deterministic);
* return ``[B, H, Sq, head_dim]`` in ``v.dtype`` (or ``v_local.dtype``
  for the CSA path);
* are autograd-friendly: fwd produces a graph; ``out.sum().backward()``
  populates ``q.grad / k.grad / v.grad / sink.grad`` (and
  ``gathered.grad`` for CSA).
"""

from __future__ import annotations

from typing import Optional

import torch

from primus.backends.megatron.core.transformer.sliding_window_kv import (
    sliding_window_causal_mask,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _softmax_with_sink(
    logits: torch.Tensor,
    sink: Optional[torch.Tensor],
) -> torch.Tensor:
    """Numerically stable softmax with optional per-head learned sink column.

    ``logits`` shape: ``[B, H, ..., Sk]`` — the head axis is at ``dim=1``.
    When ``sink`` is given (shape ``[H]``) it is joined as a virtual key
    column with notional value zero (so the V-weighted sum after
    multiplying by ``v`` is unaffected by the sink slot), then dropped
    after softmax. The head can still spend probability mass on the
    sink as a "no attention" fallback.

    **dtype contract.** This is the *only* fp32 step in V4 attention.
    ``logits`` may arrive in the model dtype (bf16 in production —
    coming straight out of a tensor-core ``Q @ K^T`` matmul); the
    softmax block (max-subtract + ``exp`` + sum + divide) is the
    numerically sensitive part, so we **upcast to fp32 here** and
    return probabilities in **fp32**. The caller is responsible for
    ``probs.to(v.dtype)`` before the value matmul so the V-matmul
    stays on bf16 tensor cores.

    Returns probabilities on the *real* keys of the same shape as
    ``logits`` — but in **fp32** regardless of input dtype.
    """
    # logits: [B, H, ..., Sk] (any dtype) -> logits_fp32: [B, H, ..., Sk] in fp32
    logits_fp32 = logits.float()
    if sink is None:
        logits_fp32 = logits_fp32 - logits_fp32.amax(dim=-1, keepdim=True).detach()
        return logits_fp32.softmax(dim=-1)

    # sink: [H] (any dtype) -> sink_col: [B, H, ..., 1] in fp32
    ndim = logits_fp32.dim()
    num_heads = sink.shape[0]
    view_shape = [1] * ndim
    view_shape[1] = num_heads
    view_shape[-1] = 1
    target_shape = list(logits_fp32.shape[:-1]) + [1]
    sink_col = sink.float().view(*view_shape).expand(*target_shape)
    # logits_fp32: [B, H, ..., Sk], sink_col: [B, H, ..., 1]
    # -> logits_aug: [B, H, ..., Sk+1] in fp32
    logits_aug = torch.cat([logits_fp32, sink_col], dim=-1)
    logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
    probs = logits_aug.softmax(dim=-1)
    return probs[..., :-1]


def _build_local_attention_mask(
    seq_len: int,
    swa_window: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the ``[seq_len, seq_len]`` local SWA-causal additive mask.

    Equivalent to :meth:`DeepseekV4Attention._local_mask`:

    * ``swa_window > 0`` — sliding-window causal (queries see the last
      ``swa_window`` keys);
    * otherwise — full causal (queries see all earlier keys).

    The two-call contract is deterministic: same ``(seq_len,
    swa_window, device, dtype)`` always returns the same tensor, so
    rebuilding inside this function vs. passing a pre-built mask gives
    bit-identical attention output.
    """
    window = swa_window if swa_window > 0 else seq_len
    return sliding_window_causal_mask(seq_len, window, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Public reference ops
# ---------------------------------------------------------------------------


def eager_v4_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, H, Sk, D]
    v: torch.Tensor,  # [B, H, Sk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Eager-Python V4 dense / HCA attention.

    Math::

        logits = (q @ k.T) * scale + mask        # bf16 tensor-core matmul
        probs  = softmax_with_sink(logits, sink) # softmax internally upcasts to fp32
        if attn_dropout > 0 and training: probs = dropout(probs, attn_dropout)
        out    = probs.to(v.dtype) @ v           # bf16 tensor-core matmul

    **dtype contract.** All matmuls (``Q @ K^T`` and ``probs @ V``) run
    on tensor cores in the input dtype (bf16 in production); the
    accumulator inside the matmul is fp32. The *only* fp32 step is the
    softmax block, which :func:`_softmax_with_sink` handles internally
    (input may be bf16, output is fp32). The caller-side
    ``probs.to(v.dtype)`` puts probs back on the bf16 path before the
    V-matmul.

    Mask resolution:

    * ``additive_mask is not None`` — used directly. ``swa_window`` is
      ignored. (HCA's ``compress_ratio == 128`` caller pre-builds the
      joint-softmax mask via ``cat([local_mask, hca_mask])`` and
      passes it here.)
    * ``additive_mask is None`` and ``swa_window > 0`` — sliding-window
      causal mask is built internally. Requires ``Sq == Sk``.
    * ``additive_mask is None`` and ``swa_window <= 0`` — full causal
      mask is built internally. Requires ``Sq == Sk``.

    Returns ``[B, H, Sq, D]`` in ``v.dtype``.
    """
    # q: [B, H, Sq, D], k: [B, H, Sk, D] -> Sq: scalar, Sk: scalar
    Sq = q.shape[2]
    Sk = k.shape[2]

    # mask: [Sq, Sk] in q.dtype   (broadcasts over B, H at the addition site below)
    if additive_mask is None:
        if Sq != Sk:
            raise ValueError(
                "eager_v4_attention requires `additive_mask` when Sq != Sk; "
                f"got Sq={Sq}, Sk={Sk}. The HCA caller must pre-concatenate "
                "the compressed pool to the local keys and supply the joint "
                "additive mask."
            )
        # _build_local_attention_mask(Sq, swa_window) -> mask: [Sq, Sq] (== [Sq, Sk]) in q.dtype
        mask = _build_local_attention_mask(Sq, swa_window, device=q.device, dtype=q.dtype)
    else:
        # additive_mask: [Sq, Sk] -> mask: [Sq, Sk] in caller's dtype
        mask = additive_mask

    # q: [B, H, Sq, D], k.transpose(-2,-1): [B, H, D, Sk] -> matmul(...): [B, H, Sq, Sk] in q.dtype
    # bf16 tensor-core matmul (fp32 accumulator inside, output bf16)
    # * scale: [B, H, Sq, Sk] in q.dtype
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    # logits: [B, H, Sq, Sk], mask: [Sq, Sk] -> logits: [B, H, Sq, Sk]   (mask broadcasts over B, H)
    logits = logits + mask
    # logits: [B, H, Sq, Sk] (bf16), sink: [H] or None -> probs: [B, H, Sq, Sk] in fp32
    # softmax block internally upcasts logits + sink to fp32 (numerical contract: only step in fp32)
    probs = _softmax_with_sink(logits, sink)
    # probs: [B, H, Sq, Sk] (fp32) -> probs: [B, H, Sq, Sk] (fp32)
    if attn_dropout > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=attn_dropout)
    # probs.to(v.dtype): [B, H, Sq, Sk] in v.dtype, v: [B, H, Sk, D] -> out: [B, H, Sq, D] in v.dtype
    # bf16 tensor-core matmul (fp32 accumulator inside)
    return torch.matmul(probs.to(v.dtype), v)


def eager_v4_csa_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K, D] — pre-gathered per-query top-K from compressed pool
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    sparse_mask: torch.Tensor,  # [B, Sq, K] additive (broadcasts over H)
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Eager-Python V4 CSA fused attention (joint local SWA + sparse top-K).

    Math::

        local_logits  = (q @ k_local.T) * scale + local_mask                # bf16 matmul
        sparse_logits = einsum("bhsd,bhskd->bhsk", q, gathered_h) * scale
                       + sparse_mask.unsqueeze(1)                            # bf16 einsum
        joint_logits  = cat([local_logits, sparse_logits], dim=-1)           # [B, H, Sq, Sq+K]
        probs         = softmax_with_sink(joint_logits, sink)                # JOINT softmax in fp32
        if attn_dropout > 0 and training: probs = dropout(probs, attn_dropout)
        probs_local, probs_sparse = probs[..., :Sq], probs[..., Sq:]
        out = probs_local @ v_local + einsum("bhsk,bhskd->bhsd", probs_sparse, gathered_h)
                                                                             # bf16 matmul / einsum

    where ``gathered_h = gathered.unsqueeze(1).expand(B, H, Sq, K, D)``
    (heads broadcast across the single compressor output).

    **dtype contract.** Same as :func:`eager_v4_attention`: every
    matmul / einsum runs on tensor cores in the input dtype (bf16 in
    production, fp32 accumulator inside); the *only* fp32 step is the
    joint softmax inside :func:`_softmax_with_sink`. ``probs.to(v.dtype)``
    puts the per-branch probabilities back on bf16 before the V-stage.

    The local SWA mask is built internally from ``swa_window`` (see
    :func:`_build_local_attention_mask`); the caller pre-builds
    ``sparse_mask`` to flag indexer-dropped slots (``-inf`` for
    ``topk_idx == -1``).

    Returns ``[B, H, Sq, D]`` in ``v_local.dtype``.
    """
    # q: [B, H, Sq, D], gathered: [B, Sq, K, D] -> B, H, Sq, D, K: scalars
    B, H, Sq, D = q.shape
    K = gathered.shape[2]

    # _build_local_attention_mask(Sq, swa_window) -> local_mask: [Sq, Sq] in q.dtype
    local_mask = _build_local_attention_mask(Sq, swa_window, device=q.device, dtype=q.dtype)

    # q: [B, H, Sq, D], k_local.transpose(-2,-1): [B, H, D, Sq]
    # -> matmul(...): [B, H, Sq, Sq] in q.dtype   (bf16 tensor core, fp32 accumulator inside)
    # * scale: local_logits: [B, H, Sq, Sq] in q.dtype
    local_logits = torch.matmul(q, k_local.transpose(-2, -1)) * scale
    # local_logits: [B, H, Sq, Sq], local_mask: [Sq, Sq]
    # -> local_logits: [B, H, Sq, Sq] in q.dtype   (mask broadcasts over B, H)
    local_logits = local_logits + local_mask

    # gathered: [B, Sq, K, D] -> unsqueeze(1): [B, 1, Sq, K, D]
    # -> expand: gathered_h: [B, H, Sq, K, D] in gathered.dtype   (view, no copy)
    gathered_h = gathered.unsqueeze(1).expand(B, H, Sq, K, D)
    # q: [B, H, Sq, D], gathered_h: [B, H, Sq, K, D]
    # -> einsum("bhsd,bhskd->bhsk"): [B, H, Sq, K] in q.dtype   (bf16 tensor core)
    # * scale: sparse_logits: [B, H, Sq, K] in q.dtype
    sparse_logits = torch.einsum("bhsd,bhskd->bhsk", q, gathered_h) * scale
    # sparse_logits: [B, H, Sq, K], sparse_mask.unsqueeze(1): [B, 1, Sq, K]
    # -> sparse_logits: [B, H, Sq, K] in q.dtype   (mask broadcasts over H)
    sparse_logits = sparse_logits + sparse_mask.unsqueeze(1)

    # local_logits: [B, H, Sq, Sq], sparse_logits: [B, H, Sq, K]
    # -> joint_logits: [B, H, Sq, Sq+K] in q.dtype
    joint_logits = torch.cat([local_logits, sparse_logits], dim=-1)
    # joint_logits: [B, H, Sq, Sq+K] (bf16), sink: [H] or None -> probs: [B, H, Sq, Sq+K] in fp32
    # softmax block internally upcasts to fp32 (numerical contract: only step in fp32)
    probs = _softmax_with_sink(joint_logits, sink)

    # probs: [B, H, Sq, Sq+K] (fp32) -> probs: [B, H, Sq, Sq+K] (fp32)
    if attn_dropout > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=attn_dropout)

    # probs[..., :Sq]: [B, H, Sq, Sq] in fp32 -> probs_local: [B, H, Sq, Sq] in v_local.dtype
    probs_local = probs[..., :Sq].to(v_local.dtype)
    # probs[..., Sq:]: [B, H, Sq, K] in fp32 -> probs_sparse: [B, H, Sq, K] in v_local.dtype
    probs_sparse = probs[..., Sq:].to(v_local.dtype)

    # probs_local: [B, H, Sq, Sq], v_local: [B, H, Sq, D]
    # -> matmul(...): out_local: [B, H, Sq, D] in v_local.dtype   (bf16 tensor core)
    out_local = torch.matmul(probs_local, v_local)
    # probs_sparse: [B, H, Sq, K], gathered_h.to(v_local.dtype): [B, H, Sq, K, D]
    # -> einsum("bhsk,bhskd->bhsd"): out_sparse: [B, H, Sq, D] in v_local.dtype   (bf16 tensor core)
    out_sparse = torch.einsum(
        "bhsk,bhskd->bhsd",
        probs_sparse,
        gathered_h.to(v_local.dtype),
    )

    # out_local: [B, H, Sq, D], out_sparse: [B, H, Sq, D] -> out: [B, H, Sq, D] in v_local.dtype
    return out_local + out_sparse


__all__ = [
    "eager_v4_attention",
    "eager_v4_csa_attention",
]
