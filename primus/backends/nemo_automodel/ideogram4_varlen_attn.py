###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Var-len (``cu_seqlens``) flash-attention processor for Ideogram-4 in the NeMo
AutoModel diffusion recipe — the fast, exact replacement for masked torch SDPA.

WHY:
  Ideogram-4's attention runs on **masked torch SDPA** today: the transformer builds
  a dense ``(B,1,L,L)`` block-diagonal boolean mask from ``segment_ids`` and hands it
  to ``dispatch_attention_fn``. A dense mask forces the SDPA/NATIVE backend (flash /
  aiter backends reject one), so the whole model forgoes flash — and at the customer's
  1k–2k px targets attention is O(L²) and dominates the step. The mask is, in practice,
  a per-row block-diagonal over CONTIGUOUS segments (left-padding = {pad}+{text+image};
  future multi-sample packing = several sample blocks). That maps exactly onto
  **variable-length flash attention** (``cu_seqlens``): pack the batch into one flat
  sequence and mark segment boundaries — no dense mask, full flash speed, EXACT numerics.

WHAT (NO diffusers / Automodel fork):
  A drop-in diffusers attention processor for ``Ideogram4Attention``. It reproduces the
  stock ``Ideogram4AttnProcessor`` math verbatim (q/k/v proj → q/k RMSNorm → MRoPE) and
  only replaces the attention CALL: it converts the block-diagonal boolean mask to
  ``cu_seqlens`` and runs ``aiter.flash_attn_varlen_func`` (``deterministic=False`` — the
  non-deterministic backward is a large, numerically-equivalent throughput win, and the
  *deterministic* hd=256 backward needs 150–360 GB and OOMs at 2048²). When the mask is
  absent or trivial (one full segment per row, e.g. a fixed-length/no-pad batch) it takes
  the plain dense ``flash_attn_func`` fast path.

  Install swaps the class default processor (``Ideogram4Attention._default_processor_cls``)
  BEFORE the recipe builds the model, so every attention module is constructed with it.
  Env-gated by ``PRIMUS_IDEOGRAM_VARLEN_ATTN=1`` (default off = stock SDPA path). No
  Automodel/diffusers source is modified.

GENERALITY / REUSE:
  The mask→``cu_seqlens`` transform (:func:`blockdiag_bool_mask_to_cu_seqlens`) and the
  var-len call (:func:`varlen_flash_attention`) are **model-agnostic** — they assume only
  a per-row block-diagonal boolean mask with contiguous segments, so any packed/padded
  attention (not just Ideogram) can reuse them. The only Ideogram-specific piece is the
  thin processor that wires the model's proj/norm/RoPE to them.

CORRECTNESS:
  Exact, not approximate. Every token (including padding) keeps its own segment, so the
  result matches masked SDPA on all positions up to bf16 + non-deterministic-atomic
  ordering (image-token velocity — the only positions the loss reads — matches within the
  bf16 floor). The processor falls back to the ORIGINAL dense dispatch ONLY for mask
  *types* it cannot represent as contiguous segments (a non-boolean/additive mask), and
  warns once — it never silently relays the unoptimized path for the case it exists to
  serve (Ideogram always passes a boolean block-diagonal mask).

Activation (env, no config schema change):
    PRIMUS_IDEOGRAM_VARLEN_ATTN=1        swap Ideogram-4 attention to the var-len flash path
    PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE=1  skip the mask->cu_seqlens host-sync and use dense flash
                                         directly (torch.compile-safe; EXACT only for
                                         equal-length / unpadded batches, e.g. fixed-text)
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}


def is_varlen_attn_enabled() -> bool:
    """Whether the Ideogram-4 var-len flash-attention processor should be installed."""
    return os.getenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "0") in _TRUTHY


def assume_dense_enabled() -> bool:
    """Whether to skip the block-diagonal mask analysis and use dense flash directly.

    The mask->``cu_seqlens`` transform does data-dependent host-syncs (``bool(.any())``,
    ``.max().item()``, ``nonzero``) that graph-break ``torch.compile`` and, under FSDP2
    multi-rank, desync the per-layer collectives. When every row is a single full segment
    (an equal-length / unpadded batch -> the mask is trivial anyway), set
    ``PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE=1`` to skip the analysis and go straight to dense
    flash, keeping the compiled per-layer graph break-free. EXACT only for equal-length /
    no-pad batches (e.g. fixed-text); do NOT set it for ragged/padded batches.
    """
    return os.getenv("PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE", "0") in _TRUTHY


# --------------------------------------------------------------------------- #
# Model-agnostic helpers (reusable for any packed/padded block-diagonal attn)  #
# --------------------------------------------------------------------------- #
def blockdiag_bool_mask_to_cu_seqlens(mask: Tensor) -> Tuple[Tensor, int, bool]:
    """Convert a block-diagonal boolean attention mask to a var-len packing.

    Args:
        mask: ``(B, 1, L, L)`` or ``(B, L, L)`` boolean tensor, ``True`` = "query i
            attends to key j". Each row ``b`` must be block-diagonal over CONTIGUOUS
            segments (a segment boundary lies between positions ``i`` and ``i+1``
            wherever ``mask[b, .., i, i+1]`` is ``False``); segments never span the row
            boundary. This is what an ``(seg_i == seg_j)`` mask produces for
            contiguously-assigned ``segment_ids`` (padding, or packed samples).

    Returns:
        ``(cu_seqlens, max_seqlen, is_trivial)`` describing the packed layout over the
        flattened ``(B*L)`` sequence:
          * ``cu_seqlens``: ``int32`` ``(num_segments + 1,)`` cumulative segment lengths,
            starting at 0 and ending at ``B*L`` (the ``flash_attn_varlen_func`` contract).
          * ``max_seqlen``: longest segment length.
          * ``is_trivial``: ``True`` iff every row is a single full segment (no internal
            splits) — the caller may then use plain dense attention and skip packing.
    """
    if mask.dim() == 4:
        m = mask[:, 0]
    elif mask.dim() == 3:
        m = mask
    else:
        raise ValueError(f"expected a (B,1,L,L) or (B,L,L) mask, got shape {tuple(mask.shape)}")
    if m.dtype != torch.bool:
        raise TypeError(f"expected a boolean mask, got dtype {m.dtype}")

    B, L, L2 = m.shape
    if L != L2:
        raise ValueError(f"mask must be square in the last two dims, got {(L, L2)}")
    device = m.device

    if L == 1:
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=device)
        return cu, 1, True

    # A new segment starts at position i+1 within a row wherever i and i+1 do not attend
    # (the sub-diagonal test is all we need for contiguous block-diagonal segments).
    superdiag = m.diagonal(offset=1, dim1=-2, dim2=-1)  # (B, L-1) : mask[b, i, i+1]
    splits = ~superdiag  # (B, L-1)
    is_trivial = not bool(splits.any())

    # Per-token "starts a new segment" over the flattened (B, L) sequence. The first
    # token of every row always starts a segment (segments never cross rows).
    new_seg = torch.zeros(B, L, dtype=torch.bool, device=device)
    new_seg[:, 0] = True
    new_seg[:, 1:] = splits
    seg_starts = torch.nonzero(new_seg.reshape(-1), as_tuple=False).flatten()

    total = B * L
    cu = torch.empty(seg_starts.numel() + 1, dtype=torch.int32, device=device)
    cu[:-1] = seg_starts.to(torch.int32)
    cu[-1] = total
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return cu, max_seqlen, is_trivial


def _unwrap(out):
    """aiter returns (out, lse, ...) when return_lse=True; keep only the output."""
    return out[0] if isinstance(out, (tuple, list)) else out


def dense_flash_attention(q: Tensor, k: Tensor, v: Tensor, *, deterministic: bool = False) -> Tensor:
    """Plain (unmasked) bf16 flash attention. q/k/v: ``(B, L, H, D)`` -> ``(B, L, H, D)``.

    ``return_lse=True`` is required by aiter's autograd forward (LSE is saved for the
    backward); we request it and drop the LSE.
    """
    import aiter

    return _unwrap(
        aiter.flash_attn_func(q, k, v, causal=False, deterministic=deterministic, return_lse=True)
    )


def varlen_flash_attention(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, *, deterministic: bool = False
) -> Tensor:
    """Variable-length bf16 flash attention over a packed sequence.

    q/k/v: ``(total_tokens, H, D)`` (packed, no padding between segments). ``cu_seqlens``
    and ``max_seqlen`` come from :func:`blockdiag_bool_mask_to_cu_seqlens`. Returns
    ``(total_tokens, H, D)``.
    """
    import aiter

    return _unwrap(
        aiter.flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            causal=False, deterministic=deterministic, return_lse=True,
        )
    )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate-half, matching diffusers ``transformer_ideogram4._rotate_half`` exactly."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# --------------------------------------------------------------------------- #
# Ideogram-4 var-len attention processor                                       #
# --------------------------------------------------------------------------- #
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


class Ideogram4VarlenAttnProcessor:
    """Ideogram-4 self-attention via var-len flash (exact block-diagonal packing).

    Mirrors diffusers ``Ideogram4AttnProcessor`` (q/k/v proj, q/k RMSNorm, MRoPE, output
    proj) and swaps only the attention call for a ``cu_seqlens`` flash path. Non-det
    backward (``deterministic=False``).
    """

    # kept for API-compatibility with diffusers' processor discovery / set_attention_backend
    _attention_backend = None
    _parallel_config = None

    deterministic: bool = False
    # Read once at class-definition time (torchrun sets env before import). Keeps the check a
    # constant attribute lookup inside the compiled graph (no data-dependent branch / break).
    assume_dense: bool = assume_dense_enabled()

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        image_rotary_emb: Tuple[Tensor, Tensor],
    ) -> Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        out = self._attention(query, key, value, attention_mask)
        out = out.flatten(2, 3)
        return attn.to_out[0](out)

    def _attention(self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        B, L, H, D = query.shape

        # No mask, or a mask type we cannot represent as contiguous segments: for a
        # boolean block-diagonal mask we always take the exact var-len path (that is the
        # case this processor exists for); a non-boolean/additive mask is not something
        # Ideogram emits, so defer to the original dense dispatch rather than guess.
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            _warn_once(
                "nonbool_mask",
                "[PrimusIdeogramVarlen] non-boolean attn_mask -> dense SDPA dispatch fallback.",
            )
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            return dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask,
                backend=self._attention_backend, parallel_config=self._parallel_config,
            )

        # MRoPE multiplies q/k by the float32 cos/sin, promoting them to float32. Torch
        # SDPA is autocast-aware and downcasts at its boundary; aiter's flash op is not,
        # so match that here by casting q/k to the projection (compute) dtype. ``value``
        # skips RoPE and already carries it.
        compute_dtype = value.dtype
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)

        # Dense fast path. ``attention_mask is None`` OR the equal-length assertion
        # (``assume_dense``) skips the data-dependent mask->cu_seqlens host-sync, so under
        # torch.compile the whole processor stays in one graph (aiter.flash_attn_func is
        # itself compile-safe -- fullgraph traces it with no break), which is required for
        # FSDP2 multi-rank compile to not desync collectives. Exact for equal-length/no-pad
        # batches; ragged/padded batches must leave PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE unset.
        if attention_mask is None or self.assume_dense:
            return dense_flash_attention(query, key, value, deterministic=self.deterministic)

        cu_seqlens, max_seqlen, is_trivial = blockdiag_bool_mask_to_cu_seqlens(attention_mask)
        if is_trivial:
            return dense_flash_attention(query, key, value, deterministic=self.deterministic)

        q = query.reshape(B * L, H, D)
        k = key.reshape(B * L, H, D)
        v = value.reshape(B * L, H, D)
        out = varlen_flash_attention(q, k, v, cu_seqlens, max_seqlen, deterministic=self.deterministic)
        return out.reshape(B, L, H, D)


# --------------------------------------------------------------------------- #
# Install                                                                      #
# --------------------------------------------------------------------------- #
def install(model=None) -> bool:
    """Route Ideogram-4 attention through the var-len flash processor (no-fork).

    No-op (returns False) unless ``PRIMUS_IDEOGRAM_VARLEN_ATTN`` is set. Patches the
    class default processor so every ``Ideogram4Attention`` built AFTER this call uses
    the var-len processor; if a built ``model`` is passed, also swaps its existing
    modules. Idempotent. Modifies NO Automodel/diffusers source.
    """
    if not is_varlen_attn_enabled():
        return False

    # Fail fast if aiter's flash-attention is unavailable, so the run errors clearly
    # rather than silently keeping the SDPA path.
    import aiter  # noqa: F401

    from diffusers.models.transformers.transformer_ideogram4 import Ideogram4Attention

    already = getattr(Ideogram4Attention, "_primus_varlen_installed", False)
    if not already:
        Ideogram4Attention._default_processor_cls = Ideogram4VarlenAttnProcessor
        if Ideogram4VarlenAttnProcessor not in Ideogram4Attention._available_processors:
            Ideogram4Attention._available_processors = [
                *Ideogram4Attention._available_processors,
                Ideogram4VarlenAttnProcessor,
            ]
        Ideogram4Attention._primus_varlen_installed = True

    swapped = 0
    if model is not None:
        for module in model.modules():
            if isinstance(module, Ideogram4Attention) and not isinstance(
                module.processor, Ideogram4VarlenAttnProcessor
            ):
                module.set_processor(Ideogram4VarlenAttnProcessor())
                swapped += 1

    if already and swapped == 0:
        return True

    logger.info(
        "[PrimusIdeogramVarlen] Installed var-len flash-attention processor for "
        "Ideogram4Attention (deterministic=False)%s.",
        f"; swapped {swapped} existing module(s)" if swapped else "",
    )
    return True
