###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Variable-length attention helpers for packed and padded batches.

Model-agnostic on purpose. Everything here assumes only a per-row block-diagonal
boolean attention mask over contiguous segments, which is what an
``(segment_id_i == segment_id_j)`` comparison produces for any contiguous segment
assignment -- left-padding, or several packed samples per row. Nothing about a
particular transformer appears in this file, so any packed or padded attention can
use it.

WHAT THIS IS FOR:
  A dense ``(B, 1, L, L)`` boolean mask is a poor way to express "these tokens
  attend to each other and not to those". Handing one to a dispatcher forces the
  SDPA or native backend, because the flash and AITER backends reject a dense
  mask outright -- so the model gives up flash attention entirely and pays the
  full quadratic cost in both time and memory. At high resolutions that is the
  dominant term of the step.

  A block-diagonal mask over contiguous segments carries no more information
  than the segment boundaries do. Packing the batch into one flat sequence and
  naming the boundaries in ``cu_seqlens`` says the same thing to a variable-length
  flash kernel, which is exact rather than approximate -- the same attention, in
  the same precision, with the mask expressed in the form the fast kernel wants.

WHY DERIVING THE PACKING FROM THE MASK IS THE SLOW PATH:
  :func:`blockdiag_bool_mask_to_cu_seqlens` is exact and needs nothing from the
  caller, which makes it the right fallback, but it is data-dependent: ``.any()``
  and ``.max().item()`` are device-to-host reads. Inside a compiled region each is
  a graph break, and a mid-forward graph break under FSDP2 splits the region the
  per-layer all-gather and reshard collectives are registered around, which
  desynchronizes them across ranks. That failure is silent and multi-rank.

  So a caller that already knows its segment lengths on the host should build
  ``cu_seqlens`` there and pass it in as a graph input instead. This function is
  what runs when nobody did.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def blockdiag_bool_mask_to_cu_seqlens(mask: Tensor) -> Tuple[Tensor, int, bool]:
    """Convert a block-diagonal boolean attention mask to a variable-length packing.

    Args:
        mask: ``(B, 1, L, L)`` or ``(B, L, L)`` boolean, ``True`` meaning query ``i``
            attends to key ``j``. Each row must be block-diagonal over CONTIGUOUS
            segments, and segments never span a row boundary.

    Returns:
        ``(cu_seqlens, max_seqlen, is_trivial)`` over the flattened ``(B * L)``
        sequence:

        * ``cu_seqlens`` -- ``int32`` of shape ``(num_segments + 1,)``, cumulative
          segment starts from 0 to ``B * L``, which is the varlen kernel's contract.
        * ``max_seqlen`` -- the longest segment.
        * ``is_trivial`` -- ``True`` when every row is one full segment, meaning the
          mask says nothing and the caller can use dense attention instead.

    Contains two device-to-host reads, so see the module docstring before calling
    it from anywhere that gets compiled.
    """
    if mask.dim() == 4:
        rows = mask[:, 0]
    elif mask.dim() == 3:
        rows = mask
    else:
        raise ValueError(f"expected a (B,1,L,L) or (B,L,L) mask, got shape {tuple(mask.shape)}")
    if rows.dtype != torch.bool:
        raise TypeError(
            f"expected a boolean mask, got dtype {rows.dtype}. An additive float mask "
            "cannot be reduced to segment boundaries, so it has no varlen equivalent."
        )

    batch, length, length_k = rows.shape
    if length != length_k:
        raise ValueError(f"mask must be square in its last two dims, got {(length, length_k)}")

    if length == 1:
        # Every row is a single one-token segment, so the answer is exact and needs
        # no reduction. Worth special-casing because the diagonal below is empty here.
        return (
            torch.arange(0, batch + 1, dtype=torch.int32, device=rows.device),
            1,
            True,
        )

    # For contiguous block-diagonal segments the superdiagonal is sufficient: a new
    # segment begins at i+1 exactly where i and i+1 do not attend to each other.
    # Reading the whole mask would cost more and say the same thing.
    adjacent_attends = rows.diagonal(offset=1, dim1=-2, dim2=-1)  # (B, L-1)
    splits = ~adjacent_attends
    is_trivial = not bool(splits.any())

    starts_segment = torch.zeros(batch, length, dtype=torch.bool, device=rows.device)
    starts_segment[:, 0] = True  # segments never cross rows
    starts_segment[:, 1:] = splits
    segment_starts = torch.nonzero(starts_segment.reshape(-1), as_tuple=False).flatten()

    total_tokens = batch * length
    cu_seqlens = torch.empty(segment_starts.numel() + 1, dtype=torch.int32, device=rows.device)
    cu_seqlens[:-1] = segment_starts.to(torch.int32)
    cu_seqlens[-1] = total_tokens
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return cu_seqlens, max_seqlen, is_trivial


def dense_flash_attention(q: Tensor, k: Tensor, v: Tensor, *, deterministic: bool = False) -> Tensor:
    """Unmasked flash attention. ``(B, L, H, D)`` in and out."""
    from primus_turbo.pytorch.ops import flash_attn_func

    return flash_attn_func(q, k, v, causal=False, deterministic=deterministic)


def varlen_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    *,
    deterministic: bool = False,
) -> Tensor:
    """Variable-length flash attention over a packed sequence.

    ``q``, ``k`` and ``v`` are ``(total_tokens, H, D)`` with segments packed
    back-to-back and no padding between them. Returns the same shape.

    THE CLONE IS NOT DEFENSIVE CLUTTER. The kernel treats ``cu_seqlens`` as a
    mutable argument: it saves the tensor for its backward and then writes to it,
    which bumps the tensor's autograd version counter on every call. A caller that
    shares one ``cu_seqlens`` tensor across layers -- which is the efficient thing
    to do, since the packing is identical for all of them -- would therefore have
    its version advanced once per layer, while each layer's backward still expects
    the version it saw. The step then dies during ``backward()`` with a version
    counter mismatch, at a point that names neither this call nor the sharing.
    Cloning here costs one small allocation per layer and makes the shared-buffer
    transport safe by construction rather than by convention.
    """
    from primus_turbo.pytorch.ops import flash_attn_varlen_func

    cu_seqlens = cu_seqlens.clone()
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        deterministic=deterministic,
    )


def pack_for_varlen(x: Tensor) -> Tensor:
    """Flatten ``(B, L, H, D)`` into the packed ``(B * L, H, D)`` varlen layout."""
    if x.dim() != 4:
        raise ValueError(f"expected a (B, L, H, D) tensor, got shape {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-2], x.shape[-1])


def unpack_from_varlen(x: Tensor, batch: int) -> Tensor:
    """Restore ``(B, L, H, D)`` from the packed ``(B * L, H, D)`` layout."""
    if x.dim() != 3:
        raise ValueError(f"expected a (total_tokens, H, D) tensor, got shape {tuple(x.shape)}")
    total_tokens = x.shape[0]
    if total_tokens % batch != 0:
        raise ValueError(
            f"{total_tokens} packed tokens do not divide into {batch} rows; the packing "
            "and the batch disagree about the sequence length"
        )
    return x.reshape(batch, total_tokens // batch, x.shape[-2], x.shape[-1])


def segment_lengths(cu_seqlens: Tensor) -> Tensor:
    """Segment lengths implied by ``cu_seqlens``. For assertions and logging."""
    return cu_seqlens[1:] - cu_seqlens[:-1]


def describe_packing(cu_seqlens: Tensor, max_seqlen: Optional[int] = None) -> str:
    """One-line summary of a packing, for a log line or an error message."""
    lengths = segment_lengths(cu_seqlens)
    return (
        f"{lengths.numel()} segments over {int(cu_seqlens[-1])} tokens "
        f"(min={int(lengths.min())}, max={int(lengths.max())}, bound={max_seqlen})"
    )
