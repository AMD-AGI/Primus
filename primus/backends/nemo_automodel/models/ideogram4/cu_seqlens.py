###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Host-side ``cu_seqlens`` construction for the ``[pad][text][image]`` layout.

Kept apart from the adapter that calls it because it is pure arithmetic over
Python integers, and so is worth testing on its own without building a model.

WHY BUILD IT HERE RATHER THAN DERIVE IT FROM THE MASK:
  The general mask-to-packing transform in ``attention/varlen_utils.py`` is exact
  and needs nothing from the caller, but it reads the mask, which means
  device-to-host synchronization inside the compiled region -- a graph break, and
  under FSDP2 a desynchronization of the per-layer collectives. The adapter, by
  contrast, already holds the per-sample text lengths as Python integers before
  the model is called. Building the packing from those costs one host-to-device
  copy and no reads in the other direction.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def build_cu_seqlens(
    text_lengths: Sequence[int],
    max_text_tokens: int,
    num_image_tokens: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Variable-length packing for a ``[left-pad][text][image]`` batch.

    Over the flattened ``(B * S)`` sequence, where ``S = max_text_tokens +
    num_image_tokens``, each row contributes exactly TWO segments, which is what the
    model's ``(segment_i == segment_j)`` mask encodes:

    * ``[b*S, b*S + offset)`` -- the left-pad block, attending only to itself and
      discarded downstream.
    * ``[b*S + offset, (b+1)*S)`` -- the text and image block, attended jointly.

    with ``offset = max_text_tokens - text_lengths[b]``.

    THE RESERVED PAD COLUMN, AND WHY THE SEGMENT COUNT IS FIXED:
      Because every row contributes two segments regardless of its caption, the
      result always has ``2B + 1`` entries -- a shape that does not depend on the
      data. That is what keeps the compiled graph reusable: a segment count that
      varied with the captions would change this tensor's shape and force a
      recompilation for every distinct length pattern, which for real captions
      means nearly every step.

      The count is only fixed if no row is entirely text, since a row with no
      padding contributes one segment rather than two. The caller reserves one
      always-pad column so ``offset >= 1`` everywhere, which makes that
      unreachable. A zero-length segment is not representable for varlen flash
      anyway, so the two requirements coincide.

    Args:
        text_lengths: per-sample real, non-pad text token counts, as integers.
        max_text_tokens: padded width of the text region, INCLUDING the reserved
            pad column.
        num_image_tokens: ``grid_h * grid_w``.
        device: destination. The one host-to-device copy this causes is the thing
            being traded for having no device-to-host reads later.

    Returns:
        ``int32`` of shape ``(2 * len(text_lengths) + 1,)``, cumulative segment
        starts from 0 to ``B * S``, which is the varlen kernel's contract.

    Raises:
        ValueError: if any row has no padding, since that breaks the fixed segment
            count the whole design rests on.
    """
    if max_text_tokens <= 0:
        raise ValueError(f"max_text_tokens must be positive, got {max_text_tokens}")
    if num_image_tokens <= 0:
        raise ValueError(f"num_image_tokens must be positive, got {num_image_tokens}")
    if len(text_lengths) == 0:
        raise ValueError("text_lengths is empty; there is nothing to pack")

    seq_len = max_text_tokens + num_image_tokens
    starts: List[int] = []
    for row, num_text in enumerate(text_lengths):
        offset = max_text_tokens - int(num_text)
        if offset < 1:
            raise ValueError(
                f"row {row} has text_length={int(num_text)} with "
                f"max_text_tokens={max_text_tokens}, leaving no padding. Every row needs at "
                "least one pad token so that the segment count -- and so cu_seqlens' shape "
                "-- is the same for every batch; otherwise the compiled graph is rebuilt "
                "whenever a batch happens to contain a full-length caption."
            )
        base = row * seq_len
        starts.append(base)
        starts.append(base + offset)
    starts.append(len(text_lengths) * seq_len)
    return torch.tensor(starts, dtype=torch.int32, device=device)


def static_max_seqlen(max_text_tokens: int, num_image_tokens: int) -> int:
    """The STATIC bound to publish alongside the packing.

    The longest segment a row can produce is its text-and-image block, which is
    largest when the padding is at its minimum of one token. Deliberately the
    static worst case rather than the batch's true maximum: an integer derived from
    the data would be guarded by value, so the graph would be rebuilt whenever the
    longest caption in a batch changed.
    """
    return max_text_tokens + num_image_tokens - 1
