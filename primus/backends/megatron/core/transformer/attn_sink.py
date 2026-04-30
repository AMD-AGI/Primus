###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Attention sink for DeepSeek-V4.

Reference: techblog §1.5 ("attn_sink: Stabilizing the Softmax").

Each head learns one scalar ``sinks[h]`` representing an "extra key column"
with value 0. The softmax is taken over ``[real_keys, sink]``, but only the
``real_keys`` probabilities are used for the value-weighted sum. This lets
each head opt out of attending — important for V4 long-context stability.

Math:

.. code-block::

    logits  : [B, H, Sq, Sk]
    sink    : sinks[h]
    combined = cat([logits, sinks_broadcast(B,H,Sq,1)], dim=-1)   # [B, H, Sq, Sk+1]
    combined -= combined.max(dim=-1, keepdim=True)                 # stabilization
    probs    = softmax(combined, dim=-1)[..., :-1]                  # drop sink col
    out      = probs @ V                                             # [B, H, Sq, D]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AttentionSink(nn.Module):
    """Per-head learnable softmax sink.

    Args:
        num_heads: number of attention heads ``H``. The sink scalars are
            initialized to ``0`` (uniform "no sink" baseline).
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        self.num_heads = num_heads
        self.sinks = nn.Parameter(torch.zeros(num_heads))

    def softmax_with_sink(
        self,
        logits: torch.Tensor,
        *,
        dim: int = -1,
    ) -> torch.Tensor:
        """Compute softmax over ``logits`` with an extra sink column appended.

        Args:
            logits: ``[B, H, Sq, Sk]`` (or any shape where dim ``-3`` is ``H``).
            dim: which dim is the key axis. Defaults to ``-1``.

        Returns:
            ``probs`` with the sink column **dropped**, same shape as ``logits``.
        """
        if dim != -1:
            raise NotImplementedError("Only dim=-1 supported for now.")

        # Broadcast sink scalar to [..., 1] matching logits' leading dims.
        # logits shape: [..., H, Sq, Sk]. We need a sink column at the last dim.
        # The H axis is at position -3. Build a [1]*L tensor and put sinks at -3.
        target_shape = list(logits.shape[:-1]) + [1]
        sinks = self.sinks
        # Find the H axis: by convention the second-to-last "non-key" axis carries H.
        # Caller is responsible for laying out logits as [..., H, Sq, Sk].
        # Reshape sinks to broadcast over (..., H, Sq, 1).
        # Number of dims in logits:
        ndim = logits.dim()
        # Construct view: replace the H axis (-3) with H, others with 1.
        view_shape = [1] * ndim
        view_shape[-3] = self.num_heads
        sink_col = sinks.view(*view_shape).expand(*target_shape).to(logits.dtype)

        combined = torch.cat([logits, sink_col], dim=-1)
        combined = combined - combined.amax(dim=-1, keepdim=True).detach()
        probs = combined.softmax(dim=-1)
        return probs[..., :-1]

    def forward(
        self,
        logits: torch.Tensor,
        value: torch.Tensor,
        *,
        dropout: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention output with sink-augmented softmax.

        Args:
            logits: ``[B, H, Sq, Sk]`` pre-softmax logits (already scaled and
                masked).
            value: ``[B, H, Sk, Dh]`` value tensor.
            dropout: if not ``None``, a probability passed to
                ``F.dropout`` on the softmax probs.

        Returns:
            ``[B, H, Sq, Dh]`` attention output.
        """
        probs = self.softmax_with_sink(logits, dim=-1)
        if dropout is not None and dropout > 0 and self.training:
            probs = torch.nn.functional.dropout(probs, p=dropout)
        probs = probs.to(value.dtype)
        return torch.matmul(probs, value)


__all__ = ["AttentionSink"]
