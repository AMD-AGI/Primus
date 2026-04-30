###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Sliding-window mask helpers for DeepSeek-V4 dense / SWA attention layers.

Reference: techblog §1 ("Hybrid Attention") — dense layers
(``compress_ratio == 0``) attend only to the last ``attn_sliding_window``
tokens (default 128) plus an optional ``attn_sink``. HCA layers also use
SWA over the raw KV in addition to the compressed-KV pool.

This module only generates the **mask** — the actual ``q @ k^T`` happens in
the surrounding attention class. Returning a plain ``[Sq, Sk]`` additive
mask keeps it composable with both eager and flash-style backends.
"""

from __future__ import annotations

import torch


def sliding_window_causal_mask(
    seq_len: int,
    window: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``[seq_len, seq_len]`` additive attention mask.

    Position ``j`` is **allowed** for query ``i`` iff:

    * ``j <= i`` (causal), and
    * ``i - j < window`` (sliding window).

    Disallowed positions get ``-inf``; allowed positions get ``0``.

    A ``window`` of ``0`` or ``>= seq_len`` degenerates to the standard
    causal mask.
    """
    q = torch.arange(seq_len, device=device).unsqueeze(1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = q - k
    if window <= 0 or window >= seq_len:
        allowed = dist >= 0
    else:
        allowed = (dist >= 0) & (dist < window)
    return torch.where(allowed, 0.0, float("-inf")).to(dtype)


def sliding_window_kv_indices(
    seq_len: int,
    window: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """For each query ``i`` return the ``window`` raw-KV indices it attends
    to: ``[max(0, i-window+1), i]``.

    Returned tensor has shape ``[seq_len, window]`` (long); positions before
    the start of the sequence are filled with ``-1`` so the caller can drop
    or zero-mask them.
    """
    if window <= 0:
        # No sliding-window restriction: nothing to gather (caller should use full causal).
        return torch.empty(seq_len, 0, dtype=torch.long, device=device)

    i = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
    offset = torch.arange(window - 1, -1, -1, device=device).unsqueeze(0)  # [1, W] (descending)
    j = i - offset  # [S, W]
    j = torch.where(j >= 0, j, torch.full_like(j, -1))
    return j


__all__ = [
    "sliding_window_causal_mask",
    "sliding_window_kv_indices",
]
