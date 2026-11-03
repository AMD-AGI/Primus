###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Closed forms shared by the per-model FLOPs reporting patches."""


def local_visible_pairs(window: int, seq_len: int) -> int:
    """Causal-visible ``(q, k)`` pair count for sliding-window attention.

    Each query at position ``q`` attends to keys in ``[max(0, q - window + 1), q]``.
    Summed over ``q in [0, seq_len)`` the count is:

    * ``window * seq_len - window * (window - 1) // 2`` for ``0 < window < seq_len``
      (queries below ``window - 1`` see only ``q + 1`` keys, the others saturate at
      ``window``);
    * ``seq_len * (seq_len + 1) // 2`` for the full-causal fallback (``window <= 0``
      or ``window >= seq_len``).
    """
    s = int(seq_len)
    w = int(window)
    if w <= 0 or w >= s:
        return s * (s + 1) // 2
    return w * s - w * (w - 1) // 2
