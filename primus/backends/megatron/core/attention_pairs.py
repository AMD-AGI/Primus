###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Causal attention (query, key) pair counts for FLOPs accounting.

Pure Python (no torch / Megatron). Used by the Muse Spark and DeepSeek-V4 FLOPs patches (one formula for both).
"""

__all__ = ["causal_visible_pairs"]


def causal_visible_pairs(seq_length: int, window=None) -> int:
    """Number of (query, key) pairs that causal attention computes over ``seq_length``
    tokens when each query sees at most ``window`` keys (itself included).

    ``window`` is an int key count (HF ``sliding_window``, the query included). None, <= 0
    (the callers' "no window", as in DeepSeek-V4's ``swa_window``) or >= ``seq_length``
    means full causal attention: ``S * (S + 1) / 2``. Otherwise queries ``0 .. W-1`` see
    ``q + 1`` keys and the rest see ``W``: ``W * S - W * (W - 1) / 2``. A Megatron
    ``window_size`` tuple ``(left, right)`` is not a key count: convert it first
    (``left + 1`` for ``right == 0``); it raises TypeError here.
    """
    if isinstance(window, (tuple, list)):
        raise TypeError(
            f"window must be an int key count, got {window!r}; for a Megatron window_size "
            "(left, 0) pass left + 1"
        )
    s = int(seq_length)
    w = int(window or 0)
    if w <= 0 or w >= s:
        return s * (s + 1) // 2
    return w * s - w * (w - 1) // 2
