###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager reference for Quantile Balancing's per-microbatch statistic.

A verbatim extraction of what ``k3_quantile_balancing.compute_margin_histogram``
used to do inline, so it can be the permanent oracle for the fused FlyDSL
kernel. ``k3_quantile_balancing`` re-exports the name, so every existing
importer and every test in ``test_quantile_balancing.py`` is untouched.

The statistic (tech report §2.3.3, and see the parent module's docstring for the
report quotes that force each detail):

.. code-block:: text

    tau_i     = (k+1)-th largest of (s_i + b^(t))    the token's BIASED cutoff
    margin_ij = s_ij - tau_i                          RAW score - biased cutoff
    hist[j, .] = histogram of margin_{:,j} over [margin_min, margin_max]

Out-of-range margins are clamped into the end bins **and counted**, so a badly
chosen range is visible rather than silently biasing the quantile.

One numerical property is load-bearing for the kernel and is stated here because
this is the file the kernel is checked against: the bin index is a ``floor`` of
an IEEE division, i.e. a **discontinuous** function of the margin. A last-bit
difference in the division moves a count into the neighbouring bin, so any
reimplementation has to reproduce ``((margin - margin_min) / width).floor()``
exactly — a reciprocal multiply is not good enough. Measured: the FlyDSL kernel
does reproduce it bit-for-bit, including on values placed exactly on bin edges.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["compute_margin_histogram", "margin_cutoff"]


def margin_cutoff(scores: torch.Tensor, expert_bias: torch.Tensor, topk: int) -> torch.Tensor:
    """``tau``, ``[num_tokens, 1]``: the ``(k+1)``-th largest **biased** score.

    Taking the cutoff from a Top-(k+1) pass is the report's own trick ("Taking
    the cutoff from Top-(k+1) routing avoids a separate token-side quantile").
    Upstream Megatron routes with Top-k and there is no clean seam to widen it,
    so this runs its own ``torch.topk``; the first ``k`` entries of a Top-(k+1)
    are the Top-k, so it is mathematically identical.

    Split out of :func:`compute_margin_histogram` because it is the one stage the
    FlyDSL backend deliberately leaves to the library — see
    :mod:`.._flydsl_v1.histogram` for the measurement that justifies that.
    """
    num_experts = scores.shape[1]
    k_plus_1 = min(int(topk) + 1, num_experts)
    biased = scores + expert_bias.to(scores.dtype)
    return torch.topk(biased, k=k_plus_1, dim=1).values[:, -1:]


def compute_margin_histogram(
    scores: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    topk: int,
    num_bins: int,
    margin_min: float,
    margin_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Histogram each expert's routing margins over a batch of tokens.

    Args:
        scores: ``[num_tokens, num_experts]`` raw sigmoid router scores, i.e.
            ``s_ij`` — **not** the biased ones.
        expert_bias: ``[num_experts]`` current ``b^(t)``.
        topk: ``k``. The cutoff is the ``(k+1)``-th largest *biased* score.
        num_bins: uniform bins over ``[margin_min, margin_max]``.
        margin_min: lower edge; margins below it land in bin 0.
        margin_max: upper edge; margins above it land in the last bin.

    Returns:
        ``(hist, clamped)`` where ``hist`` is ``[num_experts, num_bins]``
        int64 counts and ``clamped`` is ``[2]`` int64 holding
        ``(#below range, #above range)`` so saturation is observable.
    """
    num_tokens, num_experts = scores.shape
    if num_tokens == 0:
        return (
            torch.zeros(num_experts, num_bins, dtype=torch.int64, device=scores.device),
            torch.zeros(2, dtype=torch.int64, device=scores.device),
        )

    tau = margin_cutoff(scores, expert_bias, topk)
    margins = scores - tau

    width = (margin_max - margin_min) / num_bins
    raw = ((margins - margin_min) / width).floor()
    below = (raw < 0).sum().to(torch.int64)
    above = (raw > num_bins - 1).sum().to(torch.int64)
    idx = raw.clamp_(0, num_bins - 1).to(torch.int64)

    # One flat bincount over (expert, bin) pairs: cheaper and simpler than a
    # per-expert loop, and bincount already returns int64.
    offsets = torch.arange(num_experts, device=scores.device, dtype=torch.int64) * num_bins
    flat = (idx + offsets).reshape(-1)
    hist = torch.bincount(flat, minlength=num_experts * num_bins).view(num_experts, num_bins)

    return hist, torch.stack([below, above])
