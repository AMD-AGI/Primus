###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Learning-rate multipliers for the vision trainer."""

from __future__ import annotations

import math


def lr_factor(
    step: int,
    *,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    eta_min_ratio: float,
) -> float:
    """Multiplier applied to each optimizer group's base learning rate.

    ``step`` is the LambdaLR epoch, starting at 0 on the first update.
    """
    step = max(int(step), 0)
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)
    kind = (scheduler_type or "cosine").strip().lower()
    if kind == "constant":
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if kind == "constant_with_warmup":
        return 1.0
    if kind != "cosine":
        raise ValueError(
            f"Unsupported lr_scheduler_type {scheduler_type!r}. "
            "Use constant, constant_with_warmup, or cosine."
        )
    span = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / span, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = float(eta_min_ratio)
    return floor + (1.0 - floor) * cosine
