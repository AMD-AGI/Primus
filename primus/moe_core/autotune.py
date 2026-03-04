###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TuneResult:
    score: float
    config: dict[str, Any]


def grid_search(
    candidates: Iterable[dict[str, Any]],
    evaluate: Callable[[dict[str, Any]], float],
) -> TuneResult:
    """
    Minimal backend-agnostic grid-search utility.
    Lower score is considered better.
    """
    best_score = float("inf")
    best_cfg: dict[str, Any] = {}
    for cfg in candidates:
        score = float(evaluate(cfg))
        if score < best_score:
            best_score = score
            best_cfg = dict(cfg)
    return TuneResult(score=best_score, config=best_cfg)
