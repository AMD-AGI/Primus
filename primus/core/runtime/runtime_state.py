###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Runtime State Management for Training.

This module provides a dedicated RuntimeState object for storing dynamic
runtime metrics that change during training (e.g., image dimensions, timesteps).
This is separate from backend_args (which is for configuration parameters).
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RuntimeState:
    """
    Dedicated runtime state object for per-iteration training metrics.

    This is separate from backend_args (which is for configuration) and
    provides a clean place to store dynamic runtime metrics that change
    during training (e.g., image dimensions, timesteps, etc.).
    """

    # Per-iteration metrics (updated each forward step)
    last_metrics: Dict[str, Any] = field(default_factory=dict)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update last_metrics with new metrics from forward step."""
        self.last_metrics.update(metrics)
