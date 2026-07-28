###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared dataclasses for the node-smoke pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GPUResult:
    """Result of all checks for a single GPU on this node."""

    gpu: int
    status: str  # "PASS" | "FAIL" | "TIMEOUT"
    reason: str = ""
    duration_sec: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    """Whole-node verdict written to ``<dump>/smoke/<host>.json``."""

    host: str
    node_rank: int
    status: str  # "PASS" | "FAIL"
    duration_sec: float
    fail_reasons: List[str]
    tier1: Dict[str, Any]
    tier2: Dict[str, Any]
