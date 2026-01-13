###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GPU preflight checks.

Levels:
  - basic: availability/identity/safety checks (may FAIL)
  - standard: consistency/topology checks (WARN by default)
  - full: perf sanity checks (WARN only)
"""

from .gpu_basic import run_gpu_basic_checks
from .gpu_perf import run_gpu_full_checks
from .gpu_topology import run_gpu_standard_checks
from .utils import Finding, ProbeResult

__all__ = [
    "Finding",
    "ProbeResult",
    "run_gpu_basic_checks",
    "run_gpu_standard_checks",
    "run_gpu_full_checks",
]
