###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Finding:
    # "info" | "warn" | "fail"
    level: str
    message: str
    details: Dict[str, Any]


@dataclass
class NetworkProbe:
    available_nics: List[str]
    ib_devices: List[str]
    env: Dict[str, Any]
    intent: Dict[str, Any]
