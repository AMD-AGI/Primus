###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Context and Phase Management

This module defines the execution context for patches and phase constants.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ============================================================================
# Phase Constants
# ============================================================================

# Training lifecycle phases
PHASES = [
    "setup",  # Environment preparation
    "build_args",  # Argument building
    "before_train",  # Before training starts
    # Reserved extension points (future additions):
    # "after_train",
    # "before_eval",
    # "after_eval",
]


# ============================================================================
# Patch Context
# ============================================================================


@dataclass
class PatchContext:
    """
    Patch execution context.

    Attributes:
        backend:
            Backend name (e.g., "megatron", "torchtitan").
        phase:
            Execution phase (e.g., "setup", "build_args", "before_train").
        backend_version:
            Backend version identifier (e.g., "0.8.0", "commit:abc123").
        primus_version:
            Primus version (optional, used for Primus-side compatibility patches).
        model_name:
            Model name (e.g., "llama3_70B", "deepseek_v3").
        module_name:
            Primus module name (e.g., "pre_trainer", "sft_trainer") if applicable.
        platform:
            Platform identifier (e.g., "MI300X", "MI355X") if available.
        extra:
            Additional context data (e.g., {"args": megatron_args, "config": module_config}).
    """

    backend: str
    phase: str
    backend_version: Optional[str] = None
    primus_version: Optional[str] = None
    model_name: Optional[str] = None
    module_name: Optional[str] = None
    platform: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
