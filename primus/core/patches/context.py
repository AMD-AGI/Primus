###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Context and Phase Management

This module defines the execution context for patches and manages
phase constants and normalization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ============================================================================
# Phase Constants and Mapping
# ============================================================================

# Simplified phases (only 3 phases)
PHASES = [
    "setup",  # Environment preparation
    "build_args",  # Argument building
    "before_train",  # Before training starts
]

# Legacy phase mapping (for backward compatibility)
PHASE_ALIASES = {
    "before_import_backend": "setup",
    "after_import_backend": "setup",
    "before_build_args": "build_args",
    "after_build_args": "build_args",
    "after_train": None,  # Removed (no use case)
}


def normalize_phase(phase: str) -> str:
    """
    Normalize phase name using alias mapping.

    Args:
        phase: Original phase name

    Returns:
        Normalized phase name

    Raises:
        ValueError: If phase is invalid
    """
    # Check if it's already a valid phase
    if phase in PHASES:
        return phase

    # Check if it's a legacy phase
    if phase in PHASE_ALIASES:
        normalized = PHASE_ALIASES[phase]
        if normalized is None:
            raise ValueError(f"Phase '{phase}' has been removed. " f"Valid phases are: {', '.join(PHASES)}")
        log.debug(f"[PatchSystem] Normalized legacy phase '{phase}' → '{normalized}'")
        return normalized

    # Invalid phase
    raise ValueError(
        f"Invalid phase '{phase}'. "
        f"Valid phases are: {', '.join(PHASES)}. "
        f"Legacy phases: {', '.join(PHASE_ALIASES.keys())}"
    )


# ============================================================================
# Patch Context
# ============================================================================


@dataclass
class PatchContext:
    """
    Patch execution context.

    Attributes:
        backend: Backend name (e.g., "megatron", "torchtitan")
        phase: Execution phase (e.g., "setup", "build_args", "before_train")
        backend_version: Backend version (e.g., "0.8.0", "commit:abc123")
        primus_version: Primus version (optional)
        model_name: Model name (e.g., "llama3_70B", "deepseek_v3")
        extra: Additional context data (e.g., {"args": megatron_args})

    Phases (Simplified):
        - "setup": Environment preparation (set env vars, configure runtime)
        - "build_args": Argument building (modify config/args during build process)
        - "before_train": Before starting training (hook training logic)

    Legacy Phase Mapping (for backward compatibility):
        - "before_import_backend" → "setup"
        - "after_import_backend" → "setup"
        - "before_build_args" → "build_args"
        - "after_build_args" → "build_args"
        - "after_train" → removed (no use case)
    """

    backend: str
    phase: str
    backend_version: Optional[str] = None
    primus_version: Optional[str] = None
    model_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
