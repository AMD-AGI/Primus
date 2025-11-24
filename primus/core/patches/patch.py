###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Implementation

This module defines the FunctionPatch class for function-based patches.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from primus.core.patches.context import PatchContext

log = logging.getLogger(__name__)


# ============================================================================
# Patch Implementation
# ============================================================================


@dataclass
class FunctionPatch:
    """
    Function-based patch implementation.

    A patch is defined by:
        - id: Unique identifier
        - handler: Function that implements the patch logic
        - backend: Target backend (None = all backends)
        - phase: Target phase (None = all phases)
        - condition: Optional condition function for fine-grained control
    """

    id: str
    description: str
    handler: Callable[[PatchContext], None]
    backend: Optional[str] = None
    phase: Optional[str] = None
    condition: Optional[Callable[[PatchContext], bool]] = None

    def applies_to(self, ctx: PatchContext) -> bool:
        """
        Check if this patch applies to the given context.

        Filters:
            1. Backend match
            2. Phase match
            3. Custom condition (if provided)
        """
        # Backend filter
        if self.backend is not None and self.backend != ctx.backend:
            return False

        # Phase filter
        if self.phase is not None and self.phase != ctx.phase:
            return False

        # Custom condition
        if self.condition is not None and not self.condition(ctx):
            return False

        return True

    def apply(self, ctx: PatchContext) -> None:
        """Execute the patch handler."""
        log.debug(f"[Patch] Applying {self.id}: {self.description}")
        self.handler(ctx)
