###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Implementation.

Defines the FunctionPatch class used to represent executable patches.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

from primus.core.patches.context import PatchContext
from primus.core.patches.utils import version_in_range
from primus.modules.module_utils import log_rank_0

# -----------------------------------------------------------------------------
# FunctionPatch
# -----------------------------------------------------------------------------


@dataclass
class FunctionPatch:
    """
    Represents a single function-based patch.

    Attributes:
        id:
            Unique identifier of the patch.
        description:
            Human-readable description.
        handler:
            Callable implementing patch logic.
        backend:
            Backend name filter (None = all backends).
        phase:
            Execution phase filter (None = all phases).
        condition:
            Optional additional predicate taking PatchContext -> bool.
        priority:
            Sorting key; smaller values are executed earlier.
        backend_version_patterns:
            List of version patterns matched against ctx.backend_version.
        primus_version_patterns:
            List of version patterns matched against ctx.primus_version.
        tags:
            Arbitrary tags for grouping/debugging (e.g., {"llama3", "moe"}).
    """

    id: str
    description: str
    handler: Callable[[PatchContext], None]
    backend: Optional[str] = None
    phase: Optional[str] = None
    condition: Optional[Callable[[PatchContext], bool]] = None

    priority: int = 50
    backend_version_patterns: List[str] = field(default_factory=list)
    primus_version_patterns: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    # ---------------------------------------------------------------------

    def _match_backend_version(self, ctx: PatchContext) -> bool:
        if not self.backend_version_patterns:
            return True
        if ctx.backend_version is None:
            return False
        return any(
            version_in_range(ctx.backend_version, pattern) for pattern in self.backend_version_patterns
        )

    def _match_primus_version(self, ctx: PatchContext) -> bool:
        if not self.primus_version_patterns:
            return True
        if ctx.primus_version is None:
            return False
        return any(version_in_range(ctx.primus_version, pattern) for pattern in self.primus_version_patterns)

    def applies_to(self, ctx: PatchContext) -> bool:
        """Return True if this patch should be applied to the given context."""
        if self.backend is not None and self.backend != ctx.backend:
            log_rank_0(
                f"[Patch] ⊘ Skipped: {self.id} (backend mismatch: "
                f"expected={self.backend}, actual={ctx.backend})"
            )
            return False
        if self.phase is not None and self.phase != ctx.phase:
            log_rank_0(
                f"[Patch] ⊘ Skipped: {self.id} (phase mismatch: "
                f"expected={self.phase}, actual={ctx.phase})"
            )
            return False
        if not self._match_backend_version(ctx):
            log_rank_0(
                f"[Patch] ⊘ Skipped: {self.id} (backend version mismatch: "
                f"required={self.backend_version_patterns}, actual={ctx.backend_version})"
            )
            return False
        if not self._match_primus_version(ctx):
            log_rank_0(
                f"[Patch] ⊘ Skipped: {self.id} (primus version mismatch: "
                f"required={self.primus_version_patterns}, actual={ctx.primus_version})"
            )
            return False
        if self.condition is not None and not self.condition(ctx):
            log_rank_0(f"[Patch] ⊘ Skipped: {self.id} (condition not met)")
            return False
        return True

    # ---------------------------------------------------------------------

    def apply(self, ctx: PatchContext) -> None:
        """Invoke the patch handler."""
        self.handler(ctx)
