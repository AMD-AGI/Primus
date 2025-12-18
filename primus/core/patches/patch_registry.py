###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Registry and Registration.

Implements the global registry where patches are stored, and the decorator
for registering patches.
"""

import logging
from typing import Callable, Iterable, List, Optional, Sequence

from primus.core.patches.context import PatchContext
from primus.core.patches.patch import FunctionPatch

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Patch Registry
# -----------------------------------------------------------------------------


class PatchRegistry:
    """Global registry for all patches."""

    _patches: List[FunctionPatch] = []

    @classmethod
    def register(cls, patch: FunctionPatch) -> FunctionPatch:
        """Register or override a patch."""
        # Check if patch with same id already exists
        for i, existing_patch in enumerate(cls._patches):
            if existing_patch.id == patch.id:
                log.warning("Patch '%s' already registered; overriding.", patch.id)
                cls._patches[i] = patch
                return patch

        # Add new patch
        cls._patches.append(patch)
        return patch

    @classmethod
    def get(cls, patch_id: str) -> Optional[FunctionPatch]:
        """Get patch by id, returns None if not found."""
        for patch in cls._patches:
            if patch.id == patch_id:
                return patch
        return None

    @classmethod
    def list_ids(cls) -> List[str]:
        return sorted([p.id for p in cls._patches])

    @classmethod
    def iter_patches(cls, backend: Optional[str] = None, phase: Optional[str] = None) -> List[FunctionPatch]:
        """
        Get all patches, optionally pre-filtered by backend and/or phase.

        Args:
            backend: If provided, only return patches matching this backend
                    (or patches with backend=None)
            phase: If provided, only return patches matching this phase
                  (or patches with phase=None)

        Returns:
            List of FunctionPatch objects
        """
        patches = list(cls._patches)

        # Pre-filter by backend if specified
        if backend is not None:
            patches = [p for p in patches if p.backend is None or p.backend == backend]

        # Pre-filter by phase if specified
        if phase is not None:
            patches = [p for p in patches if p.phase is None or p.phase == phase]

        return patches

    @classmethod
    def clear(cls) -> None:
        cls._patches.clear()

    @classmethod
    def iter_by_tag(cls, tag: str) -> Iterable[FunctionPatch]:
        for p in cls._patches:
            if tag in p.tags:
                yield p


# -----------------------------------------------------------------------------
# Decorator: register_patch
# -----------------------------------------------------------------------------


def register_patch(
    patch_id: str,
    *,
    description: str = "",
    backend: Optional[str] = None,
    phase: Optional[str] = None,
    condition: Optional[Callable[[PatchContext], bool]] = None,
    priority: int = 50,
    backend_versions: Optional[Sequence[str]] = None,
    primus_versions: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
) -> Callable[[Callable[[PatchContext], None]], Callable[[PatchContext], None]]:
    """
    Decorator for registering a patch function.

    Arguments mirror FunctionPatch fields.
    """

    version_patterns_backend = list(backend_versions or [])
    version_patterns_primus = list(primus_versions or [])
    tag_set = set(tags or [])

    def decorator(func: Callable[[PatchContext], None]) -> Callable[[PatchContext], None]:
        patch = FunctionPatch(
            id=patch_id,
            description=description or func.__doc__ or "",
            handler=func,
            backend=backend,
            phase=phase,
            condition=condition,
            priority=priority,
            backend_version_patterns=version_patterns_backend,
            primus_version_patterns=version_patterns_primus,
            tags=tag_set,
        )
        PatchRegistry.register(patch)
        return func

    return decorator
