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
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from primus.core.patches.context import PatchContext
from primus.core.patches.patch import FunctionPatch

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Patch Registry
# -----------------------------------------------------------------------------


class PatchRegistry:
    """Global registry for all patches."""

    _patches: Dict[str, FunctionPatch] = {}

    @classmethod
    def register(cls, patch: FunctionPatch) -> FunctionPatch:
        """Register or override a patch."""
        if patch.id in cls._patches:
            log.warning("Patch '%s' already registered; overriding.", patch.id)
        cls._patches[patch.id] = patch
        return patch

    @classmethod
    def get(cls, patch_id: str) -> FunctionPatch:
        return cls._patches[patch_id]

    @classmethod
    def list_ids(cls) -> List[str]:
        return sorted(cls._patches.keys())

    @classmethod
    def iter_patches(cls) -> List[FunctionPatch]:
        return list(cls._patches.values())

    @classmethod
    def clear(cls) -> None:
        cls._patches.clear()

    @classmethod
    def iter_by_tag(cls, tag: str) -> Iterable[FunctionPatch]:
        for p in cls._patches.values():
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
