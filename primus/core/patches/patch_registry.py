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
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from primus.core.patches.context import PatchContext
from primus.core.patches.patch import FunctionPatch

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Patch Registry
# -----------------------------------------------------------------------------


class PatchRegistry:
    """
    Global registry for all patches.

    Patches are organized by (backend, phase) for efficient lookup.
    Uses a nested dict structure: {backend: {phase: [patches]}}

    Special keys:
    - None: represents patches that apply to all backends/phases
    """

    # Nested dict: {backend: {phase: [FunctionPatch]}}
    _patches_by_backend_phase: Dict[Optional[str], Dict[Optional[str], List[FunctionPatch]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Flat list for operations that need all patches (like get by id, list_ids)
    _all_patches: List[FunctionPatch] = []

    @classmethod
    def register(cls, patch: FunctionPatch) -> FunctionPatch:
        """
        Register a patch, organized by backend and phase.

        A patch is stored in the appropriate bucket based on its backend/phase.
        If backend or phase is None, it goes into the None bucket.
        """
        # Remove existing patch with same id if it exists
        for i, existing_patch in enumerate(cls._all_patches):
            if existing_patch.id == patch.id:
                log.warning("Patch '%s' already registered; overriding.", patch.id)
                # Remove from old bucket
                old_backend = existing_patch.backend
                old_phase = existing_patch.phase
                if old_backend in cls._patches_by_backend_phase:
                    if old_phase in cls._patches_by_backend_phase[old_backend]:
                        try:
                            cls._patches_by_backend_phase[old_backend][old_phase].remove(existing_patch)
                        except ValueError:
                            pass
                # Remove from all_patches
                cls._all_patches[i] = patch
                break
        else:
            # New patch
            cls._all_patches.append(patch)

        # Add to bucket
        backend_key = patch.backend
        phase_key = patch.phase
        cls._patches_by_backend_phase[backend_key][phase_key].append(patch)

        return patch

    @classmethod
    def get(cls, patch_id: str) -> Optional[FunctionPatch]:
        """Get patch by id, returns None if not found."""
        for patch in cls._all_patches:
            if patch.id == patch_id:
                return patch
        return None

    @classmethod
    def list_ids(cls) -> List[str]:
        """Get sorted list of all patch ids."""
        return sorted([p.id for p in cls._all_patches])

    @classmethod
    def iter_patches(cls, backend: Optional[str] = None, phase: Optional[str] = None) -> List[FunctionPatch]:
        """
        Get patches filtered by backend and/or phase.

        This method efficiently retrieves patches using the pre-classified structure.

        Args:
            backend: If provided, only return patches for this backend (+ generic patches)
            phase: If provided, only return patches for this phase (+ generic patches)

        Returns:
            List of FunctionPatch objects

        Lookup strategy:
            - If both backend and phase are specified:
              Returns patches from: (backend, phase) + (None, phase) + (backend, None) + (None, None)
            - If only backend is specified:
              Returns patches from: (backend, *) + (None, *)
            - If only phase is specified:
              Returns patches from: (*, phase) + (*, None)
            - If neither is specified:
              Returns all patches
        """
        if backend is None and phase is None:
            # Return all patches
            return list(cls._all_patches)

        result = []
        seen_ids = set()

        if backend is not None and phase is not None:
            # Most specific case: both backend and phase specified
            # Order: (backend, phase), (None, phase), (backend, None), (None, None)
            for b_key, p_key in [
                (backend, phase),  # Exact match
                (None, phase),  # Generic backend, specific phase
                (backend, None),  # Specific backend, generic phase
                (None, None),  # Generic backend and phase
            ]:
                if b_key in cls._patches_by_backend_phase:
                    if p_key in cls._patches_by_backend_phase[b_key]:
                        for patch in cls._patches_by_backend_phase[b_key][p_key]:
                            if patch.id not in seen_ids:
                                result.append(patch)
                                seen_ids.add(patch.id)

        elif backend is not None:
            # Only backend specified, all phases
            # Look in (backend, *) and (None, *)
            for b_key in [backend, None]:
                if b_key in cls._patches_by_backend_phase:
                    for phase_dict in cls._patches_by_backend_phase[b_key].values():
                        for patch in phase_dict:
                            if patch.id not in seen_ids:
                                result.append(patch)
                                seen_ids.add(patch.id)

        elif phase is not None:
            # Only phase specified, all backends
            # Look in (*, phase) and (*, None)
            for p_key in [phase, None]:
                for backend_dict in cls._patches_by_backend_phase.values():
                    if p_key in backend_dict:
                        for patch in backend_dict[p_key]:
                            if patch.id not in seen_ids:
                                result.append(patch)
                                seen_ids.add(patch.id)

        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registered patches."""
        cls._patches_by_backend_phase.clear()
        cls._all_patches.clear()

    @classmethod
    def iter_by_tag(cls, tag: str) -> Iterable[FunctionPatch]:
        """Iterate patches that have the specified tag."""
        for patch in cls._all_patches:
            if tag in patch.tags:
                yield patch


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
