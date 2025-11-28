###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Registry and Registration

This module manages the global patch registry and provides the
@register_patch decorator for patch registration.
"""

import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from primus.core.patches.context import PatchContext
from primus.core.patches.patch import FunctionPatch

log = logging.getLogger(__name__)


# ============================================================================
# Patch Registry
# ============================================================================


class PatchRegistry:
    """
    Global patch registry.

    Usage:
        1. Register patches via @register_patch decorator
        2. Execute patches via run_patches()
    """

    _patches: Dict[str, FunctionPatch] = {}

    # ------------------------ Registration ------------------------ #

    @classmethod
    def register(cls, patch: FunctionPatch) -> FunctionPatch:
        """Register a patch."""
        if patch.id in cls._patches:
            log.warning("Patch '%s' already registered, overriding", patch.id)
        cls._patches[patch.id] = patch
        log.debug(
            "[PatchRegistry] Registered patch: %s (backend=%s, phase=%s, priority=%s)",
            patch.id,
            patch.backend,
            patch.phase,
            patch.priority,
        )
        return patch

    # ------------------------- Lookup APIs ------------------------ #

    @classmethod
    def get(cls, patch_id: str) -> FunctionPatch:
        """Get a patch by ID."""
        return cls._patches[patch_id]

    @classmethod
    def list_ids(cls) -> List[str]:
        """List all registered patch IDs."""
        return sorted(cls._patches.keys())

    @classmethod
    def iter_patches(cls) -> List[FunctionPatch]:
        """Iterate over all registered patches."""
        # Do not sort here, leave sorting to the runner to avoid frequent list creation
        return list(cls._patches.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all patches (useful for testing)."""
        cls._patches.clear()

    # ----------------------- Helper Queries ----------------------- #

    @classmethod
    def iter_by_tag(cls, tag: str) -> Iterable[FunctionPatch]:
        """Iterate patches that contain the given tag."""
        for p in cls._patches.values():
            if tag in p.tags:
                yield p


# ============================================================================
# Decorator: register_patch
# ============================================================================


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
    Decorator to register a function as a patch.

    Args:
        patch_id:
            Unique patch identifier (e.g., "megatron.deepseek_v3.fix_moe").
        description:
            Human-readable description.
        backend:
            Target backend (e.g., "megatron", "torchtitan"). None = all backends.
        phase:
            Target phase (e.g., "before_train", "build_args"). None = all phases.
        condition:
            Optional condition function for fine-grained control.

        priority:
            Execution priority. Smaller values run earlier. Default = 50.
            Recommendations:
                0-19   : Framework-level compatibility patches
                20-49  : Backend argument correction patches
                50-79  : Model/Scenario related patches
                80-100 : Experimental / Debug patches

        backend_versions:
            Optional list/tuple of backend version patterns (see version_matches).
        primus_versions:
            Optional list/tuple of Primus version patterns.
        tags:
            Optional list/tuple of tags (e.g., ["megatron", "llama3", "args"]).

    Example:

        @register_patch(
            "megatron.deepseek_v3.fix_moe",
            backend="megatron",
            phase="before_train",
            description="Fix MoE load balancing for DeepSeek V3",
            backend_versions=["0.8.*"],
            priority=30,
            tags=["megatron", "deepseek", "moe"],
            condition=lambda ctx: ctx.model_name == "deepseek_v3",
        )
        def fix_deepseek_moe(ctx: PatchContext):
            args = ctx.extra.get("args")
            if args and hasattr(args, "moe_aux_loss_coeff"):
                args.moe_aux_loss_coeff = 0.001
    """

    backend_versions = list(backend_versions or [])
    primus_versions = list(primus_versions or [])
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
            backend_version_patterns=backend_versions,
            primus_version_patterns=primus_versions,
            tags=tag_set,
        )
        PatchRegistry.register(patch)
        return func

    return decorator
