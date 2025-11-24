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
from typing import Callable, Dict, List, Optional

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

    @classmethod
    def register(cls, patch: FunctionPatch) -> FunctionPatch:
        """Register a patch."""
        if patch.id in cls._patches:
            log.warning(f"Patch '{patch.id}' already registered, overriding")
        cls._patches[patch.id] = patch
        log.debug(f"[PatchRegistry] Registered patch: {patch.id}")
        return patch

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
        return list(cls._patches.values())

    @classmethod
    def clear(cls):
        """Clear all patches (useful for testing)."""
        cls._patches.clear()


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
) -> Callable[[Callable[[PatchContext], None]], Callable[[PatchContext], None]]:
    """
    Decorator to register a function as a patch.

    Args:
        patch_id: Unique patch identifier (e.g., "megatron.deepseek_v3.fix_moe")
        description: Human-readable description
        backend: Target backend (e.g., "megatron", "torchtitan")
        phase: Target phase (e.g., "before_train", "after_build_args")
        condition: Optional condition function for fine-grained control

    Example:
        @register_patch(
            "megatron.deepseek_v3.fix_moe",
            backend="megatron",
            phase="before_train",
            description="Fix MoE load balancing for DeepSeek V3",
            condition=lambda ctx: ctx.model_name == "deepseek_v3",
        )
        def fix_deepseek_moe(ctx: PatchContext):
            args = ctx.extra.get("args")
            if args and hasattr(args, "moe_aux_loss_coeff"):
                args.moe_aux_loss_coeff = 0.001
    """

    def decorator(func: Callable[[PatchContext], None]) -> Callable[[PatchContext], None]:
        patch = FunctionPatch(
            id=patch_id,
            description=description or func.__doc__ or "",
            handler=func,
            backend=backend,
            phase=phase,
            condition=condition,
        )
        PatchRegistry.register(patch)
        return func

    return decorator
