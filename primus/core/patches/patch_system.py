###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Patch System

A lightweight, phase-aware patching system for backend frameworks.

Design Goals:
    1. Handle version compatibility issues across Megatron/TorchTitan/etc
    2. Apply hotfixes without modifying upstream framework code
    3. Support model-specific patches (DeepSeek, Llama, Mixtral, etc)
    4. Phase-based execution (before_import, after_build_args, before_train, etc)
    5. Environment variable control (PRIMUS_PATCHES)

Architecture:
    - PatchContext: Runtime context (backend, phase, version, extra data)
    - FunctionPatch: Function-based patch implementation
    - PatchRegistry: Central registry with decorator-based registration
    - run_patches(): Execute applicable patches for given context

Usage:
    # 1. Define a patch
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

    # 2. Execute patches
    run_patches(
        backend="megatron",
        phase="before_train",  # Can also use legacy names like "before_import_backend"
        backend_version="0.8.0",
        model_name="deepseek_v3",
        extra={"args": megatron_args},
    )
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


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


# ============================================================================
# Run Patches
# ============================================================================


def _parse_enabled_patches_from_env() -> Optional[List[str]]:
    """
    Parse enabled patches from environment variable.

    Environment variable: PRIMUS_PATCHES
        - "all" or empty: Enable all patches (default)
        - "none": Disable all patches
        - "id1,id2,id3": Enable only specified patches

    Returns:
        - None: Use default (all patches)
        - []: No patches
        - ["id1", "id2"]: Only these patches
    """
    raw = os.getenv("PRIMUS_PATCHES", "").strip()
    if not raw or raw.lower() == "all":
        return None
    if raw.lower() == "none":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def run_patches(
    *,
    backend: str,
    phase: str,
    backend_version: Optional[str] = None,
    primus_version: Optional[str] = None,
    model_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    enabled_ids: Optional[List[str]] = None,
) -> int:
    """
    Execute applicable patches for the given context.

    Args:
        backend: Backend name (e.g., "megatron", "torchtitan")
        phase: Execution phase (e.g., "setup", "build_args", "before_train")
                Legacy phases are automatically normalized
        backend_version: Backend version (e.g., "0.8.0")
        primus_version: Primus version
        model_name: Model name (e.g., "llama3_70B")
        extra: Additional context data (e.g., {"args": megatron_args})
        enabled_ids: Limit to specific patch IDs (None = use env var)

    Returns:
        Number of patches applied

    Environment Control:
        Set PRIMUS_PATCHES to control which patches are enabled:
        - PRIMUS_PATCHES="all" (default): Enable all patches
        - PRIMUS_PATCHES="none": Disable all patches
        - PRIMUS_PATCHES="patch1,patch2": Enable only specified patches
    """
    # Normalize phase (handles legacy phase names)
    normalized_phase = normalize_phase(phase)

    # Create context
    ctx = PatchContext(
        backend=backend,
        phase=normalized_phase,
        backend_version=backend_version,
        primus_version=primus_version,
        model_name=model_name,
        extra=extra or {},
    )

    # Parse enabled patches from env if not specified
    if enabled_ids is None:
        enabled_ids = _parse_enabled_patches_from_env()

    # Log execution (show phase normalization if applicable)
    phase_info = f"{phase}" if phase == normalized_phase else f"{phase} → {normalized_phase}"
    log.debug(
        f"[PatchSystem] Running patches: backend={backend}, phase={phase_info}, "
        f"version={backend_version}, model={model_name}, enabled={enabled_ids}"
    )

    applied_count = 0
    applied_patches = []

    for patch in PatchRegistry.iter_patches():
        # Filter by enabled_ids
        if enabled_ids is not None and patch.id not in enabled_ids:
            log.debug(f"[PatchSystem] Skipped {patch.id} (not in enabled list)")
            continue

        # Check if patch applies
        if not patch.applies_to(ctx):
            log.debug(f"[PatchSystem] Skipped {patch.id} (does not apply to context)")
            continue

        # Apply patch
        try:
            log.debug(f"[PatchSystem] Applying {patch.id}: {patch.description}")
            patch.apply(ctx)
            applied_count += 1
            applied_patches.append(patch.id)
            log.info(f"[PatchSystem] ✓ Applied patch: {patch.id}")
        except Exception as e:
            log.exception(f"[PatchSystem] ✗ Patch {patch.id} failed: {e}")
            # Continue with other patches (non-fatal)
            continue

    if applied_count > 0:
        log.info(
            f"[PatchSystem] Applied {applied_count} patches for {backend}/{normalized_phase}: {', '.join(applied_patches)}"
        )
    else:
        log.debug(f"[PatchSystem] No patches applied for {backend}/{normalized_phase}")

    return applied_count


# ============================================================================
# Utility Functions
# ============================================================================


def version_matches(version: str, pattern: str) -> bool:
    """
    Check if version matches a pattern.

    Patterns:
        - Exact match: "0.8.0"
        - Prefix match: "0.8.*"
        - Contains: "*2024-09*"

    Args:
        version: Version string to check
        pattern: Pattern to match against

    Returns:
        True if version matches pattern
    """
    if "*" not in pattern:
        return version == pattern

    # Simple wildcard matching
    regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
    return re.match(f"^{regex_pattern}$", version) is not None
