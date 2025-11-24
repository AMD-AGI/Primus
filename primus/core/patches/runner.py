###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Execution Runner

This module handles the execution of patches based on context.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from primus.core.patches.context import PatchContext, normalize_phase
from primus.core.patches.registry import PatchRegistry

log = logging.getLogger(__name__)


# ============================================================================
# Environment Parsing
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


# ============================================================================
# Run Patches
# ============================================================================


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
