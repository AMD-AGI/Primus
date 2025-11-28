###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Execution Runner

This module handles the execution of patches based on context.
"""

import os
from typing import Any, Dict, List, Optional

from primus.core.patches.context import PatchContext
from primus.core.patches.registry import PatchRegistry
from primus.core.utils.distributed_logging import log_rank_0

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
    module_name: Optional[str] = None,
    platform: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    enabled_ids: Optional[List[str]] = None,
    dry_run: bool = False,
    stop_on_error: bool = False,
) -> int:
    """
    Execute applicable patches for the given context.

    Args:
        backend:
            Backend name (e.g., "megatron", "torchtitan").
        phase:
            Execution phase (e.g., "setup", "build_args", "before_train").
        backend_version:
            Backend version (e.g., "0.8.0", "commit:abc123").
        primus_version:
            Primus version string used by Primus itself.
        model_name:
            Model name (e.g., "llama3_70B").
        module_name:
            Primus module name (e.g., "pre_trainer", "sft_trainer"), if applicable.
        platform:
            Platform identifier (e.g., "MI300X", "MI355X").
        extra:
            Additional context data (e.g., {"args": megatron_args}).
        enabled_ids:
            Limit to specific patch IDs. If None, use env var PRIMUS_PATCHES.

        dry_run:
            If True, only log what would be applied, but do not execute patches.
        stop_on_error:
            If True, raise the first exception from a patch.
            If False (default), log the error and continue with other patches.

    Returns:
        Number of patches successfully applied (or would-be-applied in dry_run mode).

    Environment Control:
        Set PRIMUS_PATCHES to control which patches are enabled:
        - PRIMUS_PATCHES="all" (default): Enable all patches
        - PRIMUS_PATCHES="none": Disable all patches
        - PRIMUS_PATCHES="patch1,patch2": Enable only specified patches
    """
    # ---------------------------- Context ---------------------------- #
    ctx = PatchContext(
        backend=backend,
        phase=phase,
        backend_version=backend_version,
        primus_version=primus_version,
        model_name=model_name,
        module_name=module_name,
        platform=platform,
        extra=extra or {},
    )

    # Parse enabled patches from env if not specified
    if enabled_ids is None:
        enabled_ids = _parse_enabled_patches_from_env()

    applied_count = 0
    applied_patches: List[str] = []

    # ------------------------- Patch Iteration ------------------------ #
    patches = PatchRegistry.iter_patches()

    # sort by priority first, then by id for deterministic order
    patches = sorted(patches, key=lambda p: (p.priority, p.id))

    log_rank_0(
        f"[PatchSystem] Running patches for backend={backend}, phase={phase}, "
        f"backend_version={backend_version}, primus_version={primus_version}, "
        f"model={model_name}, module={module_name}, platform={platform}, "
        f"dry_run={dry_run}, enabled_ids={enabled_ids}"
    )

    for patch in patches:
        # Filter by enabled_ids
        if enabled_ids is not None and patch.id not in enabled_ids:
            continue

        # Check if patch applies
        if not patch.applies_to(ctx):
            continue

        if dry_run:
            log_rank_0(
                f"[PatchSystem] (dry-run) Would apply patch: {patch.id} " f"(priority={patch.priority})"
            )
            applied_count += 1
            applied_patches.append(patch.id)
            continue

        # Apply patch
        try:
            patch.apply(ctx)
            applied_count += 1
            applied_patches.append(patch.id)
            log_rank_0(f"[PatchSystem] ✓ Applied patch: {patch.id} (priority={patch.priority})")
        except Exception as e:  # noqa: BLE001
            log_rank_0(f"[PatchSystem] ✗ Patch {patch.id} failed: {e}")
            if stop_on_error:
                # Raise exception directly, handled by upper layer (commonly used in testing environment)
                raise

            # Continue with other patches (non-fatal)
            continue

    log_rank_0(f"[PatchSystem] Applied {applied_count} patches for {backend}/{phase}: " f"{applied_patches}")

    return applied_count
