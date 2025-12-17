###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Execution Runner.

This module selects and executes patches based on the given PatchContext.
"""

import os
from typing import Any, Dict, List, Optional

from primus.core.patches.context import PatchContext
from primus.core.patches.patch_registry import PatchRegistry
from primus.modules.module_utils import log_rank_0

# -----------------------------------------------------------------------------
# Parse PRIMUS_PATCHES Environment Variable
# -----------------------------------------------------------------------------


def _parse_enabled_patches_from_env() -> Optional[List[str]]:
    """
    Environment variable: PRIMUS_PATCHES

        "all" or ""   -> enable all patches (default)
        "none"        -> disable all patches
        "a,b,c"       -> enable only {a, b, c}
    """
    raw = os.getenv("PRIMUS_PATCHES", "").strip()
    if not raw or raw.lower() == "all":
        return None
    if raw.lower() == "none":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


# -----------------------------------------------------------------------------
# Run Patches
# -----------------------------------------------------------------------------


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
    Execute all applicable patches for the given context.

    Returns the number of patches applied (or would-be-applied in dry_run).
    """

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

    if enabled_ids is None:
        enabled_ids = _parse_enabled_patches_from_env()

    patches = PatchRegistry.iter_patches()

    # Deterministic ordering: (priority ASC, id ASC)
    patches = sorted(patches, key=lambda p: (p.priority, p.id))

    applied_count = 0
    applied_ids: List[str] = []

    log_rank_0("--------------------------------------------------------------------------------")
    log_rank_0(
        f"[Patch] Executing patches: backend={backend}, phase={phase}, "
        f"backend_version={backend_version}, primus_version={primus_version}, "
        f"model={model_name}, module={module_name}, platform={platform}, "
        f"dry_run={dry_run}, enabled_ids={enabled_ids}"
    )

    for patch in patches:
        # log_rank_0(f"[Patch] Applying patch: {patch.id} (priority={patch.priority}) {patch}")
        # ID filter
        if enabled_ids is not None and patch.id not in enabled_ids:
            continue

        # Applicability filter
        if not patch.applies_to(ctx):
            log_rank_0(f"[Patch] ⊘ Skipped: {patch.id} (condition not met or version mismatch)")
            continue

        # Dry-run mode
        if dry_run:
            log_rank_0(f"[Patch] (dry-run) Would apply: {patch.id} " f"(priority={patch.priority})")
            applied_count += 1
            applied_ids.append(patch.id)
            continue

        # Execute patch
        try:
            log_rank_0("--------------------------------------------------------------------------------")
            log_rank_0(f"[Patch] Applying {patch.id}: {patch.description}")
            patch.apply(ctx)
            applied_count += 1
            applied_ids.append(patch.id)
            log_rank_0(f"[Patch] ✓ Applied: {patch.id} (priority={patch.priority})")
        except Exception as e:
            log_rank_0(f"[Patch] ✗ Patch '{patch.id}' failed: {e}")
            if stop_on_error:
                raise

    total_patches = len(patches)
    log_rank_0(
        f"[Patch] Applied {applied_count}/{total_patches} patches for {backend}/{phase}: " f"{applied_ids}"
    )
    log_rank_0("--------------------------------------------------------------------------------")

    return applied_count
