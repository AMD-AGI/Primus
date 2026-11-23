###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus.core.patches.patch_registry import PatchRegistry


@pytest.fixture(autouse=True)
def _restore_patch_registry():
    """These tests empty the process-wide registry; the patch modules fill it only once, at import."""
    by_phase = {
        backend: {phase: list(patches) for phase, patches in phases.items()}
        for backend, phases in PatchRegistry._patches_by_backend_phase.items()
    }
    all_patches = list(PatchRegistry._all_patches)
    yield
    PatchRegistry.clear()
    for backend, phases in by_phase.items():
        for phase, patches in phases.items():
            PatchRegistry._patches_by_backend_phase[backend][phase] = patches
    PatchRegistry._all_patches.extend(all_patches)
