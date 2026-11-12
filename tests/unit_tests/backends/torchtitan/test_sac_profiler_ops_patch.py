###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.utils.checkpoint as checkpoint

import primus.backends.torchtitan.patches.sac_profiler_ops_patches as sac_patch
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.torch.sac_ignore_profiler_ops"


def test_patch_registered():
    patch = PatchRegistry.get(PATCH_ID)
    assert patch is not None
    assert patch.backend == "torchtitan"
    assert patch.phase == "setup"


@pytest.mark.skipif(not hasattr(checkpoint, "SAC_IGNORED_OPS"), reason="torch without SAC_IGNORED_OPS")
def test_profiler_ops_added_to_sac_ignored_ops(monkeypatch):
    monkeypatch.setattr(checkpoint, "SAC_IGNORED_OPS", set(checkpoint.SAC_IGNORED_OPS))
    sac_patch.patch_sac_ignore_profiler_ops(None)
    assert torch.ops.profiler._record_function_enter_new.default in checkpoint.SAC_IGNORED_OPS
    assert torch.ops.profiler._record_function_exit._RecordFunction in checkpoint.SAC_IGNORED_OPS
