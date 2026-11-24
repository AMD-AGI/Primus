###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

import primus.backends.megatron.patches.mamba_rocm_patches as patch_mod
from primus.core.patches.context import PatchContext


@pytest.mark.parametrize(
    "hip, params, expected",
    [
        ("6.4", {"model_type": "mamba"}, True),
        ("6.4", {"model_type": "hylo"}, True),
        ("6.4", {"model_type": "gpt"}, False),
        ("6.4", {"recipe": "zebra_llama_1B"}, True),
        (None, {"model_type": "hylo"}, False),
    ],
)
def test_patch_applies_only_to_mamba_family_runs_on_rocm(monkeypatch, hip, params, expected):
    monkeypatch.setattr(patch_mod.torch.version, "hip", hip, raising=False)
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
    )
    assert patch_mod._is_rocm_mamba_run(ctx) is expected
