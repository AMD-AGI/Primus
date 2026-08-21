###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for the TorchTitan v0.2.2 DeepSeek memory patch.

DeepSeek keeps low-fragment whole-block compilation for dense layers, but MoE
TransformerBlocks and ``MoE.forward`` must remain eager. Wrapping a MoE block
silently produced a NaN forward loss even with a graph break around its MoE.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

import primus.backends.torchtitan.patches.dsv3_v022_perf_patches as dsv3_patch
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

COMBINE_PATCH_ID = "torchtitan.dsv3.moe_bf16_combine"
COMPILE_PATCH_ID = "torchtitan.dsv3.whole_block_compile"


def _ctx(model_name, compile_enable=True):
    params = SimpleNamespace(compile=SimpleNamespace(enable=compile_enable))
    module_config = SimpleNamespace(params=params)
    return PatchContext(
        backend="torchtitan",
        phase="setup",
        model_name=model_name,
        extra={"module_config": module_config},
    )


class TestDsv3V022PatchRegistration:
    def test_safe_whole_block_compile_is_registered(self):
        patch = PatchRegistry.get(COMPILE_PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"

    def test_bf16_combine_remains_registered(self):
        patch = PatchRegistry.get(COMBINE_PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"


class TestDsv3V022PatchCondition:
    def test_compile_patch_requires_deepseek_and_compile(self):
        patch = PatchRegistry.get(COMPILE_PATCH_ID)
        assert patch is not None
        assert patch.condition(_ctx("deepseek_v3", True)) is True
        assert patch.condition(_ctx("deepseek_v3", False)) is False
        assert patch.condition(_ctx("llama3", True)) is False

    def test_bf16_combine_applies_to_deepseek(self):
        patch = PatchRegistry.get(COMBINE_PATCH_ID)
        assert patch is not None
        assert patch.condition(_ctx("deepseek_v3")) is True
        assert patch.condition(_ctx(SimpleNamespace(name="DeepSeek-V3"))) is True

    def test_bf16_combine_does_not_apply_to_other_models(self):
        patch = PatchRegistry.get(COMBINE_PATCH_ID)
        assert patch is not None
        assert patch.condition(_ctx("llama3")) is False
        assert patch.condition(_ctx(None)) is False


class _Block(nn.Module):
    def __init__(self, moe_enabled):
        super().__init__()
        self.moe_enabled = moe_enabled


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "dense": _Block(moe_enabled=False),
                "moe": _Block(moe_enabled=True),
            }
        )


def test_safe_compile_only_compiles_dense_blocks(monkeypatch):
    # Load TorchTitan before replacing torch.compile; its attention module
    # compiles a helper at import time with a different signature.
    import torchtitan.tools.logging  # noqa: F401

    model = _Model()
    calls = []

    def fake_compile(module, *, backend, fullgraph):
        calls.append((module.moe_enabled, backend, fullgraph))
        return module

    monkeypatch.setattr(dsv3_patch.torch, "compile", fake_compile)
    dsv3_patch._apply_dense_only_compile(
        model,
        SimpleNamespace(backend="inductor"),
        ep_enabled=True,
    )

    assert calls == [(False, "inductor", True)]


def test_moe_forward_is_installed_behind_compiler_disable(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    def eager_boundary(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        dsv3_patch.torch.compiler,
        "disable",
        lambda fn: eager_boundary,
    )
    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)

    with patch.object(moe_module.MoE, "forward", moe_module.MoE.forward):
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is eager_boundary
