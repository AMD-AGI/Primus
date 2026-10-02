###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for the TorchTitan v0.2.2 DeepSeek memory patch.

DeepSeek uses whole-block compilation with an explicit graph break around
GroupedExperts so EP/FSDP hooks remain eager.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
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


class _Leaf(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class _MoE(nn.Module):
    def __init__(self, *, optional_modules=True):
        super().__init__()
        self.experts = _Leaf("experts")
        self.router = _Leaf("router")
        self.reorderer = _Leaf("reorderer") if optional_modules else None
        self.shared_experts = _Leaf("shared_experts") if optional_modules else None


class _Block(nn.Module):
    def __init__(self, moe_enabled):
        super().__init__()
        self.moe_enabled = moe_enabled
        self.name = "moe_block" if moe_enabled else "dense_block"
        self.attention = _Leaf("attention")
        self.attention_norm = _Leaf("attention_norm")
        self.ffn_norm = _Leaf("ffn_norm")
        if moe_enabled:
            self.moe = _MoE()
        else:
            self.feed_forward = _Leaf("feed_forward")


class _Compiled(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "dense": _Block(moe_enabled=False),
                "moe": _Block(moe_enabled=True),
            }
        )


def test_unified_compile_breaks_at_grouped_experts(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    model = _Model()
    compile_calls = []
    disable_calls = []

    def fake_compile(module, *, backend, fullgraph):
        compile_calls.append((module.name, backend, fullgraph))
        return _Compiled(module)

    def fake_disable(fn, *, recursive):
        disable_calls.append((fn, recursive))

        def eager_boundary(*args, **kwargs):
            return fn(*args, **kwargs)

        return eager_boundary

    monkeypatch.setattr(dsv3_patch.torch, "compile", fake_compile)
    monkeypatch.setattr(dsv3_patch.torch.compiler, "disable", fake_disable)
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", False)
    with patch.object(moe_module.GroupedExperts, "forward", moe_module.GroupedExperts.forward):
        dsv3_patch._apply_unified_compile(
            model,
            SimpleNamespace(backend="inductor"),
            ep_enabled=True,
        )

    assert compile_calls == [
        ("dense_block", "inductor", False),
        ("moe_block", "inductor", False),
    ]
    assert torch._dynamo.config.capture_scalar_outputs is True
    assert len(disable_calls) == 1
    assert disable_calls[0][1] is False


def test_compile_patch_rebinds_source_and_deepseek_alias(monkeypatch):
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)
    with (
        patch.object(llama4_parallelize, "apply_compile", object()),
        patch.object(deepseek_parallelize, "apply_compile", object()),
    ):
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        assert llama4_parallelize.apply_compile is dsv3_patch._apply_unified_compile
        assert deepseek_parallelize.apply_compile is dsv3_patch._apply_unified_compile


def test_compile_patch_repeated_install_is_idempotent(monkeypatch):
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)
    with (
        patch.object(llama4_parallelize, "apply_compile", object()),
        patch.object(deepseek_parallelize, "apply_compile", object()),
    ):
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        assert llama4_parallelize.apply_compile is dsv3_patch._apply_unified_compile
        assert deepseek_parallelize.apply_compile is dsv3_patch._apply_unified_compile


def test_moe_forward_is_installed_directly(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)

    with patch.object(moe_module.MoE, "forward", moe_module.MoE.forward):
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is dsv3_patch._moe_forward_bf16_combine
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is dsv3_patch._moe_forward_bf16_combine
