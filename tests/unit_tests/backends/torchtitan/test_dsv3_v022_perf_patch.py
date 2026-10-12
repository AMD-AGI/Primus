###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for the TorchTitan v0.2.2 DeepSeek memory patch.

DeepSeek keeps whole-block compilation for dense layers. MoE parents remain
eager while numerically safe pure-compute children compile independently.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _grouped_mm_sentinel(*args, **kwargs):
    return args, kwargs


_grouped_mm_sentinel.__qualname__ = dsv3_patch._GROUPED_MM_SENTINEL


def test_selective_compile_preserves_outer_moe_and_experts(monkeypatch):
    # Load TorchTitan before replacing torch.compile; its attention module
    # compiles a helper at import time with a different signature.
    import torchtitan.models.moe.moe as moe_module
    import torchtitan.tools.logging  # noqa: F401

    model = _Model()
    dense = model.layers["dense"]
    moe_block = model.layers["moe"]
    experts = moe_block.moe.experts
    calls = []

    def fake_compile(module, *, backend, fullgraph):
        calls.append((module.name, backend, fullgraph))
        return _Compiled(module)

    monkeypatch.setattr(dsv3_patch.torch, "compile", fake_compile)
    with patch.object(
        moe_module,
        "_run_experts_grouped_mm",
        _grouped_mm_sentinel,
    ):
        dsv3_patch._apply_selective_compile(
            model,
            SimpleNamespace(backend="inductor"),
            ep_enabled=True,
        )

    assert isinstance(model.layers["dense"], _Compiled)
    assert model.layers["dense"].original is dense
    assert model.layers["moe"] is moe_block
    assert moe_block.moe.experts is experts
    assert [name for name, _, _ in calls] == [
        "dense_block",
        "attention",
        "attention_norm",
        "ffn_norm",
        "router",
        "reorderer",
        "shared_experts",
    ]
    assert all(backend == "inductor" and fullgraph for _, backend, fullgraph in calls)


def test_selective_compile_skips_absent_optional_moe_modules(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    model = _Model()
    model.layers["moe"].moe = _MoE(optional_modules=False)
    calls = []

    def fake_compile(module, *, backend, fullgraph):
        calls.append(module.name)
        return _Compiled(module)

    monkeypatch.setattr(dsv3_patch.torch, "compile", fake_compile)
    with patch.object(
        moe_module,
        "_run_experts_grouped_mm",
        _grouped_mm_sentinel,
    ):
        dsv3_patch._apply_selective_compile(
            model,
            SimpleNamespace(backend="inductor"),
            ep_enabled=False,
        )

    assert "reorderer" not in calls
    assert "shared_experts" not in calls
    assert "experts" not in calls


def test_selective_compile_updates_checkpoint_wrapped_moe_children(monkeypatch):
    import torchtitan.models.moe.moe as moe_module
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    inner = _Block(moe_enabled=True)
    wrapped = checkpoint_wrapper(inner)
    model = nn.Module()
    model.layers = nn.ModuleDict({"moe": wrapped})

    monkeypatch.setattr(
        dsv3_patch.torch,
        "compile",
        lambda module, **kwargs: _Compiled(module),
    )
    with patch.object(
        moe_module,
        "_run_experts_grouped_mm",
        _grouped_mm_sentinel,
    ):
        dsv3_patch._apply_selective_compile(
            model,
            SimpleNamespace(backend="inductor"),
            ep_enabled=True,
        )

    assert model.layers["moe"] is wrapped
    assert isinstance(inner.attention, _Compiled)
    assert isinstance(inner.moe.router, _Compiled)
    assert not isinstance(inner.moe.experts, _Compiled)


def test_compile_patch_rebinds_source_and_deepseek_alias(monkeypatch):
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)
    with (
        patch.object(llama4_parallelize, "apply_compile", object()),
        patch.object(deepseek_parallelize, "apply_compile", object()),
    ):
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        assert llama4_parallelize.apply_compile is dsv3_patch._apply_selective_compile
        assert deepseek_parallelize.apply_compile is dsv3_patch._apply_selective_compile


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
        assert llama4_parallelize.apply_compile is dsv3_patch._apply_selective_compile
        assert deepseek_parallelize.apply_compile is dsv3_patch._apply_selective_compile


def test_grouped_mm_sentinel_keeps_turbo_function_eager(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    compile_mock = MagicMock()
    monkeypatch.setattr(dsv3_patch.torch, "compile", compile_mock)
    with patch.object(
        moe_module,
        "_run_experts_grouped_mm",
        _grouped_mm_sentinel,
    ):
        dsv3_patch._compile_grouped_mm("inductor", ep_enabled=True)
        assert moe_module._run_experts_grouped_mm is _grouped_mm_sentinel
    compile_mock.assert_not_called()


def test_standard_grouped_mm_is_compiled(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    original = lambda *args: args
    compiled = lambda *args: args
    compile_mock = MagicMock(return_value=compiled)
    monkeypatch.setattr(dsv3_patch.torch, "compile", compile_mock)
    with patch.object(moe_module, "_run_experts_grouped_mm", original):
        dsv3_patch._compile_grouped_mm("inductor", ep_enabled=False)
        assert moe_module._run_experts_grouped_mm is compiled
    compile_mock.assert_called_once_with(
        original,
        backend="inductor",
        fullgraph=True,
    )


def test_ep_grouped_mm_marks_dynamic_token_dimension(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    original = lambda *args: args
    compiled = MagicMock(return_value="result")
    monkeypatch.setattr(dsv3_patch.torch, "compile", lambda *args, **kwargs: compiled)
    mark_dynamic = MagicMock()
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", mark_dynamic)

    with patch.object(moe_module, "_run_experts_grouped_mm", original):
        dsv3_patch._compile_grouped_mm("inductor", ep_enabled=True)
        dynamic_fn = moe_module._run_experts_grouped_mm
        x = object()
        assert dynamic_fn(1, 2, 3, x, 4) == "result"

    mark_dynamic.assert_called_once_with(x, 0)
    compiled.assert_called_once_with(1, 2, 3, x, 4)


def test_moe_forward_is_installed_directly(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)

    with patch.object(moe_module.MoE, "forward", moe_module.MoE.forward):
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is dsv3_patch._moe_forward_bf16_combine
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is dsv3_patch._moe_forward_bf16_combine
