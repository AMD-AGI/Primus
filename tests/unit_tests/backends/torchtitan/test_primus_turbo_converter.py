###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TorchTitan's Primus-Turbo model converter gating."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import primus.backends.torchtitan.patches.turbo.attention_patches as attention_patch
import primus.backends.torchtitan.patches.turbo.attention_safety as attention_safety
import primus.backends.torchtitan.primus_turbo_extensions.primus_turbo_converter as converter_module
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry


def _job_config(
    *,
    model_name="llama3",
    enable_turbo=True,
    use_turbo_attention=True,
    attention_float8=False,
    use_turbo_float8_linear=False,
    use_moe_fp8=False,
    use_classic_attention=False,
):
    turbo = SimpleNamespace(
        enable_primus_turbo=enable_turbo,
        use_turbo_attention=use_turbo_attention,
        enable_attention_float8=attention_float8,
        use_turbo_float8_linear=use_turbo_float8_linear,
        use_moe_fp8=use_moe_fp8,
        use_classic_attention=use_classic_attention,
    )
    return SimpleNamespace(
        model=SimpleNamespace(name=model_name),
        primus_turbo=turbo,
    )


def _fake_low_precision_module():
    module = ModuleType("primus_turbo.pytorch.core.low_precision")
    module.ScalingGranularity = SimpleNamespace(BLOCKWISE="blockwise")
    module.Float8QuantConfig = lambda **kwargs: ("fp8", kwargs)
    return module


@pytest.mark.parametrize(
    "enable_turbo,use_turbo_attention",
    [(False, True), (True, False), (False, False)],
)
def test_converter_skips_attention_replacement_when_disabled(
    monkeypatch,
    enable_turbo,
    use_turbo_attention,
):
    replace = MagicMock()
    monkeypatch.setattr(converter_module, "replace_turbo_attention_modules", replace)

    converter = converter_module.PrimusTubroConverter(
        _job_config(
            enable_turbo=enable_turbo,
            use_turbo_attention=use_turbo_attention,
        ),
        parallel_dims=None,
    )
    model = object()

    assert converter.enabled is False
    assert converter.fp8_config is None
    assert converter.convert(model) is None
    replace.assert_not_called()


def test_converter_replaces_attention_when_both_flags_enabled(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.core.low_precision",
        _fake_low_precision_module(),
    )
    replace = MagicMock()
    monkeypatch.setattr(converter_module, "replace_turbo_attention_modules", replace)

    converter = converter_module.PrimusTubroConverter(
        _job_config(attention_float8=True),
        parallel_dims=None,
    )
    model = object()

    assert converter.enabled is True
    assert converter.fp8_config == (
        "fp8",
        {"granularity": "blockwise", "block_size": 64},
    )
    assert converter.convert(model) is model
    replace.assert_called_once_with(model, converter.fp8_config)


def _deepseek_fp8_config():
    return _job_config(
        model_name="deepseek_v3",
        use_turbo_float8_linear=True,
        use_moe_fp8=True,
        use_classic_attention=False,
    )


def test_converter_keeps_turbo_attention_enabled_for_gfx942_deepseek_fp8(
    monkeypatch,
):
    monkeypatch.setattr(attention_safety, "_is_gfx942", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "primus_turbo.pytorch.core.low_precision",
        _fake_low_precision_module(),
    )
    replace = MagicMock()
    monkeypatch.setattr(converter_module, "replace_turbo_attention_modules", replace)

    converter = converter_module.PrimusTubroConverter(
        _deepseek_fp8_config(),
        parallel_dims=None,
    )
    model = object()

    assert converter.enabled is True
    assert converter.convert(model) is model
    replace.assert_called_once_with(model, converter.fp8_config)


def test_setup_patch_uses_the_same_runtime_safety_gate(monkeypatch):
    monkeypatch.setattr(attention_safety, "_is_gfx942", lambda: True)
    monkeypatch.setattr(attention_patch, "log_rank_0", lambda *args, **kwargs: None)
    config = _deepseek_fp8_config()
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        model_name=config.model,
        extra={"module_config": SimpleNamespace(params=config)},
    )
    patch = PatchRegistry.get("torchtitan.primus_turbo.turbo_attention")
    assert patch is not None
    assert patch.condition(ctx) is True

    monkeypatch.setattr(attention_safety, "_is_gfx942", lambda: False)
    assert patch.condition(ctx) is True
