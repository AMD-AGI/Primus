###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The kernel-path config fields: their defaults, the recipes that set them, the modules that read them,
and the build-log line."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)
from test_zaya1_model import _tiny_config  # noqa: E402

import primus
from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm
from primus.backends.megatron.core.models.zaya1.zaya1_transformer_config import (
    Zaya1TransformerConfig,
)

_PATHS = (
    "zaya_attn_flash",
    "zaya_fuse_norm",
    "zaya_fuse_cca",
    "zaya_conv_gemm",
    "zaya_fuse_moe",
    "zaya_sort_dispatch",
    "zaya_router_defer_ar",
)
_CHUNKED_CE = ("fused_ce_mode", "fused_ce_chunk_tokens", "fused_ce_compile")


def test_every_path_is_off_by_default():
    base = Path(primus.__file__).parent / "configs" / "models" / "megatron" / "zaya1_base.yaml"
    values = yaml.safe_load(base.read_text())
    for name in _PATHS + _CHUNKED_CE:
        assert values[name] == getattr(Zaya1TransformerConfig, name), name
    for name in _PATHS + ("fused_ce_compile",):
        assert getattr(Zaya1TransformerConfig, name) is False, name
    # Unset rather than 0, which would also run the FLA runtime-knob patch.
    assert values["fused_ce_mode"] is None
    # The unit tests' stand-in configs have none of the fields.
    assert not any(zm._path_on(SimpleNamespace(), name) for name in _PATHS)


def test_on_off_fields_take_only_true_or_false():
    # A CLI override such as --zaya_fuse_norm off arrives as the string "off".
    for name in _PATHS + ("fused_ce_compile",):
        assert getattr(_tiny_config(**{name: 1}), name) is True
        assert getattr(_tiny_config(**{name: 0}), name) is False
        for value in ("off", "false", "", None, 2, 1.0):
            with pytest.raises(ValueError, match=f"{name} must be true or false"):
                _tiny_config(**{name: value})


_TURBO = (
    "enable_primus_turbo",
    "use_turbo_grouped_gemm",
    "use_turbo_fused_act_with_probs",
    "use_turbo_attention",
)


def test_only_the_mi355x_recipes_turn_paths_on():
    compute = {"zaya_attn_flash", "zaya_fuse_norm", "zaya_fuse_cca", "zaya_conv_gemm", "zaya_fuse_moe"}
    turbo = {"enable_primus_turbo", "use_turbo_grouped_gemm"}
    expected = {
        "MI355X/zaya1_8B-BF16-pretrain.yaml": compute
        | turbo
        | {"zaya_sort_dispatch", "zaya_router_defer_ar"},
        "MI355X/zaya1_8B-FP8-pretrain.yaml": compute | turbo | {"zaya_sort_dispatch"},
    }
    configs = Path(primus.__file__).parents[1] / "examples" / "megatron" / "configs"
    recipes = sorted(configs.glob("*/zaya1_8B-*-pretrain.yaml"))
    assert len(recipes) == 6
    for path in recipes:
        overrides = yaml.safe_load(path.read_text())["modules"]["pre_trainer"]["overrides"]
        on = {name for name in _PATHS + ("fused_ce_compile",) + _TURBO if overrides.get(name) is True}
        if overrides.get("fused_ce_mode"):
            on.add("fused_ce_mode")
        assert on == expected.get(path.relative_to(configs).as_posix(), set()), path


def test_modules_read_the_fields_when_built():
    for flag in (True, False):
        stack = zm.ZayaStack(_cfg(num_layers=2, **{name: flag for name in _PATHS}))
        cca, moe = stack.layers[0].self_attn, stack.layers[1].zaya_block
        assert stack.fuse_norm is stack.attn_flash is stack.layers[0].fuse_norm is flag
        assert cca.attn_flash is cca.fuse_cca is cca._conv_gemm_ok is flag
        assert moe.fuse_moe is moe.router.fuse_moe is moe.router.defer_bias_update is flag
        assert moe.sort_dispatch is flag
        assert all(expert.fuse_moe is flag for expert in moe.local_experts)
    # The GEMM convolutions cover kernel 2 only; CCA falls back to nn.Conv1d.
    assert zm.CCA(_cfg(zaya_conv_gemm=True, cca_time1=3))._conv_gemm_ok is False


def test_switch_summary_names_the_paths_that_run(monkeypatch):
    turbo = SimpleNamespace(
        enable_primus_turbo=True, use_turbo_grouped_gemm=True, turbo_grouped_gemm_backend="triton"
    )
    monkeypatch.setattr(zm, "_megatron_args", lambda: turbo)
    line = zm.switch_summary(_cfg(zaya_sort_dispatch=True))
    assert line.endswith("; experts: Primus Turbo grouped GEMM (backend: triton)")
    assert "zaya_sort_dispatch=on (unused: the grouped GEMMs use Turbo's moe_permute)" in line
    turbo.use_turbo_fused_act_with_probs = True
    assert zm.switch_summary(_cfg()).endswith(
        "; experts: Primus Turbo grouped GEMM (backend: triton), gate in Turbo's swiglu_with_probs"
    )
    turbo.use_turbo_fused_act_with_probs = False
    line = zm.switch_summary(_cfg(fp8="hybrid", fp8_recipe="tensorwise", zaya_sort_dispatch=True))
    assert line.endswith("; experts: Primus Turbo FP8 grouped GEMM (tensorwise, backend: triton)")
    assert "zaya_sort_dispatch=on (unused: the grouped GEMMs use Turbo's moe_permute)" in line
    line = zm.switch_summary(_cfg(fp8="hybrid", fp8_recipe="delayed", zaya_sort_dispatch=True))
    assert line.endswith("; experts: split (Turbo's FP8 grouped GEMM runs only the tensorwise recipe here)")
    assert "zaya_sort_dispatch=on (sort)" in line
    monkeypatch.setattr(zm, "_megatron_args", lambda: SimpleNamespace(enable_primus_turbo=True))
    assert zm.switch_summary(_cfg()).endswith("; experts: split")
    assert "zaya_sort_dispatch=off (boolean masks)" in zm.switch_summary(_cfg())
    line = zm.switch_summary(_cfg(zaya_attn_flash=False, zaya_fuse_norm=False, zaya_router_defer_ar=True))
    assert "zaya_attn_flash=off (dense mask)" in line and "zaya_fuse_norm=off" in line
    turbo_attention = SimpleNamespace(enable_primus_turbo=True, use_turbo_attention=True)
    monkeypatch.setattr(zm, "_megatron_args", lambda: turbo_attention)
    assert "zaya_attn_flash=on (Primus Turbo flash_attn_func)" in zm.switch_summary(
        _cfg(zaya_attn_flash=True)
    )
    assert "zaya_attn_flash=off (dense mask; use_turbo_attention needs zaya_attn_flash)" in zm.switch_summary(
        _cfg(zaya_attn_flash=False)
    )
    monkeypatch.setattr(zm, "_megatron_args", lambda: SimpleNamespace(enable_primus_turbo=True))
    assert "zaya_router_defer_ar=on" in line and "zaya_conv_gemm=off (nn.Conv1d)" in line
    assert "zaya_conv_gemm=on (GEMM)" in zm.switch_summary(_cfg(zaya_conv_gemm=True))
    assert (
        "zaya_conv_gemm=on (nn.Conv1d: the GEMM path needs cca_time0 and cca_time1 of 2, got 2 and 3)"
        in zm.switch_summary(_cfg(zaya_conv_gemm=True, cca_time1=3))
    )
