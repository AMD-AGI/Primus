###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU checks for the Llama2-70B FlyDSL recipe helpers.

These tests do not build a GEMM, attention kernel, or RoPE kernel, and they do
not import ``llama2_custom`` (that import installs process-wide log suppression).
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_RECIPE = Path(__file__).resolve().parents[4] / "primus/backends/megatron_bridge/recipes/mlperf_llama2_70b"
if str(_RECIPE) not in sys.path:
    sys.path.insert(0, str(_RECIPE))


def _load_recipe(name):
    spec = importlib.util.spec_from_file_location(name, _RECIPE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load by file path. Importing the recipe package also imports the Megatron
# backend and its patches, which these checks do not need.
pre_quantize_mxfp4 = _load_recipe("pre_quantize_mxfp4")
turbo_fp4_frozen = _load_recipe("turbo_fp4_frozen")
turbo_fp8_frozen = _load_recipe("turbo_fp8_frozen")
turbo_lora_fp4 = _load_recipe("turbo_lora_fp4")
turbo_rope = _load_recipe("turbo_rope")
turbo_attention_thd = _load_recipe("turbo_attention_thd")

_KERNEL = "primus_turbo.flydsl.gemm.gemm_mxfp4_kernel"
_QUANT = "primus_turbo.pytorch.core.quantized_tensor"


def _ensure_package(monkeypatch, name):
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)


def _stub_flydsl_kernel(monkeypatch, glu=None, dglu=None):
    for name in ("primus_turbo", "primus_turbo.flydsl", "primus_turbo.flydsl.gemm"):
        _ensure_package(monkeypatch, name)
    kernel = types.ModuleType(_KERNEL)
    kernel.dense_glu_epi_quant_supported = glu or (lambda *_args, **_kwargs: False)
    kernel.dense_dglu_epi_quant_supported = dglu or (lambda *_args, **_kwargs: False)
    monkeypatch.setitem(sys.modules, _KERNEL, kernel)


def _stub_quantized_tensor(monkeypatch):
    for name in ("primus_turbo", "primus_turbo.pytorch", "primus_turbo.pytorch.core"):
        _ensure_package(monkeypatch, name)

    class QuantizedTensor:
        pass

    class QuantizedTensorPair:
        pass

    module = types.ModuleType(_QUANT)
    module.QuantizedTensor = QuantizedTensor
    module.QuantizedTensorPair = QuantizedTensorPair
    monkeypatch.setitem(sys.modules, _QUANT, module)
    return QuantizedTensor, QuantizedTensorPair


def test_truthy_env(monkeypatch):
    monkeypatch.delenv("PRE_QUANTIZED_MODEL", raising=False)
    assert pre_quantize_mxfp4._truthy_env("PRE_QUANTIZED_MODEL") is False
    assert pre_quantize_mxfp4._truthy_env("PRE_QUANTIZED_MODEL", default=True) is True
    for raw in ("1", "true", "YES", "on"):
        monkeypatch.setenv("PRE_QUANTIZED_MODEL", raw)
        assert pre_quantize_mxfp4._truthy_env("PRE_QUANTIZED_MODEL") is True
    for raw in ("0", "false", "no", ""):
        monkeypatch.setenv("PRE_QUANTIZED_MODEL", raw)
        assert pre_quantize_mxfp4._truthy_env("PRE_QUANTIZED_MODEL") is False


def test_is_pre_quantized_enabled_reads_only_its_env(monkeypatch):
    monkeypatch.delenv("PRE_QUANTIZED_MODEL", raising=False)
    assert pre_quantize_mxfp4.is_pre_quantized_enabled() is False
    monkeypatch.setenv("PRE_QUANTIZED_MODEL", "true")
    assert pre_quantize_mxfp4.is_pre_quantized_enabled() is True


def test_fused_mlp_eligible_rejects_before_shape_check(monkeypatch):
    _stub_flydsl_kernel(monkeypatch)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    assert turbo_fp4_frozen.fused_mlp_eligible(x, x, x, None) is False

    class Preshuffle:
        use_preshuffle = True

    assert turbo_fp4_frozen.fused_mlp_eligible(x, x, x, Preshuffle()) is False
    assert turbo_fp4_frozen.fused_mlp_eligible(x.unsqueeze(0), x, x, object()) is False
    assert turbo_fp4_frozen.fused_mlp_eligible(x.float(), x, x, object()) is False


def test_fused_mlp_eligible_rejects_weight_that_is_not_a_frozen_pair(monkeypatch):
    _stub_flydsl_kernel(monkeypatch)
    _stub_quantized_tensor(monkeypatch)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    assert turbo_fp4_frozen.fused_mlp_eligible(x, torch.zeros(8, 4), torch.zeros(4, 4), object()) is False


def test_fused_mlp_eligible_passes_shapes_to_the_kernel_check(monkeypatch):
    seen = {}

    def glu(k, inner, m, dtype):
        seen["glu"] = (k, inner, m, dtype)
        return True

    def dglu(n, inner, m, dtype):
        seen["dglu"] = (n, inner, m, dtype)
        return True

    _stub_flydsl_kernel(monkeypatch, glu=glu, dglu=dglu)
    quant_cls, pair_cls = _stub_quantized_tensor(monkeypatch)

    class Recipe:
        use_2d_block = False
        shuffle_scale = False
        shuffle_out = False

    def half():
        tensor = quant_cls()
        tensor.requires_grad = False
        tensor.scaling_recipe = Recipe()
        return tensor

    def pair(shape):
        weight = pair_cls()
        weight.data = half()
        weight.data_t = half()
        weight.shape = shape
        return weight

    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    assert turbo_fp4_frozen.fused_mlp_eligible(x, pair((8, 4)), pair((6, 4)), object()) is True
    assert seen["glu"] == (4, 4, 2, torch.bfloat16)
    assert seen["dglu"] == (6, 4, 2, torch.bfloat16)


def test_install_returns_when_already_installed():
    modules = (turbo_fp4_frozen, turbo_fp8_frozen, turbo_lora_fp4, turbo_rope, turbo_attention_thd)
    saved = [(module, module._STATE["installed"]) for module in modules]
    try:
        for module, _previous in saved:
            module._STATE["installed"] = True
            assert module.install() is True
    finally:
        for module, previous in saved:
            module._STATE["installed"] = previous
