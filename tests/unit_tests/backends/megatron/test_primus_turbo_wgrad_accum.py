# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Guards for Primus-Turbo fused weight-gradient accumulation."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    PrimusTurboLowPrecisionGlobalStateManager,
    _fuse_wgrad_accum_pattern,
)


class _FakeMXFP4QuantConfig:
    def __init__(self, *, mxfp4=True, use_preshuffle=False, expose_preshuffle=True):
        self._mxfp4 = mxfp4
        values = {"use_preshuffle": use_preshuffle} if expose_preshuffle else {}
        self._data = SimpleNamespace(**values)

    def mxfp4_scaling(self):
        return self._mxfp4

    def data(self):
        return self._data


@pytest.fixture(autouse=True)
def _turbo_fp4_state(monkeypatch):
    manager = PrimusTurboLowPrecisionGlobalStateManager
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", True)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", False)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_QUANT_CONFIG", _FakeMXFP4QuantConfig())


def _weight(*, dtype=torch.bfloat16, main_grad_dtype=None):
    weight = torch.nn.Parameter(torch.empty(8, 8, dtype=dtype))
    weight.main_grad = torch.zeros_like(
        weight, dtype=dtype if main_grad_dtype is None else main_grad_dtype
    )
    weight.grad_added_to_main_grad = False
    return weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_matching_16bit_main_grad_enables_fusion(dtype):
    config = SimpleNamespace(gradient_accumulation_fusion=True)

    assert _fuse_wgrad_accum_pattern(config, _weight(dtype=dtype)) == "megatron"


@pytest.mark.parametrize(
    "quant_config",
    [
        None,
        _FakeMXFP4QuantConfig(mxfp4=False),
        _FakeMXFP4QuantConfig(use_preshuffle=True),
        _FakeMXFP4QuantConfig(expose_preshuffle=False),
    ],
)
def test_mxfp4_unsupported_quantization_fails_closed(monkeypatch, quant_config):
    monkeypatch.setattr(
        PrimusTurboLowPrecisionGlobalStateManager,
        "PRIMUS_TURBO_QUANT_CONFIG",
        quant_config,
    )
    config = SimpleNamespace(gradient_accumulation_fusion=True)

    assert _fuse_wgrad_accum_pattern(config, _weight()) is None


@pytest.mark.parametrize(
    "weight_dtype,main_grad_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
    ],
)
def test_mxfp4_unsupported_accumulator_dtype_fails_closed(weight_dtype, main_grad_dtype):
    config = SimpleNamespace(gradient_accumulation_fusion=True)
    weight = _weight(dtype=weight_dtype, main_grad_dtype=main_grad_dtype)

    assert _fuse_wgrad_accum_pattern(config, weight) is None


def test_mxfp4_non_tensor_main_grad_fails_closed():
    config = SimpleNamespace(gradient_accumulation_fusion=True)
    weight = _weight()
    weight.main_grad = None

    assert _fuse_wgrad_accum_pattern(config, weight) is None


def test_mxfp4_gradient_accumulation_fusion_flag_still_controls_gate():
    config = SimpleNamespace(gradient_accumulation_fusion=False)

    assert _fuse_wgrad_accum_pattern(config, _weight()) is None
