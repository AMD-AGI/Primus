# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Guards for Primus-Turbo fused weight-gradient accumulation."""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.backends.megatron.conftest import requires_mxfp4
from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus_turbo.pytorch as primus_turbo_torch  # noqa: E402  isort:skip

from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    Format,
    PrimusTurboLowPrecisionGlobalStateManager,
    PrimusTurboQuantConfig,
    PrimusTurboQuantizedTensorPair,
    ScaleDtype,
    ScalingGranularity,
    _bridge_weight_grad,
    _fuse_wgrad_accum_pattern,
    _maybe_create_quantized_weight_buffers,
    float4_e2m1fn_x2,
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
    weight.main_grad = torch.zeros_like(weight, dtype=dtype if main_grad_dtype is None else main_grad_dtype)
    weight.grad_added_to_main_grad = False
    return weight


def _accumulate_mxfp4_wgrad(weight, inputs, grad_outputs, quant_config, *, fused):
    quantized_weight, quantized_weight_trans = _maybe_create_quantized_weight_buffers(
        weight,
        float4_e2m1fn_x2,
        quant_config,
        disable_parameter_transpose_cache=False,
    )

    for x, grad_output in zip(inputs, grad_outputs):
        bridged_x, bridged_weight = _bridge_weight_grad(
            x.detach().clone().requires_grad_(True),
            weight,
            PrimusTurboQuantizedTensorPair(
                data=quantized_weight,
                data_t=quantized_weight_trans,
            ),
            fuse_wgrad_accum=fused,
        )
        output = primus_turbo_torch.ops.gemm_fp4(
            bridged_x,
            bridged_weight,
            trans_b=True,
            config=quant_config.data(),
            fuse_bgrad_accum_pattern="megatron" if fused else None,
        )
        output.backward(grad_output)

    return weight.main_grad


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_matching_16bit_main_grad_enables_fusion(dtype):
    config = SimpleNamespace(gradient_accumulation_fusion=True)

    assert _fuse_wgrad_accum_pattern(config, _weight(dtype=dtype)) == "megatron"


@requires_mxfp4
def test_mxfp4_fused_wgrad_matches_multi_microbatch_reference(monkeypatch):
    """Exercise the real beta=1 GEMM path with an existing gradient in place."""
    quant_config = PrimusTurboQuantConfig(
        format=Format.E2M1_X2,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        scale_dtype=ScaleDtype.E8M0,
        block_size=32,
        use_gradient_sr=False,
    )
    monkeypatch.setattr(
        PrimusTurboLowPrecisionGlobalStateManager,
        "PRIMUS_TURBO_QUANT_CONFIG",
        quant_config,
    )

    torch.manual_seed(42)
    weight_data = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda") / 16
    inputs = [torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") / 16 for _ in range(2)]
    grad_outputs = [torch.randn(128, 256, dtype=torch.bfloat16, device="cuda") / 16 for _ in range(2)]
    initial_main_grad = torch.randn_like(weight_data) / 16

    fused_weight = torch.nn.Parameter(weight_data.clone())
    fused_weight.main_grad = initial_main_grad.clone()
    fused_weight.grad_added_to_main_grad = False
    fused_main_grad = _accumulate_mxfp4_wgrad(
        fused_weight,
        inputs,
        grad_outputs,
        quant_config,
        fused=True,
    )

    reference_weight = torch.nn.Parameter(weight_data.clone())
    reference_weight.main_grad = initial_main_grad.clone()
    reference_weight.grad_added_to_main_grad = False
    reference_main_grad = _accumulate_mxfp4_wgrad(
        reference_weight,
        inputs,
        grad_outputs,
        quant_config,
        fused=False,
    )

    assert fused_weight.grad_added_to_main_grad
    assert reference_weight.grad_added_to_main_grad
    torch.testing.assert_close(fused_main_grad, reference_main_grad, rtol=5e-3, atol=5e-3)


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
