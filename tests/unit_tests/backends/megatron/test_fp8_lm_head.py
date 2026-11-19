# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()
if torch.version.hip is None:
    pytest.skip("the Turbo FP8 head requires ROCm", allow_module_level=True)

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    check_fp8_support,
)

from primus.backends.megatron.patches.turbo import (
    gpt_output_layer_patches as head_patch,
)


def test_requested_head_rejects_unsupported_parallelism():
    from primus.backends.megatron.patches.args.rocm_arg_validation import (
        validate_args_on_rocm,
    )

    args = SimpleNamespace(
        turbo_fp8_lm_head=True,
        enable_primus_turbo=True,
        bf16=True,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        tensor_model_parallel_size=2,
        gradient_accumulation_fusion=False,
        defer_embedding_wgrad_compute=False,
    )
    with pytest.raises(ValueError, match="turbo_fp8_lm_head requires .*TP1"):
        validate_args_on_rocm(args)


def test_unknown_native_argument_is_forwarded():
    original = Mock()
    head = SimpleNamespace(_forward_impl=original, training=True)
    head_patch._install_on_head(head, None)
    x, weight = object(), object()
    kwargs = dict.fromkeys(head_patch._NATIVE_HEAD_ARGUMENTS)
    kwargs["future_option"] = True
    assert head._forward_impl(x, weight, **kwargs) is original.return_value
    original.assert_called_once_with(x, weight, **kwargs)


def _assert_relative_error(actual, reference, threshold):
    actual, reference = actual.float(), reference.float()
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    assert (actual - reference).norm() < threshold * reference.norm()


@pytest.mark.parametrize("tied", [True, False])
def test_native_head_gradients_and_eval(init_parallel_state, monkeypatch, tied):
    supported, reason = check_fp8_support()
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(7331)
    config = ModelParallelConfig(params_dtype=torch.bfloat16, perform_initialization=False)
    head = ColumnParallelLinear(
        256,
        1024,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_weight_param_allocation=tied,
    )
    reference = torch.nn.Parameter(torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16) * 0.02)
    weight = torch.nn.Parameter(reference.detach().clone()) if tied else head.weight
    with torch.no_grad():
        weight.copy_(reference)
    passed_weights = []
    original = head_patch._fp8_projection

    def record_projection(inputs, supplied_weight, recipe):
        passed_weights.append(supplied_weight)
        return original(inputs, supplied_weight, recipe)

    monkeypatch.setattr(head_patch, "_fp8_projection", record_projection)
    head_patch._install_on_head(head, Float8QuantConfig(format=Format.HYBRID))

    for _ in range(2):
        hidden = torch.randn(128, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref_hidden = hidden.detach().clone().requires_grad_()
        ids = torch.randint(1024, (128, 2), device="cuda")
        output, bias = head(
            hidden + F.embedding(ids, weight) if tied else hidden,
            weight=weight if tied else None,
            runtime_gather_output=True,
        )
        expected = F.linear(ref_hidden + F.embedding(ids, reference) if tied else ref_hidden, reference)
        assert bias is None and output.shape == expected.shape and output.dtype == torch.bfloat16
        _assert_relative_error(output, expected, 0.08)
        labels = torch.randint(1024, (256,), device="cuda")
        loss = F.cross_entropy(output.flatten(0, 1).float(), labels)
        ref_loss = F.cross_entropy(expected.flatten(0, 1).float(), labels)
        torch.testing.assert_close(loss, ref_loss, atol=0.01, rtol=0.002)
        loss.backward()
        ref_loss.backward()
        _assert_relative_error(hidden.grad, ref_hidden.grad, 0.13)
        _assert_relative_error(weight.grad, reference.grad, 0.13)

    assert len(passed_weights) == 2 and all(supplied is weight for supplied in passed_weights)
    head.eval()
    with torch.no_grad():
        output, _ = head(hidden, weight=weight if tied else None)
        torch.testing.assert_close(output, F.linear(hidden, weight), atol=0, rtol=0)
    assert len(passed_weights) == 2

    head.train()
    weight.requires_grad_(False)
    frozen_input = hidden.detach().requires_grad_()
    ref_input = hidden.detach().requires_grad_()
    logits, _ = head(frozen_input, weight=weight if tied else None)
    reference_logits = F.linear(ref_input, weight)
    dy = torch.randn_like(logits)
    torch.testing.assert_close(logits, reference_logits, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(logits, frozen_input, dy)[0],
        torch.autograd.grad(reference_logits, ref_input, dy)[0],
        atol=0,
        rtol=0,
    )
    assert len(passed_weights) == 2
