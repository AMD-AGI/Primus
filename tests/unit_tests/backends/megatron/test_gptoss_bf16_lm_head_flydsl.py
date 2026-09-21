###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from torch._subclasses.fake_tensor import FakeTensorMode

import primus.backends.megatron.core.tensor_parallel.layers as tp_layers
from primus.backends.megatron.patches.turbo import gptoss_bf16_lm_head_patches
from primus.core.patches import PatchContext


def _lm_head_tensors():
    input_ = torch.empty((8192, 4, 2880), device="cuda", dtype=torch.bfloat16)
    grad_output = torch.empty((8192, 4, 128256), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((128256, 2880), device="cuda", dtype=torch.bfloat16)
    return input_, grad_output, weight


def test_gptoss_bf16_lm_head_shape_lock():
    with FakeTensorMode():
        input_, grad_output, weight = _lm_head_tensors()
        assert tp_layers._is_gptoss_bf16_lm_head_forward(input_, weight)
        assert tp_layers._is_gptoss_bf16_lm_head_wgrad(input_, grad_output, weight)

        wrong_weight = torch.empty((128255, 2880), device="cuda", dtype=torch.bfloat16)
        assert not tp_layers._is_gptoss_bf16_lm_head_forward(input_, wrong_weight)

        fp16_input = torch.empty((8192, 4, 2880), device="cuda", dtype=torch.float16)
        assert not tp_layers._is_gptoss_bf16_lm_head_forward(fp16_input, weight)


def test_gptoss_bf16_lm_head_routes_all_three_roles(monkeypatch):
    calls = []

    def fake_gemm(a, trans_a, b, trans_b, out_dtype, trans_c=False):
        calls.append(("gemm", a.shape, trans_a, b.shape, trans_b, out_dtype, trans_c))
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[0] if trans_b else b.shape[1]
        if trans_c:
            m, n = n, m
        return torch.empty((m, n), device=a.device, dtype=out_dtype)

    def fake_gemm_accum(a, trans_a, b, trans_b, out_dtype, trans_c, out):
        calls.append(
            ("accum", a.shape, trans_a, b.shape, trans_b, out_dtype, trans_c, out.shape)
        )

    monkeypatch.setattr(tp_layers, "_turbo_gemm", fake_gemm)
    monkeypatch.setattr(tp_layers, "_turbo_gemm_accum", fake_gemm_accum)

    with FakeTensorMode():
        input_, grad_output, weight = _lm_head_tensors()

        output = tp_layers._gptoss_bf16_lm_head_forward(input_, weight)
        grad_input = tp_layers._gptoss_bf16_lm_head_dgrad(
            grad_output, weight, input_.shape
        )
        handled = tp_layers._gptoss_bf16_lm_head_wgrad_accum(
            input_, grad_output, weight
        )

    assert output.shape == (8192, 4, 128256)
    assert grad_input.shape == input_.shape
    assert handled
    assert calls == [
        (
            "gemm",
            (32768, 2880),
            False,
            (128256, 2880),
            True,
            torch.bfloat16,
            False,
        ),
        (
            "gemm",
            (32768, 128256),
            False,
            (128256, 2880),
            False,
            torch.bfloat16,
            False,
        ),
        (
            "accum",
            (32768, 2880),
            True,
            (32768, 128256),
            False,
            torch.bfloat16,
            True,
            (128256, 2880),
        ),
    ]


def _patch_context(**overrides):
    params = {
        "use_turbo_gemm": True,
        "bf16": True,
        "hidden_size": 2880,
        "seq_length": 8192,
        "micro_batch_size": 4,
        "tensor_model_parallel_size": 1,
    }
    params.update(overrides)
    return PatchContext(
        backend="megatron",
        phase="before_train",
        model_name="gpt_oss_20B.yaml",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
    )


def test_gptoss_bf16_lm_head_patch_condition_is_workload_specific():
    assert gptoss_bf16_lm_head_patches._is_gptoss_bf16_lm_head_can_patch(
        _patch_context()
    )
    assert not gptoss_bf16_lm_head_patches._is_gptoss_bf16_lm_head_can_patch(
        _patch_context(use_turbo_gemm=False)
    )

    wrong_model = _patch_context()
    wrong_model.model_name = "llama3_8B.yaml"
    assert not gptoss_bf16_lm_head_patches._is_gptoss_bf16_lm_head_can_patch(
        wrong_model
    )


def test_gptoss_bf16_lm_head_patch_wraps_only_exact_shape(monkeypatch):
    import megatron.core.tensor_parallel.layers as megatron_layers

    calls = []

    def original_linear(*args):
        calls.append(("megatron", args[0].shape, args[1].shape))
        return "megatron"

    original_linear.warned = False

    class FakePrimusLinear:
        @staticmethod
        def apply(*args):
            calls.append(("primus", args[0].shape, args[1].shape))
            return "primus"

    monkeypatch.setattr(
        megatron_layers,
        "linear_with_grad_accumulation_and_async_allreduce",
        original_linear,
    )
    original_class = megatron_layers.LinearWithGradAccumulationAndAsyncCommunication
    monkeypatch.setattr(
        tp_layers, "LinearWithGradAccumulationAndAsyncCommunication", FakePrimusLinear
    )
    monkeypatch.setattr(gptoss_bf16_lm_head_patches, "log_rank_0", lambda *_: None)

    gptoss_bf16_lm_head_patches.patch_gptoss_bf16_lm_head(_patch_context())

    linear = megatron_layers.linear_with_grad_accumulation_and_async_allreduce
    fallback = linear(
        torch.empty((2, 3)),
        torch.empty((4, 3)),
        None,
        False,
        False,
        False,
    )
    with FakeTensorMode():
        input_, _, weight = _lm_head_tensors()
        routed = linear(input_, weight, None, True, False, False)

    assert fallback == "megatron"
    assert routed == "primus"
    assert calls == [
        ("megatron", (2, 3), (4, 3)),
        ("primus", (8192, 4, 2880), (128256, 2880)),
    ]
    assert (
        megatron_layers.LinearWithGradAccumulationAndAsyncCommunication
        is original_class
    )
