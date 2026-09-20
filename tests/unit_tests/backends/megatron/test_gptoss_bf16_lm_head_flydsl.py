###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from torch._subclasses.fake_tensor import FakeTensorMode

import primus.backends.megatron.core.tensor_parallel.layers as tp_layers


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
