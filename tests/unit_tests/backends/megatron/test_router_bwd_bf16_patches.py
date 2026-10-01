###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU unit tests for the BF16 router-backward Primus patch."""

from types import SimpleNamespace

import torch

from primus.backends.megatron.patches.moe_patches import (
    router_bwd_bf16_patches as router_bwd,
)


def _ctx(inp, weight, bias=None):
    return SimpleNamespace(
        saved_tensors=(inp, weight, bias),
        router_dtype=torch.float32,
        input_dtype=inp.dtype,
        weight_dtype=weight.dtype,
    )


class _FakeTEGemm:
    def __init__(self):
        self.calls = []

    def __call__(self, a, b, out_dtype, *, layout, grad):
        self.calls.append((a, b, out_dtype, layout, grad))
        if layout == "NN":
            result = torch.full((b.shape[0], a.shape[1]), 2.0, dtype=out_dtype)
        elif layout == "NT":
            result = torch.full((b.shape[1], a.shape[1]), 3.0, dtype=out_dtype)
        else:  # pragma: no cover - the helper only uses NN/NT
            raise AssertionError(layout)
        return [result]


class _FakeAccumGemm:
    def __init__(self):
        self.calls = []

    def __call__(self, a, b, *, out_dtype, layout, out, grad, accumulate):
        self.calls.append(
            {
                "a": a,
                "b": b,
                "out_dtype": out_dtype,
                "layout": layout,
                "out": out,
                "grad": grad,
                "accumulate": accumulate,
            }
        )
        value = torch.full_like(out, 5.0)
        if accumulate:
            out.add_(value)
        else:
            out.copy_(value)
        return [out]


def _dummy(shape, dtype, zero=False):
    return torch.zeros(shape, dtype=dtype) if zero else torch.empty(shape, dtype=dtype)


def test_direct_wgrad_overwrites_fp32_main_grad_and_returns_bf16_dummy():
    inp = torch.randn(8, 6, dtype=torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
    weight.main_grad = torch.full_like(weight, 17.0, dtype=torch.float32)
    weight.overwrite_main_grad = True
    weight.grad_added_to_main_grad = False
    grad_output = torch.randn(8, 4, dtype=torch.float32)
    te_gemm = _FakeTEGemm()
    accum_gemm = _FakeAccumGemm()

    grad_input, grad_weight, grad_bias, trailing = router_bwd._router_backward_bf16(
        _ctx(inp, weight),
        grad_output,
        te_general_gemm=te_gemm,
        te_general_gemm_accumulate=accum_gemm,
        get_dummy_wgrad=_dummy,
        fuse_main_grad=True,
    )

    assert grad_input.shape == inp.shape
    assert grad_input.dtype == torch.bfloat16
    assert grad_weight.shape == weight.shape
    assert grad_weight.dtype == torch.bfloat16
    assert grad_bias is None
    assert trailing is None
    assert torch.equal(weight.main_grad, torch.full_like(weight.main_grad, 5.0))
    assert weight.overwrite_main_grad is False
    assert weight.grad_added_to_main_grad is True
    assert accum_gemm.calls[0]["out"] is weight.main_grad
    assert accum_gemm.calls[0]["out_dtype"] == torch.float32
    assert accum_gemm.calls[0]["accumulate"] is False
    assert accum_gemm.calls[0]["a"].dtype == torch.bfloat16
    assert accum_gemm.calls[0]["b"].dtype == torch.bfloat16


def test_direct_wgrad_accumulates_after_first_microbatch():
    inp = torch.randn(8, 6, dtype=torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
    weight.main_grad = torch.ones_like(weight, dtype=torch.float32)
    weight.overwrite_main_grad = False
    weight.grad_added_to_main_grad = False
    accum_gemm = _FakeAccumGemm()

    router_bwd._router_backward_bf16(
        _ctx(inp, weight),
        torch.randn(8, 4, dtype=torch.float32),
        te_general_gemm=_FakeTEGemm(),
        te_general_gemm_accumulate=accum_gemm,
        get_dummy_wgrad=_dummy,
        fuse_main_grad=True,
    )

    assert torch.equal(weight.main_grad, torch.full_like(weight.main_grad, 6.0))
    assert accum_gemm.calls[0]["accumulate"] is True


def test_without_fusion_returns_bf16_param_grad_from_fp32_gemm_output():
    inp = torch.randn(8, 6, dtype=torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
    grad_output = torch.randn(8, 4, dtype=torch.float32)
    te_gemm = _FakeTEGemm()

    _, grad_weight, _, _ = router_bwd._router_backward_bf16(
        _ctx(inp, weight),
        grad_output,
        te_general_gemm=te_gemm,
        te_general_gemm_accumulate=lambda *args, **kwargs: None,
        get_dummy_wgrad=_dummy,
        fuse_main_grad=False,
    )

    assert grad_weight.dtype == torch.bfloat16
    assert torch.equal(grad_weight, torch.full_like(weight, 3.0))
    wgrad_call = te_gemm.calls[1]
    assert wgrad_call[0].dtype == torch.bfloat16
    assert wgrad_call[1].dtype == torch.bfloat16
    assert wgrad_call[2] == torch.float32
    assert wgrad_call[3] == "NT"


def test_precision_gate_requires_fp32_router_and_bf16_input_and_weight():
    ctx = SimpleNamespace(
        router_dtype=torch.float32,
        input_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
    )

    assert router_bwd._uses_bf16_router_backward(ctx, object())

    ctx.router_dtype = torch.bfloat16
    assert not router_bwd._uses_bf16_router_backward(ctx, object())
    ctx.router_dtype = torch.float32
    ctx.input_dtype = torch.float32
    assert not router_bwd._uses_bf16_router_backward(ctx, object())
    ctx.input_dtype = torch.bfloat16
    ctx.weight_dtype = torch.float32
    assert not router_bwd._uses_bf16_router_backward(ctx, object())
    ctx.weight_dtype = torch.bfloat16
    assert not router_bwd._uses_bf16_router_backward(ctx, None)


def test_non_fp32_or_noncontiguous_main_grad_falls_back_to_param_grad():
    inp = torch.randn(8, 6, dtype=torch.bfloat16)
    weight = torch.nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
    weight.main_grad = torch.zeros_like(weight, dtype=torch.bfloat16)
    te_gemm = _FakeTEGemm()

    _, grad_weight, _, _ = router_bwd._router_backward_bf16(
        _ctx(inp, weight),
        torch.randn(8, 4, dtype=torch.float32),
        te_general_gemm=te_gemm,
        te_general_gemm_accumulate=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not directly accumulate")
        ),
        get_dummy_wgrad=_dummy,
        fuse_main_grad=True,
    )

    assert grad_weight.dtype == torch.bfloat16
    assert len(te_gemm.calls) == 2
