###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU numerical and dispatch tests for the BF16 router backward patch."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus.core.patches import PatchContext

# Load just this patch so CPU tests do not import unrelated GPU patch modules.
_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/moe_patches/router_bwd_bf16_patches.py"
)
_spec = importlib.util.spec_from_file_location("router_bwd_bf16_under_test", _PATCH_PATH)
router_bwd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router_bwd)


def _patch_context(**kwargs):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kwargs))},
    )


def _ctx(inp, weight, bias=None, router_dtype=torch.float32):
    return SimpleNamespace(
        saved_tensors=(inp, weight, bias),
        router_dtype=router_dtype,
        input_dtype=inp.dtype,
        weight_dtype=weight.dtype,
    )


class _CPUGemm:
    """Emulate TE layouts with FP32 accumulation and requested output dtype."""

    def __init__(self):
        self.calls = []

    def __call__(self, a, b, out_dtype, *, layout, grad):
        self.calls.append((a, b, out_dtype, layout, grad))
        if layout == "NN":
            result = b.float() @ a.float()
        elif layout == "NT":
            result = b.float().t() @ a.float()
        else:
            raise AssertionError(layout)
        return (result.to(out_dtype),)


@pytest.mark.parametrize("shape", [(8, 6), (2, 4, 6)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_backward_preserves_mlperf_precision_contract(shape, with_bias):
    generator = torch.Generator().manual_seed(42)
    inp = torch.randn(shape, generator=generator).bfloat16()
    weight = torch.randn(4, 6, generator=generator).bfloat16()
    bias = torch.zeros(4, dtype=torch.bfloat16) if with_bias else None
    grad_output = torch.randn((*shape[:-1], 4), generator=generator)
    gemm = _CPUGemm()

    dx, dw, db, trailing = router_bwd._router_backward_bf16(
        _ctx(inp, weight, bias), grad_output, te_general_gemm=gemm
    )

    rounded_grad = grad_output.reshape(-1, 4).bfloat16().float()
    expected_dx = (rounded_grad @ weight.float()).bfloat16().reshape(shape)
    expected_dw = (rounded_grad.t() @ inp.reshape(-1, 6).float()).bfloat16()
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(dw, expected_dw, rtol=0, atol=0)
    if with_bias:
        expected_db = grad_output.reshape(-1, 4).sum(0).bfloat16()
        torch.testing.assert_close(db, expected_db, rtol=0, atol=0)
        assert not torch.equal(expected_db, rounded_grad.sum(0).bfloat16())
    else:
        assert db is None
    assert trailing is None
    assert len(gemm.calls) == 2
    assert gemm.calls[0][2:] == (torch.bfloat16, "NN", True)
    assert gemm.calls[1][2:] == (torch.float32, "NT", True)
    for a, b, *_ in gemm.calls:
        assert a.dtype == b.dtype == torch.bfloat16


@pytest.fixture
def installed_patch(monkeypatch):
    original = Mock(return_value=object())
    forward = Mock()
    router_function = type(
        "RouterGatingLinearFunction",
        (),
        {"backward": staticmethod(original), "forward": staticmethod(forward)},
    )
    moe_utils = SimpleNamespace(RouterGatingLinearFunction=router_function, te_general_gemm=_CPUGemm())
    moe = ModuleType("megatron.core.transformer.moe")
    moe.moe_utils = moe_utils
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe", moe)
    monkeypatch.setattr(router_bwd, "log_rank_0", lambda *args: None)
    router_bwd.patch_router_bwd_bf16(_patch_context(moe_router_bwd_bf16=True))
    return moe_utils, original, forward


@pytest.mark.parametrize(
    "router_dtype,input_dtype,weight_dtype,has_te",
    [
        (torch.float64, torch.bfloat16, torch.bfloat16, True),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, True),
        (torch.float32, torch.float32, torch.bfloat16, True),
        (torch.float32, torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, torch.bfloat16, False),
    ],
)
def test_other_precisions_and_no_te_delegate_to_original(
    installed_patch, router_dtype, input_dtype, weight_dtype, has_te
):
    moe_utils, original, _ = installed_patch
    if not has_te:
        moe_utils.te_general_gemm = None
    ctx = _ctx(torch.zeros(8, 6, dtype=input_dtype), torch.zeros(4, 6, dtype=weight_dtype))
    ctx.router_dtype = router_dtype
    grad_output = torch.randn(8, 4)
    result = moe_utils.RouterGatingLinearFunction.backward(ctx, grad_output)
    assert result is original.return_value
    original.assert_called_once_with(ctx, grad_output)


def test_install_is_idempotent_preserves_forward_and_routes_bf16(installed_patch):
    moe_utils, original, forward = installed_patch
    router_function = moe_utils.RouterGatingLinearFunction
    backward = router_function.backward
    router_bwd.patch_router_bwd_bf16(_patch_context(moe_router_bwd_bf16=True))
    assert router_function.backward is backward
    assert router_function.forward is forward
    ctx = _ctx(torch.ones(8, 6, dtype=torch.bfloat16), torch.ones(4, 6, dtype=torch.bfloat16))
    dx, dw, db, _ = router_function.backward(ctx, torch.ones(8, 4))
    torch.testing.assert_close(dx, torch.full_like(ctx.saved_tensors[0], 4))
    torch.testing.assert_close(dw, torch.full_like(ctx.saved_tensors[1], 8))
    assert db is None
    original.assert_not_called()


def test_patch_requires_explicit_opt_in():
    assert not router_bwd._enabled(_patch_context())
    assert not router_bwd._enabled(_patch_context(moe_router_bwd_bf16=False))
    assert router_bwd._enabled(_patch_context(moe_router_bwd_bf16=True))
