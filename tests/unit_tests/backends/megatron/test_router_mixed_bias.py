###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe import router as mcore_router
from torch import nn

from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter


@pytest.fixture
def make_router(monkeypatch):
    # The upstream constructor normally allocates these Parameters on CPU, then
    # converts them to config.params_dtype in reset_parameters.
    def init(self, config, *args, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.weight = nn.Parameter(torch.randn(128, 256, dtype=config.params_dtype))
        self.bias = (
            nn.Parameter(torch.randn(128, dtype=config.params_dtype)) if config.add_bias_linear else None
        )

    monkeypatch.setattr(PrimusTopKRouter.__bases__[0], "__init__", init)

    def make(bias=True, route_dtype="fp32", dtype=torch.bfloat16, cls=PrimusTopKRouter):
        return cls(SimpleNamespace(add_bias_linear=bias, moe_router_dtype=route_dtype, params_dtype=dtype))

    return make


@pytest.mark.parametrize(
    "bias,route_dtype,dtype,hip",
    [
        (False, "fp32", torch.bfloat16, "test"),
        (True, "fp64", torch.bfloat16, "test"),
        (True, None, torch.bfloat16, "test"),
        (True, "fp32", torch.float32, "test"),
        (True, "fp32", torch.bfloat16, None),
    ],
)
def test_other_router_instances_keep_megatron_gating(make_router, monkeypatch, bias, route_dtype, dtype, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    original = mcore_router.Router.gating
    router = make_router(bias=bias, route_dtype=route_dtype, dtype=dtype)
    assert PrimusTopKRouter.gating is original
    assert router.gating.__func__ is original
    assert "gating" not in vars(router)


def test_compatibility_preserves_custom_gating_and_falls_back_without_operator(make_router, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")

    class CustomRouter(PrimusTopKRouter):
        def gating(self, value):
            return value

    assert make_router(cls=CustomRouter).gating.__func__ is CustomRouter.gating
    monkeypatch.setattr(mcore_router, "router_gating_linear", None, raising=False)
    assert make_router().gating.__func__ is mcore_router.Router.gating


@pytest.mark.skipif(not (torch.version.hip and torch.cuda.is_available()), reason="requires ROCm")
def test_mixed_router_uses_public_autograd_and_retains_bfloat16_leaf(make_router):
    router = make_router()
    assert router.gating.__func__ is PrimusTopKRouter._gating_with_fp32_bias
    weight, bias = router.weight, router.bias
    assert weight.device.type == bias.device.type == "cpu"
    x = torch.randn(16, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = router.gating(x)
    dy = torch.randn_like(y)
    grad = torch.autograd.grad(y, (x, weight, bias), dy)
    xx = x.detach().requires_grad_()
    ww = weight.detach().requires_grad_()
    bb = bias.detach().requires_grad_()
    reference = mcore_router.router_gating_linear(xx, ww, bb.float(), torch.float32)
    expected = torch.autograd.grad(reference, (xx, ww, bb), dy)
    for actual, answer in zip((y, *grad), (reference, *expected), strict=True):
        torch.testing.assert_close(actual, answer, atol=0, rtol=0)
    assert y.shape == (16, 2, 128) and y.dtype == torch.float32
    assert weight is router.weight and bias is router.bias
    assert weight.is_leaf and bias.is_leaf and bias.dtype == grad[-1].dtype == torch.bfloat16
    assert "bias" in router.state_dict() and router.state_dict()["bias"].dtype == torch.bfloat16
