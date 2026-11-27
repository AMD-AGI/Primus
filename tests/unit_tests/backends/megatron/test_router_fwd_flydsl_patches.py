###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests for the gfx1250 FlyDSL FP32 router forward GEMM and the patch that runs Megatron's router
forward on it."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext
from primus.core.utils.gpu_arch import is_gfx1250

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

needs_kernel = pytest.mark.skipif(
    not (is_gfx1250() and importlib.util.find_spec("flydsl") is not None), reason="gfx1250 FlyDSL kernel"
)


def _patch_context(**kwargs):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kwargs))},
    )


def _inputs(tokens, hidden, experts, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, hidden, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(experts, hidden, device="cuda", generator=g) * hidden**-0.5).to(torch.bfloat16)
    dy = torch.randn(tokens, experts, device="cuda", generator=g)
    return x, w, dy


def _rel_err(a, ref):
    return ((a.double() - ref).abs().max() / ref.abs().max()).item()


@needs_kernel
@pytest.mark.parametrize(
    "tokens, experts, hidden",
    [
        (8192, 896, 7168),  # Kimi K3.1 Pro, one rank's micro-batch
        (1024, 256, 2048),
        (384, 128, 256),
        (384, 128, 128),  # 4 k-steps, fewer than the loads issued ahead
    ],
)
def test_logits_match_fp64(tokens, experts, hidden):
    from primus.backends.megatron.core.extensions._flydsl import router_gemm

    x, w, _ = _inputs(tokens, hidden, experts)
    assert router_gemm.router_logits_unsupported_reason(x, w, None, torch.float32) is None
    out = router_gemm.flydsl_router_logits(x, w)
    assert out.dtype == torch.float32 and out.shape == (tokens, experts)
    # products of BF16 inputs are exact in FP32, so the error is the FP32 accumulation's
    assert _rel_err(out, x.double() @ w.double().t()) < 1e-5


@needs_kernel
def test_unsupported_operands_are_named():
    from primus.backends.megatron.core.extensions._flydsl import router_gemm

    x, w, _ = _inputs(1024, 2048, 256)
    ok = router_gemm.router_logits_unsupported_reason
    assert ok(x, w, None, torch.float32) is None
    assert ok(x, w, torch.zeros(256, device="cuda", dtype=torch.bfloat16), torch.float32) is not None
    assert ok(x, w, None, torch.bfloat16) is not None
    assert ok(x.float(), w, None, torch.float32) is not None
    assert ok(x[:1000], w, None, torch.float32) is not None  # tokens not a multiple of 128
    assert ok(x, w[:224], None, torch.float32) is not None  # experts not a multiple of 128
    assert ok(x[:, :1024], w[:, :1024].contiguous(), None, torch.float32) is not None  # non-contiguous x
    x2, w2, _ = _inputs(1024, 2016, 256)
    assert ok(x2, w2, None, torch.float32) is not None  # hidden not a multiple of 128


def test_enabled_reads_the_argument():
    from primus.backends.megatron.patches.moe_patches import (
        router_fwd_flydsl_patches as patch,
    )

    assert patch._enabled(_patch_context(moe_router_fwd_flydsl=True))
    assert not patch._enabled(_patch_context(moe_router_fwd_flydsl=False))
    assert not patch._enabled(_patch_context())


@pytest.fixture
def router_function():
    """Megatron's ``RouterGatingLinearFunction``, unpatched again after the test."""
    from megatron.core.transformer.moe import moe_utils

    fn = moe_utils.RouterGatingLinearFunction
    saved = fn.__dict__["forward"], fn.__dict__["backward"]
    yield fn
    fn.forward, fn.backward = saved
    if "_primus_router_fwd_flydsl_patched" in fn.__dict__:
        del fn._primus_router_fwd_flydsl_patched


@needs_kernel
@pytest.mark.parametrize("experts, on_kernel", [(256, True), (224, False)])
def test_patched_function_against_megatron(router_function, monkeypatch, experts, on_kernel):
    """Megatron's router GEMM before and after the patch, on [s, b, h] as its router calls it: the logits on
    the kernel as accurate as Megatron's FP32 GEMM, the backward unchanged; 224 experts stay on Megatron's path.
    """
    from primus.backends.megatron.core.extensions._flydsl import router_gemm
    from primus.backends.megatron.patches.moe_patches import (
        router_fwd_flydsl_patches as patch,
    )

    tokens, hidden = 1024, 2048
    x, w, dy = _inputs(tokens, hidden, experts)
    x, dy = x.view(tokens // 2, 2, hidden), dy.view(tokens // 2, 2, experts)

    def run():
        inp, weight = x.clone().requires_grad_(), w.clone().requires_grad_()
        out = router_function.apply(inp, weight, None, torch.float32)
        out.backward(dy)
        return out.detach(), inp.grad, weight.grad

    reference = run()
    patch.patch_router_fwd_flydsl(_patch_context(moe_router_fwd_flydsl=True))
    calls = []
    kernel = router_gemm.flydsl_router_logits
    monkeypatch.setattr(router_gemm, "flydsl_router_logits", lambda *a: calls.append(1) or kernel(*a))
    out, dx, dw = run()

    assert len(calls) == int(on_kernel)
    assert out.dtype == torch.float32 and out.shape == reference[0].shape
    if on_kernel:
        torch.testing.assert_close(out, reference[0], rtol=1e-5, atol=1e-5)
    else:
        assert torch.equal(out, reference[0])
    assert torch.equal(dx, reference[1]) and torch.equal(dw, reference[2])


def test_not_installed_with_moe_router_triton_gemm(router_function):
    from primus.backends.megatron.patches.moe_patches import (
        router_fwd_flydsl_patches as patch,
    )

    forward = router_function.forward
    patch.patch_router_fwd_flydsl(_patch_context(moe_router_fwd_flydsl=True, moe_router_triton_gemm=True))
    assert router_function.forward is forward
    assert not getattr(router_function, "_primus_router_fwd_flydsl_patched", False)
