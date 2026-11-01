###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests for the Triton FP32 router GEMMs and the patch that runs Megatron's router GEMM on them."""

from types import SimpleNamespace

import pytest
import torch
import triton

from primus.backends.megatron.core.extensions._triton import router_gemm
from primus.backends.megatron.patches.moe_patches import (
    router_triton_gemm_patches as router_patch,
)
from primus.core.patches import PatchContext
from primus.core.utils.gpu_arch import is_gfx1250

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# DeepSeek-V3's hidden size and a smaller one the kernels take
HIDDENS = [7168, 2048]


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


@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("experts", [32, 128, 256])
@pytest.mark.parametrize("tokens", [8192, 1000])
def test_kernels_match_fp64(tokens, experts, hidden):
    x, w, dy = _inputs(tokens, hidden, experts)
    assert router_gemm.supported(x, w, None, torch.float32)
    x64, w64, dy64 = x.double(), w.double(), dy.double()

    def rel_err(a, ref):
        return ((a.double() - ref).abs().max() / ref.abs().max()).item()

    # Each result must be about as accurate as Megatron's FP32 path (torch FP32 GEMMs here).
    out = router_gemm.forward(x, w)
    ref = x64 @ w64.t()
    assert rel_err(out, ref) <= max(2 * rel_err(x.float() @ w.float().t(), ref), 1e-6)

    # dX is returned in BF16: within one BF16 rounding of the FP64 result
    dx = router_gemm._dx(dy, w)
    torch.testing.assert_close(dx, (dy64 @ w64).to(torch.bfloat16), rtol=1.6e-2, atol=1e-2)

    # dW: dlogits enter as a BF16 hi + lo pair (about 16 mantissa bits), so the result stays close to
    # FP32 and far from what rounding dlogits to BF16 (moe_router_bwd_bf16) gives
    dw = router_gemm._dw(dy, x)
    ref_dw = dy64.t() @ x64
    err = rel_err(dw, ref_dw)
    err_bf16 = rel_err(dy.to(torch.bfloat16).float().t() @ x.float(), ref_dw)
    assert err < 2e-5 and err < 0.05 * err_bf16, f"dW relative error {err:.2e} (BF16 dlogits {err_bf16:.2e})"


@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("experts", [32, 128, 256])
@pytest.mark.parametrize("tokens", [8192, 1000])
def test_gfx1250_launch_configs_are_bit_identical(tokens, experts, hidden):
    if not is_gfx1250():
        pytest.skip("the gfx1250 launch configs")
    x, w, _ = _inputs(tokens, hidden, experts)
    # the default forward config, launched directly
    out = torch.empty(tokens, experts, dtype=torch.float32, device="cuda")
    router_gemm._router_fwd_kernel[(triton.cdiv(tokens, 32),)](
        x,
        w,
        out,
        tokens,
        x.stride(0),
        w.stride(0),
        out.stride(0),
        K=hidden,
        N=experts,
        BLOCK_M=32,
        BLOCK_K=64,
        num_warps=4,
    )
    assert torch.equal(router_gemm.forward(x, w), out)


@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("experts", [32, 128, 256])
@pytest.mark.parametrize("tokens", [8192, 1000])
def test_gfx1250_split_backward_is_bit_identical(tokens, experts, hidden):
    if not is_gfx1250():
        pytest.skip("the gfx1250 pre-split backward")
    x, w, dy = _inputs(tokens, hidden, experts)
    dy[::7] *= 1e-6  # small dlogits too, so the lo half carries bits
    hi, lo = router_gemm._split(dy)
    assert torch.equal(hi, dy.to(torch.bfloat16))
    assert torch.equal(lo, (dy - hi.float()).to(torch.bfloat16))
    assert torch.equal(router_gemm._dx_split(hi, lo, w), router_gemm._dx(dy, w))
    assert torch.equal(router_gemm._dw_split(hi, lo, x), router_gemm._dw(dy, x))


def test_unsupported_shapes_are_rejected():
    x, w, _ = _inputs(64, 7168, 96)
    assert not router_gemm.supported(x, w, None, torch.float32)  # not a power of two
    x, w, _ = _inputs(64, 1152, 32)
    assert not router_gemm.supported(x, w, None, torch.float32)  # hidden not a multiple of 256
    x, w, _ = _inputs(64, 7168, 32)
    assert not router_gemm.supported(x, w, None, torch.bfloat16)  # not an FP32 router
    assert not router_gemm.supported(x.float(), w, None, torch.float32)  # FP32 input
    bias = torch.zeros(32, device="cuda", dtype=torch.bfloat16)
    assert not router_gemm.supported(x, w, bias, torch.float32)


def test_enabled_reads_the_argument():
    assert router_patch._enabled(_patch_context(moe_router_triton_gemm=True))
    assert not router_patch._enabled(_patch_context(moe_router_triton_gemm=False))
    assert not router_patch._enabled(_patch_context())


@pytest.fixture
def router_function():
    """Megatron's ``RouterGatingLinearFunction``, unpatched again after the test."""
    from megatron.core.transformer.moe import moe_utils

    fn = moe_utils.RouterGatingLinearFunction
    saved = fn.__dict__["forward"], fn.__dict__["backward"]
    yield fn
    fn.forward, fn.backward = saved
    if "_primus_router_triton_gemm_patched" in fn.__dict__:
        del fn._primus_router_triton_gemm_patched


def _counted(kernel, calls):
    def counted(*args):
        calls.append(kernel.__name__)
        return kernel(*args)

    return counted


@pytest.mark.parametrize("hidden", HIDDENS + [1152])
def test_patched_function_against_megatron(router_function, monkeypatch, hidden):
    """Megatron's router GEMM before and after the patch, on [s, b, h] as its router calls it: on the
    kernels where they take the shapes, as accurate as Megatron's FP32 path; on that path for hidden 1152."""
    tokens, experts = 1000, 128
    x, w, dy = _inputs(tokens, hidden, experts)
    x, dy = x.view(tokens // 2, 2, hidden), dy.view(tokens // 2, 2, experts)

    def run():
        inp, weight = x.clone().requires_grad_(), w.clone().requires_grad_()
        out = router_function.apply(inp, weight, None, torch.float32)
        out.backward(dy)
        return out.detach(), inp.grad, weight.grad

    reference = run()
    router_patch.patch_router_triton_gemm(_patch_context(moe_router_triton_gemm=True))
    calls = []
    for name in ("forward", "backward"):
        monkeypatch.setattr(router_gemm, name, _counted(getattr(router_gemm, name), calls))
    out, dx, dw = run()

    assert calls == (["forward", "backward"] if hidden % 256 == 0 else [])
    assert (out.dtype, dx.dtype, dw.dtype) == (torch.float32, torch.bfloat16, torch.bfloat16)
    torch.testing.assert_close(out, reference[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dx, reference[1], rtol=1.6e-2, atol=1e-2)
    torch.testing.assert_close(dw, reference[2], rtol=1.6e-2, atol=1e-2)


def test_not_installed_with_moe_router_bwd_bf16(router_function):
    forward = router_function.forward
    router_patch.patch_router_triton_gemm(
        _patch_context(moe_router_triton_gemm=True, moe_router_bwd_bf16=True)
    )
    assert router_function.forward is forward
    assert not getattr(router_function, "_primus_router_triton_gemm_patched", False)
