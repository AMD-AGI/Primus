###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton MoE router gating GEMM patch: plumbing on CPU (is_cuda stand-ins), numerics on a GPU."""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.triton_router_gemm_patches as RGP  # noqa: E402
from primus.backends.megatron.core.fusions import triton_router_gemm as R  # noqa: E402
from tests.unit_tests.backends.megatron.conftest import (  # noqa: E402
    cuda_like,
    patch_ctx,
    requires_gpu,
)


def test_tile_sizes_divide_the_eligibility_multiples():
    # No masks in the kernels: ineligible_reason() admits only shapes that every tile divides exactly.
    for bm in (R.FWD_BM, R.DG_BM, R.WG_SPLIT_M * R.WG_BM):
        assert R.M_MULTIPLE % bm == 0
    for bk in (R.FWD_BK, R.DG_BN, R.WG_BN):
        assert R.K_MULTIPLE % bk == 0


def _bf16(*shape):
    return cuda_like(torch.zeros(*shape, dtype=torch.bfloat16))


def test_ineligible_reasons():
    def reason(x, w, bias=None, router_dtype=torch.float32):
        return R.ineligible_reason(x, w, bias, router_dtype)

    x, w = _bf16(2048, 256), _bf16(32, 256)
    assert reason(x, w) is None
    assert reason(_bf16(4, 512, 256), w) is None  # 3-D [s, b, h]
    assert reason(x, w, bias=cuda_like(torch.zeros(32))) == "router bias"
    assert reason(x, w, router_dtype=torch.bfloat16) == "router dtype torch.bfloat16"
    assert reason(_bf16(256, 2048).t(), w) == "non-contiguous input"
    assert reason(_bf16(4, 256, 256).permute(1, 0, 2), w) == "non-contiguous input"  # [s, b, h] view
    assert "CUDA" in reason(torch.zeros(2048, 256, dtype=torch.bfloat16), w)
    assert "bf16" in reason(cuda_like(torch.zeros(2048, 256)), w)  # fp32 input
    assert "bf16" in reason(x, cuda_like(torch.zeros(32, 256)))  # fp32 weight
    assert reason(x, _bf16(256, 32).t()) == "weight not 2-D contiguous"
    assert "hidden size" in reason(x, _bf16(32, 384))
    assert reason(x, _bf16(64, 256)) == "64 experts (supported: (32,))"
    assert "tokens" in reason(_bf16(2000, 256), w)
    assert "tokens" in reason(_bf16(0, 256), w)
    assert "hidden size not a multiple" in reason(_bf16(2048, 200), _bf16(32, 200))


def test_drop_in_falls_back_to_original(monkeypatch):
    warned = []
    monkeypatch.setattr(R, "warn_fallback_once", lambda key, why: warned.append((key, why)))

    def orig(inp, weight, bias, router_dtype):
        return "orig"

    f = R.make_router_gating_linear(orig, "key")
    assert f.__wrapped__ is orig
    assert f(_bf16(1024, 128), _bf16(32, 128), None, torch.bfloat16) == "orig"
    assert warned == [("key", "router dtype torch.bfloat16")]


def test_autograd_plumbing_with_cpu_stand_ins(monkeypatch):
    """The drop-in + RouterGatingLinearTriton with CPU stand-ins for the kernels == an fp32 reference."""
    seen = []

    def fwd(x2d, w):
        seen.append("fwd")
        return x2d.float() @ w.float().t()

    def dgrad(g, w):
        seen.append("dgrad")
        return (g @ w.float()).bfloat16()

    def wgrad(g, x2d, out_dtype=torch.bfloat16):
        seen.append("wgrad")
        return (g.t() @ x2d.float()).to(out_dtype)

    monkeypatch.setattr(R, "ineligible_reason", lambda *a: None)
    monkeypatch.setattr(R, "router_fwd", fwd)
    monkeypatch.setattr(R, "router_dgrad", dgrad)
    monkeypatch.setattr(R, "router_wgrad", wgrad)
    f = R.make_router_gating_linear(lambda *a: pytest.fail("original path taken"), "key")
    gen = torch.Generator().manual_seed(0)
    for shape in ((64, 48), (8, 4, 48)):
        x = torch.randn(*shape, generator=gen).bfloat16().requires_grad_()
        w = (0.1 * torch.randn(32, 48, generator=gen)).bfloat16().requires_grad_()
        y = f(x, w, None, torch.float32)
        assert y.dtype == torch.float32 and y.shape == (*shape[:-1], 32)
        g = torch.randn(y.shape, generator=gen)
        y.backward(g)
        xr = x.detach().float().requires_grad_()
        wr = w.detach().float().requires_grad_()
        (xr @ wr.t()).backward(g)
        torch.testing.assert_close(y.detach(), x.detach().float() @ w.detach().float().t())
        assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.bfloat16
        assert x.grad.shape == x.shape and w.grad.shape == w.shape
        torch.testing.assert_close(x.grad, xr.grad.bfloat16())
        torch.testing.assert_close(w.grad, wr.grad.bfloat16())
    assert seen == ["fwd", "dgrad", "wgrad"] * 2


def test_patch_installs_once_on_megatron_router_module(restore_patched):
    from megatron.core.transformer.moe import router

    restore_patched(router, "router_gating_linear")
    before = router.router_gating_linear
    ctx = patch_ctx(use_primus_triton_router_gemm=True, moe_router_dtype="fp32")
    RGP.patch_triton_router_gemm(ctx)
    first = router.router_gating_linear
    assert first is not before and first.__wrapped__ is before
    RGP.patch_triton_router_gemm(ctx)
    assert router.router_gating_linear is first  # idempotent


def _sqnr_db(x, ref):
    x, ref = x.double(), ref.double()
    return (10 * torch.log10(ref.square().sum() / (x - ref).square().sum().clamp_min(1e-300))).item()


@requires_gpu
@pytest.mark.parametrize("shape,K", [((1024,), 128), ((1024, 2), 1024), ((4096,), 7168)])
def test_triton_router_gemm_matches_reference(shape, K):
    from megatron.core.transformer.moe.moe_utils import router_gating_linear as orig

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    x = torch.randn(*shape, K, generator=gen, device=dev).bfloat16()
    # Router-like scales: weights ~ a small init std, logits grads ~ an aux-loss-sized gradient.
    w = (0.008 * torch.randn(32, K, generator=gen, device=dev)).bfloat16()
    g = 1e-4 * torch.randn(*shape, 32, generator=gen, device=dev)
    f = R.make_router_gating_linear(lambda *a: pytest.fail("original path taken"), "key")

    def run(fn):
        xi = x.clone().requires_grad_()
        wi = w.clone().requires_grad_()
        y = fn(xi, wi, None, torch.float32)
        y.backward(g)
        return y.detach(), xi.grad, wi.grad

    y_t, dx_t, dw_t = run(f)
    y_o, dx_o, dw_o = run(orig)
    xd, wd, gd = x.double(), w.double(), g.double().reshape(-1, 32)
    y_r = xd @ wd.t()
    dx_r = (gd @ wd).reshape(x.shape)
    dw_r = gd.t() @ xd.reshape(-1, K)
    assert torch.isfinite(dx_t).all() and torch.isfinite(dw_t).all()
    assert _sqnr_db(y_t, y_r) > 100  # fp32 accumulation of exact bf16 products
    torch.testing.assert_close(y_t, y_o, rtol=1e-5, atol=1e-5)
    for t, o, r in ((dx_t, dx_o, dx_r), (dw_t, dw_o, dw_r)):
        assert t.dtype == torch.bfloat16
        # bf16-output bound (~55 dB); the Triton path is as close to fp64 as the original fp32 GEMM path
        assert _sqnr_db(t, r) > 50
        assert _sqnr_db(t, r) >= _sqnr_db(o, r) - 0.5
    # deterministic
    y2, dx2, dw2 = run(f)
    assert torch.equal(y2, y_t) and torch.equal(dx2, dx_t) and torch.equal(dw2, dw_t)
