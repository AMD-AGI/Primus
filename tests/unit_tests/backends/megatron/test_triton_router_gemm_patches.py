###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the pinned-Triton MoE router gating GEMM patch.

CPU part (always): bounds proof of every launch the eligibility contract admits (the kernels' index arithmetic
at the pinned tile sizes), the contract itself (device, dtypes, contiguity, shapes, experts), the drop-in's
fallbacks (CPU tensors, bias, non-fp32 router dtype, non-contiguous 2-D / 3-D input), the autograd plumbing with CPU
stand-ins for the three kernels (2-D and 3-D input, dtypes of the returned grads), and the patch installing
once on Megatron's router module.
GPU part (skipped without a device): the Triton kernels against an fp64 reference and against Megatron's
original router_gating_linear, through autograd.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.triton_router_gemm_patches as RGP  # noqa: E402
from primus.backends.megatron.core.fusions import triton_router_gemm as R  # noqa: E402
from primus.core.patches.context import PatchContext  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class _CudaLike(torch.Tensor):
    """CPU tensor that reports is_cuda (exercises the shape / dtype part of the contract on CPU)."""

    @property
    def is_cuda(self):
        return True


def _cl(t):
    return t.as_subclass(_CudaLike)


def _launch_extents(M, K, E=32):
    """Every (tensor, max element offset touched + 1, numel) of one fwd/dgrad/wgrad call, from the kernels'
    index arithmetic at the pinned tile sizes."""
    rows = []
    # fwd: pid_m < M/BM, one K split of length K
    pm = M // R.FWD_BM - 1
    rows.append(("fwd.x", (pm * R.FWD_BM + R.FWD_BM - 1) * K + (K - R.FWD_BK) + R.FWD_BK - 1 + 1, M * K))
    rows.append(("fwd.w", (E - 1) * K + (K - R.FWD_BK) + R.FWD_BK - 1 + 1, E * K))
    rows.append(("fwd.y", (pm * R.FWD_BM + R.FWD_BM - 1) * E + E - 1 + 1, M * E))
    pm, pn = M // R.DG_BM - 1, K // R.DG_BN - 1
    rows.append(("dgrad.g", (pm * R.DG_BM + R.DG_BM - 1) * E + E - 1 + 1, M * E))
    rows.append(("dgrad.w", (E - 1) * K + pn * R.DG_BN + R.DG_BN - 1 + 1, E * K))
    rows.append(("dgrad.dx", (pm * R.DG_BM + R.DG_BM - 1) * K + pn * R.DG_BN + R.DG_BN - 1 + 1, M * K))
    rps = M // R.WG_SPLIT_M
    pn, ps = K // R.WG_BN - 1, R.WG_SPLIT_M - 1
    last_m = ps * rps + (rps - R.WG_BM) + R.WG_BM - 1
    rows.append(("wgrad.g", last_m * E + E - 1 + 1, M * E))
    rows.append(("wgrad.x", last_m * K + pn * R.WG_BN + R.WG_BN - 1 + 1, M * K))
    rows.append(
        ("wgrad.part", ps * E * K + (E - 1) * K + pn * R.WG_BN + R.WG_BN - 1 + 1, R.WG_SPLIT_M * E * K)
    )
    return rows


@pytest.mark.parametrize("M,K", [(1024, 128), (8192, 7168), (4096, 2048), (16384, 7168)])
def test_launch_extents_stay_inside_tensors(M, K):
    for name, hi, numel in _launch_extents(M, K):
        assert 0 < hi <= numel, (name, hi, numel)


def test_eligibility_contract():
    x = _cl(torch.zeros(2048, 256, dtype=torch.bfloat16))
    w = _cl(torch.zeros(32, 256, dtype=torch.bfloat16))
    assert R.eligible(x, w)
    assert not R.eligible(torch.zeros(2048, 256, dtype=torch.bfloat16), w)  # CPU
    assert not R.eligible(_cl(torch.zeros(2000, 256, dtype=torch.bfloat16)), w)  # tokens % 1024
    assert not R.eligible(
        _cl(torch.zeros(2048, 200, dtype=torch.bfloat16)), _cl(torch.zeros(32, 200).bfloat16())
    )
    assert not R.eligible(x, _cl(torch.zeros(64, 256, dtype=torch.bfloat16)))  # 64 experts
    assert not R.eligible(_cl(torch.zeros(2048, 256)), w)  # fp32 input
    assert not R.eligible(_cl(torch.zeros(256, 2048, dtype=torch.bfloat16).t()), w)  # non-contiguous
    assert not R.eligible(_cl(torch.zeros(0, 256, dtype=torch.bfloat16)), w)  # no tokens
    assert not R.eligible(x, _cl(torch.zeros(32, 256)))  # fp32 weight
    assert not R.eligible(x, _cl(torch.zeros(256, 32, dtype=torch.bfloat16).t()))  # non-contiguous weight
    assert not R.eligible(x, _cl(torch.zeros(32, 384, dtype=torch.bfloat16)))  # weight K != input K


def test_drop_in_falls_back_to_original():
    calls = []

    def orig(inp, weight, bias, router_dtype):
        calls.append(router_dtype)
        return "orig"

    f = R.make_router_gating_linear(orig)
    assert f.__wrapped__ is orig and getattr(f, R.ORIGINAL_ATTR) is orig
    x = torch.zeros(1024, 128, dtype=torch.bfloat16)
    w = torch.zeros(32, 128, dtype=torch.bfloat16)
    assert f(x, w, None, torch.float32) == "orig"  # CPU tensors
    xc, wc = _cl(x), _cl(w)
    assert f(xc, wc, _cl(torch.zeros(32)), torch.float32) == "orig"  # bias
    assert f(xc, wc, None, torch.bfloat16) == "orig"  # router dtype
    xt = _cl(torch.zeros(128, 1024, dtype=torch.bfloat16).t())
    assert f(xt, wc, None, torch.float32) == "orig"  # non-contiguous input
    x3 = _cl(torch.zeros(4, 256, 128, dtype=torch.bfloat16).permute(1, 0, 2))  # non-contiguous [s, b, h] view
    assert f(x3, wc, None, torch.float32) == "orig"  # non-contiguous 3-D input
    assert len(calls) == 5


def test_autograd_plumbing_with_cpu_stand_ins(monkeypatch):
    """The drop-in + RouterGatingLinearTriton with CPU stand-ins for the kernels == an fp32 reference."""
    seen = []
    monkeypatch.setattr(R, "eligible", lambda x2d, w: True)
    monkeypatch.setattr(R, "router_fwd", lambda x2d, w: (seen.append("fwd"), x2d.float() @ w.float().t())[1])
    monkeypatch.setattr(R, "router_dgrad", lambda g, w: (seen.append("dgrad"), (g @ w.float()).bfloat16())[1])
    monkeypatch.setattr(
        R,
        "router_wgrad",
        lambda g, x2d, out_dtype=torch.bfloat16: (seen.append("wgrad"), (g.t() @ x2d.float()).to(out_dtype))[
            1
        ],
    )
    f = R.make_router_gating_linear(lambda *a: pytest.fail("original path taken"))
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


def _ctx(**args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))},
    )


def test_patch_installs_once_on_megatron_router_module():
    from megatron.core.transformer.moe import router

    before = router.router_gating_linear
    try:
        RGP.patch_triton_router_gemm(_ctx(use_primus_triton_router_gemm=True, moe_router_dtype="fp32"))
        first = router.router_gating_linear
        assert getattr(first, R.ORIGINAL_ATTR) is before
        RGP.patch_triton_router_gemm(_ctx(use_primus_triton_router_gemm=True, moe_router_dtype="fp32"))
        assert router.router_gating_linear is first  # idempotent
    finally:
        router.router_gating_linear = before


# ---------------------------------------------------------------- GPU


def _sqnr_db(x, ref):
    x, ref = x.double(), ref.double()
    return (10 * torch.log10(ref.square().sum() / (x - ref).square().sum().clamp_min(1e-300))).item()


@cuda
@pytest.mark.parametrize("shape,K", [((1024,), 128), ((1024, 2), 1024), ((4096,), 7168)])
def test_triton_router_gemm_matches_reference(shape, K):
    from megatron.core.transformer.moe.moe_utils import router_gating_linear as orig

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    x = torch.randn(*shape, K, generator=gen, device=dev).bfloat16()
    w = (0.008 * torch.randn(32, K, generator=gen, device=dev)).bfloat16()
    g = 1e-4 * torch.randn(*shape, 32, generator=gen, device=dev)
    f = R.make_router_gating_linear(lambda *a: pytest.fail("original path taken"))

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
    assert _sqnr_db(y_t, y_r) > 100
    torch.testing.assert_close(y_t, y_o, rtol=1e-5, atol=1e-5)
    for t, o, r in ((dx_t, dx_o, dx_r), (dw_t, dw_o, dw_r)):
        assert t.dtype == torch.bfloat16
        # bf16-output bound (~55 dB); the Triton path is as close to fp64 as the original fp32 GEMM path
        assert _sqnr_db(t, r) > 50
        assert _sqnr_db(t, r) >= _sqnr_db(o, r) - 0.5
    # deterministic
    y2, dx2, dw2 = run(f)
    assert torch.equal(y2, y_t) and torch.equal(dx2, dx_t) and torch.equal(dw2, dw_t)
