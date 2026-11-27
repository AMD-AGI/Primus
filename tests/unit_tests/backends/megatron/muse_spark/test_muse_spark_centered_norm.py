###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused centered RMSNorm (``norm(x) * (1 + w)``) against the eager fp32 formula."""

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.transformer.muse_spark import centered_norm, norms


def _eager(x, w, eps):
    # independent of the module: fp64 math, one rounding
    xd = x.double()
    out = xd / torch.sqrt((xd * xd).mean(-1, keepdim=True) + eps) * (1 + w.double())
    return out.to(x.dtype)


def test_the_module_on_cpu_is_the_eager_formula():
    torch.manual_seed(0)
    m = norms.MuseSparkCenteredRMSNorm(None, 16, eps=1e-5)
    with torch.no_grad():
        m.weight.normal_(0, 0.1)
    x = (torch.randn(5, 16) * 3).to(torch.bfloat16)
    ref = centered_norm.centered_rms_norm_reference(x, m.weight, 1e-5)
    assert torch.equal(m(x), ref)
    assert (m(x).float() - _eager(x, m.weight.detach(), 1e-5).float()).abs().max() <= 2**-6


def test_the_shared_eager_path_is_the_centered_formula():
    # the shared fused_rms_norm's eager reference with weight 1 + w: same forward, dx and dw
    torch.manual_seed(0)
    x = (torch.randn(64, 256) * 3).to(torch.bfloat16)
    w = torch.randn(256) * 0.02
    g = torch.randn(64, 256).to(torch.bfloat16)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    wa, wb = w.clone().requires_grad_(), w.clone().requires_grad_()
    out_a = centered_norm.centered_rms_norm(xa, wa, 1e-5)
    out_b = centered_norm.centered_rms_norm_reference(xb, wb, 1e-5)
    assert torch.equal(out_a, out_b)
    out_a.backward(g)
    out_b.backward(g)
    assert torch.equal(xa.grad, xb.grad) and torch.allclose(wa.grad, wb.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("h", [512, 1000, 4096, 6144, 8192])  # 1000 / 6144: partial column mask
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_kernel_matches_eager_forward_and_backward(h, dtype):
    torch.manual_seed(0)
    x = (torch.randn(2, 37, h, device="cuda") * 3).to(dtype)
    w = (torch.randn(h, device="cuda") * 0.1).requires_grad_()
    g = torch.randn_like(x)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    wa, wb = w.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    out_f = centered_norm.centered_rms_norm(
        xa, wa, 1e-5
    )  # Primus-Turbo's rmsnorm, else the shared Triton one
    out_e = centered_norm.centered_rms_norm_reference(xb, wb, 1e-5)
    out_f.backward(g)
    out_e.backward(g)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert out_f.shape == x.shape and out_f.dtype == dtype
    assert (out_f.float() - out_e.float()).abs().max() <= tol
    assert (xa.grad.float() - xb.grad.float()).abs().max() <= tol * 4
    assert torch.allclose(wa.grad, wb.grad, rtol=1e-3, atol=1e-2)


def _fake_turbo(calls):
    def rmsnorm(x, gamma, eps=1e-6, zero_centered=False):
        calls.append((gamma, eps, zero_centered))
        return x

    return rmsnorm


def test_turbo_rmsnorm_takes_gpu_calls_when_installed(monkeypatch):
    gpu, cpu = SimpleNamespace(is_cuda=True), SimpleNamespace(is_cuda=False)
    fake = _fake_turbo([])
    monkeypatch.setattr(centered_norm, "_turbo_rmsnorm", lambda: fake)
    assert centered_norm._turbo_rmsnorm_for(gpu, gpu) is fake
    assert centered_norm._turbo_rmsnorm_for(cpu, cpu) is None
    monkeypatch.setattr(centered_norm, "_turbo_rmsnorm", lambda: None)  # not installed / too old
    assert centered_norm._turbo_rmsnorm_for(gpu, gpu) is None


def test_centered_norm_hands_turbo_the_raw_weight_zero_centered(monkeypatch):
    calls = []
    monkeypatch.setattr(centered_norm, "_turbo_rmsnorm_for", lambda x, w: _fake_turbo(calls))
    x, w = torch.randn(4, 8), torch.randn(8)
    centered_norm.centered_rms_norm(x, w, 1e-5)
    centered_norm.weighted_rms_norm(x, w.float(), 1e-6)
    assert calls[0][0] is w and calls[0][1:] == (1e-5, True)
    assert calls[1][1:] == (1e-6, False)
    centered_norm.centered_rms_norm(x, w, 1e-5, fused=False)  # the eager formula: no kernel
    assert len(calls) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(centered_norm._turbo_rmsnorm() is None, reason="needs Primus-Turbo's rmsnorm")
def test_turbo_path_matches_the_shared_kernel_and_is_reproducible(monkeypatch):
    torch.manual_seed(0)
    x = (torch.randn(512, 1, 8192, device="cuda") * 3).to(torch.bfloat16)
    w = (torch.randn(8192, device="cuda") * 0.05).to(torch.bfloat16)
    g = torch.randn_like(x)

    def run():
        xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
        out = centered_norm.centered_rms_norm(xa, wa, 1e-5)
        out.backward(g)
        return out, xa.grad, wa.grad

    turbo = run()
    assert all(torch.equal(a, b) for a, b in zip(turbo, run()))  # deterministic weight gradient
    monkeypatch.setattr(centered_norm, "_turbo_rmsnorm", lambda: None)
    shared = run()
    eager_x, eager_w = x.clone().requires_grad_(), w.clone().requires_grad_()
    eager = centered_norm.centered_rms_norm_reference(eager_x, eager_w, 1e-5)
    eager.backward(g)
    for out, dx, dw in (turbo, shared):
        assert (out.float() - eager.float()).abs().max() <= 2e-2
        assert (dx.float() - eager_x.grad.float()).abs().max() <= 8e-2
        assert torch.allclose(dw.float(), eager_w.grad.float(), rtol=2e-2, atol=5e-1)
