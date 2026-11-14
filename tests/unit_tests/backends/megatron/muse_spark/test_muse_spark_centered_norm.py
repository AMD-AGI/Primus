###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused centered RMSNorm (``norm(x) * (1 + w)``) against the eager fp32 formula."""

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
    out_f = centered_norm.centered_rms_norm(xa, wa, 1e-5)  # the shared Triton RMSNorm
    out_e = centered_norm.centered_rms_norm_reference(xb, wb, 1e-5)
    out_f.backward(g)
    out_e.backward(g)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert out_f.shape == x.shape and out_f.dtype == dtype
    assert (out_f.float() - out_e.float()).abs().max() <= tol
    assert (xa.grad.float() - xb.grad.float()).abs().max() <= tol * 4
    assert torch.allclose(wa.grad, wb.grad, rtol=1e-3, atol=1e-2)
