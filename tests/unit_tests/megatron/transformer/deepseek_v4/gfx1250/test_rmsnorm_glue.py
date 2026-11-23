###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RMSNorm glue kernels (``PRIMUS_GLUE_RMSNORM``).

* ``fused_rms_norm``: with the switch on, y and dx are bit-equal to the switch-off kernel and dw is as accurate
  as the switch-off result (fp64 reference); both ``main_grad`` write modes (first write overwrites, later writes
  accumulate) equal the ``param.grad`` + DDP-hook semantics.
* ``rmsnorm_g90``: the direct ``main_grad`` write (first write / accumulate) equals ``param.grad`` and the hook.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


@pytest.fixture
def first_write_policy():
    from primus_turbo.pytorch.ops import utils as U

    state = {"first": True}
    U.set_main_grad_first_write_policy(lambda p: state["first"])
    yield state
    U.set_main_grad_first_write_policy(None)


@pytest.mark.parametrize("N,D", [(64, 512), (130, 512), (1000, 384), (32, 2048)])
@pytest.mark.parametrize("wdt", [torch.float32, torch.bfloat16])
def test_fused_rms_norm_glue_matches_default(monkeypatch, N, D, wdt):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        rmsnorm as R,
    )

    torch.manual_seed(N + D)
    x = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
    w = (torch.rand(D, device=DEV) + 0.5).to(wdt)
    out_dtype = torch.promote_types(x.dtype, w.dtype)
    g = torch.randn(N, D, device=DEV, dtype=out_dtype)
    res = {}
    for glue in ("0", "1"):
        monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", glue)
        xx = x.clone().requires_grad_(True)
        ww = torch.nn.Parameter(w.clone())
        y = R.fused_rms_norm(xx, ww, eps=1e-6, mid_cast=True, out_dtype=out_dtype)
        y.backward(g)
        res[glue] = (y.detach(), xx.grad.detach(), ww.grad.detach().float())
    (y0, dx0, dw0), (y1, dx1, dw1) = res["0"], res["1"]
    x64, g64 = x.double(), g.double()
    rstd = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-6)
    dw_ref = (g64 * x64 * rstd).sum(0)
    e0 = ((dw0.double() - dw_ref).abs().max() / dw_ref.abs().max()).item()
    e1 = ((dw1.double() - dw_ref).abs().max() / dw_ref.abs().max()).item()
    assert bits_equal(y0, y1), "y differs"
    assert bits_equal(dx0, dx1), "dx differs"
    assert e1 <= max(2 * e0, 1e-6), f"dw error {e1:.2e} vs default {e0:.2e}"


@pytest.mark.parametrize("mgdt", [torch.float32, torch.bfloat16])
def test_fused_rms_norm_main_grad_direct_write(monkeypatch, first_write_policy, mgdt):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        rmsnorm as R,
    )

    torch.manual_seed(0)
    N, D = 256, 512
    x = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(N, D, device=DEV, dtype=torch.float32)
    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "1")
    w = torch.nn.Parameter(torch.rand(D, device=DEV) + 0.5)
    w.grad_added_to_main_grad = False
    w.main_grad = torch.full((D,), 7.0, device=DEV, dtype=mgdt)

    first_write_policy["first"] = True
    R.fused_rms_norm(
        x.clone().requires_grad_(True), w, eps=1e-6, mid_cast=True, out_dtype=torch.float32
    ).backward(g)
    mg1 = w.main_grad.clone()
    w2 = torch.nn.Parameter(w.detach().clone())  # reference: plain param.grad
    R.fused_rms_norm(
        x.clone().requires_grad_(True), w2, eps=1e-6, mid_cast=True, out_dtype=torch.float32
    ).backward(g)
    ref1 = w2.grad.to(mgdt)
    assert torch.equal(mg1, ref1) and w.grad_added_to_main_grad, "first write differs"

    first_write_policy["first"] = False
    R.fused_rms_norm(
        x.clone().requires_grad_(True), w, eps=1e-6, mid_cast=True, out_dtype=torch.float32
    ).backward(g)
    ref2 = ref1.clone()
    ref2.add_(w2.grad)  # the DDP hook: main_grad.add_(param.grad)
    assert torch.equal(w.main_grad, ref2), "accumulate differs"


@pytest.mark.parametrize("M,H", [(256, 512), (512, 1024)])
def test_rmsnorm_g90_main_grad_direct_write(monkeypatch, first_write_policy, M, H):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        rmsnorm_g90 as G,
    )

    torch.manual_seed(M + H)
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
    gam = torch.rand(H, device=DEV, dtype=torch.bfloat16) + 0.5
    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "0")
    p0 = torch.nn.Parameter(gam.clone())
    x0 = x.clone().requires_grad_(True)
    G.rmsnorm_g90(x0, p0, 1e-6).backward(g)

    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "1")
    p1 = torch.nn.Parameter(gam.clone())
    x1 = x.clone().requires_grad_(True)
    p1.grad_added_to_main_grad = False
    p1.main_grad = torch.full((H,), 3.0, device=DEV, dtype=torch.bfloat16)
    first_write_policy["first"] = True
    G.rmsnorm_g90(x1, p1, 1e-6).backward(g)
    assert torch.equal(p1.main_grad, p0.grad) and torch.equal(x1.grad, x0.grad), "first write differs"
    first_write_policy["first"] = False
    G.rmsnorm_g90(x.clone().requires_grad_(True), p1, 1e-6).backward(g)
    ref = p0.grad.clone()
    ref.add_(p0.grad)
    assert torch.equal(p1.main_grad, ref), "accumulate differs"
