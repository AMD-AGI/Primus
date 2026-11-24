###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HyperHead: fused lift / collapse and the rms-linear backward that also sums the two incoming gradients.

* ``hc_lift`` / ``hc_head_collapse`` (``PRIMUS_HC_LIFT_FUSED`` / ``PRIMUS_HC_HEAD_FUSED``) give y, d(h) and d(mixes)
  byte-identical to eager autograd (mode 1; mode 2 within a relative 1e-2).
* ``PRIMUS_LAUNCH_HEAD_MG=1`` (HyperHead parameter gradients written into main_grad inside the backward) gives y, dx and
  every main_grad byte-identical to autograd plus the DDP hook (first write and accumulate).
* ``PRIMUS_FUSE2_HEAD_GADD=1`` (the two x-gradients summed inside the rms-linear backward) gives out, dx and all
  parameter gradients of ``HyperHead`` byte-identical to autograd's add.
* The one-pass collapse backward (D = 4096) gives dx and dpre byte-identical to the product kernel + torch's reduce
  and to eager autograd (zero / -0 / tiny / huge / cancelling rows).
* The rms-linear launch config (forward row tile, backward warps / stages) does not change out / dx / dW.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

B, S, K, D = 1, 512, 4, 1024


@pytest.mark.parametrize("mode", ["1", "2"])
@pytest.mark.parametrize("seed", [0, 1])
def test_lift_and_collapse_match_eager(monkeypatch, mode, seed):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        hc_head as hh,
    )

    monkeypatch.setenv("PRIMUS_HC_HEAD_FUSED", mode)
    monkeypatch.setenv("PRIMUS_HC_LIFT_FUSED", "1")
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mixes = torch.randn(B, S, K, device="cuda", generator=g)
    gy = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, generator=g)
    mix = torch.randn(B, S, K, D, device="cuda", dtype=torch.bfloat16, generator=g)  # make the streams differ
    res = []
    for fused in (False, True):
        hr = h.clone().requires_grad_(True)
        mr = mixes.clone().requires_grad_(True)
        x = hh.hc_lift(hr) if fused else hr.unsqueeze(2).expand(B, S, K, D).contiguous()
        x = x * mix
        pre = torch.sigmoid(mr) + 1e-6
        y = hh.hc_head_collapse(x, pre) if fused else (pre.unsqueeze(-1) * x).sum(dim=-2).to(x.dtype)
        y.backward(gy)
        torch.cuda.synchronize()
        res.append((y.detach(), hr.grad, mr.grad))
    for a, b in zip(*res):
        if mode == "1":
            assert bits_equal(a, b)
        else:  # mode 2 keeps the same math with a different reduction tree
            assert ((a.float() - b.float()).norm() / b.float().norm()).item() < 1e-2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_head_grad_add_fusion_matches_autograd(monkeypatch, seed):
    from primus.backends.megatron.core.transformer.hyper_connection import HyperHead

    res = []
    for on in ("0", "1"):
        monkeypatch.setenv("PRIMUS_FUSE2_HEAD_GADD", on)
        torch.manual_seed(seed)
        head = HyperHead(hidden_size=1024, hc_mult=4).cuda()
        x = (torch.randn(1, S, 4, 1024, device="cuda") * (1 + seed)).to(torch.bfloat16)
        x0 = x.clone().requires_grad_(True)
        y = head(x0 * 1)  # non-leaf, like the trunk output
        y.backward(torch.randn_like(y))
        torch.cuda.synchronize()
        res.append([y.detach().clone(), x0.grad.clone()] + [p.grad.clone() for p in head.parameters()])
    assert all(bits_equal(a, b) for a, b in zip(*res))


@pytest.mark.parametrize("first", [True, False], ids=["first_write", "accumulate"])
@pytest.mark.parametrize("seed", [0, 1])
def test_head_main_grad_in_backward_matches_ddp_hook(monkeypatch, seed, first):
    import primus_turbo.pytorch.ops.utils as U

    from primus.backends.megatron.core.transformer.hyper_connection import HyperHead

    dev = "cuda"
    res = []
    for on in ("0", "1"):
        monkeypatch.setenv("PRIMUS_LAUNCH_HEAD_MG", on)
        torch.manual_seed(seed)
        hh = HyperHead(hidden_size=D, hc_mult=4).to(dev)
        with torch.no_grad():
            hh.base.copy_(torch.randn(4) * 0.1)
            hh.scale.fill_(0.7)
        hh = hh.to(torch.bfloat16)
        g = torch.Generator(device=dev).manual_seed(seed + 10)
        for n, p in hh.named_parameters():
            p.grad_added_to_main_grad = False
            p.main_grad = torch.randn(p.shape, device=dev, generator=g).to(torch.bfloat16)
        U.set_main_grad_first_write_policy(lambda p: first)
        x = (torch.randn(1, S, 4, D, device=dev, generator=g) * 0.5).to(torch.bfloat16).requires_grad_(True)
        dy = torch.randn(1, S, D, device=dev, generator=g).to(torch.bfloat16)
        hh(x).backward(dy)
        torch.cuda.synchronize()
        for n, p in hh.named_parameters():  # the DDP hook for parameters not written in the kernel
            if not p.grad_added_to_main_grad and p.grad is not None:
                p.main_grad.copy_(p.grad) if first else p.main_grad.add_(p.grad)
        res.append([x.grad.clone()] + [p.main_grad.clone() for _, p in hh.named_parameters()])
    assert all(bits_equal(a, b) for a, b in zip(*res))


def _head_bwd_inputs(seed, M, D):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gy = (torch.randn(M, D, device="cuda", generator=g) * (1 + seed)).to(torch.bfloat16)
    x = torch.randn(M, 4, D, device="cuda", generator=g).to(torch.bfloat16)
    pre = torch.rand(M, 4, device="cuda", generator=g) * 2 - 0.5
    if seed == 1 and M > 11:
        gy[3] = 0
        gy[5] = -0.0
        x[7] = 1e-38
        x[9, 2] = 3e38
        gy[11, ::2] = -gy[11, 1::2]
    if seed == 2:
        x = x * 1e-20
    if seed == 3:
        pre[0] = -0.0
    return gy, x, pre


@pytest.mark.parametrize("M", [4096, 37, 4, 3])  # M < 4 (fewer than 16 outputs) keeps the torch reduce
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_one_pass_collapse_backward_matches_torch_reduce(seed, M):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        hc_head as hh,
    )

    D = 4096
    gy, x, pre = _head_bwd_inputs(seed, M, D)
    # the code path it replaces: fp32 products from the prod kernel, summed by torch
    dx_ref = torch.empty_like(x)
    prod = torch.empty((M, 4, D), dtype=torch.float32, device="cuda")
    hh._hc_head_bwd_prod_kernel[(M, 1)](
        gy, x, pre, dx_ref, prod, D, BD=4096, num_warps=8, enable_fp_fusion=False
    )
    dpre_ref = prod.sum(dim=(prod.dim() - 1,), keepdim=True).view(M, 4)
    # eager autograd
    xe = x.clone().requires_grad_(True)
    pe = pre.clone().requires_grad_(True)
    (pe.unsqueeze(-1) * xe).sum(dim=-2).to(x.dtype).backward(gy)
    # fused autograd function (one-pass kernel for D = 4096)
    xf = x.clone().requires_grad_(True)
    pf = pre.clone().requires_grad_(True)
    hh.hc_head_collapse(xf.view(1, M, 4, D), pf.view(1, M, 4)).backward(gy.view(1, M, D))
    torch.cuda.synchronize()
    assert bits_equal(xf.grad, dx_ref) and bits_equal(pf.grad, dpre_ref)
    assert bits_equal(xf.grad, xe.grad) and bits_equal(pf.grad, pe.grad)


@pytest.mark.parametrize(
    "N,KD,wdtype", [(4, 16384, torch.float32), (24, 16384, torch.bfloat16), (4, 4096, torch.float32)]
)
@pytest.mark.parametrize("seed", [0, 1])
def test_rms_linear_launch_config_keeps_bits(monkeypatch, seed, N, KD, wdtype):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        hc_rms_linear as rl,
    )

    T = 4096
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, KD, device="cuda", generator=g).to(torch.bfloat16)
    if seed == 1:
        x[5] = 0
        x[7] = x[7] * 1e-20
    w = (torch.randn(N, KD, device="cuda", generator=g) / KD**0.5).to(wdtype)
    dy = torch.randn(T, N, device="cuda", generator=g)
    res = []
    for cfg in ({}, dict(fwd_bt=32, bwd_warps=8, bwd_stages=2)):  # the code default, the previous config
        with monkeypatch.context() as m:
            for k, v in cfg.items():
                m.setitem(rl.HC_CFG, k, v)
            xr = x.clone().requires_grad_(True)
            wr = w.clone().requires_grad_(True)
            out = rl.hc_rms_linear(xr, wr, 1e-6)
            out.backward(dy)
            torch.cuda.synchronize()
            res.append((out.detach(), xr.grad, wr.grad))
    assert all(bits_equal(a, b) for a, b in zip(*res))
