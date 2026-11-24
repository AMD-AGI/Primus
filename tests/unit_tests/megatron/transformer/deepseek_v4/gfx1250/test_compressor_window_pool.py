###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compressor window pool.

* ``fused_window_pool`` (``PRIMUS_GLUE_COMPRESS_POOL=1``) equals the reference chain (overlap transform of the
  projection followed by ``fused_softmax_weighted_pool``): output, d(proj) and d(ape) byte-equal.
* The d(ape) fan-in kernel (``PRIMUS_FUSE2_WPOOL_DAPE=1``): output and d(proj) byte-equal to the switch-off
  backward; d(ape) equal up to the fp32 summation order (checked at bf16-rounding tolerance); ``main_grad``
  first-write (overwrite) and accumulate semantics equal ``param.grad`` plus the DDP hook.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


def _reference_pool(CP, proj, ape, ratio, hd, overlap):
    coff = 2 if overlap else 1
    kv_proj, score_proj = proj.split(coff * hd, dim=-1)
    B, S, C = kv_proj.shape
    kv = kv_proj.reshape(B, S // ratio, ratio, C)
    sc = score_proj.reshape(B, S // ratio, ratio, C)
    if overlap:

        def ot(t):
            a, b = torch.chunk(t, 2, dim=-1)
            bp = torch.cat([torch.zeros_like(b[:, :1]), b[:, :-1]], dim=1)
            return torch.cat([a, bp], dim=2)

        kv, sc = ot(kv), ot(sc)
    return CP.fused_softmax_weighted_pool(kv, sc, ape)


@pytest.mark.parametrize(
    "B,S,ratio,hd,overlap", [(1, 512, 4, 512, True), (1, 1024, 128, 512, False), (2, 512, 4, 128, True)]
)
def test_window_pool_matches_reference_chain(monkeypatch, B, S, ratio, hd, overlap):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        compressor_pool as CP,
    )

    monkeypatch.setenv("PRIMUS_GLUE_COMPRESS_POOL", "1")
    coff = 2 if overlap else 1
    torch.manual_seed(S + ratio)
    proj0 = torch.randn(B, S, 2 * coff * hd, device=DEV, dtype=torch.bfloat16)
    ape0 = torch.randn((2 * ratio if overlap else ratio), hd, device=DEV) * 0.02
    gout = torch.randn(B, S // ratio, hd, device=DEV, dtype=torch.bfloat16)
    assert CP.glue_window_pool_ok(proj0, ape0, ratio, hd, overlap)
    res = []
    for fn in (lambda *a: _reference_pool(CP, *a), CP.fused_window_pool):
        proj = proj0.clone().requires_grad_(True)
        ape = ape0.clone().requires_grad_(True)
        y = fn(proj, ape, ratio, hd, overlap)
        y.backward(gout)
        res.append((y.detach(), proj.grad, ape.grad))
    for name, a, b in zip(("out", "dproj", "dape"), *res):
        assert bits_equal(a, b), f"{name} differs from the reference chain"


@pytest.mark.parametrize("ratio,overlap", [(4, True), (128, False)])
@pytest.mark.parametrize("mg_mode", [None, "first", "acc"])
def test_window_pool_dape_fanin(monkeypatch, ratio, overlap, mg_mode):
    import primus_turbo.pytorch.ops.utils as U

    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        compressor_pool as CP,
    )

    S, hd = 1024, 512
    monkeypatch.setenv("PRIMUS_GLUE_COMPRESS_POOL", "1")

    def run(fused, seed=0):
        monkeypatch.setenv("PRIMUS_FUSE2_WPOOL_DAPE", "1" if fused else "0")
        g = torch.Generator(device=DEV).manual_seed(seed)
        coff = 2 if overlap else 1
        R = 2 * ratio if overlap else ratio
        proj = (
            torch.randn(1, S, 2 * coff * hd, device=DEV, generator=g).to(torch.bfloat16).requires_grad_(True)
        )
        ape = torch.nn.Parameter(torch.randn(R, hd, device=DEV, generator=g) * 0.02)
        mg0 = torch.randn(R, hd, device=DEV, generator=g)
        if mg_mode is not None:
            ape.grad_added_to_main_grad = False
            ape.main_grad = mg0.clone()
            U.set_main_grad_first_write_policy(lambda p: mg_mode == "first")
        dout = torch.randn(1, S // ratio, hd, device=DEV, generator=g).to(torch.bfloat16)
        out = CP.fused_window_pool(proj, ape, ratio, hd, overlap)
        out.backward(dout)
        torch.cuda.synchronize()
        if mg_mode is not None and not ape.grad_added_to_main_grad:  # the DDP hook of the unfused path
            if mg_mode == "first":
                ape.main_grad.copy_(ape.grad)
            else:
                ape.main_grad.add_(ape.grad)
        dape = ape.main_grad.clone() if mg_mode is not None else ape.grad.clone()
        return out.detach().clone(), proj.grad.clone(), dape

    try:
        o0, dp0, da0 = run(False)
        o1, dp1, da1 = run(True)
    finally:
        U.set_main_grad_first_write_policy(None)
    assert bits_equal(o0, o1), "pool output differs"
    assert bits_equal(dp0, dp1), "d(proj) differs"
    rel = ((da0 - da1).abs().max() / da0.abs().max()).item()
    assert rel < 2e-2, f"d(ape) max relative difference {rel:.2e}"
