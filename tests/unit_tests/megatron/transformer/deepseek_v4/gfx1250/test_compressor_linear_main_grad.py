###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``linear_main_grad`` (``PRIMUS_FUSE2_COMP_MG=1``, used by the compressor's ``wkv_gate``) against
``nn.Linear`` plus the DDP hook: output, input gradient and ``main_grad`` are byte-equal at the compressor
projection shapes (CSA 2048 x 4096, HCA 1024 x 4096), for the first write of a step (overwrite) and for
accumulation.
"""

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


@pytest.mark.parametrize("nout", [2048, 1024])
@pytest.mark.parametrize("first", [True, False])
def test_linear_main_grad_matches_linear_plus_hook(monkeypatch, nout, first):
    import primus_turbo.pytorch.ops.utils as U

    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        fuse2_main_grad as F2,
    )

    S, K = 1024, 4096
    res = []
    try:
        for fused in (False, True):
            monkeypatch.setenv("PRIMUS_FUSE2_COMP_MG", "1" if fused else "0")
            g = torch.Generator(device=DEV).manual_seed(nout)
            lin = torch.nn.Linear(K, nout, bias=False, device=DEV, dtype=torch.bfloat16)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(nout, K, device=DEV, generator=g) * 0.02)
            w = lin.weight
            w.grad_added_to_main_grad = False
            w.main_grad = torch.randn(nout, K, device=DEV, generator=g).to(torch.bfloat16)
            x = torch.randn(1, S, K, device=DEV, generator=g).to(torch.bfloat16).requires_grad_(True)
            dy = torch.randn(1, S, nout, device=DEV, generator=g).to(torch.bfloat16)
            U.set_main_grad_first_write_policy(lambda p: first)
            if F2.fuse2_linear_mg_ok(x, w):
                y = F2.linear_main_grad(x, w)
            else:
                assert not fused, "fused path was not selected"
                y = lin(x)
            y.backward(dy)
            torch.cuda.synchronize()
            if not w.grad_added_to_main_grad:
                if first:
                    w.main_grad.copy_(w.grad)
                else:
                    w.main_grad.add_(w.grad)
            res.append((y.detach().clone(), x.grad.clone(), w.main_grad.clone()))
    finally:
        U.set_main_grad_first_write_policy(None)
    for name, a, b in zip(("out", "dx", "main_grad"), *res):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name} differs"
