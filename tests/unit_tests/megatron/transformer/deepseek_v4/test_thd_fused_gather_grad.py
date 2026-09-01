###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused compact gather must match the PyTorch one in VALUE and in GRADIENT.

dsv4_cp ships compressor_input_compact fwd/bwd as two bare functions with nothing
connecting them to autograd, so Primus supplies the ``autograd.Function``. A wrong
backward there is the worst kind of bug available here: the forward still produces
sensible compressed keys, training still converges to something, and only the gradient
that reaches the hidden states is wrong. Nothing surfaces it except a test like this.

The forward is a pure gather -- every output row copies exactly one input row -- so the
reference backward is a scatter-add, which is precisely what an equivalent PyTorch gather
gets from autograd for free. Comparing against that is a genuine check, not a restatement.
"""

import os
import sys

import pytest
import torch

# Out-of-tree package: not in the repo and not in the image. Point PRIMUS_DSV4_CP_DIR
# at a checkout to enable these tests; they skip cleanly when it is absent.
_PKG = os.environ.get("PRIMUS_DSV4_CP_DIR", "")
pytestmark = pytest.mark.skipif(not os.path.isdir(_PKG), reason="dsv4_cp not present")
if os.path.isdir(_PKG) and _PKG not in sys.path:
    sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)


@pytest.mark.parametrize("ratio", [4, 8])
@pytest.mark.parametrize("global_start", [0, 24])
def test_fused_gather_value_and_grad_match_torch(ratio, global_start):
    from dsv4_cp_layout.backends import available, get

    if "torch_native" not in available():
        pytest.skip("no reference backend")
    backend = get("torch_native")

    C = 16
    lens = [13, 9, 21, 7]  # ragged on purpose: no length is a multiple of ratio
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    total = int(cu[-1])
    c = Compressor(hidden_size=C, head_dim=C, ratio=ratio)
    d_comp = c.thd_d_comp
    d_window = d_comp

    torch.manual_seed(0)
    # Under CP this rank holds rows [global_start, global_start + l_local); a window whose
    # earlier rows fall before that reads them from the boundary buffer. global_start = 0
    # is rank 0, where the boundary is never touched -- only the nonzero case exercises
    # the boundary gradient, which is the one that would silently vanish.
    l_local = total - global_start
    base = torch.randn(l_local, C, dtype=torch.float64)
    bnd_base = torch.randn(d_window, C, dtype=torch.float64)

    # --- fused path ---
    h1 = base.clone().requires_grad_(True)
    b1 = bnd_base.clone().requires_grad_(True)
    row_idx, comp_ids, _ = c.thd_compact_plan(cu.to(torch.int64), global_start, l_local)
    c_cap = comp_ids.numel()
    out1, _ = Compressor._FusedCompactGather.apply(
        h1,
        b1,
        cu,
        backend,
        global_start,
        l_local,
        ratio,
        d_comp,
        d_window,
        c_cap * ratio,
        C,
    )
    # Only the slots this rank actually owns carry meaning; the rest are capacity padding
    # and must not contribute gradient. Zero their share so both paths see the same
    # upstream gradient.
    n_used = int((comp_ids >= 0).sum().item()) * ratio
    g = torch.randn_like(out1)
    g[n_used:] = 0
    out1.backward(g)

    # --- PyTorch reference: the same gather, letting autograd derive the scatter-add ---
    h2 = base.clone().requires_grad_(True)
    b2 = bnd_base.clone().requires_grad_(True)
    buf = torch.cat([b2, h2], dim=0)  # [boundary ++ local]
    # global row -> [boundary ++ local] buffer row
    idx = (row_idx[: n_used // ratio] - global_start + d_window).reshape(-1)
    out2 = buf[idx]
    out2.backward(g[:n_used])

    torch.testing.assert_close(out1[:n_used], out2, rtol=0, atol=0)
    torch.testing.assert_close(
        h1.grad,
        h2.grad,
        rtol=1e-10,
        atol=1e-10,
        msg=lambda m: f"hidden gradient differs (ratio={ratio}):\n{m}",
    )
    torch.testing.assert_close(
        b1.grad,
        b2.grad,
        rtol=1e-10,
        atol=1e-10,
        msg=lambda m: f"boundary gradient differs (ratio={ratio}) -- this one would be "
        f"invisible in training, since it only affects the neighbour's "
        f"share of the update:\n{m}",
    )
