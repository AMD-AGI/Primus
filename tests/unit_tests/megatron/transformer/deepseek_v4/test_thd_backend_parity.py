###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused and PyTorch packed-compressor paths must agree, forward and backward.

``PRIMUS_THD_COMPACT_BACKEND`` selects between Primus's own window gather and the
dsv4_cp layout kernel. They are meant to be interchangeable, so this compares them
directly on the same weights and the same input rather than only checking that each
passes its own tests -- two paths can each be self-consistent and still disagree.

The gradient half matters more than the value half: the kernel's backward is a
hand-written scatter wrapped by Primus in an ``autograd.Function``, and an error there
leaves the forward perfectly correct while training a different model.
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


def _reference_groups(cu, global_start, l_local, ratio, d_comp):
    """dsv4_cp reference.py::_compressed_groups, transcribed -- the window set the KERNEL
    fills its compact buffer with. Deliberately reaches back d_comp rows, so it is a
    superset of Primus's strict last-row ownership."""
    cu = [int(x) for x in cu.tolist()]
    groups, g_end = [], global_start + l_local
    for seq, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
        if max(s, global_start) >= min(e, g_end):
            continue
        n_full = (e - s) // ratio
        numer = max(0, global_start - d_comp - s)
        first = (numer + ratio - 1) // ratio if numer > 0 else 0
        stop = min((min(e, g_end) - s) // ratio, n_full)
        groups.extend((seq, c) for c in range(first, max(first, stop)))
    return groups


def _run(backend_name, ratio, cu, hidden, boundary, global_start):
    torch.manual_seed(0)  # identical weights on both paths
    c = Compressor(hidden_size=hidden.shape[-1], head_dim=16, ratio=ratio).double()
    prev = os.environ.get("PRIMUS_THD_COMPACT_BACKEND")
    os.environ["PRIMUS_THD_COMPACT_BACKEND"] = backend_name
    try:
        h = hidden.clone().requires_grad_(True)
        b = None if boundary is None else boundary.clone().requires_grad_(True)
        out = c(h, cu_seqlens=cu, global_start=global_start, boundary_hidden=b)
        out.sum().backward()
        return out.detach(), h.grad, (None if b is None else b.grad)
    finally:
        if prev is None:
            os.environ.pop("PRIMUS_THD_COMPACT_BACKEND", None)
        else:
            os.environ["PRIMUS_THD_COMPACT_BACKEND"] = prev


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("with_boundary", [False, True])
def test_fused_matches_torch_path(ratio, with_boundary):
    from dsv4_cp_layout.backends import available

    if "torch_native" not in available():
        pytest.skip("no reference backend")

    C = 32
    # Ragged AND long enough to contain whole windows at ratio=128: with lengths under
    # the ratio no sequence has a single window, both paths return a constant zero, and
    # the comparison passes while testing nothing.
    lens = [333, 191, 277, 223]  # no length is a multiple of either ratio
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64)
    total = int(cu[-1])  # 1024
    global_start = total // 2 if with_boundary else 0  # rank 1 of cp=2
    l_local = total - global_start

    torch.manual_seed(1)
    hidden = torch.randn(1, l_local, C, dtype=torch.float64)
    d_comp = Compressor(hidden_size=C, head_dim=16, ratio=ratio).thd_d_comp
    boundary = torch.randn(1, d_comp, C, dtype=torch.float64) if with_boundary else None

    comp = Compressor(hidden_size=C, head_dim=16, ratio=ratio)
    _, comp_ids, seq_ids = comp.thd_compact_plan(cu, global_start, l_local)
    keep = comp_ids >= 0
    primus_set = set(zip(seq_ids[keep].tolist(), comp_ids[keep].tolist()))
    assert primus_set, f"ratio={ratio} owns no window here -- the comparison would be vacuous"
    kernel_set = set(_reference_groups(cu, global_start, l_local, ratio, comp.thd_d_comp))

    if kernel_set != primus_set:
        # Not interchangeable here, and that is not a bug in either one: the kernel uses
        # upstream's reach-back rule and also emits windows whose last row precedes
        # global_start, which upstream's seq_to_rank_row simply never addresses. Primus
        # concatenates the ranks' pools and uses the column number directly, so those
        # extra windows would become duplicate, doubly-attended keys. The ONLY acceptable
        # behaviour is a refusal -- silently pairing the kernel's rows with Primus's
        # comp_ids shifts every compressed RoPE phase.
        extra = kernel_set - primus_set
        assert extra and not (primus_set - kernel_set), (
            f"expected the kernel to be a strict superset (reach-back only), got "
            f"missing={sorted(primus_set - kernel_set)[:5]}"
        )
        cu_l = cu.tolist()
        for seq, cid in extra:
            last_row = cu_l[seq] + (cid + 1) * ratio - 1
            assert last_row < global_start, (
                f"extra window {(seq, cid)} ends at {last_row} >= global_start "
                f"{global_start}; the difference is not pure reach-back, so the two "
                f"enumerations disagree for some other reason that needs diagnosing"
            )
        with pytest.raises(RuntimeError, match="enumerates different windows"):
            _run("torch_native", ratio, cu, hidden, boundary, global_start)
        return

    o_t, gh_t, gb_t = _run("torch", ratio, cu, hidden, boundary, global_start)
    o_f, gh_f, gb_f = _run("torch_native", ratio, cu, hidden, boundary, global_start)

    torch.testing.assert_close(o_f, o_t, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        gh_f,
        gh_t,
        rtol=1e-9,
        atol=1e-9,
        msg=lambda m: f"hidden gradient differs (ratio={ratio}, boundary={with_boundary}):\n{m}",
    )
    if with_boundary:
        torch.testing.assert_close(
            gb_f,
            gb_t,
            rtol=1e-9,
            atol=1e-9,
            msg=lambda m: f"boundary gradient differs -- this is the neighbour's share of "
            f"the update, invisible in a single-rank run:\n{m}",
        )
