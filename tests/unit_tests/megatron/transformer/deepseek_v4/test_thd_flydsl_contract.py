###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Contract test: the FlyDSL compressor gather must agree with Primus's own.

Primus reached packed (THD) DeepSeek-V4 with a pure-PyTorch window gather
(``Compressor.thd_window_plan`` + an advanced index). The FlyDSL port of upstream's
CuTeDSL layout kernels does the same gather fused. Before swapping one for the other,
pin that they compute the SAME thing -- otherwise a later numerical difference in
training is impossible to attribute.

Both are "collect, for every compressed window, the ``ratio`` rows it pools, anchored at
the window's own sequence start". Primus expresses that as row indices; the kernel
writes a dense compact buffer. This compares the gathered rows directly.

Skipped when the dsv4_cp package is absent -- it is a separate deliverable, not a
dependency of Primus.
"""

import os
import sys

import pytest
import torch

# Out-of-tree package: not in the repo and not in the image. Point PRIMUS_DSV4_CP_DIR
# at a checkout to enable these tests; they skip cleanly when it is absent.
_PKG = os.environ.get("PRIMUS_DSV4_CP_DIR", "")
pytestmark = pytest.mark.skipif(not os.path.isdir(_PKG), reason="dsv4_cp package not present")
if os.path.isdir(_PKG) and _PKG not in sys.path:
    sys.path.insert(0, _PKG)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)


def _primus_gather(hidden, cu, ratio):
    """Rows each window pools, via Primus's plan: [N, ratio, C]."""
    c = Compressor(hidden_size=hidden.shape[-1], head_dim=hidden.shape[-1], ratio=ratio)
    row_idx, _, cu_pool = c.thd_window_plan(cu)
    if row_idx is None:
        return None, cu_pool
    return hidden[row_idx.reshape(-1)].reshape(row_idx.shape[0], ratio, -1), cu_pool


@pytest.mark.parametrize("ratio", [4, 8])
@pytest.mark.parametrize("lens", [(8, 12, 8), (16, 8, 8)])
def test_flydsl_gather_matches_primus(ratio, lens):
    from dsv4_cp_layout.backends import available, get

    if "torch_native" not in available():
        pytest.skip("no reference backend")
    backend = get("torch_native")  # same contract as flydsl, runs on CPU

    C = 16
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    total = int(cu[-1])
    torch.manual_seed(0)
    hidden = torch.randn(total, C)

    mine, cu_pool = _primus_gather(hidden, cu, ratio)
    n_win = int(cu_pool[-1])
    if n_win == 0:
        pytest.skip("no whole windows at this ratio/lengths")

    # No CP here: the whole pack is local, so there is no boundary region.
    d_window = ratio
    out = backend.compressor_input_compact_fwd(
        hidden,
        hidden.new_zeros(d_window, C),
        cu,
        0,  # global_start
        total,  # l_local
        ratio,
        n_win,  # d_comp: number of compressed groups
        d_window,
        n_win * ratio,  # compact_len
        C,  # row_width
    )
    theirs = (out[0] if isinstance(out, (tuple, list)) else out)[: n_win * ratio]
    theirs = theirs.reshape(n_win, ratio, C)

    torch.testing.assert_close(
        theirs,
        mine,
        rtol=0,
        atol=0,
        msg=lambda m: f"FlyDSL/native gather disagrees with Primus (ratio={ratio}, lens={lens}):\n{m}",
    )
