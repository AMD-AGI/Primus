###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``quant_dual_t_e8m0_v2`` with ``tmode=2`` (8 warps) against ``tmode=0`` on dense-linear activation shapes:
FP8 row data, row scales, transposed data and transposed scales are byte-equal, including rows / columns of
zeros and an ``inf`` element.
"""

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

F8 = torch.float8_e4m3fn


@pytest.mark.parametrize(
    "M,N,row_layout,t_layout",
    [
        (1024, 4096, "packed", "packed"),
        (1024, 8192, "packed", "packed"),
        (1024, 2048, "packed", "packed"),
        (1024, 1024, "packed", "packed"),
        (1024, 4096, "g90", "packed"),
    ],
)
def test_quant_dual_tmode2_matches_tmode0(M, N, row_layout, t_layout):
    pytest.importorskip("primus_turbo")
    from primus_turbo.triton.quantization.quantization_fp8blk_dual_kernel import (
        quant_dual_t_e8m0_v2 as Q,
    )

    torch.manual_seed(N)
    x = (
        torch.randn(M, N, device="cuda") * torch.exp2(torch.randint(-8, 8, (M, 1), device="cuda").float())
    ).to(torch.bfloat16)
    x[3] = 0
    x[:, 7] = 0
    x[5, 9] = float("inf")
    a = Q(x, F8, row_layout=row_layout, t_layout=t_layout, tmode=0, num_warps=2)
    b = Q(x, F8, row_layout=row_layout, t_layout=t_layout, tmode=2, num_warps=8)
    for i in (0, 2, 3, 4):
        assert torch.equal(a[i].view(torch.uint8), b[i].view(torch.uint8)), f"output {i} differs"
