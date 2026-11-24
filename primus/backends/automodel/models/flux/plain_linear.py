###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Which FLUX projections the fused ops may compute from their weights.

The fused ops read a projection's ``weight`` and ``bias`` and run the GEMM themselves, so they
may only stand in for a stock ``torch.nn.Linear``. The low-precision linear swaps replace
linears with subclasses whose forward is the low-precision GEMM; computing one of those from
its weight would skip that forward, with nothing in the log to say so.
"""

import torch


def is_plain_linear(m) -> bool:
    """A stock torch.nn.Linear with a bias and no hooks, which F.linear computes exactly."""
    return (
        type(m) is torch.nn.Linear
        and m.bias is not None
        and not m._forward_hooks
        and not m._forward_pre_hooks
    )
