###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Which FLUX projections and norms the fused ops may compute from their parameters.

The fused ops read a projection's ``weight`` and ``bias`` and run the GEMM themselves, so they
may only stand in for a stock ``torch.nn.Linear``. The low-precision linear swaps replace
linears with subclasses whose forward is the low-precision GEMM; computing one of those from
its weight would skip that forward, with nothing in the log to say so. Likewise for a norm the
ops compute from its ``eps``, and for any module whose call they skip: its hooks would not run.
"""

import torch


def has_no_hooks(m) -> bool:
    return not m._forward_hooks and not m._forward_pre_hooks


def is_plain_linear(m) -> bool:
    """A stock torch.nn.Linear with a bias and no hooks, which F.linear computes exactly."""
    return type(m) is torch.nn.Linear and m.bias is not None and has_no_hooks(m)


def is_plain_layer_norm(m) -> bool:
    """A stock torch.nn.LayerNorm over the last dimension, without affine parameters or hooks."""
    return (
        type(m) is torch.nn.LayerNorm
        and m.weight is None
        and m.bias is None
        and len(m.normalized_shape) == 1
        and has_no_hooks(m)
    )
