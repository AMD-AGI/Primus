###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compatibility shim — Triton RMSNorm has moved to Primus-Turbo.

The kernels and autograd wrappers are now maintained at
``primus_turbo.pytorch.ops.triton_rmsnorm``. This module re-exports the
public API for backward compatibility with existing Primus call sites
(notably ``primus_turbo.PrimusTurboRMSNorm``).
"""
from primus_turbo.pytorch.ops.triton_rmsnorm import (  # noqa: F401
    TritonRMSNormFn,
    TritonRMSNormResidualFn,
    triton_rmsnorm,
    triton_rmsnorm_residual,
)

__all__ = [
    "TritonRMSNormFn",
    "TritonRMSNormResidualFn",
    "triton_rmsnorm",
    "triton_rmsnorm_residual",
]
