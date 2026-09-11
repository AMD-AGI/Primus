###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FlyDSL Quantile Balancing backend, gfx950 / CDNA4 only.

Importing this package pulls in ``flydsl``, so it must only ever be reached
through :func:`...qb_kernels.load_flydsl_qb_backend`, never from module scope of
anything on the default import path.
"""

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1.histogram import (
    KERNEL_MAX_TOKENS,
    flydsl_compute_margin_histogram,
    inject_defect,
    kernel_beats_eager,
    supports_histogram_inputs,
)
from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1.qb_margin_histogram_kernel import (
    INJECTIONS,
    WIDTH_DEPENDENT_VARIANTS,
)

__all__ = [
    "flydsl_compute_margin_histogram",
    "supports_histogram_inputs",
    "kernel_beats_eager",
    "KERNEL_MAX_TOKENS",
    # test-only: proving the bit-exactness assertion has discrimination power
    "inject_defect",
    "INJECTIONS",
    "WIDTH_DEPENDENT_VARIANTS",
]
