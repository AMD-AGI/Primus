###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager Quantile Balancing statistic: the permanent numerical oracle."""

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._eager.reference import (
    compute_margin_histogram,
    margin_cutoff,
)

__all__ = ["compute_margin_histogram", "margin_cutoff"]
