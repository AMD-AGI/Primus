###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FSDP2 dtype-island opt-in attribute (a leaf module: no imports).

A module class sets ``fsdp2_dtype_island = True`` when its direct parameters must keep
their own dtype under FSDP2 (e.g. Muse Spark's fp32 router weight and residual gates).
Shared by the FSDP2 wrapper (``torch_fully_sharded_data_parallel``), which shards
islands as their own groups, and the FP8 all-gather path (``fsdp2_fp8_all_gather``),
which leaves them unwrapped.
"""

__all__ = ["ISLAND_ATTR"]

ISLAND_ATTR = "fsdp2_dtype_island"
