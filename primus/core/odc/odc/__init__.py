# Adapted from ODC (https://github.com/sail-sg/odc), which is distributed under
# the MIT License per its package metadata (pyproject.toml / setup.py
# classifiers). The upstream repository ships no LICENSE file or per-file
# copyright headers; upstream copyright is held by the ODC authors (Sea AI Lab).
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# See LICENSE for license information.

import logging

from odc.primitives.gather import GatherService
from odc.primitives.scatter_accumulate import ReductionService
from odc.primitives.utils import SymmBufferRegistry, finalize_distributed, init_shmem

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


__all__ = [
    "init_shmem",
    "SymmBufferRegistry",
    "ReductionService",
    "GatherService",
    "finalize_distributed",
]
