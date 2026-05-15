###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.cpp_extension import IS_HIP_EXTENSION
from transformer_engine.pytorch.utils import get_device_compute_capability


def check_fp8_block_scaling_support() -> Tuple[bool, str]:
    """Return if fp8 block scaling support is available"""
    if IS_HIP_EXTENSION:
        return True, ""
    if (
        get_device_compute_capability() >= (9, 0)
        and get_device_compute_capability() < (10, 0)
        and float(torch.version.cuda) >= 12.9
    ):
        return True, ""

    return False, "FP8 block scaled GEMM requires Hopper and CUDA >= 12.9."
