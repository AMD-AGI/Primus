###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import enum


class Fp4Recipe(str, enum.Enum):
    """FP4 recipe names: nvfp4."""

    nvfp4 = "nvfp4"
    mxfp4 = "mxfp4"
