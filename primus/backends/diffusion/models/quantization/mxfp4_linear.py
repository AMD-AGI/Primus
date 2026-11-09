###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FLUX Linear implementations selected before graph compilation."""

import os

if os.getenv("FLUX_FP4_H16_STOCK", "0") != "1":
    from . import mxfp4_linear_dual as _implementation
else:
    from . import mxfp4_linear_stock as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
