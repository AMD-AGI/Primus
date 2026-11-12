###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FLUX Linear implementations selected before graph compilation."""

import os

_dispatch = (os.getenv("FLUX_FP4_DISPATCH") or "fusion").lower()
_fused = _dispatch in ("fusion", "host_dispatch", "mxfp4_mm") or any(
    os.getenv(name, "0") == "1"
    for name in ("FLUX_FP4_FUSED_H16_QUANT", "FLUX_FP4_HOST_DISPATCH", "FLUX_FP4_MXFP4_MM")
)
if _fused and os.getenv("FLUX_FP4_H16_STOCK", "0") != "1":
    from . import mxfp4_linear_dual as _implementation
else:
    from . import mxfp4_linear_stock as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
