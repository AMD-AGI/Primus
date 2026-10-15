###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FLUX Linear implementations selected before graph compilation."""

import os


def _resolve_fp4_dispatch() -> str:
    explicit = os.getenv("FLUX_FP4_DISPATCH")
    if explicit is not None and explicit.strip():
        dispatch = explicit.strip().lower()
        if dispatch not in ("fusion", "host_dispatch", "mxfp4_mm"):
            raise ValueError(
                f"FLUX_FP4_DISPATCH={dispatch!r} is invalid; expected fusion, "
                "host_dispatch, mxfp4_mm, or empty"
            )
        return dispatch
    if os.getenv("FLUX_FP4_MXFP4_MM", "0") == "1":
        return "mxfp4_mm"
    if os.getenv("FLUX_FP4_HOST_DISPATCH", "0") == "1":
        return "host_dispatch"
    return "fusion"


_dispatch = _resolve_fp4_dispatch()
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
