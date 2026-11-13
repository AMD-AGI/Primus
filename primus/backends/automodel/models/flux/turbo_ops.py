###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo's fused FLUX ops (``primus_turbo.pytorch.ops.flux``), which the FLUX patches call."""

import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)


def flux_ops(tag: str, *, needs_gfx950: bool = False) -> Optional[ModuleType]:
    """The ops module, or None after logging why the ``tag`` patch is not applied: this Primus-Turbo has
    no FLUX ops, or the patch's ops are FlyDSL-only (``needs_gfx950``) and this GPU is not gfx950."""
    try:
        from primus_turbo.pytorch.core.utils import is_gfx950
        from primus_turbo.pytorch.ops import flux
    except ImportError as exc:
        logger.warning("[%s] Primus-Turbo has no FLUX ops (%s); not applied.", tag, exc)
        return None
    if needs_gfx950 and not is_gfx950():
        logger.warning("[%s] Primus-Turbo's FLUX adaLN ops need gfx950; not applied.", tag)
        return None
    return flux
