###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The Primus-Turbo ops the FLUX patches call: the fused FLUX ops (``primus_turbo.pytorch.ops.flux``) and
the DiT q/k norm + RoPE (``primus_turbo.pytorch.ops.dit_qk_norm_rope``)."""

import logging
from types import ModuleType
from typing import Callable, Optional

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


def dit_qk_norm_rope(tag: str) -> Optional[Callable]:
    """Primus-Turbo's ``dit_qk_norm_rope``, or None after logging that this Primus-Turbo has none and the
    ``tag`` patch is not applied."""
    try:
        from primus_turbo.pytorch.ops.dit_qk_norm_rope import dit_qk_norm_rope as op
    except ImportError as exc:
        logger.warning("[%s] Primus-Turbo has no dit_qk_norm_rope (%s); not applied.", tag, exc)
        return None
    return op
