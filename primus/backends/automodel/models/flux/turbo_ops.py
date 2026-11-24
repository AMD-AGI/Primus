###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The Primus-Turbo ops the FLUX patches call: ``adaln_modulate``, ``gated_residual_adaln`` and
``linear_gelu`` (DiT adaLN and linear + GELU) and ``dit_qk_norm_rope`` (q/k RMSNorm + RoPE), all from
``primus_turbo.pytorch.ops``."""

import logging
from types import ModuleType
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def turbo_ops(tag: str, *names: str, gain_on_gfx950_only: bool = False) -> Optional[ModuleType]:
    """``primus_turbo.pytorch.ops`` if it has every one of ``names``, else None after logging why the
    ``tag`` patch is not applied. With ``gain_on_gfx950_only`` the patch also declines on other GPUs: the ops
    run there too (eager PyTorch arithmetic) but are faster than the unfused path only on gfx950."""
    try:
        import primus_turbo.pytorch.ops as ops
        from primus_turbo.pytorch.core.utils import is_gfx950
    except ImportError as exc:
        logger.warning("[%s] Primus-Turbo is not importable (%s); not applied.", tag, exc)
        return None
    missing = [name for name in names if not hasattr(ops, name)]
    if missing:
        logger.warning("[%s] this Primus-Turbo has no %s; not applied.", tag, ", ".join(missing))
        return None
    if gain_on_gfx950_only and not is_gfx950():
        logger.warning("[%s] Primus-Turbo's %s gain only on gfx950; not applied.", tag, ", ".join(names))
        return None
    return ops


def dit_qk_norm_rope(tag: str) -> Optional[Callable]:
    """Primus-Turbo's ``dit_qk_norm_rope``, or None after logging that this Primus-Turbo has none and the
    ``tag`` patch is not applied."""
    ops = turbo_ops(tag, "dit_qk_norm_rope")
    return None if ops is None else ops.dit_qk_norm_rope
