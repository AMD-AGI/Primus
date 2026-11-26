###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The local GPU's architecture, for paths tuned and validated on one architecture only."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def gfx_arch() -> str:
    """The current GPU's architecture without its feature flags (``"gfx1250"`` for
    ``"gfx1250:sramecc+:xnack-"``), or ``""`` without a GPU. Never raises."""
    try:
        if not torch.cuda.is_available():
            return ""
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return getattr(props, "gcnArchName", "").split(":")[0]
    except Exception:  # noqa: BLE001 - an arch probe must never abort a run
        return ""


def is_gfx1250() -> bool:
    """Whether the current GPU is a gfx1250 (MI455X). False without a GPU; never raises."""
    return gfx_arch() == "gfx1250"
