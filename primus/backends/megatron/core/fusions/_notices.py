###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Log helpers shared by the opt-in Primus fusion and optimizer-step patches."""

import functools

from primus.core.utils.module_utils import warning_rank_0


@functools.lru_cache(maxsize=None)
def warn_fallback_once(key: str, reason: str) -> None:
    """Log once per (patch key, reason) that a call took the original path."""
    warning_rank_0(f"[Patch:{key}] original path: {reason}")


@functools.lru_cache(maxsize=None)
def _gpu_arch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001 - no device query, no notice
        pass
    return ""


def warn_if_not_gfx1250(key: str, arg: str) -> None:
    """Warn (not refuse) when ``arg`` is enabled on another GPU: correct there, but not tuned or validated."""
    arch = _gpu_arch()
    if arch and arch != "gfx1250":
        warning_rank_0(
            f"[Patch:{key}] {arg} is tuned and validated on gfx1250 (MI455X) only; this GPU is {arch}"
        )
