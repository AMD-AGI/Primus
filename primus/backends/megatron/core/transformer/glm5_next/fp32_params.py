###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hold a module's small gating parameters at FP32 through the bf16 model cast.

GLM-5.3 ships ``A_log`` / ``dt_bias`` (KDA decay gate), the mHC
``fn`` / ``base`` / ``scale`` and the indexer's ``index_kpool_compress_ape`` in
FP32, and every one of them feeds a sigmoid / softmax / exp directly.
``Float16Module`` casts the whole model with ``module.bfloat16()``; this mixin
re-casts the listed parameters after any ``_apply`` so they keep their full
resolution. Controlled by ``config.glm5_keep_fp32_params`` (default on).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

__all__ = ["KeepFp32ParamsMixin"]


class KeepFp32ParamsMixin:
    """Mix into an ``nn.Module``; set ``self._fp32_param_names`` in ``__init__``."""

    _fp32_param_names: Iterable[str] = ()

    def _apply(self, fn, *args, **kwargs):  # type: ignore[override]
        out = super()._apply(fn, *args, **kwargs)
        if not getattr(self, "_keep_fp32_enabled", True):
            return out
        for name in self._fp32_param_names:
            p = getattr(self, name, None)
            if isinstance(p, nn.Parameter) and p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.float()
        return out
