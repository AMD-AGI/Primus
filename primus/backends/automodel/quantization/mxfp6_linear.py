###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP6-forward/MXFP4-backward ``nn.Linear`` for AutoModel training.

The implementation delegates both formats and GEMMs to Primus-Turbo.  Primus
owns only model eligibility, token padding, backward routing and the module
swap.  Bias remains in BF16 outside the low-precision GEMM.

Settings (module config, ``primus_turbo:`` section):
  mxfp6_linear: false          enable the swap
  mxfp6_keep_sensitive: true   keep conditioning projections in BF16
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch.nn as nn

from primus.backends.automodel.quantization import _common, _fp6_common, mxfp6_gemm

logger = logging.getLogger(__name__)

BACKEND_NAME = _fp6_common.BACKEND_NAME
_LOG_PREFIX = "[PrimusTurbo-MXFP6]"


class Float6Linear(nn.Linear):
    """Linear whose matrix multiplication runs through MXFP6 A6W6."""

    def __init__(self, in_features, out_features, bias=True, config=None, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        if config is None:
            config = (
                _fp6_common.resolve_fp6_config(),
                _fp6_common.resolve_mxfp4_backward_config(),
            )
        self.fp6_config, self.fp4_config = config

    def forward(self, x):
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        out = mxfp6_gemm.mxfp6_fwd_mxfp4_bwd(
            x_2d,
            self.weight,
            self.fp6_config,
            self.fp4_config,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.view(*leading_shape, self.out_features)

    def extra_repr(self):
        return f"{super().extra_repr()}, forward=mxfp6, backward=mxfp4"


def replace_linears(module, module_name: str, *, fp8_safe_only: bool = False) -> int:
    """Replace eligible AutoModel linears with :class:`Float6Linear`."""
    del fp8_safe_only  # signature compatibility with AutoModel's replacement seam
    config = (
        _fp6_common.resolve_fp6_config(),
        _fp6_common.resolve_mxfp4_backward_config(),
    )

    def factory(linear):
        return Float6Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            config=config,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

    converted, skipped = _common.replace_linears(
        module,
        module_name,
        factory=factory,
        should_convert=_fp6_common.is_fp6_training_safe_linear,
        already_converted=(Float6Linear,),
        log_prefix=_LOG_PREFIX,
    )
    logger.info(
        "%s replaced %d torch.nn.Linear with Float6Linear in %s; skipped=%d",
        _LOG_PREFIX,
        converted,
        module_name,
        skipped,
    )
    return converted


def install(model_config: Optional[Any] = None) -> bool:
    """Install the MXFP6 swap at AutoModel's existing linear-replacement seam."""
    _common.install_linear_swap(replace_linears, _LOG_PREFIX, model_config)
    return True
