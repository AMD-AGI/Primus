###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tiny RMSNorm fallback used by DeepSeek-V4 modules.

Plan-2 P17 introduces this shared helper to retire three nearly
identical ``_RMSNorm`` implementations that lived in:

* ``primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_block.py``
* ``primus/backends/megatron/core/transformer/deepseek_v4_attention.py``
  (a closure-built fallback for the attention's ``q_norm`` / ``kv_norm``
  no-spec path)
* ``primus/backends/megatron/core/transformer/compressor.py``

All three computed the same RMSNorm with a learnable per-channel
``weight`` and worked on the last dim. The goal of this module is to
expose **one** implementation so:

* dead-code audits stay clean,
* state-dict round-trips treat the four call sites uniformly,
* CPU-only unit tests can build V4 norms without dragging in
  TransformerEngine / Megatron's TE-backed norm kernel.

The spec-driven path (``DeepSeekV4SpecProvider.v4_norm_module()``) is
unchanged — it returns Megatron's TE-backed RMSNorm or the local
fallback, depending on the active runtime mode. This file is the local
fallback only.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class LocalRMSNorm(nn.Module):
    """Single canonical RMSNorm fallback used across V4 modules.

    Args:
        dim: hidden dimension to normalize over (last axis).
        eps: numerical stability epsilon.
        hidden_size: alias for ``dim``; if both are provided ``dim``
            wins. Provided for compatibility with Megatron-style norm
            constructors that expect ``hidden_size=`` instead.
        config: ignored (consumed for compatibility with Megatron's norm
            factory signature).

    Notes:
        * The internal compute happens in fp32 to match the V4 reference
          (``inference/model.py`` uses fp32 RMS even when activations
          are bf16); the result is cast back to the input dtype.
        * The ``weight`` parameter has shape ``(dim,)`` and is initialized
          to ones so a freshly built norm is the identity transform —
          this matches the V4 / HF reference checkpoint layout exactly,
          so state-dict round-trips work without remapping.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        eps: float = 1e-6,
        *,
        hidden_size: Optional[int] = None,
        config: Optional[Any] = None,
    ) -> None:
        del config
        super().__init__()
        if dim is None:
            dim = hidden_size
        if dim is None:
            raise ValueError("LocalRMSNorm requires `dim` (or the alias `hidden_size`).")
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rsqrt = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rsqrt).to(in_dtype) * self.weight


__all__ = ["LocalRMSNorm"]
