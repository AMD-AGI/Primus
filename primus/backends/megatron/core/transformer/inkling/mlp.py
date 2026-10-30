###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling's dense SwiGLU MLP: Megatron's :class:`MLP` times a learned scalar.

``InklingMLP`` in ``modeling_inkling.py`` multiplies the output by
``global_scale`` (one element, initialised to 1). With
``inkling_fp32_activations`` the SwiGLU and the scale run in fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.transformer.mlp import MLP

from primus.backends.megatron.core.transformer.keep_in_fp32 import (
    KeepInFp32Mixin,
    is_marked_keep_in_fp32,
    mark_keep_in_fp32,
)

__all__ = ["InklingMLP", "fp32_swiglu", "inkling_global_scale"]


def fp32_swiglu(x: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
    """``silu(gate) * up [* scale]`` in fp32 on ``x = [gate | up]``, returned in ``x.dtype``."""
    gate, up = x.float().chunk(2, dim=-1)
    out = F.silu(gate) * up
    if scale is not None:
        out = out * scale.float()
    return out.to(x.dtype)


def inkling_global_scale(config) -> torch.nn.Parameter:
    """``global_scale`` (ones). Stored in fp32 under ``inkling_fp32_activations``
    when the ``keep_in_fp32`` contract is enabled; the owner must mix in
    :class:`KeepInFp32Mixin`."""
    scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    if config.inkling_fp32_activations:
        mark_keep_in_fp32(scale)
    if not is_marked_keep_in_fp32(scale):
        scale.data = scale.data.to(config.params_dtype)
    setattr(scale, "sequence_parallel", config.sequence_parallel)
    return scale


class InklingMLP(KeepInFp32Mixin, MLP):
    def __init__(self, config, submodules, **kwargs):
        if kwargs.get("ffn_hidden_size") is None:
            kwargs["ffn_hidden_size"] = config.ffn_hidden_size
        super().__init__(config=config, submodules=submodules, **kwargs)
        self.global_scale = inkling_global_scale(config)

    def forward(self, hidden_states, per_token_scale=None, **kwargs):
        if not self.config.inkling_fp32_activations:
            output, bias = super().forward(hidden_states, per_token_scale, **kwargs)
            return output * self.global_scale, bias
        if per_token_scale is not None:
            raise NotImplementedError("inkling_fp32_activations does not support per_token_scale")
        intermediate, bias = self.linear_fc1(hidden_states)
        if bias is not None:
            raise ValueError("Inkling's MLP has no biases")
        output, bias = self.linear_fc2(fp32_swiglu(intermediate))
        return (output.float() * self.global_scale.float()).to(output.dtype), bias
