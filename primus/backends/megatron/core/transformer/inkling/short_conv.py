###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling's depthwise causal short convolution with a residual.

``y = x + conv1d_causal(x)`` per channel, computed in fp32
(``InklingShortConvolution`` in ``modeling_inkling.py``). Inkling applies
it to K and V inside attention, and to the attention and MLP outputs before
each residual add.

The weight keeps the checkpoint's ``nn.Conv1d`` layout ``[channels, 1,
kernel]`` so the importer copies it unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import get_tensor_model_parallel_group_if_none

__all__ = ["InklingShortConv"]


class InklingShortConv(MegatronModule):
    """Depthwise causal conv + residual over the last dim of ``[s, b, c]``.

    Args:
        config: transformer config (for ``params_dtype``).
        channels: channels held on this rank.
        kernel_size: conv taps.
        tensor_parallel: the channels are a TP shard (K/V convs); the
            attention/MLP output convs act on the full hidden size and are
            replicated.
        tp_group: the tensor-parallel group; Megatron's global one if ``None``.
    """

    def __init__(self, config, channels: int, kernel_size: int, tensor_parallel: bool = False, tp_group=None):
        super().__init__(config=config)
        self.channels = channels
        self.kernel_size = kernel_size
        self.tensor_parallel = tensor_parallel
        self.tp_group = tp_group
        self.weight = torch.nn.Parameter(torch.empty(channels, 1, kernel_size, dtype=config.params_dtype))
        if tensor_parallel:
            self.weight.tensor_model_parallel = True
            self.weight.partition_dim = 0
            self.weight.partition_stride = 1
        else:
            setattr(self.weight, "sequence_parallel", config.sequence_parallel)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight.device.type != "meta" and self.config.perform_initialization:
            self.config.init_method(self.weight)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """K/V convs are sharded on the channel axis like ``linear_qkv``'s groups; the rest are replicated."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0} if self.tensor_parallel else {},
            sharded_offsets,
            tp_group=get_tensor_model_parallel_group_if_none(self.tp_group),
            dp_cp_group=metadata["dp_cp_group"],
        )

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: ``[s, b, c]``. Packed sequences (``seq_lengths``) are optional item O8.

        The conv runs over the whole row and ignores the attention mask, so
        padded rows and per-document masking (``reset_attention_mask``) are
        unsupported; the builder rejects the latter.
        """
        if seq_lengths is not None:
            raise NotImplementedError("Packed-sequence short convolution is optional item O8")
        in_dtype = x.dtype
        xf = x.float()
        h = xf.permute(1, 2, 0)  # [b, c, s]
        out = F.conv1d(h, self.weight.float(), padding=self.kernel_size - 1, groups=self.channels)
        out = out[..., : h.shape[-1]].permute(2, 0, 1)
        return (out + xf).to(in_dtype)
