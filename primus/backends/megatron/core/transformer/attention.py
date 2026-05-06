###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""

Primus Lfm2ShortConv

Primus implementation of Megatron Lfm2ShortConv.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from torch import Tensor, nn
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_torchdynamo_compiling,
)

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    print(f"Unable to import causal_conv1d!!!")
    causal_conv1d_fn, causal_conv1d_update = None, None

kernel_modules = (causal_conv1d_fn, causal_conv1d_update)
is_fast_path_available = all(kernel_modules)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None:
        # hidden_states: [batch, seq, hidden_size]
        # support attention_mask as:
        #   1) [batch, seq]                      (True/1 means keep token)
        #   2) [batch, 1, seq, seq]             (Megatron style, True means masked)
        if attention_mask.dim() == 2:
            token_keep_mask = attention_mask
        elif attention_mask.dim() == 4:
            assert attention_mask.shape[0] == hidden_states.shape[0], (
                f"Batch mismatch between hidden_states {hidden_states.shape} "
                f"and attention_mask {attention_mask.shape}"
            )
            assert (
                attention_mask.shape[1] == 1
            ), f"Expected attention_mask.shape[1] == 1 for 4D mask, got {attention_mask.shape}"
            assert attention_mask.shape[2] == hidden_states.shape[1], (
                f"Seq mismatch between hidden_states {hidden_states.shape} "
                f"and attention_mask {attention_mask.shape}"
            )
            # Megatron 4D mask uses True for masked positions. If an entire query row
            # is masked, that token is padding and should be zeroed.
            token_keep_mask = ~attention_mask[:, 0].all(dim=-1)
        else:
            raise ValueError(
                f"Unsupported attention_mask shape: {attention_mask.shape}. "
                "Expected [batch, seq] or [batch, 1, seq, seq]."
            )

        dtype = hidden_states.dtype
        hidden_states = (hidden_states * token_keep_mask[:, :, None]).to(dtype)

    return hidden_states


@dataclass
class Lfm2ShortConvSubmodules:
    """
    Configuration class for specifying the submodules of a Lfm2ShortConv.
    """

    in_proj: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class Lfm2ShortConv(MegatronModule, ABC):
    """Lfm2ShortConv layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[Lfm2ShortConvSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_idx = layer_number - 1

        args = get_args()
        self.L_cache = args.conv_L_cache
        self.bias = args.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )

        if self.config.perform_initialization:
            self.config.init_method(self.conv.weight)

        if self.conv.bias is not None:
            with torch.no_grad():
                self.conv.bias.zero_()

        # self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        # self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

        # In projection.
        self.in_proj = build_module(
            submodules.in_proj,
            self.config.hidden_size,
            3 * self.config.hidden_size,
            parallel_mode=None,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.bias,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )

        # Out projection.
        self.out_proj = build_module(
            submodules.out_proj,
            self.config.hidden_size,
            self.config.hidden_size,
            parallel_mode=None,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.bias,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )

        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

    def cuda_kernels_forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x)[0].transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        conv_out = causal_conv1d_fn(Bx, conv_weights, self.conv.bias, activation=None)

        y = C * conv_out
        y = self.out_proj(y.transpose(-1, -2).contiguous())[0]
        return y

    def slow_forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        # x: [batch, seq, hidden_size]
        # attention_mask: [batch, 1, seq, seq]
        seqlen = x.shape[1]
        x = apply_mask_to_padding_states(x, attention_mask)

        # BCx: [batch, 3 * hidden_size, seq]
        BCx = self.in_proj(x)[0].transpose(-1, -2)

        # B: [batch, hidden_size, seq]
        # C: [batch, hidden_size, seq]
        # x: [batch, hidden_size, seq]
        B, C, x = BCx.chunk(3, dim=-2)

        # Bx: [batch, hidden_size, seq]
        Bx = B * x

        conv_out = self.conv(Bx)[..., :seqlen]

        # y: [batch, hidden_size, seq]
        y = C * conv_out

        # y: [batch, seq, hidden_size]
        y = y.transpose(-1, -2).contiguous()

        # y: [batch, seq, hidden_size]
        y = self.out_proj(y)[0]

        return y

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[Tensor, Tensor]:

        # print(f"layer_idx: {self.layer_idx}")
        # hidden_states.shape: [seq, batch, hidden_size]
        # attention_mask: [batch, 1, seq, seq]
        hidden_states = hidden_states.transpose(0, 1)

        if is_fast_path_available and "cuda" in hidden_states.device.type and not is_torchdynamo_compiling():
            out = self.cuda_kernels_forward(hidden_states, attention_mask)
        else:
            out = self.slow_forward(hidden_states, attention_mask)
        return out.transpose(0, 1), None
