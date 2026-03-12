# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params


@dataclass
class KimiDeltaAttentionLayerSubmodules:
    """Configuration class for specifying the submodules of a KDA layer.

    No separate norm is needed here because the LayerNorm is fused into the
    mixer's in_proj (TELayerNormColumnParallelLinear), matching the GDN
    layer pattern.
    """

    mixer: Union[ModuleSpec, type] = IdentityOp
    kda_bda: Union[ModuleSpec, type] = IdentityOp

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class KimiDeltaAttentionLayer(MegatronModule):
    """A single Kimi Delta Attention layer wrapping the KDA mixer.

    Analogous to GatedDeltaNetLayer. The pre-normalization is handled
    inside the mixer via the fused TELayerNormColumnParallelLinear in_proj,
    so this wrapper only manages residual + bias-dropout-add.

    The forward interface matches what HybridStack expects for the
    mamba_layer slot.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32: bool = False,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for KimiDeltaAttentionLayer"

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout

        self.mixer = build_module(
            submodules.mixer,
            self.config,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )
        self.kda_bda = build_module(submodules.kda_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """Forward pass through the KDA layer.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h].
            attention_mask (Tensor, optional): Attention mask forwarded to the mixer.
            inference_context (BaseInferenceContext, optional): Inference context.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings (unused).

        Returns:
            Tensor: Transformed hidden states of shape [s, b, h].
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)

        mixer_out_with_bias = self.mixer(
            hidden_states, attention_mask, inference_context=inference_context
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.kda_bda(
                training=self.training, fused=self.config.bias_dropout_fusion
            )(mixer_out_with_bias, residual, self.hidden_dropout)

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
