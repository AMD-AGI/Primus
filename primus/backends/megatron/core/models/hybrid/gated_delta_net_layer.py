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
class GatedDeltaNetLayerSubmodules:
    """
    Configuration class for specifying the submodules of a GatedDeltaNet layer.

    Unlike MambaLayerSubmodules, this does not pass d_model to the mixer and
    forwards attention_mask through to the GatedDeltaNet mixer.
    """

    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    gdn_bda: Union[ModuleSpec, type] = IdentityOp

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class GatedDeltaNetLayer(MegatronModule):
    """
    A single GatedDeltaNet layer wrapping the GatedDeltaNet mixer.

    This is analogous to MambaLayer but avoids the interface mismatches:
    - Does not pass d_model to the mixer (GatedDeltaNet reads hidden_size from config)
    - Forwards attention_mask to the mixer
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32: bool = False,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNetLayer"

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
        self.norm = build_module(submodules.norm, self.config, self.config.hidden_size)
        self.gdn_bda = build_module(submodules.gdn_bda)
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
        """
        Forward pass through the GatedDeltaNet layer.

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
        hidden_states = self.norm(hidden_states)

        mixer_out_with_bias = self.mixer(
            hidden_states, attention_mask, inference_context=inference_context
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.gdn_bda(
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
