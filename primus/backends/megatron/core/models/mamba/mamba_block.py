# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from Megatron-LM release/v26.2 MambaStack implementation.
# This version uses allocate_layers() internally so pure-Mamba models
# work without requiring hybrid_layer_pattern in the config.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    make_viewless_tensor,
)
from torch import Tensor, nn

try:
    from megatron.core.transformer.enums import CudaGraphScope

    HAS_CUDA_GRAPH_SCOPE = True
except ImportError:
    CudaGraphScope = None
    HAS_CUDA_GRAPH_SCOPE = False


@dataclass
class MambaStackSubmodules:
    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp


class MambaStack(MegatronModule):
    """
    v26.2-compatible MambaStack that uses allocate_layers() internally.

    When MambaModel (v26.3) passes layer_type_list, it is used directly.
    When layer_type_list is empty or not provided, allocate_layers() generates
    a pure-Mamba layer list from config.num_layers.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaStackSubmodules = None,
        pre_process: bool = True,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        layer_type_list=None,
        pp_layer_offset=0,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process

        assert pg_collection is not None, "pg_collection must be provided for MambaStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp
        self.input_tensor = None

        if layer_type_list:
            self.layer_type_list = layer_type_list
        else:
            self.layer_type_list = [LayerSymbols.MAMBA] * self.config.num_layers

            pp_layer_offset = 0
            if self.pp_group.size() > 1:
                pp_layer_offset, self.layer_type_list = self._select_layers_for_pipeline_parallel(
                    self.layer_type_list
                )

        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(self.layer_type_list):
            fp8_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            with fp8_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        layer_number=i + 1 + pp_layer_offset,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.ATTENTION:
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=i + 1,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=i + 1,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=i + 1,
                        pg_collection=pg_collection,
                    )
                else:
                    assert False, f"unexpected layer_type: {layer_type}"
            self.layers.append(layer)

        self.num_layers_per_pipeline_rank = len(self.layers)

        if self.post_process and self.post_layer_norm:
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def _select_layers_for_pipeline_parallel(self, layer_type_list):
        num_layers_per_pipeline_rank = self.config.num_layers // self.pp_group.size()

        assert self.config.virtual_pipeline_model_parallel_size is None, (
            "The Mamba model does not currently support virtual/interleaved pipeline parallelism"
        )

        offset = self.pp_group.rank() * num_layers_per_pipeline_rank
        selected_list = layer_type_list[offset : offset + num_layers_per_pipeline_rank]

        return offset, selected_list

    def set_input_tensor(self, input_tensor: Tensor):
        self.input_tensor = input_tensor

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        for layer_type, layer in zip(self.layer_type_list, self.layers):
            if layer_type == LayerSymbols.MAMBA:
                return layer.mamba_state_shapes_per_request()
        return None

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask=None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if not self.pre_process:
            hidden_states = self.input_tensor

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if inference_context and inference_context.is_static_batching():
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            HAS_CUDA_GRAPH_SCOPE
            and (
                (
                    self.config.cuda_graph_impl == "local"
                    and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
                )
                or self.config.flash_decode
            )
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            current_batch_size = hidden_states.shape[1]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            sequence_len_offset = None

        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with outer_fp8_context:
            for layer in self.layers:
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                with inner_fp8_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                    else:
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                        )

                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return hidden_states

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Optional[tuple] = None,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict = {}
        layer_prefix = f"{prefix}layers."

        for local_layer_idx, layer in enumerate(self.layers):
            global_layer_offset = layer.layer_number - 1
            state_dict_prefix = f"{layer_prefix}{local_layer_idx}."
            sharded_prefix = f"{layer_prefix}{global_layer_offset}."
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        for name, module in self.named_children():
            if module is not self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        f"{prefix}{name}.",
                        sharded_offsets,
                        metadata,
                        tp_group=self.tp_group,
                    )
                )

        return sharded_state_dict
