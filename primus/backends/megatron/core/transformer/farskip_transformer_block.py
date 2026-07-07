###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from contextlib import nullcontext

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor


def forward(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,
    attention_bias=None,
    inference_context=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    *,
    inference_params=None,
    dynamic_inference_decode_only=None,
):
    inference_context = deprecate_inference_params(inference_context, inference_params)

    # Delete the obsolete reference to the initial input tensor if necessary
    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()

    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        from megatron.core.fp8_utils import get_fp8_context
        from megatron.core.transformer.transformer_config import Fp8Recipe

        use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
        outer_quantization_context = (
            get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
        )
    elif self.config.fp4:
        from megatron.core.fp4_utils import get_fp4_context

        use_outer_quantization_context = False
        use_inner_quantization_context = True
        outer_quantization_context = nullcontext()
    else:
        use_outer_quantization_context = False
        use_inner_quantization_context = False
        outer_quantization_context = nullcontext()

    with rng_context, outer_quantization_context:
        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states, farskip_state = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_quantization_context=use_inner_quantization_context,
                farskip_state={},
            )
        else:
            farskip_state = {}
            for l_no, layer in enumerate(self.layers):
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                with self.offload_context, inner_quantization_context:

                    layer_output = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        rotary_pos_cos_sin=rotary_pos_cos_sin,
                        attention_bias=attention_bias,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        **farskip_state,
                    )
                    if len(layer_output) == 3:
                        hidden_states, context, farskip_state = layer_output
                    else:
                        hidden_states, context = layer_output
                        farskip_state = {}

                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

            # handle synchronization in overlapped settings
            if farskip_state:
                if 'combine_handle' in farskip_state and farskip_state['combine_handle'] is not None:
                    farskip_state['combine_handle'].wait()
                if 'combined_experts' in farskip_state and farskip_state['combined_experts'] is not None:
                    from megatron.core.transformer.moe.moe_utils import unpermute
                    unpermute_inputs = farskip_state.pop('dispatcher_state', None)
                    reversed_local_input_permutation_mapping = unpermute_inputs.pop(
                        'reversed_local_input_permutation_mapping')
                    combined_experts = unpermute(
                        farskip_state['combined_experts'],
                        reversed_local_input_permutation_mapping, **unpermute_inputs
                    )
                    combined_experts = combined_experts.view(hidden_states.shape)
                    if 'last_residual_with_attention' in farskip_state and 'shared_expert' in farskip_state:
                        if os.environ.get('VERBOSE_DEBUG_PRINT', '0') == '1':
                            print('Adding farskip using the same associative add order (a +(b+c)) as simple and regular')
                        hidden_states = farskip_state['last_residual_with_attention'] + (combined_experts + farskip_state['shared_expert'])
                    else:
                        hidden_states = hidden_states + combined_experts

    # Final layer norm.
    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

    if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
        hidden_states = hidden_states.clone()

    return hidden_states


def _checkpointed_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor,
    context_mask: Tensor,
    rotary_pos_emb: Tensor,
    attention_bias: Tensor,
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    farskip_state: dict = None,
):
    """Forward method with activation checkpointing that threads farskip state
    across the checkpoint boundary.

    Farskip layers return a 3-tuple (hidden_states, context, farskip_state). To
    survive activation recomputation, the tensor-valued entries of farskip_state
    are flattened into positional inputs/outputs of the checkpointed function and
    rebuilt into a dict on the other side; non-tensor entries are captured by
    closure.
    """

    if farskip_state is None:
        farskip_state = {}

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
            farskip_state_dict=None, farskip_tensor_keys_for_output=None
        ):
            local_farskip_state = farskip_state_dict.copy() if farskip_state_dict else {}

            for index in range(start, end):
                layer = self._get_layer(index)

                if use_inner_quantization_context:
                    if self.config.fp8:
                        from megatron.core.fp8_utils import get_fp8_context

                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    elif self.config.fp4:
                        from megatron.core.fp4_utils import get_fp4_context

                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                with inner_quantization_context:
                    layer_output = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        inference_context=None,
                        packed_seq_params=packed_seq_params,
                        **local_farskip_state,
                    )

                    if len(layer_output) == 3:
                        hidden_states, context, local_farskip_state = layer_output
                    else:
                        hidden_states, context = layer_output
                        local_farskip_state = {}

            # Return flattened tensor outputs for checkpoint
            if farskip_tensor_keys_for_output and local_farskip_state:
                farskip_output_tensors = tuple(
                    local_farskip_state.get(k) for k in farskip_tensor_keys_for_output
                )
                return (hidden_states, context) + farskip_output_tensors
            else:
                return hidden_states, context, local_farskip_state

        return custom_forward

    def checkpoint_handler(forward_func):
        farskip_tensor_keys = []
        farskip_tensors = []
        farskip_non_tensors = {}

        if farskip_state:
            for k, v in sorted(farskip_state.items()):
                if torch.is_tensor(v):
                    farskip_tensor_keys.append(k)
                    farskip_tensors.append(v)
                else:
                    farskip_non_tensors[k] = v

        def wrapped_forward(hidden_states, attention_mask, context, context_mask,
                           rotary_pos_emb, *farskip_tensors_tuple):
            if farskip_tensor_keys:
                rebuilt_farskip_state = dict(zip(farskip_tensor_keys, farskip_tensors_tuple))
                rebuilt_farskip_state.update(farskip_non_tensors)
            else:
                rebuilt_farskip_state = None

            return forward_func(hidden_states, attention_mask, context, context_mask,
                              rotary_pos_emb, rebuilt_farskip_state, farskip_tensor_keys)

        if self.config.fp8 or self.config.fp4:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            checkpoint_output = te_checkpoint(
                wrapped_forward,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                self.pg_collection.tp,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                *farskip_tensors,
            )
        else:
            checkpoint_output = tensor_parallel.checkpoint(
                wrapped_forward,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                *farskip_tensors,
            )

        # Reconstruct farskip_state dict from flattened tensor outputs
        if len(checkpoint_output) > 2 and farskip_tensor_keys:
            hidden_states_out = checkpoint_output[0]
            context_out = checkpoint_output[1]
            farskip_tensor_values = checkpoint_output[2:]
            farskip_state_out = dict(zip(farskip_tensor_keys, farskip_tensor_values))
            farskip_state_out.update(farskip_non_tensors)
            return hidden_states_out, context_out, farskip_state_out
        else:
            return checkpoint_output

    if self.config.recompute_method == 'uniform':
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            checkpoint_output = checkpoint_handler(
                custom(layer_idx, layer_idx + self.config.recompute_num_layers)
            )
            if len(checkpoint_output) == 3:
                hidden_states, context, farskip_state = checkpoint_output
            else:
                hidden_states, context = checkpoint_output
                farskip_state = {}

            layer_idx += self.config.recompute_num_layers

    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            # Skip recomputation when input grad computation is not needed.
            # Need to have at least one input tensor with gradient computation
            # for re-enterant autograd engine.
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            if (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            ):
                checkpoint_output = checkpoint_handler(
                    custom(layer_idx, layer_idx + 1)
                )
                if len(checkpoint_output) == 3:
                    hidden_states, context, farskip_state = checkpoint_output
                else:
                    hidden_states, context = checkpoint_output
                    farskip_state = {}
            else:
                layer_output = custom(layer_idx, layer_idx + 1)(
                    hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
                    farskip_state
                )
                if len(layer_output) == 3:
                    hidden_states, context, farskip_state = layer_output
                else:
                    hidden_states, context = layer_output
                    farskip_state = {}
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states, farskip_state