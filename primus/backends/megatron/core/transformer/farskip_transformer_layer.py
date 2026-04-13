###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import (
    deprecate_inference_params,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.training.global_vars import get_args


# Backward-compatible additions in TransformerLayer to support FarSkip in derived classes
# backward-compatible additional arg in _forward_mlp, and the split attention forward functions (_forward_attention_part_a, _forward_attention_part_b)
def _forward_mlp(self, hidden_states, inference_context=None, input_to_mlp=None):
    """
    Perform a forward pass through the feed-forward layer.

    Args:
        hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

    Returns:
        output (Tensor): Transformed hidden states of shape [s, b, h].
    """
    if input_to_mlp is None:
        input_to_mlp = hidden_states

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm post the cross-attention.
    if self.recompute_pre_mlp_layernorm:
        self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
        pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
            self.pre_mlp_layernorm, input_to_mlp
        )
    else:
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(input_to_mlp)

    nvtx_range_push(suffix="mlp")
    # Potentially chunk the MLP computation during prefill to minimize the peak activation size
    should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
    )

    if self.recompute_mlp:
        if self.config.fp8:
            # import here to avoid circular import
            from megatron.core.extensions.transformer_engine import te_checkpoint

            mlp_output_with_bias = te_checkpoint(
                self.mlp,
                False,
                tensor_parallel.random.get_cuda_rng_tracker,
                self.pg_collection.tp,
                pre_mlp_layernorm_output,
            )
        else:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, pre_mlp_layernorm_output
            )
    elif should_chunk_mlp_for_prefill:
        # Chunk input along sequence dimension
        num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
        chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

        # Compute outputs for each chunk
        outputs = [self.mlp(chunk) for chunk in chunks]

        # Aggregate chunk outputs
        mlp_output = torch.cat([out for out, _ in outputs], dim=0)
        bias_chunks = [bias for _, bias in outputs if bias is not None]
        bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
        mlp_output_with_bias = (mlp_output, bias_output)

    else:
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

    if self.recompute_pre_mlp_layernorm:
        # discard the output of the pre-mlp layernorm and register the recompute
        # as a gradient hook of mlp_output_with_bias[0]
        self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
            mlp_output_with_bias[0]
        )
    nvtx_range_pop(suffix="mlp")

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    nvtx_range_push(suffix="mlp_bda")
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, residual, self.hidden_dropout
        )
    nvtx_range_pop(suffix="mlp_bda")

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    return output


def _forward_attention_part_a(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin=None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
):
    """
    Perform a forward pass through the attention layer and the layernorms before and after
    the attention operations.

    Args:
        hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
            b is batch size, and h is hidden size.
        attention_mask (Tensor): Mask tensor for self-attention.
        context (Tensor, optional): Context tensor for cross-attention.
        context_mask (Tensor, optional): Mask tensor for cross-attention.
        rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
        attention_bias (Tensor, optional): Bias tensor for Q * K.T.
        inference_context (object, optional): Parameters for inference-time optimizations.
        packed_seq_params (object, optional): Parameters for packed sequence processing.
        sequence_len_offset (Tensor, optional): Offset along sequence dimension
            during inference.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
            context (Tensor): Updated context tensor if cross-attention is used,
            otherwise None.
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    # Residual connection.
    residual = hidden_states

    # Optional Input Layer norm
    if self.recompute_input_layernorm:
        self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
        input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
            self.input_layernorm, hidden_states
        )
    else:
        input_layernorm_output = self.input_layernorm(hidden_states)

    # Self attention.
    nvtx_range_push(suffix="self_attention")
    forward_a_output = self.self_attention.forward_a(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        rotary_pos_cos_sin=rotary_pos_cos_sin,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
    )
    return tuple(forward_a_output), residual


def _forward_attention_part_b(
        self,
        forward_a_outputs: Tuple,
        residual: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin=None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
):
    """
    Perform a forward pass through the attention layer and the layernorms before and after
    the attention operations.

    Args:
        hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
            b is batch size, and h is hidden size.
        attention_mask (Tensor): Mask tensor for self-attention.
        context (Tensor, optional): Context tensor for cross-attention.
        context_mask (Tensor, optional): Mask tensor for cross-attention.
        rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
        attention_bias (Tensor, optional): Bias tensor for Q * K.T.
        inference_context (object, optional): Parameters for inference-time optimizations.
        packed_seq_params (object, optional): Parameters for packed sequence processing.
        sequence_len_offset (Tensor, optional): Offset along sequence dimension
            during inference.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
            context (Tensor): Updated context tensor if cross-attention is used,
            otherwise None.
    """
    inference_context = forward_a_outputs[-1]  # overwrite context with context from forward_a_outputs
    forward_b_args_input = forward_a_outputs[:-1]

    attention_output_with_bias = self.self_attention.forward_b(
        *forward_b_args_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        rotary_pos_cos_sin=rotary_pos_cos_sin,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
    )
    nvtx_range_pop(suffix="self_attention")

    if self.recompute_input_layernorm:
        # discard the output of the input layernorm and register the recompute
        # as a gradient hook of attention_output_with_bias[0]
        self.input_layernorm_checkpoint.discard_output_and_register_recompute(
            attention_output_with_bias[0]
        )

    # inside the module provided in the `bias_dropout_add_spec` module?
    nvtx_range_push(suffix="self_attn_bda")
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )
    nvtx_range_pop(suffix="self_attn_bda")

    # Residual connection.
    residual = hidden_states

    # Optional Layer norm after self-attention
    pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

    # Cross attention.
    attention_output_with_bias = self.cross_attention(
        pre_cross_attn_layernorm_output,
        attention_mask=context_mask,
        key_value_states=context,
        inference_context=inference_context,
    )

    if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
        context = attention_output_with_bias["context"]

    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    return hidden_states, context


def _forward_attention_combined(self, *args, **kwargs):
        forward_a_outputs, residual = self._forward_attention_a(*args, **kwargs)
        kwargs.pop('hidden_states', None) # remove hidden states from kwargs to avoid passing it twice
        hidden_states, context = self._forward_attention_b(forward_a_outputs, residual, *args[1:], **kwargs)
        return hidden_states, context

TransformerLayer._forward_mlp = _forward_mlp
TransformerLayer._forward_attention_a = _forward_attention_part_a
TransformerLayer._forward_attention_b = _forward_attention_part_b


# auxilary functions for farskip overlapping
def wait_for_async_backward_comm(backward_handle_dicts):
    """
    Create a gradient hook that waits for async backward communication to complete.

    This hook ensures that when gradients arrive at input tensors, we wait for the
    async all-to-all backward communication to finish before allowing gradients to
    propagate further backward.

    Args:
        backward_handle_dicts: Tuple/list of dictionaries containing 'backward' keys
                              that map to distributed work handles.

    Returns:
        A hook function that can be registered on a tensor.
    """
    def hook(grad):
        # Wait for all backward communication handles to complete
        for handle_dict in backward_handle_dicts:
            if 'backward' in handle_dict and handle_dict['backward'] is not None:
                handle_dict['backward'].wait()
                handle_dict['backward'] = None # TODO added
        return grad
    return hook


def set_tensor_grad_fn_sequence_sr(tensor, value):
    if tensor is not None and tensor.grad_fn is not None:
        tensor.grad_fn._set_sequence_nr(value)

# reference logical implementation of farskip transformer layer no overlapping
class SimpleFarSkipTransformerLayer(TransformerLayer):
    """A transformer layer with simple far skip connections.

    This layer extends TransformerLayer to support farskip connections, but without explicit enabled communication-computation overlap
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        self.attn_only_farskip = get_args().attn_only_farskip
        self.mlp_only_farskip = get_args().mlp_only_farskip

    def forward(self, *args, **kwargs):
        from megatron.core.transformer.moe.moe_layer import MoELayer

        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)
        hidden_states_no_routed = kwargs.get("hidden_states_no_routed", None)
        if hidden_states_no_routed is None:
            hidden_states_no_routed = kwargs.get("hidden_states")
            kwargs['hidden_states_no_routed'] = hidden_states_no_routed

        if self.mlp_only_farskip:
            kwargs['hidden_states_no_routed'] = kwargs.get("hidden_states")
        residual_with_attention_output, residual, context = self._simple_farskip_forward_attention(*args, **kwargs)

        if self.attn_only_farskip:
            residual = residual_with_attention_output
        if isinstance(self.mlp, MoELayer):


            output, output_no_routed_experts = self._simple_farskip_forward_mlp(residual_with_attention=residual_with_attention_output, input_to_mlp=residual, inference_context=kwargs.get("inference_context", None))
        else:
            output = self._forward_mlp(residual_with_attention_output, kwargs.get("inference_context", None), input_to_mlp=residual)
            output_no_routed_experts = output
        farskip_state = {'hidden_states_no_routed': output_no_routed_experts}
        return output, context, farskip_state

    def _simple_farskip_forward_attention(
        self,
        hidden_states: Tensor,
        hidden_states_no_routed: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states
        if hidden_states_no_routed is not None:
            input_to_attention = hidden_states_no_routed
        else:
            raise NotImplementedError("hidden_states_no_routed must be provided for SimpleFarSkipTransformerLayer")

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, input_to_attention
            )
        else:
            input_layernorm_output = self.input_layernorm(input_to_attention)

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            residual_with_attention_output = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        # Residual connection.

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(residual_with_attention_output)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            residual_with_attention_output = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual_with_attention_output, self.hidden_dropout
            )
        return residual_with_attention_output, residual, context

    def _simple_farskip_forward_mlp(self, residual_with_attention, input_to_mlp, inference_context=None):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, input_to_mlp
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(input_to_mlp)

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
        )
        self.mlp.forward = functools.partial(self.mlp.forward, return_shared_expert=True)

        if self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
            outputs = [self.mlp(chunk) for chunk in chunks]

            # Aggregate chunk outputs; changed to enable aggregation of output and output_no_residual
            mlp_output = torch.cat([out for out, _, _ in outputs], dim=0)
            shared_mlp_output = torch.cat([shared_output for _, shared_output, _ in outputs], dim=0)
            bias_chunks = [bias for _, _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, shared_mlp_output, bias_output)

        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        nvtx_range_pop(suffix="mlp")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        with self.bias_dropout_add_exec_handler():
            mlp_shared_output_with_bias = (mlp_output_with_bias[1], None)  # shared expert no bias
            mlp_full_output_with_bias = (mlp_output_with_bias[0], mlp_output_with_bias[-1])

            #
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_full_output_with_bias, residual_with_attention, self.hidden_dropout
            )
            hidden_states_no_routed_experts = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_shared_output_with_bias, residual_with_attention, self.hidden_dropout
            )
        nvtx_range_pop(suffix="mlp_bda")

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        output_no_routed_experts = make_viewless_tensor(
            inp=hidden_states_no_routed_experts, requires_grad=hidden_states_no_routed_experts.requires_grad, keep_graph=True
        )
        return output, output_no_routed_experts


class OverlappedFarSkipTransformerLayer(TransformerLayer):
    """
    Overlapped implementation of farskip transformer layer
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )


    def forward(self, *args, **kwargs):
        kwargs.pop('hidden_states_no_routed', None) # hidden_states_no_routed can be None if previous SimpleFarSkipTransformerLayer MLP was dense
        residual = kwargs['hidden_states']
        combine_handle = kwargs.pop('combine_handle', None)
        combined_experts = kwargs.pop('combined_experts', None)
        unpermute_inputs = kwargs.pop('dispatcher_state', None)

        if combine_handle is None:
            input_to_mlp = self.pre_mlp_layernorm(residual)
            input_to_dispatch, new_probs_pre, input_to_shared = self.mlp.router_and_preprocess(input_to_mlp)
            dispatched_input, new_probs, dispatch_handles, dispatch_backward_handles = self.mlp.dispatch(input_to_dispatch, new_probs_pre, async_op=True)

            # Register hooks to wait for backward communication
            input_to_dispatch.register_hook(wait_for_async_backward_comm(dispatch_backward_handles))
            new_probs_pre.register_hook(wait_for_async_backward_comm(dispatch_backward_handles))

            forward_a_outputs, residual = self._forward_attention_a(*args, **kwargs)
        else:
            if get_args().farskip_overlap_associative_add:
                shared_expert = kwargs.pop('shared_expert', None)
                last_residual_with_attention = kwargs.pop('last_residual_with_attention', None)

            from megatron.core.transformer.moe.moe_utils import unpermute
            forward_a_outputs, residual = self._forward_attention_a(*args, **kwargs)
            combine_handle.wait()
            reversed_local_input_permutation_mapping = unpermute_inputs.pop('reversed_local_input_permutation_mapping')
            combined_experts = unpermute(
                combined_experts,
                reversed_local_input_permutation_mapping, **unpermute_inputs
            )
            combined_experts = combined_experts.view(residual.shape)
            if get_args().farskip_overlap_associative_add:
                residual = last_residual_with_attention + (combined_experts + shared_expert)
            else:
                residual = residual + combined_experts


            input_to_mlp = self.pre_mlp_layernorm(residual)
            input_to_dispatch, new_probs_pre, input_to_shared = self.mlp.router_and_preprocess(input_to_mlp)
            dispatched_input, new_probs, dispatch_handles, dispatch_backward_handles = self.mlp.dispatch(input_to_dispatch, new_probs_pre, async_op=True)

            # Register hooks to wait for backward communication to complete
            input_to_dispatch.register_hook(wait_for_async_backward_comm(dispatch_backward_handles))
            new_probs_pre.register_hook(wait_for_async_backward_comm(dispatch_backward_handles))

        set_tensor_grad_fn_sequence_sr(input_to_dispatch, 0)
        set_tensor_grad_fn_sequence_sr(new_probs_pre, 0)
        set_tensor_grad_fn_sequence_sr(input_to_shared, 0)
        [set_tensor_grad_fn_sequence_sr(some_tensor, 2) for some_tensor in forward_a_outputs if torch.is_tensor(some_tensor)]
        set_tensor_grad_fn_sequence_sr(residual, 1)
        set_tensor_grad_fn_sequence_sr(dispatched_input, torch.iinfo(torch.int).max)
        set_tensor_grad_fn_sequence_sr(new_probs, torch.iinfo(torch.int).max)

        kwargs.pop('hidden_states', None)  # remove hidden states from kwargs to avoid passing it twice
        residual_with_attention, context = self._forward_attention_b(forward_a_outputs, residual, *args[1:], **kwargs)
        if dispatch_handles[0] is not None:
            dispatch_handles[0].wait()
            dispatch_handles[1].wait()
        dispatched_input_post, tokens_per_expert, permuted_probs = (
            self.mlp.token_dispatcher.dispatch_postprocess(dispatched_input, new_probs)
        )
        expert_output, mlp_bias = self.mlp.experts(dispatched_input_post, tokens_per_expert, permuted_probs)
        output = self.mlp.token_dispatcher.combine_preprocess(expert_output)
        new_combined_experts, new_combine_handle, new_combine_backward_handle = self.mlp.token_dispatcher.token_combine(output, async_op=True)
        output.register_hook(wait_for_async_backward_comm([new_combine_backward_handle]))

        shared_expert_output = self.mlp.shared_experts(input_to_shared)
        hidden_states = residual_with_attention + shared_expert_output

        if new_combine_handle is None: # adding for EP=1
            from megatron.core.transformer.moe.moe_utils import unpermute
            reversed_local_input_permutation_mapping = self.mlp.token_dispatcher.reversed_local_input_permutation_mapping
            unpermute_inputs =  {'restore_shape': self.mlp.token_dispatcher.hidden_shape_before_permute,
            'routing_map': self.mlp.token_dispatcher.routing_map,
            'fused': self.mlp.token_dispatcher.config.moe_permute_fusion,
            'drop_and_pad': self.mlp.token_dispatcher.drop_and_pad,
            }
            new_combined_experts = unpermute(
                new_combined_experts,
                reversed_local_input_permutation_mapping, **unpermute_inputs
            )
            hidden_states = hidden_states + new_combined_experts.view(hidden_states.shape)
            new_combined_experts = None

        # Make viewless tensor to avoid checkpoint issues with JIT-compiled view tensors
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        dispatcher_state = {
            'reversed_local_input_permutation_mapping': self.mlp.token_dispatcher.reversed_local_input_permutation_mapping.clone(),
            'restore_shape': self.mlp.token_dispatcher.hidden_shape_before_permute,
            'routing_map': self.mlp.token_dispatcher.routing_map.clone(),
            'fused': self.mlp.token_dispatcher.config.moe_permute_fusion,
            'drop_and_pad': self.mlp.token_dispatcher.drop_and_pad,
        }

        set_tensor_grad_fn_sequence_sr(output, 3)
        set_tensor_grad_fn_sequence_sr(new_combined_experts, torch.iinfo(torch.int).max)

        set_tensor_grad_fn_sequence_sr(dispatched_input_post, torch.iinfo(torch.int).max)
        set_tensor_grad_fn_sequence_sr(tokens_per_expert, torch.iinfo(torch.int).max)
        set_tensor_grad_fn_sequence_sr(permuted_probs, torch.iinfo(torch.int).max)

        set_tensor_grad_fn_sequence_sr(expert_output, torch.iinfo(torch.int).max)
        if mlp_bias is not None:
            set_tensor_grad_fn_sequence_sr(mlp_bias, torch.iinfo(torch.int).max)


        farskip_state = {'combined_experts': new_combined_experts, 'combine_handle': new_combine_handle}
        farskip_state['dispatcher_state'] = dispatcher_state
        if get_args().farskip_overlap_associative_add:
            farskip_state['shared_expert'] = shared_expert_output
            farskip_state['last_residual_with_attention'] = residual_with_attention
        return hidden_states, context, farskip_state