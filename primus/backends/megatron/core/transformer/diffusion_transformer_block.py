# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
DiffusionTransformerBlock for diffusion models with timestep conditioning.

This module provides a specialized TransformerBlock for diffusion models that properly
handles timestep embeddings and other conditioning parameters through gradient checkpointing.
"""

from typing import Optional, Union

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import WrappedTensor
from torch import Tensor


class DiffusionTransformerBlock(TransformerBlock):
    """
    TransformerBlock for diffusion models with timestep conditioning.

    Extends upstream Megatron TransformerBlock with explicit timestep_emb and
    guidance_emb parameters, routed through gradient checkpointing in a
    thread-safe manner (no instance attribute storage for conditioning).

    Call chain:
        Flux.forward() -> transformer(timestep_emb=...) ->
        DiffusionTransformerBlock.forward() -> checkpoint(custom_forward, timestep_emb) ->
        layer(timestep_emb=...) -> adaln(timestep_emb)
    """

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        # Diffusion-specific conditioning parameters
        timestep_emb: Optional[Tensor] = None,
        guidance_emb: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with explicit diffusion conditioning.

        Timestep and guidance embeddings are passed as tensors and participate in
        gradient checkpointing like other forward inputs.

        Args:
            hidden_states: Input tensor [seq, batch, hidden]
            attention_mask: Attention mask tensor
            context: Context tensor for cross-attention (e.g., text embeddings)
            context_mask: Mask for context in cross-attention
            timestep_emb: Timestep conditioning [batch, hidden] for diffusion models
            guidance_emb: Guidance conditioning [batch, hidden] for classifier-free guidance
            All other args: Same as TransformerBlock.forward()

        Returns:
            Output tensor [seq, batch, hidden], or a tuple (hidden_states, context)
            when cross-attention context is present.

        Note:
            This implementation is thread-safe because no conditioning parameters are
            stored as instance attributes. All conditioning flows through function arguments.
        """
        # Validate diffusion parameters if model type requires them
        if hasattr(self.config, "model_type") and self.config.model_type in ["flux", "diffusion"]:
            if timestep_emb is None:
                raise ValueError(
                    f"DiffusionTransformerBlock requires timestep_emb for model_type={self.config.model_type}"
                )

        # Build conditioning_kwargs for passing to layers
        # Only include non-None values to avoid passing unused parameters
        conditioning_kwargs = {}
        if timestep_emb is not None:
            conditioning_kwargs["timestep_emb"] = timestep_emb
        if guidance_emb is not None:
            conditioning_kwargs["guidance_emb"] = guidance_emb

        # Extract kwargs that need to be passed to parent (excluding conditioning)
        parent_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "context": context,
            "context_mask": context_mask,
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "rotary_pos_cos_sin": rotary_pos_cos_sin,
            "attention_bias": attention_bias,
            "inference_context": inference_context,
            "packed_seq_params": packed_seq_params,
            "sequence_len_offset": sequence_len_offset,
        }
        # Filter out conditioning parameters from kwargs before updating
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("timestep_emb", "guidance_emb")}
        parent_kwargs.update(filtered_kwargs)

        # Check if checkpointing is needed (based on parent's logic)
        if self.config.recompute_granularity == "full" and self.training:
            # Checkpointed path: Need to override _checkpointed_forward to handle conditioning
            # Delete the obsolete reference to the initial input tensor if necessary
            if isinstance(hidden_states, WrappedTensor):
                hidden_states = hidden_states.unwrap()

            if not self.pre_process:
                # See set_input_tensor()
                hidden_states = self.input_tensor

            # Determine inner quantization context usage (from parent logic)
            from megatron.core.enums import Fp8Recipe

            if self.config.fp8:
                use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            elif self.config.fp4:
                use_inner_quantization_context = True
            else:
                use_inner_quantization_context = False

            # Call custom checkpointed forward with conditioning
            return self._checkpointed_forward_with_conditioning(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_quantization_context=use_inner_quantization_context,
                **conditioning_kwargs,  # Pass conditioning to checkpoint
            )
        else:
            # Normal forward path - manually iterate layers with conditioning
            # Parent TransformerBlock doesn't accept timestep_emb/guidance_emb
            # So we need to manually process layers when conditioning is present
            if conditioning_kwargs:
                # Process layers manually with conditioning
                current_hidden = hidden_states
                current_context = context

                for layer in self.layers:
                    layer_kwargs = {
                        "hidden_states": current_hidden,
                        "attention_mask": attention_mask,
                        "context": current_context,
                        "context_mask": context_mask,
                        "rotary_pos_emb": rotary_pos_emb,
                        "attention_bias": attention_bias,
                        "inference_context": inference_context,
                        "packed_seq_params": packed_seq_params,
                    }
                    # Add conditioning
                    layer_kwargs.update(conditioning_kwargs)

                    layer_output = layer(**layer_kwargs)
                    if isinstance(layer_output, tuple):
                        current_hidden, current_context = layer_output
                    else:
                        current_hidden = layer_output
                        current_context = None

                return current_hidden if current_context is None else (current_hidden, current_context)
            else:
                # No conditioning - use parent's forward
                return super().forward(**parent_kwargs)

    def _checkpointed_forward_with_conditioning(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
        timestep_emb: Optional[Tensor] = None,
        guidance_emb: Optional[Tensor] = None,
    ):
        """
        Checkpointed forward with conditioning support.

        This method extends the parent's _checkpointed_forward to handle
        diffusion conditioning parameters (timestep_emb, guidance_emb).

        Note: Unlike the non-checkpointed path in forward(), this method
        only returns hidden_states and discards context.
        """
        from contextlib import nullcontext

        from megatron.core import tensor_parallel
        from megatron.core.fp4_utils import get_fp4_context
        from megatron.core.fp8_utils import get_fp8_context
        from megatron.core.transformer.transformer_layer import (
            get_transformer_layer_offset,
        )

        try:
            pass

            HAVE_TE = True
        except ImportError:
            HAVE_TE = False

        te_checkpoint = None
        if HAVE_TE:
            from megatron.core.extensions.transformer_engine import te_checkpoint

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                timestep_emb=None,
                guidance_emb=None,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        # Build layer kwargs with conditioning
                        layer_kwargs = {
                            "hidden_states": hidden_states,
                            "attention_mask": attention_mask,
                            "context": context,
                            "context_mask": context_mask,
                            "rotary_pos_emb": rotary_pos_emb,
                            "attention_bias": attention_bias,
                            "inference_context": None,
                            "packed_seq_params": packed_seq_params,
                        }
                        # Add conditioning if present
                        if timestep_emb is not None:
                            layer_kwargs["timestep_emb"] = timestep_emb
                        if guidance_emb is not None:
                            layer_kwargs["guidance_emb"] = guidance_emb

                        hidden_states, context = layer(**layer_kwargs)
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    (
                        timestep_emb if timestep_emb is not None else context
                    ),  # Use context as placeholder if None
                    (
                        guidance_emb if guidance_emb is not None else context
                    ),  # Use context as placeholder if None
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    (
                        timestep_emb if timestep_emb is not None else context
                    ),  # Use context as placeholder if None
                    (
                        guidance_emb if guidance_emb is not None else context
                    ),  # Use context as placeholder if None
                )

        recompute_layer_ids = getattr(self.config, "recompute_layer_ids", None)

        if self.config.recompute_method == "uniform":
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )
                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == "block":
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    # Build args for non-checkpointed path
                    forward_args = (
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        timestep_emb,
                        guidance_emb,
                    )
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(*forward_args)

        elif recompute_layer_ids is not None:
            for block_layer_idx in range(self.num_layers_per_pipeline_rank):
                layer_idx = block_layer_idx + get_transformer_layer_offset(self.config, self.vp_stage)
                if layer_idx not in recompute_layer_ids or (
                    (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad
                ):
                    forward_args = (
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        timestep_emb,
                        guidance_emb,
                    )
                    hidden_states, context = custom(block_layer_idx, block_layer_idx + 1)(*forward_args)
                else:
                    hidden_states, context = checkpoint_handler(custom(block_layer_idx, block_layer_idx + 1))
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states
