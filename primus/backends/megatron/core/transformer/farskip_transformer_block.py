"""
FarSkip TransformerBlock forward.

This is the original TransformerBlock.forward from Megatron-LM, to be monkey-patched
onto TransformerBlock by the trainer. Future commits will add farskip state threading
and async finalization logic.
"""
from contextlib import nullcontext

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor
from megatron.core.transformer.custom_layers.transformer_engine import WrappedTensor


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
            hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_quantization_context=use_inner_quantization_context,
            )
        else:
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
                    hidden_states, context = layer(
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
                    )

                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    # Final layer norm.
    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

    if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
        hidden_states = hidden_states.clone()

    return hidden_states