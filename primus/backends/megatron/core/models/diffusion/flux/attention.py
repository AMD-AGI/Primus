# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux attention mechanisms.

This module implements specialized attention for Flux's MMDiT architecture:
    - JointSelfAttention: Processes concatenated image + text tokens
    - FluxSingleAttention: Processes image tokens only
    - JointSelfAttentionSubmodules: Configuration for joint attention

These implementations follow Megatron-Core's attention patterns with
customizations for diffusion model conditioning.

Reference:
    - MMDiT: "Scaling Rectified Flow Transformers"
    - Megatron-Core: megatron.core.transformer.attention
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from megatron.core.transformer.attention import (
    Attention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

try:
    from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None


@dataclass
class JointSelfAttentionSubmodules:
    """
    Submodules configuration for Joint Self-Attention layer.

    Joint attention processes both image and text tokens together (MMDiT architecture).
    It requires separate QKV projections for image and text (context) streams.

    Attributes:
        linear_qkv: QKV projection for main stream (image tokens)
        added_linear_qkv: QKV projection for added stream (text/context tokens)
        core_attention: Core attention computation module
        linear_proj: Output projection for main stream
        q_layernorm: Optional layer norm for queries (main stream)
        k_layernorm: Optional layer norm for keys (main stream)
        added_q_layernorm: Optional layer norm for queries (added stream)
        added_k_layernorm: Optional layer norm for keys (added stream)

    Note:
        Flux uses RMSNorm for Q/K normalization to improve training stability.

    Reference:
        - Paper: "Scaling Rectified Flow Transformers"
    """

    linear_qkv: Union[ModuleSpec, type] = None
    added_linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    added_q_layernorm: Union[ModuleSpec, type] = None
    added_k_layernorm: Union[ModuleSpec, type] = None


class JointSelfAttention(Attention):
    """
    Joint Self-Attention for MMDiT (Multimodal Diffusion Transformer).

    Processes two token streams jointly -- main (image) and added (text) --
    by projecting each through separate QKV layers, concatenating, computing
    joint attention, then splitting back. This enables cross-modal interaction
    in Flux's "double blocks".

    Args:
        config: Transformer configuration
        submodules: JointSelfAttentionSubmodules with layer specifications
        layer_number: Layer index in the model
        attn_mask_type: Type of attention mask (default: padding)
        context_pre_only: If True, only compute Q/K/V for context (default: False)

    Input/Output:
        hidden_states [seq_main, B, H] + additional_hidden_states [seq_added, B, H]
        -> (main_output, added_output) with same shapes

    Reference:
        - "Scaling Rectified Flow Transformers"
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: JointSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        context_pre_only: bool = False,
        **kwargs,
    ):
        # Use RMSNorm for Q/K normalization (improves stability)
        config.normalization = "RMSNorm"

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            **kwargs,
        )

        # QKV projection for main stream (image tokens)
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        # QKV projection for added stream (text tokens)
        if submodules.added_linear_qkv is not None:
            self.added_linear_qkv = build_module(
                submodules.added_linear_qkv,
                self.config.hidden_size,
                self.query_projection_size + 2 * self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
            )

        # Output projection for added stream (text tokens)
        if not context_pre_only:
            self.added_linear_proj = build_module(
                submodules.linear_proj,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name="proj",
            )

        if (
            not context_pre_only
            and getattr(self.config, "use_dual_fp8_output_projection", False)
            and hasattr(self.linear_proj, "_fp8_config")
        ):
            from primus_turbo.pytorch.core.low_precision import ScalingGranularity

            if self.linear_proj._fp8_config.granularity == ScalingGranularity.TENSORWISE:
                from primus_turbo.pytorch.core.backend import BackendType

                from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
                    DualFP8LinearTensorwiseFunction,
                    _get_fp8_dtype,
                )

                self._dual_fp8_fn = DualFP8LinearTensorwiseFunction
                cfg = self.linear_proj._fp8_config
                self._dual_fp8_fwd_dtype = _get_fp8_dtype(cfg.format, is_fwd=True)
                self._dual_fp8_bwd_dtype = _get_fp8_dtype(cfg.format, is_fwd=False)
                self._dual_fp8_gran_value = ScalingGranularity.TENSORWISE.value
                self._dual_fp8_backend_value = BackendType.HIPBLASLT.value

        # Optional Q/K layer normalization for main stream
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        # Optional Q/K layer normalization for added stream
        if submodules.added_q_layernorm is not None:
            self.added_q_layernorm = build_module(
                submodules.added_q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_q_layernorm = None

        if submodules.added_k_layernorm is not None:
            self.added_k_layernorm = build_module(
                submodules.added_k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_k_layernorm = None

    def _split_qkv(self, mixed_qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Split mixed QKV tensor into separate Q, K, V tensors.

        Args:
            mixed_qkv: Combined QKV tensor [seq, batch, hidden]

        Returns:
            Tuple of (query, key, value) tensors
        """
        # Reshape: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Define split sizes for Q, K, V
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # Split tensor
        if SplitAlongDim is not None:
            # Use Transformer Engine's optimized split if available
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # Fallback to PyTorch split
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Reshape query: [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value

    def get_query_key_value_tensors(
        self, hidden_states: Tensor, key_value_states: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Derive Q, K, V tensors from main stream hidden states.

        Args:
            hidden_states: Main stream tokens [seq, batch, hidden]
            key_value_states: Not used for self-attention

        Returns:
            Tuple of (query, key, value) tensors
        """
        # Project to QKV: [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # Split into Q, K, V
        query, key, value = self._split_qkv(mixed_qkv)

        # Apply optional Q/K normalization
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def get_added_query_key_value_tensors(
        self, added_hidden_states: Tensor, key_value_states: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Derive Q, K, V tensors from added stream (text) hidden states.

        Args:
            added_hidden_states: Added stream tokens [seq, batch, hidden]
            key_value_states: Not used for self-attention

        Returns:
            Tuple of (query, key, value) tensors
        """
        # Project to QKV
        mixed_qkv, _ = self.added_linear_qkv(added_hidden_states)

        # Split into Q, K, V
        query, key, value = self._split_qkv(mixed_qkv)

        # Apply optional Q/K normalization
        if self.added_q_layernorm is not None:
            query = self.added_q_layernorm(query)

        if self.added_k_layernorm is not None:
            key = self.added_k_layernorm(key)

        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        additional_hidden_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: Joint attention over image and text tokens.

        Args:
            hidden_states: Main stream (image) tokens [seq_main, batch, hidden]
            attention_mask: Attention mask
            key_value_states: Not used for self-attention
            inference_params: Parameters for inference (e.g., KV cache)
            rotary_pos_emb: RoPE position embeddings
            packed_seq_params: Parameters for packed sequences
            additional_hidden_states: Added stream (text) tokens [seq_added, batch, hidden]

        Returns:
            Tuple of (main_output, added_output):
                - main_output: Processed main stream [seq_main, batch, hidden]
                - added_output: Processed added stream [seq_added, batch, hidden]
        """
        # Ensure rotary_pos_emb is a tuple for Q and K
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Get Q, K, V for both streams
        query, key, value = self.get_query_key_value_tensors(hidden_states)
        added_query, added_key, added_value = self.get_added_query_key_value_tensors(additional_hidden_states)

        # Concatenate streams: [added; main]
        query = torch.cat([added_query, query], dim=0)
        key = torch.cat([added_key, key], dim=0)
        value = torch.cat([added_value, value], dim=0)

        # Adjust for inference (KV caching, etc.)
        query, key, value, rotary_pos_emb, attn_mask_type, *_ = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )

        # Handle packed sequences
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # Apply RoPE position embeddings
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            cu_seqlens_q = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None

            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # Core attention computation
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # Handle packed sequences output
        if packed_seq_params is not None:
            # Reshape: (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Split output back into added and main streams
        encoder_attention_output = core_attn_out[: additional_hidden_states.shape[0], :, :]
        attention_output = core_attn_out[additional_hidden_states.shape[0] :, :, :]

        # Project outputs
        if hasattr(self, "_dual_fp8_fn"):
            from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
                _extract_fp8_weight,
            )

            w_fp8_a, w_scale_a = _extract_fp8_weight(
                self.linear_proj.weight,
                self._dual_fp8_fwd_dtype,
            )
            w_fp8_b, w_scale_b = _extract_fp8_weight(
                self.added_linear_proj.weight,
                self._dual_fp8_fwd_dtype,
            )
            result = self._dual_fp8_fn.apply(
                attention_output,
                self.linear_proj.weight,
                w_fp8_a,
                w_scale_a,
                encoder_attention_output,
                self.added_linear_proj.weight,
                w_fp8_b,
                w_scale_b,
                self._dual_fp8_fwd_dtype,
                self._dual_fp8_bwd_dtype,
                self._dual_fp8_gran_value,
                self._dual_fp8_backend_value,
            )
            output, encoder_output = result[0], result[1]
            if self.linear_proj.bias is not None:
                output = output + self.linear_proj.bias
            if self.added_linear_proj.bias is not None:
                encoder_output = encoder_output + self.added_linear_proj.bias
        else:
            output, bias = self.linear_proj(attention_output)
            encoder_output, encoder_bias = self.added_linear_proj(encoder_attention_output)
            output = output + bias
            encoder_output = encoder_output + encoder_bias

        return output, encoder_output


class FluxSingleAttention(SelfAttention):
    """
    Single-stream Self-Attention for Flux (image tokens only).

    Standard self-attention without cross-modal interaction. Used in Flux's
    "single blocks" after the joint MMDiT blocks.

    Args:
        config: Transformer configuration
        submodules: SelfAttentionSubmodules with layer specifications
        layer_number: Layer index in the model
        attn_mask_type: Type of attention mask (default: padding)
        cp_comm_type: Communication type for context parallelism
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        **kwargs,
    ):
        # Use RMSNorm for Q/K normalization
        config.normalization = "RMSNorm"

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            **kwargs,
        )

        # The original Flux proj_out (Diffusers) / linear2 (TorchTitan) is a single fused
        # projection with one bias. Megatron splits it into linear_proj + linear_fc2, so the
        # bias only needs to be on one path (linear_fc2) to preserve mathematical equivalence.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: Self-attention on image tokens.

        Args:
            hidden_states: Image tokens [seq, batch, hidden]
            attention_mask: Attention mask
            key_value_states: Not used for self-attention
            inference_params: Parameters for inference (e.g., KV cache)
            rotary_pos_emb: RoPE position embeddings
            packed_seq_params: Parameters for packed sequences

        Returns:
            Tuple of (output, bias):
                - output: Projected attention output [seq, batch, hidden]
                - bias: Projection bias (None when linear_proj has bias=False)
        """
        # Ensure rotary_pos_emb is a tuple for Q and K
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Get Q, K, V
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # Adjust for inference
        query, key, value, rotary_pos_emb, attn_mask_type, *_ = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )

        # Handle packed sequences
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # Apply RoPE position embeddings
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            cu_seqlens_q = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None

            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # Core attention computation
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # Handle packed sequences output
        if packed_seq_params is not None:
            # Reshape: (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Project output (return both output and bias for skip_bias_add pattern)
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
