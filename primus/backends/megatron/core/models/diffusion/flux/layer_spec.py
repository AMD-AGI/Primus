# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux layer specifications for Megatron-Core integration.

This module implements transformer layers for Flux's MMDiT architecture:
    - MMDiTLayer: Joint image-text transformer block
    - FluxSingleTransformerBlock: Image-only transformer block
    - Factory functions for creating layer specs with Transformer Engine

Key Innovation: Heterogeneous layer support via TransformerBlock
    Instead of separate ModuleLists, Primus uses a unified TransformerBlock
    with heterogeneous layer specifications, enabling better pipeline
    parallelism and cleaner checkpoint management.

Reference:
    - Flux Paper: "Flux: A Scalable Diffusion Model"
    - MMDiT Paper: "Scaling Rectified Flow Transformers"
    - Megatron-Core: Heterogeneous layer patterns
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from primus.backends.megatron.core.models.diffusion.common.normalization import (
    AdaLN,
    AdaLNContinuous,
)
from primus.backends.megatron.core.models.diffusion.flux.attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)

try:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.attention import SelfAttentionSubmodules

    HAVE_TE_SPEC_PROVIDER = True
except ImportError:
    HAVE_TE_SPEC_PROVIDER = False
    SelfAttentionSubmodules = None

# Try to import PrimusTurboLocalSpecProvider (may not exist in all versions)
try:
    from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
        PrimusTurboFloat8LocalSpecProvider,
        PrimusTurboLocalSpecProvider,
    )

    HAVE_PRIMUS_TURBO_LOCAL = True
except ImportError:
    HAVE_PRIMUS_TURBO_LOCAL = False
    PrimusTurboLocalSpecProvider = None
    PrimusTurboFloat8LocalSpecProvider = None

# MXFP4 provider in a separate guard so a missing primus_turbo_mxfp4_local
# doesn't break FP8 imports.
PrimusTurboMXFP4LocalSpecProvider = None
try:
    from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
        PrimusTurboMXFP4LocalSpecProvider,
    )
except ImportError:
    pass


class MMDiTLayer(TransformerLayer):
    """
    Multimodal Diffusion Transformer (MMDiT) Layer.

    Processes image and text streams jointly via AdaLN-conditioned attention
    and separate MLPs with gated residual connections. Used in Flux's "double
    blocks" for cross-modal image-text interaction.

    Args:
        config: Transformer configuration
        submodules: Submodule specifications for attention, MLP, etc.
        layer_number: Layer index in the model (default: 1)
        context_pre_only: If True, context stream only computes pre-attention norm

    Input/Output:
        (hidden_states [S_img, B, H], context [S_txt, B, H], timestep_emb [B, H])
        -> (hidden_states, context) with same shapes

    Reference:
        - "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        context_pre_only: bool = False,
        **kwargs,  # Accept additional kwargs from TransformerBlock (e.g., pg_collection)
    ):
        hidden_size = config.hidden_size
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)

        # Enable per-layer CUDA graph if configured
        if config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config, share_cudagraph_io_buffers=False)

        # Adaptive layer normalization for main stream (image).
        # init_method=nn.init.normal_ draws RNG matching NeMo's DiT init
        # sequence so cross-framework convergence comparisons line up.
        # init_weights() (model.py) immediately re-zeroes these weights,
        # so the only observable effect is RNG-sequence alignment.
        self.adaln = AdaLN(
            config,
            modulation_bias=True,
            n_adaln_chunks=6,
            use_second_norm=True,
            init_method=nn.init.normal_,
        )

        # Adaptive layer normalization for context stream (text)
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continuous" if context_pre_only else "ada_norm_zero"

        if context_norm_type == "ada_norm_continuous":
            # Continuous AdaLN for context (simpler, used when context is pre-only)
            self.adaln_context = AdaLNContinuous(
                config, hidden_size, modulation_bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            # Full AdaLN for context (used when context has full processing).
            # See note on self.adaln above re: init_method choice.
            self.adaln_context = AdaLN(
                config,
                modulation_bias=True,
                n_adaln_chunks=6,
                use_second_norm=True,
                init_method=nn.init.normal_,
            )
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, "
                f"currently only support `ada_norm_continuous`, `ada_norm_zero`"
            )

        # Context MLP (only if not pre-only)
        if not context_pre_only:
            # Disable context parallelism for context MLP
            cp_override_config = copy.deepcopy(config)
            cp_override_config.context_parallel_size = 1
            cp_override_config.tp_comm_overlap = False

            from megatron.core.transformer.spec_utils import build_module

            self.context_mlp = build_module(submodules.mlp, config=cp_override_config)
        else:
            self.context_mlp = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        timestep_emb: Optional[Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: Joint processing of image and text tokens.

        Args:
            hidden_states: Image tokens [seq_img, batch, hidden]
            context: Text tokens [seq_txt, batch, hidden]
            timestep_emb: Timestep conditioning [batch, hidden] (required)

        Returns:
            Tuple of (hidden_states, context) - both updated
        """
        # Map TransformerBlock's 'context' to internal 'encoder_hidden_states'
        encoder_hidden_states = context

        if timestep_emb is None:
            raise ValueError("MMDiTLayer requires timestep_emb for AdaLN conditioning.")

        emb = timestep_emb

        # Get modulation parameters for main stream (image)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(emb)

        # Apply modulated layer norm to image tokens
        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_msa, scale=scale_msa, layernorm_idx=0
        )

        # Apply modulated layer norm to text tokens
        if self.context_pre_only:
            # Continuous AdaLN (simpler)
            norm_encoder_hidden_states = self.adaln_context(encoder_hidden_states, emb)
        else:
            # Full AdaLN with modulation parameters
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.adaln_context(
                emb
            )
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_msa, scale=c_scale_msa, layernorm_idx=0
            )

        # Joint self-attention
        attn_output, context_attn_output = self.self_attention(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            additional_hidden_states=norm_encoder_hidden_states,
        )

        # MLP for main stream (image)
        # Fused operation: gated residual + modulated layernorm for MLP input
        hidden_states, norm_hidden_states = self.adaln.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attn_output,
            gate=gate_msa,
            shift=shift_mlp,
            scale=scale_mlp,
            layernorm_idx=1,
        )
        mlp_output, mlp_bias = self.mlp(norm_hidden_states)
        hidden_states = self.adaln.scale_add(hidden_states, x=(mlp_output + mlp_bias), gate=gate_mlp)

        # MLP for context stream (text) - only if not pre-only
        if not self.context_pre_only:
            # Fused operation: gated residual + modulated layernorm for context MLP input
            encoder_hidden_states, norm_encoder_hidden_states = self.adaln_context.scaled_modulated_layernorm(
                residual=encoder_hidden_states,
                x=context_attn_output,
                gate=c_gate_msa,
                shift=c_shift_mlp,
                scale=c_scale_mlp,
                layernorm_idx=1,
            )
            context_mlp_output, context_mlp_bias = self.context_mlp(norm_encoder_hidden_states)
            encoder_hidden_states = self.adaln_context.scale_add(
                encoder_hidden_states, x=(context_mlp_output + context_mlp_bias), gate=c_gate_mlp
            )

        # Make output viewless for MPU checkpoint compatibility
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        encoder_hidden_states = make_viewless_tensor(
            inp=encoder_hidden_states, requires_grad=encoder_hidden_states.requires_grad, keep_graph=True
        )

        return hidden_states, encoder_hidden_states

    def __call__(self, *args, **kwargs):
        """Override call to support CUDA graphs."""
        if hasattr(self, "cudagraph_manager"):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)


class FluxSingleTransformerBlock(TransformerLayer):
    """
    Flux Single Transformer Block (image-only processing).

    Processes image tokens with parallel attention + MLP paths, AdaLN
    conditioning, and gated residual connections. Used in Flux's "single
    blocks" after the joint MMDiT layers.

    Args:
        config: Transformer configuration
        submodules: Submodule specifications for attention, MLP, etc.
        layer_number: Layer index in the model (default: 1)
        mlp_ratio: MLP hidden size ratio (default: 4)
        n_adaln_chunks: AdaLN modulation chunks (default: 3 for shift, scale, gate)
        modulation_bias: Whether to use bias in modulation layers (default: True)

    Input/Output:
        (hidden_states [S, B, H], timestep_emb [B, H]) -> (hidden_states, None)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        n_adaln_chunks: int = 3,
        modulation_bias: bool = True,
        **kwargs,  # Accept additional kwargs from TransformerBlock (e.g., pg_collection)
    ):
        # Override add_bias_linear with single_block_bias for this block
        # This allows independent control of bias in single blocks vs joint blocks
        original_add_bias_linear = config.add_bias_linear
        if hasattr(config, "single_block_bias"):
            config.add_bias_linear = config.single_block_bias

        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)

        # Restore original value
        config.add_bias_linear = original_add_bias_linear

        # Enable per-layer CUDA graph if configured
        if config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config, share_cudagraph_io_buffers=False)

        # Adaptive layer normalization
        # n_adaln_chunks=3 for (shift, scale, gate)
        # init_method=nn.init.normal_: see note on FluxTransformerBlock's
        # self.adaln above -- NeMo-aligned RNG draw, re-zeroed in init_weights().
        self.adaln = AdaLN(
            config=config,
            n_adaln_chunks=n_adaln_chunks,
            modulation_bias=modulation_bias,
            use_second_norm=False,  # Single block only uses one norm
            init_method=nn.init.normal_,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        timestep_emb: Optional[Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        """
        Single-stream block: first single layer may receive separate image and text streams
        (concatenates them); later layers see one concatenated stream with context=None.

        Args:
            hidden_states: Image tokens [seq_img, B, H] or concatenated [seq_total, B, H]
            context: Text tokens [seq_txt, B, H] for the first single block only
            timestep_emb: Timestep conditioning [B, H] (required)

        Returns:
            Tuple of (hidden_states, None)
        """
        if timestep_emb is None:
            raise ValueError("FluxSingleTransformerBlock requires timestep_emb for AdaLN conditioning.")

        emb = timestep_emb

        # TRANSITION HANDLING: Concatenate if this is first single block
        if context is not None:
            # This is layer 19 - concatenate text and image
            hidden_states = torch.cat([context, hidden_states], dim=0)

        residual = hidden_states

        # Get modulation parameters (shift, scale, gate)
        shift, scale, gate = self.adaln(emb)

        # Apply modulated layer normalization
        norm_hidden_states = self.adaln.modulated_layernorm(hidden_states, shift=shift, scale=scale)

        # MLP path
        mlp_hidden_states, mlp_bias = self.mlp(norm_hidden_states)

        # Attention path
        attention_output, attention_bias = self.self_attention(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Combine MLP and attention (parallel paths)
        hidden_states = mlp_hidden_states + mlp_bias + attention_output
        if attention_bias is not None:
            hidden_states = hidden_states + attention_bias

        # Gated residual connection
        hidden_states = self.adaln.scale_add(residual, x=hidden_states, gate=gate)

        return hidden_states, None

    def __call__(self, *args, **kwargs):
        """Override call to support CUDA graphs."""
        if hasattr(self, "cudagraph_manager"):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)


def get_flux_single_transformer_spec_for_backend(
    backend: BackendSpecProvider,
) -> ModuleSpec:
    """
    Get ModuleSpec for Flux single transformer block with backend support.

    This factory function creates the specification for a Flux single block
    using the provided backend spec provider for layer implementations.

    Args:
        backend: BackendSpecProvider (e.g., TESpecProvider, PrimusTurboSpecProvider)

    Returns:
        ModuleSpec for FluxSingleTransformerBlock with backend submodules
    """
    return ModuleSpec(
        module=FluxSingleTransformerBlock,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=FluxSingleAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                    k_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                    linear_proj=backend.row_parallel_linear(),
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=backend.column_parallel_linear(),
                    linear_fc2=backend.row_parallel_linear(),
                ),
            ),
        ),
    )


def get_flux_double_transformer_spec_for_backend(
    backend: BackendSpecProvider,
) -> ModuleSpec:
    """
    Get ModuleSpec for Flux double (joint) transformer block with backend support.

    This factory function creates the specification for a Flux MMDiT layer
    using the provided backend spec provider for layer implementations.

    Args:
        backend: BackendSpecProvider (e.g., TESpecProvider, PrimusTurboSpecProvider)

    Returns:
        ModuleSpec for MMDiTLayer with backend submodules
    """
    return ModuleSpec(
        module=MMDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                    k_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                    added_q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                    added_k_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                    linear_qkv=backend.column_parallel_linear(),
                    added_linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=backend.column_parallel_linear(),
                    linear_fc2=backend.row_parallel_linear(),
                ),
            ),
        ),
    )


def get_flux_layer_spec(
    config,
    backend: Optional[BackendSpecProvider] = None,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """
    Create heterogeneous layer specifications for Flux TransformerBlock.

    Builds a list of ModuleSpec objects for a unified TransformerBlock:
    - Layers 0 to num_joint_layers-1: MMDiTLayer specs (joint image-text processing)
    - Layers num_joint_layers to (num_joint_layers + num_single_layers - 1):
      FluxSingleTransformerBlock specs (concatenated processing)

    The layer specs are automatically sliced for pipeline parallelism based on
    pp_rank and vp_stage, following the pattern from get_gpt_heterogeneous_layer_spec.

    Args:
        config: FluxConfig with num_joint_layers, num_single_layers, hidden_size, etc.
        backend: BackendSpecProvider (auto-selected based on config.transformer_impl
            and available providers if None)
        vp_stage: Virtual pipeline stage number (for interleaved PP, optional)
        pp_rank: Pipeline parallel rank (optional, auto-detected if None)

    Returns:
        TransformerBlockSubmodules containing layer_specs and layer_norm

    Example:
        >>> from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        >>> config = FluxConfig.flux_12b()
        >>> backend = TESpecProvider()
        >>> spec = get_flux_layer_spec(config, backend=backend)
        >>> transformer = TransformerBlock(config=config, spec=spec)

    Reference:
        Similar to megatron/core/models/gpt/heterogeneous/heterogeneous_layer_specs.py
    """
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    # Default backend selection based on config
    sensitive_backend = None

    if backend is None:
        if config.transformer_impl == "local":
            if config.fp4 is not None and PrimusTurboMXFP4LocalSpecProvider is not None:
                backend = PrimusTurboMXFP4LocalSpecProvider()

                # Resolve sensitive layer backend
                sensitive_precision = getattr(config, "sensitive_layer_precision", "bf16")
                if sensitive_precision == "tw_fp8":
                    sensitive_backend = PrimusTurboFloat8LocalSpecProvider()
                elif sensitive_precision == "bf16":
                    sensitive_backend = PrimusTurboLocalSpecProvider()
            elif (
                config.fp8 is not None
                and HAVE_PRIMUS_TURBO_LOCAL
                and PrimusTurboFloat8LocalSpecProvider is not None
            ):
                backend = PrimusTurboFloat8LocalSpecProvider()
            elif HAVE_PRIMUS_TURBO_LOCAL and PrimusTurboLocalSpecProvider is not None:
                backend = PrimusTurboLocalSpecProvider()
            else:
                from megatron.core.models.backends import LocalSpecProvider

                backend = LocalSpecProvider()
        elif HAVE_TE_SPEC_PROVIDER:
            # Use TransformerEngine (current default)
            backend = TESpecProvider()
        else:
            # Fallback to pure native Megatron
            from megatron.core.models.backends import LocalSpecProvider

            backend = LocalSpecProvider()

    # Build per-layer specs with optional sensitive-layer heterogeneity
    sensitive_enabled = getattr(config, "sensitive_layers_enabled", False)
    num_start = getattr(config, "sensitive_layers_start", 0) if sensitive_enabled else 0
    num_end = getattr(config, "sensitive_layers_end", 0) if sensitive_enabled else 0
    total = config.num_joint_layers + config.num_single_layers

    layer_specs = []
    for i in range(total):
        is_sensitive = sensitive_backend is not None and ((i < num_start) or (i >= total - num_end))
        layer_backend = sensitive_backend if is_sensitive else backend

        if i < config.num_joint_layers:
            layer_specs.append(get_flux_double_transformer_spec_for_backend(layer_backend))
        else:
            layer_specs.append(get_flux_single_transformer_spec_for_backend(layer_backend))

    # Slice for pipeline parallelism (only if parallel state is initialized)
    try:
        if parallel_state.model_parallel_is_initialized():
            offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
            num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
            layer_specs = layer_specs[offset : offset + num_layers_to_build]
    except (AssertionError, RuntimeError):
        # Parallel state not initialized - use all layers (for single-device testing)
        pass

    # Get layer norm from backend
    layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)

    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=layer_norm)
