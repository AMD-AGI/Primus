# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Configuration for Flux diffusion model.

Flux is a flow-based diffusion model with MMDiT (Multimodal Diffusion Transformer)
architecture that uses separate "joint" and "single" transformer blocks.

Reference:
    - https://github.com/black-forest-labs/flux
    - NeMo's Flux implementation
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..common.config import BaseDiffusionConfig

# Try to import erf_gelu from megatron, fallback to custom implementation
try:
    from megatron.core.transformer.utils import erf_gelu
except ImportError:
    # Fallback used when Megatron's erf_gelu is unavailable
    def erf_gelu(x):
        """GELU activation using error function approximation."""
        return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


__all__ = ["FluxConfig", "openai_gelu_no_jit", "erf_gelu"]


# Custom non-JIT compiled openai_gelu to avoid ROCm bugs
def openai_gelu_no_jit(x):
    """
    OpenAI's GELU implementation without JIT compilation (for ROCm compatibility).

    This is the tanh-based approximation of GELU used in the original Flux model.
    We use a non-JIT version to avoid ROCm compilation bugs that cause NaN values.
    """
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


@dataclass
class FluxConfig(BaseDiffusionConfig):
    """
    Flux-specific configuration.

    Configuration for Flux flow-based diffusion model. Flux uses a dual-stream
    architecture with joint (multimodal) and single (image-only) transformer layers.

    Standard Configurations:
        - Flux.1 [dev/schnell]: 12B parameters (19 joint + 38 single layers)
        - Flux 535M: Minimal config for testing (1 joint + 1 single layer)

    Key Configuration Groups:
        - Architecture: num_joint_layers, num_single_layers, hidden_size
        - Context: context_dim (T5), vec_in_dim (CLIP), model_channels (timestep)
        - Position Encoding: theta, axes_dim, rotary_interleaved
        - Guidance: guidance_embed, guidance_scale

    For architecture details, see Flux class documentation.

    Attributes:
        num_joint_layers: Number of joint (multimodal) transformer layers
        num_single_layers: Number of single (image-only) transformer layers
        context_dim: Dimension of text context embeddings (T5-XXL: 4096)
        vec_in_dim: Dimension of pooled text embeddings (CLIP-L: 768)
        model_channels: Channels for timestep embedding (default: 256)
        guidance_embed: Whether to use guidance embedding for CFG
        guidance_scale: Guidance scale for classifier-free guidance (default: 3.5)
        theta: Base for RoPE position embeddings (default: 10000)
        axes_dim: Dimension for each axis in 3D RoPE (default: [16, 56, 56])
        rotary_interleaved: Whether RoPE dimensions are interleaved (default: True)
        apply_rope_fusion: Whether to apply RoPE fusion optimization (default: False)
        add_qkv_bias: Whether to add bias to QKV projections (default: True)
        single_block_bias: Whether to add bias to single block linear layers (default: True)
        activation_func: Activation function (default: openai_gelu_no_jit)
        use_te_rng_tracker: Whether to use Transformer Engine RNG tracker (default: False)
    """

    # Model identification
    model_type: str = "flux"

    # Dummy layer for compatibility (actual layers defined below)
    num_layers: int = 1  # Not used in Flux, kept for compatibility

    # Architecture: Number of layers
    num_joint_layers: int = 19  # Default: Flux 12B
    num_single_layers: int = 38  # Default: Flux 12B

    # Architecture: Dimensions (Flux standard: 3072)
    hidden_size: int = 3072
    num_attention_heads: int = 24

    # Input dimensions
    in_channels: int = 64  # Packed latent channels (16 VAE channels * 4 from 2x2 patch packing)

    # Context dimensions
    context_dim: int = 4096  # T5-XXL hidden dimension
    vec_in_dim: int = 768  # CLIP-L pooled dimension
    model_channels: int = 256  # Channels for timestep embedding

    # Guidance (for classifier-free guidance)
    guidance_embed: bool = False  # Set True for CFG support
    guidance_scale: float = 3.5  # CFG guidance scale
    cfg_dropout_prob: float = (
        0.0  # Probability of replacing text embeddings with empty encodings (MLPerf: 0.1)
    )

    # Training: timestep sampling strategy
    # Options: "logit_normal" (SD3 default), "direct_uniform" (NVIDIA MLPerf), "uniform", "mode"
    timestep_sampling_strategy: str = "logit_normal"

    # Position embeddings (RoPE)
    theta: int = 10000  # Base for RoPE frequencies
    axes_dim: tuple = (16, 56, 56)  # 3D RoPE: (channels, height, width)
    rotary_interleaved: bool = True  # Whether RoPE dimensions are interleaved
    apply_rope_fusion: bool = False  # Whether to apply RoPE fusion optimization

    # Patchification (Flux uses 1x1 patches by default)
    patch_size: int = 1

    # Attention configuration
    add_qkv_bias: bool = True  # Whether to add bias to QKV projections

    # Single block configuration
    single_block_bias: bool = True  # Whether to add bias to single block linear layers

    # Default: non-JIT tanh GELU (ROCm-safe). Override via YAML activation_func:
    # "openai_gelu" -> fused F.gelu(approximate="tanh"); "erf_gelu" -> erf-based GELU.
    activation_func: Callable = openai_gelu_no_jit

    # Dropout (Flux typically uses 0)
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # Normalization
    layernorm_epsilon: float = 1e-6

    # Initialization
    use_cpu_initialization: bool = True

    # Optimization flags
    gradient_accumulation_fusion: bool = False
    use_dual_fp8_output_projection: bool = False
    use_triton_ops: bool = False
    adaln_plain_ops: bool = False
    adaln_always_jit_fuser: bool = False

    # FSDP2 prefetch depth: number of layers to prefetch ahead for all-gather overlap.
    # 1 = prefetch next layer (default), 2 = prefetch next 2 layers, 0 = no prefetch.
    fsdp_prefetch_depth: int = 1

    # FP8 all-gather data cache: batch-quantize all weights upfront and cache
    # the FP8 data so fsdp_pre_all_gather skips per-layer quantization.
    # When false, only scales are precomputed; quantization happens on-demand.
    fp8_precompute_data_cache: bool = True

    # Optimizer foreach batching: use _foreach_copy_ for batched grad/weight
    # copies instead of per-parameter loops.
    optimizer_foreach: bool = True

    # Overlap grad norm with reduce-scatter: accumulate squared norms
    # incrementally inside FSDP2's post-reduce stream via
    # register_post_accumulate_grad_hook, replacing the full recompute
    # in clip_grad_norm with a single all-reduce + sqrt + clip.
    overlap_grad_norm: bool = False

    # Use the C++ quantize_fp8 kernel from primus_turbo for tensorwise FP8
    # quantization instead of the default inline implementation.
    use_cpp_fp8_quantize: bool = False

    # CUDA graph support
    enable_cuda_graph: bool = False
    cuda_graph_scope: Optional[str] = None  # Options: "full", "full_iteration"
    cuda_graph_warmup_steps: int = 2

    # Transformer Engine
    use_te_rng_tracker: bool = False

    # Torch compile configuration (applied AFTER distributed setup)
    enable_torch_compile: bool = False  # Enable compilation (applied after FSDP/DDP wrapping)
    torch_compile_backend: str = "inductor"  # Compilation backend
    torch_compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    torch_compile_fullgraph: bool = False  # Allow graph breaks (recommended for distributed)
    torch_compile_optimizer: bool = False  # Whether to compile optimizer step
    torch_compile_optimizer_scope: str = (
        "full"  # "full" = compile FSDP2FP32Optimizer.step(), "inner_only" = compile only inner AdamW.step()
    )

    # Selective stack compilation (TE spec only — local spec falls back to per_block)
    torch_compile_strategy: str = "per_block"
    # "per_block": compile each transformer layer individually (default, recommended)
    # "whole_model": compile entire forward (incompatible with overlap_param_gather)
    # "double_stack": compile only the double (joint) block loop
    # "single_stack": compile only the single block loop
    # "stack": compile both double and single block loops
    # "full_dit": compile the entire DiT (double + cat + single + output) as one region
    torch_compile_replace_qk_rmsnorm: bool = False
    torch_compile_disable_inductor_cudagraphs: bool = True
    torch_compile_emulate_precision_casts: bool = True  # Preserve eager BF16 precision in Triton kernels
    torch_compile_fused_ln_modulate: bool = True  # Use fused LN+modulate Triton kernel in AdaLN

    def __post_init__(self):
        """Post-initialization processing."""
        # MUST set num_layers BEFORE super().__post_init__() -- convention for BaseDiffusionConfig
        # Required because BaseDiffusionConfig maps sensitive_layer_* fields using num_layers,
        # and TransformerConfig validates num_layers_at_start_in_bf16 against num_layers.
        self.num_layers = self.num_joint_layers + self.num_single_layers

        super().__post_init__()  # BaseDiffusionConfig (mapping) -> TransformerConfig (validation)

        # Xavier uniform for Megatron parallel linear layers (common Flux reference default).
        self.init_method = nn.init.xavier_uniform_
        self.output_layer_init_method = nn.init.xavier_uniform_

        # Ensure axes_dim is a tuple
        if isinstance(self.axes_dim, list):
            self.axes_dim = tuple(self.axes_dim)

    def validate(self):
        """
        Validate Flux-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation
        super().validate()

        # Flux-specific validations
        if self.num_joint_layers <= 0:
            raise ValueError(f"num_joint_layers must be positive, got {self.num_joint_layers}")

        if self.num_single_layers <= 0:
            raise ValueError(f"num_single_layers must be positive, got {self.num_single_layers}")

        if self.context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {self.context_dim}")

        if self.vec_in_dim <= 0:
            raise ValueError(f"vec_in_dim must be positive, got {self.vec_in_dim}")

        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")

        if len(self.axes_dim) != 3:
            raise ValueError(f"axes_dim must have 3 elements for 3D RoPE, got {len(self.axes_dim)}")

        if any(d <= 0 for d in self.axes_dim):
            raise ValueError(f"All axes_dim values must be positive, got {self.axes_dim}")

        # Validate RoPE fusion constraints
        if self.apply_rope_fusion:
            import warnings

            warnings.warn(
                "\n"
                "=" * 80 + "\n"
                "RoPE Fusion Enabled: Same-Resolution Batch Requirement\n"
                "=" * 80 + "\n"
                "RoPE fusion optimization requires ALL images in each training batch to have\n"
                "the SAME resolution (height and width). Variable-resolution batches will\n"
                "produce incorrect positional encodings and degrade model quality.\n"
                "\n"
                "Recommended data pipeline configurations:\n"
                "  1. Fixed resolution: All images resized to same size (e.g., 512x512)\n"
                "  2. Resolution bucketing: Group images by resolution in separate batches\n"
                "  3. Aspect ratio bucketing: Use bucketing with consistent dimensions\n"
                "\n"
                "=" * 80,
                UserWarning,
                stacklevel=2,
            )

    def get_num_layers(self):
        """
        Get total number of transformer layers.

        Returns:
            Total number of layers (joint + single)
        """
        return self.num_joint_layers + self.num_single_layers

    @classmethod
    def flux_535m(cls, **kwargs):
        """
        Create configuration for Flux 535M (minimal model for testing).

        Args:
            **kwargs: Override default parameters

        Returns:
            FluxConfig instance
        """
        defaults = {
            "num_joint_layers": 1,
            "num_single_layers": 1,
            "hidden_size": 3072,
            "num_attention_heads": 24,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def flux_12b(cls, **kwargs):
        """
        Create configuration for Flux 12B (standard model).

        Args:
            **kwargs: Override default parameters

        Returns:
            FluxConfig instance
        """
        defaults = {
            "num_joint_layers": 19,
            "num_single_layers": 38,
            "hidden_size": 3072,
            "num_attention_heads": 24,
        }
        defaults.update(kwargs)
        return cls(**defaults)
