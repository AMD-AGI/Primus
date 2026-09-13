# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Configuration classes for diffusion models.

This module defines configuration dataclasses that extend Megatron-Core's
TransformerConfig to include diffusion-specific parameters.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class BaseDiffusionConfig(TransformerConfig):
    """
    Base configuration for all diffusion models in Primus.

    This class extends Megatron-Core's TransformerConfig to add common
    diffusion model parameters. Model-specific configurations (FluxConfig,
    DiTConfig, etc.) should inherit from this class.

    Attributes:
        model_type: Type of diffusion model (e.g., 'flux', 'dit', 'moviegen')
        in_channels: Number of input channels in latent space
        out_channels: Number of output channels (default: same as in_channels)
        patch_size: Patch size for patchification (if applicable)
        fp8_scaling_strategy: FP8 scaling strategy for local spec provider (default: 'dynamic')
        fp8_force_nt_layout: FP8 backward GEMM layout (default: False)
        fp8_reduce_amax: Whether to allreduce amax across ranks (default: False)
        mxfp4_backward_precision: MXFP4 backward precision, 'mxfp4' or 'fp8' (default: 'mxfp4')
        mxfp4_gradient_stochastic_rounding: Stochastic rounding on gradients (default: False)
        sensitive_layers_enabled: Enable sensitive layer configuration (default: False)
        sensitive_layers_start: Number of sensitive layers at start (default: 0)
        sensitive_layers_end: Number of sensitive layers at end (default: 0)
        sensitive_layer_precision: Precision for sensitive layers (default: 'bf16')
        outer_sensitive_layers_start: Number of outermost starting layers that
            override the sensitive-layer precision (default: 0)
        outer_sensitive_layers_end: Number of outermost ending layers that
            override the sensitive-layer precision (default: 0)
        outer_sensitive_layer_precision: Precision for the outer override
            (default: 'bf16')

    Inherited from TransformerConfig:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        ffn_hidden_size: FFN intermediate dimension
        layernorm_epsilon: LayerNorm epsilon value
        bf16, fp16, params_dtype: Precision settings
        And many more Megatron-Core transformer parameters...
    """

    # Model identification
    model_type: str = "base"

    # Input/output dimensions
    in_channels: int = 64
    out_channels: Optional[int] = None  # Defaults to in_channels if None

    # Patchification
    patch_size: int = 1

    # FP8 scaling strategy for local spec provider
    fp8_scaling_strategy: str = "dynamic"

    # FP8 backward GEMM layout for the local spec provider (tensorwise path only).
    # False (default) = native layouts (dgrad=NN, wgrad=TN), the validated 0-NaN path
    # on hipBLASLt 1.3. True = forced-NT (every GEMM normalized to NT via pre-transposed
    # operands); faster on some stacks but NaN-prone on hipBLASLt 1.3 (gfx950).
    # Only affects ScalingGranularity.TENSORWISE; rowwise/blockwise ignore it.
    fp8_force_nt_layout: bool = False

    # Whether to allreduce amax across DP/TP ranks for delayed FP8 scaling
    fp8_reduce_amax: bool = False

    # MXFP4 backward precision: "mxfp4" (pure) or "fp8" (hybrid)
    mxfp4_backward_precision: str = "mxfp4"

    # Stochastic rounding on MXFP4 gradients (paper Section 4.4)
    mxfp4_gradient_stochastic_rounding: bool = False

    # Sensitive layer configuration (clean naming, maps to Megatron internals)
    sensitive_layers_enabled: bool = False
    sensitive_layers_start: int = 0
    sensitive_layers_end: int = 0
    sensitive_layer_precision: str = "bf16"  # "bf16", "tw_fp8", or "mxfp8" (future)
    outer_sensitive_layers_start: int = 0
    outer_sensitive_layers_end: int = 0
    outer_sensitive_layer_precision: str = "bf16"

    def __post_init__(self):
        """Post-initialization processing."""
        # Pipeline parallelism is not implemented for diffusion models: the
        # forward path runs embeddings/output head on every rank and does not
        # relay activations between stages, so PP > 1 would silently
        # miscompute. Reject it explicitly (before TransformerConfig validation)
        # rather than producing wrong results.
        if self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "Diffusion models do not support pipeline parallelism; "
                f"got pipeline_model_parallel_size={self.pipeline_model_parallel_size}. "
                "Set pipeline_model_parallel_size=1."
            )

        layer_count_fields = (
            "sensitive_layers_start",
            "sensitive_layers_end",
            "outer_sensitive_layers_start",
            "outer_sensitive_layers_end",
        )
        for field_name in layer_count_fields:
            value = getattr(self, field_name)
            if type(value) is not int:
                raise ValueError(f"{field_name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")

        sensitive_count = self.sensitive_layers_start + self.sensitive_layers_end
        outer_sensitive_count = self.outer_sensitive_layers_start + self.outer_sensitive_layers_end
        if (sensitive_count or outer_sensitive_count) and not self.sensitive_layers_enabled:
            raise ValueError("sensitive layer counts require sensitive_layers_enabled=True")

        active_precisions = set()
        if self.sensitive_layers_enabled:
            if self.num_layers <= 1:
                raise ValueError(
                    "sensitive_layers_enabled=True requires num_layers to be set by the child config "
                    "BEFORE calling super().__post_init__(). Set self.num_layers in your model config's "
                    "__post_init__ before the super() call."
                )
            if self.sensitive_layers_start + self.sensitive_layers_end <= 0:
                raise ValueError("sensitive_layers_enabled=True but both start and end counts are 0")
            if self.sensitive_layers_start + self.sensitive_layers_end > self.num_layers:
                raise ValueError(
                    f"sensitive_layers_start ({self.sensitive_layers_start}) + "
                    f"sensitive_layers_end ({self.sensitive_layers_end}) exceeds "
                    f"num_layers ({self.num_layers})"
                )
            if self.outer_sensitive_layers_start > self.sensitive_layers_start:
                raise ValueError(
                    "outer_sensitive_layers_start "
                    f"({self.outer_sensitive_layers_start}) exceeds "
                    f"sensitive_layers_start ({self.sensitive_layers_start})"
                )
            if self.outer_sensitive_layers_end > self.sensitive_layers_end:
                raise ValueError(
                    "outer_sensitive_layers_end "
                    f"({self.outer_sensitive_layers_end}) exceeds "
                    f"sensitive_layers_end ({self.sensitive_layers_end})"
                )

            if self.transformer_impl != "local":
                raise ValueError(
                    "sensitive layer routing requires transformer_impl='local'; "
                    f"got {self.transformer_impl!r}"
                )
            if self.fp4 != "mxfp4" or self.fp4_recipe != "mxfp4":
                raise ValueError(
                    "sensitive layer routing requires fp4='mxfp4' and "
                    f"fp4_recipe='mxfp4'; got fp4={self.fp4!r}, "
                    f"fp4_recipe={self.fp4_recipe!r}"
                )

            inner_sensitive_count = (
                self.sensitive_layers_start + self.sensitive_layers_end - outer_sensitive_count
            )
            if inner_sensitive_count > 0:
                active_precisions.add(self.sensitive_layer_precision)
            if outer_sensitive_count > 0:
                active_precisions.add(self.outer_sensitive_layer_precision)
            unsupported_precisions = active_precisions - {"bf16", "tw_fp8"}
            if unsupported_precisions:
                raise ValueError(
                    "sensitive layer precision must be 'bf16' or 'tw_fp8'; "
                    f"got {sorted(unsupported_precisions)!r}"
                )

            if outer_sensitive_count > 0:
                collapsed_start = (
                    self.outer_sensitive_layers_start > 0
                    and self.outer_sensitive_layers_start == self.sensitive_layers_start
                )
                collapsed_end = (
                    self.outer_sensitive_layers_end > 0
                    and self.outer_sensitive_layers_end == self.sensitive_layers_end
                )
                if collapsed_start or collapsed_end:
                    raise ValueError(
                        "each graduated outer boundary requires at least one "
                        "inner sensitive layer on the same side"
                    )
                if (
                    self.sensitive_layer_precision != "tw_fp8"
                    or self.outer_sensitive_layer_precision != "bf16"
                ):
                    raise ValueError(
                        "graduated sensitive routing requires inner precision "
                        "'tw_fp8' and outer precision 'bf16'"
                    )

            if "tw_fp8" in active_precisions:
                if self.fp8 not in (None, "e4m3"):
                    raise ValueError(
                        "tw_fp8 sensitive layers require fp8=None or fp8='e4m3'; " f"got {self.fp8!r}"
                    )
                if self.fp8_recipe not in (
                    None,
                    Fp8Recipe.delayed,
                    Fp8Recipe.tensorwise,
                ):
                    raise ValueError(
                        "tw_fp8 sensitive layers require fp8_recipe='tensorwise' "
                        f"or the deferred 'delayed' default; got {self.fp8_recipe!r}"
                    )
            if "bf16" in active_precisions and (
                not self.bf16 or self.fp16 or self.params_dtype != torch.bfloat16
            ):
                raise ValueError(
                    "bf16 sensitive layers require bf16=True, fp16=False, "
                    f"and params_dtype=torch.bfloat16; got bf16={self.bf16!r}, "
                    f"fp16={self.fp16!r}, params_dtype={self.params_dtype!r}"
                )

            # The FP4 context uses these legacy fields to exclude the complete
            # heterogeneous boundary from MXFP4. The Flux layer spec chooses
            # BF16 versus FP8 within that excluded boundary.
            self.first_last_layers_bf16 = True
            self.num_layers_at_start_in_bf16 = self.sensitive_layers_start
            self.num_layers_at_end_in_bf16 = self.sensitive_layers_end

        uses_tw_fp8 = self.sensitive_layers_enabled and "tw_fp8" in active_precisions
        if uses_tw_fp8:
            # Megatron rejects global FP4+FP8 before it knows these formats are
            # assigned to disjoint layers. Hide validated FP8 settings during
            # parent validation, then restore the canonical local FP8 state.
            _deferred_fp8 = "e4m3"
            _deferred_fp8_recipe = Fp8Recipe.tensorwise
            self.fp8 = None
        else:
            _deferred_fp8 = None
            _deferred_fp8_recipe = None

        super().__post_init__()

        # Apply deferred FP8 settings for sensitive layers (set after super to
        # avoid Megatron's "fp4 and fp8 cannot coexist" validation).
        if _deferred_fp8 is not None:
            self.fp8 = _deferred_fp8
        if _deferred_fp8_recipe is not None:
            self.fp8_recipe = _deferred_fp8_recipe

        # Re-run the FP8 validations that Megatron skipped because self.fp8 was
        # None during super().__post_init__() (TransformerConfig lines 988-1017).
        if self.fp8 and self.sensitive_layers_enabled:
            if self.first_last_layers_bf16 and self.fp8_recipe == Fp8Recipe.delayed:
                raise ValueError("Delayed scaling does not support first / last layer in BF16.")
            max_bf16 = self.num_layers // self.pipeline_model_parallel_size
            if self.first_last_layers_bf16:
                if not (0 <= self.num_layers_at_start_in_bf16 <= max_bf16):
                    raise ValueError(
                        f"num_layers_at_start_in_bf16 ({self.num_layers_at_start_in_bf16}) "
                        f"must be between 0 and {max_bf16}."
                    )
                if not (0 <= self.num_layers_at_end_in_bf16 <= max_bf16):
                    raise ValueError(
                        f"num_layers_at_end_in_bf16 ({self.num_layers_at_end_in_bf16}) "
                        f"must be between 0 and {max_bf16}."
                    )

        if self.out_channels is None:
            self.out_channels = self.in_channels

        # Run configuration validation on construction. (Subclass fields used by
        # validate() are plain dataclass fields, so they are already populated.)
        self.validate()

    def validate(self):
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")

        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")

        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
