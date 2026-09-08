# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Configuration classes for diffusion models.

This module defines configuration dataclasses that extend Megatron-Core's
TransformerConfig to include diffusion-specific parameters.
"""

from dataclasses import dataclass
from typing import Optional

from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig

# Accepted values for `fp6`. Only one 6-bit format exists on gfx950 (E2M3, via AITER's
# A6W6 kernels), but keeping this a tuple mirrors how `fp4` names its format.
MXFP6_FORMATS = ("mxfp6",)

# Accepted values for `mxfp6_backward_precision`.
MXFP6_BACKWARD_PRECISIONS = ("mxfp6", "fp8")


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
        fp6: Set to 'mxfp6' to run linears in MXFP6 (E2M3). None disables (default: None)
        mxfp6_backward_precision: MXFP6 backward precision, 'mxfp6' or 'fp8' (default: 'mxfp6')
        mxfp6_fused_wgrad_accum: MXFP6 wgrad writes weight.main_grad in place (default: False)
        sensitive_layers_enabled: Enable sensitive layer configuration (default: False)
        sensitive_layers_start: Number of sensitive layers at start (default: 0)
        sensitive_layers_end: Number of sensitive layers at end (default: 0)
        sensitive_layer_precision: Precision for sensitive layers (default: 'bf16')

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

    # MXFP6 (E2M3). Declared here rather than on TransformerConfig because Megatron has
    # no notion of a 6-bit format, so unlike `fp4` this is Primus-owned -- which also
    # means Megatron's "fp4 and fp8 cannot coexist" validation never sees it and the
    # cross-checks below are the only place those combinations can be rejected.
    fp6: Optional[str] = None

    # MXFP6 backward precision: "mxfp6" (pure) or "fp8" (hybrid), mirroring
    # mxfp4_backward_precision.
    mxfp6_backward_precision: str = "mxfp6"

    # Have the MXFP6 wgrad GEMM write weight.main_grad itself, replacing the elementwise
    # add Megatron's DDP hook would otherwise run over every gradient.
    #
    # Deliberately not Megatron's `gradient_accumulation_fusion`: that flag is read by
    # every plain linear too, and switching it on routes Flux's 76 AdaLN projections
    # through `wgrad_gemm_accum_fp16`, which at their M=32 shapes is slower than the
    # separate add it replaces -- measured at +11% step time on 8x MI355X, swamping the
    # saving on the MXFP6 linears. This field moves only the MXFP6 ones.
    #
    # The A6W6 store has no beta=1 accumulate epilogue, so it overwrites main_grad and is
    # only valid at one microbatch per optimizer step. Enforced per module, not here,
    # because the microbatch count is not known at config time.
    mxfp6_fused_wgrad_accum: bool = False

    # Sensitive layer configuration (clean naming, maps to Megatron internals)
    sensitive_layers_enabled: bool = False
    sensitive_layers_start: int = 0
    sensitive_layers_end: int = 0
    sensitive_layer_precision: str = "bf16"  # "bf16", "tw_fp8", or "mxfp8" (future)

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
            self.first_last_layers_bf16 = True
            self.num_layers_at_start_in_bf16 = self.sensitive_layers_start
            self.num_layers_at_end_in_bf16 = self.sensitive_layers_end

        # MXFP6 cross-checks. Since `fp6` is Primus-owned, nothing downstream would
        # notice a nonsense combination -- the layer spec would just pick one provider
        # and silently ignore the other request.
        if self.fp6 is not None:
            if self.fp6 not in MXFP6_FORMATS:
                raise ValueError(f"Unknown fp6 '{self.fp6}'. Choose from: {list(MXFP6_FORMATS)}.")
            if getattr(self, "fp4", None) is not None:
                raise ValueError(
                    f"fp4 ('{self.fp4}') and fp6 ('{self.fp6}') cannot both be set: the "
                    "layer spec selects one linear implementation per model."
                )
            if self.fp8 is not None:
                raise ValueError(
                    f"fp6 ('{self.fp6}') and fp8 ('{self.fp8}') cannot both be set, "
                    "mirroring Megatron's fp4/fp8 exclusion. For an MXFP6 forward with "
                    "an FP8 backward use mxfp6_backward_precision='fp8' instead."
                )
            # Read through getattr because the MXFP4 -> FP8 switch is a separate change
            # that may not be present; the check has to hold once both are, without
            # making this branch depend on it.
            switch_iter = int(getattr(self, "mxfp4_to_fp8_switch_iter", 0) or 0)
            if switch_iter > 0:
                # The switch patch walks the model for MXFP4ColumnParallelLinear /
                # MXFP4RowParallelLinear and *skips* anything else, so with fp6 it would
                # build a plan over zero layers and quietly never switch. Reject the
                # combination rather than extend the patch: its prewarm and ramp logic
                # are written around MXFP4 and there is no verified MXFP6 equivalent.
                raise ValueError(
                    f"fp6 ('{self.fp6}') cannot be combined with mxfp4_to_fp8_switch_iter="
                    f"{switch_iter}. The switch only converts MXFP4 "
                    "linears, of which an MXFP6 model has none. Use "
                    "mxfp6_backward_precision='fp8' for a hybrid MXFP6 run."
                )
        if self.mxfp6_backward_precision not in MXFP6_BACKWARD_PRECISIONS:
            raise ValueError(
                f"Unknown mxfp6_backward_precision '{self.mxfp6_backward_precision}'. "
                f"Choose from: {list(MXFP6_BACKWARD_PRECISIONS)}."
            )
        if self.mxfp6_backward_precision != "mxfp6" and self.fp6 is None:
            raise ValueError(
                f"mxfp6_backward_precision='{self.mxfp6_backward_precision}' requires fp6 "
                "to be set (e.g. fp6: mxfp6); with no MXFP6 linears it has no effect."
            )
        if self.mxfp6_fused_wgrad_accum and self.fp6 is None:
            raise ValueError(
                "mxfp6_fused_wgrad_accum=True requires fp6 to be set (e.g. fp6: mxfp6); "
                "with no MXFP6 linears it has no effect."
            )

        if self.sensitive_layers_enabled and self.sensitive_layer_precision == "tw_fp8":
            _deferred_fp8 = "e4m3" if self.fp8 is None else None
            _deferred_fp8_recipe = (
                Fp8Recipe.tensorwise
                if self.fp8_recipe is None or self.fp8_recipe == Fp8Recipe.delayed
                else None
            )
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
