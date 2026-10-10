# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Configuration classes for diffusion models.

This module defines configuration dataclasses that extend Megatron-Core's
TransformerConfig to include diffusion-specific parameters.
"""

from dataclasses import dataclass
from dataclasses import fields as dataclasses_fields
from typing import Optional

from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.transformer_config import TransformerConfig

from . import mxfp6_gates

# The all-default gate set, used to tell "the recipe asked for this" from "this
# is just the default" when rejecting gates set without fp6.
_GATE_DEFAULTS = mxfp6_gates.Mxfp6Gates()

# Accepted values for `fp6`. Only one 6-bit format exists on gfx950 (E2M3, via AITER's
# A6W6 kernels), but keeping this a tuple mirrors how `fp4` names its format.
MXFP6_FORMATS = ("mxfp6",)

# Accepted values for `mxfp6_backward_precision`.
MXFP6_BACKWARD_PRECISIONS = ("mxfp6", "fp8")

# Accepted values for `mxfp6_weight_format`: the format the *weight* operand is packed in
# while activations stay MXFP6. "mxfp4" selects AITER's A6W4 GEMM for the forward and
# dgrad. wgrad is unaffected in either case -- it contracts the token dimension, so
# neither of its operands is the weight and there is nothing to narrow.
MXFP6_WEIGHT_FORMATS = ("mxfp6", "mxfp4")


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
        mxfp6_weight_format: Weight operand format, 'mxfp6' or 'mxfp4' (A6W4) (default: 'mxfp6')
        mxfp6_fused_wgrad_accum: MXFP6 wgrad writes weight.main_grad in place (default: False)
        mxfp6_apply_rope_fusion: Use Megatron's fused RoPE kernels (default: False)
        mxfp6_fused_mlp: Fold MLP bias+GELU into the packer, 'auto'/'on'/'off' (default: 'auto')
        mxfp6_strided_v: Feed attention V to the packer strided (default: False)
        mxfp6_grouped_mlp: Run both MLP linears as one grouped A6W6 GEMM (default: False)
        mxfp6_joint_qkv: Project Q, K and V in a single GEMM (default: False)
        mxfp6_fused_small_grads: Route small reduction grads into main_grad (default: False)
        mxfp6_grouped_gemm_functional: Call the grouped-MLP GEMM through a functional op (default: False)
        mxfp6_shared_grad_pack: Single block runs MLP + out-proj as one Function sharing a grad pack (default: False)
        mxfp6_shared_grad_pack_bias: ...and takes fc2's bias gradient from that pack's column sums (default: False)
        mxfp6_ln_bwd_fused_sum: Fold the LN-modulate backward's gradient sum into its kernel (default: False)
        mxfp6_joint_proj: Joint block's two out-projections as one Function sharing dO (default: False)
        mxfp6_gate_mul_pack: Single block's shared pack multiplies dy by the gate while packing (default: False)
        mxfp6_gate_mul_pack_bias: ...and takes fc2's bias gradient from that pack's column sums (default: False)
        mxfp6_ln_bwd_ns16: LN-modulate backward over 16 sequence slices instead of 8 (default: False)
        mxfp6_adaln_wgrad_main_grad: AdaLN modulation wgrad written straight into main_grad (default: False)
        mxfp6_fused_qk_rope: Fuse QK-norm and RoPE into one kernel (default: False)
        mxfp6_fused_qkv: Fuse the QKV norm+RoPE prologue, 'auto'/'on'/'off' (default: 'off')
        mxfp6_fused_ln_mod_bwd: Single-pass LN-modulate backward (default: False)
        mxfp6_norm_rope_pin: Pin the norm/RoPE autotune configs (default: False)
        mxfp6_rope_slice_legacy: Pre-fusion RoPE slice order; changes numerics (default: False)
        mxfp6_wgrad_a6w4: Pack the wgrad column operand as MXFP4; needs A6W4 (default: False)
        mxfp6_bwd_fp4_dgrad: dgrad GEMMs as A4W4 (MXFP4 both operands, f4gemm) (default: False)
        mxfp6_bwd_fp4_wgrad: wgrad GEMMs as A4W4 (MXFP4 both operands, f4gemm) (default: False)
        mxfp6_bwd_fp4_sr: stochastic rounding for the A4W4 gradient operand (default: False)
        mxfp6_bwd_fp4_a6w6_first / _last: keep the A6W6 backward in the first / last N blocks
            while the A4W4 backward gates are on (default: 0)
        mxfp6_fp4_scale_rounding_{grad,actw_hp,actw_fp4fwd}: E8M0 scale rule of the MXFP4 packs,
            'rceil' / 'm0' / 'm1' / 'm2' (default: 'rceil')
        mxfp6_fp4_hadamard_{fwd,dgrad,wgrad}: Hadamard of each A4W4 GEMM's operands, 'h32' / 'h16' / 'none'
            (default: 'h32')
        mxfp6_fp4_weight_2d: 32x32 block scaling of the weights' MXFP4 copies (default: False)
        mxfp6_fp4_sr_actw: stochastic rounding of the activations' / weights' MXFP4 backward copies (default: False)
        mxfp6_fwd_bf16_joint_img_fc1: joint blocks' image-stream fc1 forward in bf16 (default: False)
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

    # Format of the weight operand while activations stay MXFP6. "mxfp4" runs the forward
    # and dgrad GEMMs on AITER's A6W4 kernels, which halve the weight's operand traffic.
    #
    # It buys less than the narrower format suggests. A6W4 issues the same
    # v_mfma_scale_f32_16x16x128_f8f6f4 that A6W6 does, only with blgp=FP4 instead of
    # blgp=FP6, so the matrix pipe runs at exactly the same rate and the gain is operand
    # traffic alone, a modest per-GEMM gain that shrinks further once GEMM's share of the step
    # and the two-thirds eligibility are applied. And it costs accuracy: cosine against bf16 falls from 0.99919 to 0.99293.
    # Default "mxfp6" accordingly; this is opt-in and needs its own convergence validation.
    mxfp6_weight_format: str = "mxfp6"

    # Have the MXFP6 wgrad GEMM write weight.main_grad itself, replacing the elementwise
    # add Megatron's DDP hook would otherwise run over every gradient.
    #
    # Deliberately not Megatron's `gradient_accumulation_fusion`: that flag is read by
    # every plain linear too, so switching it on also routes Flux's 76 AdaLN projections
    # through `wgrad_gemm_accum_fp16`, which costs end-to-end step time. This field
    # moves only the MXFP6 ones.
    #
    # Note the reason is *not* that the fused wgrad is slower at those M=32 shapes -- an
    # earlier version of this comment said so and was wrong. Measured directly with an
    # fp32 main_grad it is faster at both production shapes. End to end it still lost,
    # because the change perturbs grad-reduce overlap, which is worth far more than the
    # kernel saving. Isolated kernel timings do not price changes to the gradient pipeline.
    #
    # The A6W6 store has no beta=1 accumulate epilogue, so it overwrites main_grad and is
    # only valid at one microbatch per optimizer step. Enforced per module, not here,
    # because the microbatch count is not known at config time.
    mxfp6_fused_wgrad_accum: bool = False

    # Turn on Megatron's fused RoPE kernels for this model.
    #
    # Primus-owned deliberately. Megatron's own `apply_rope_fusion` is cleared by
    # validate_args whenever position_embedding_type != "rope" (arguments.py:1232-1233),
    # and Flux never sets that type, so the Megatron-owned flag can never be true here --
    # setting it in YAML looks like it works and silently does nothing. A name Megatron
    # does not know survives validation, the same way `fp6` does.
    mxfp6_apply_rope_fusion: bool = False

    # ------------------------------------------------------------------
    # MXFP6 fusion gates
    #
    # These were environment variables until they were migrated here. They are
    # the difference between the measured recipe and a run that is several
    # percent slow while looking structurally identical -- every hand-written
    # kernel untouched, the compiled Triton layer roughly doubled -- so they
    # belong in the file that defines the run.
    #
    # All default off ("auto" where tri-state), matching the environment
    # defaults they replace: a config setting none of them behaves exactly as
    # the unset environment did. The values are pushed into the registry in
    # `mxfp6_gates` at the end of __post_init__; see that module for why the
    # read happens at point of use rather than at import.
    #
    # "on" versus "auto" matters for a submission: "auto" lets a module whose
    # shape the fusion cannot reproduce fall back silently, so the disclosed
    # implementation and the measured throughput can diverge. "on" makes that
    # an error.
    # ------------------------------------------------------------------
    mxfp6_fused_mlp: str = "auto"
    mxfp6_strided_v: bool = False
    mxfp6_grouped_mlp: bool = False
    mxfp6_joint_qkv: bool = False
    mxfp6_fused_small_grads: bool = False
    mxfp6_grouped_gemm_functional: bool = False
    mxfp6_shared_grad_pack: bool = False
    mxfp6_shared_grad_pack_bias: bool = False
    mxfp6_ln_bwd_fused_sum: bool = False
    mxfp6_joint_proj: bool = False
    mxfp6_gate_mul_pack: bool = False
    mxfp6_gate_mul_pack_bias: bool = False
    mxfp6_ln_bwd_ns16: bool = False
    mxfp6_adaln_wgrad_main_grad: bool = False
    # Pinned TunableOp solutions for the AdaLN GEMMs; see Mxfp6Gates.adaln_tunableop.
    mxfp6_adaln_tunableop: bool = False
    # "hipblaslt" or "turbo" ("aiter": deprecated alias of "turbo"); see Mxfp6Gates.adaln_gemm_backend.
    mxfp6_adaln_gemm_backend: str = "hipblaslt"
    mxfp6_fused_qk_rope: bool = False
    mxfp6_fused_qkv: str = "off"
    mxfp6_fused_ln_mod_bwd: bool = False
    mxfp6_norm_rope_pin: bool = False
    # Changes numerics; kept only so an A/B against pre-fusion results is
    # possible. A recipe should not set this.
    mxfp6_rope_slice_legacy: bool = False
    mxfp6_wgrad_a6w4: bool = False
    # A4W4 backward GEMMs (MXFP4 both operands); see Mxfp6Gates.bwd_fp4_*.
    mxfp6_bwd_fp4_dgrad: bool = False
    mxfp6_bwd_fp4_wgrad: bool = False
    mxfp6_bwd_fp4_sr: bool = False
    mxfp6_bwd_fp4_grouped_mlp: bool = False
    mxfp6_fwd_fp4_single_linear2: bool = False
    mxfp6_fwd_fp4_single_fc1: bool = False
    mxfp6_fwd_fp4_joint_mlp: bool = False
    # "all" or a comma list of img_fc1, img_fc2, txt_fc1, txt_fc2; see Mxfp6Gates.fwd_fp4_joint_mlp_parts.
    mxfp6_fwd_fp4_joint_mlp_parts: str = "all"
    mxfp6_fwd_fp4_joint_txt_proj: bool = False
    # Operand layout of every MX GEMM, "blob" or "tilescale"; see Mxfp6Gates.gemm_layout.
    mxfp6_gemm_layout: str = "blob"
    mxfp6_fwd_a6w4: bool = False
    mxfp6_packed_param_gather: bool = False
    mxfp6_packed_param_gather_fused_adam: bool = False
    mxfp6_packed_param_gather_prob_bits: int = 0
    mxfp6_packed_param_gather_neutral: str = ""
    mxfp6_packed_param_gather_transport: str = "planes"
    mxfp6_dgrad_emit_attn_delta: bool = False
    mxfp6_single_linear2_cat: bool = False
    mxfp6_attn_q_norm_rope: bool = False
    mxfp6_bwd_fp4_a6w6_first: int = 0
    mxfp6_bwd_fp4_a6w6_last: int = 0
    # MXFP4 quantization options of the A4W4 packs; see Mxfp6Gates.fp4_*.
    mxfp6_fp4_scale_rounding_grad: str = "rceil"
    mxfp6_fp4_scale_rounding_actw_hp: str = "rceil"
    mxfp6_fp4_scale_rounding_actw_fp4fwd: str = "rceil"
    mxfp6_fp4_hadamard_fwd: str = "h32"
    mxfp6_fp4_hadamard_dgrad: str = "h32"
    mxfp6_fp4_hadamard_wgrad: str = "h32"
    mxfp6_fp4_weight_2d: bool = False
    mxfp6_fp4_sr_actw: bool = False
    mxfp6_fwd_bf16_joint_img_fc1: bool = False

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
        if self.mxfp6_weight_format not in MXFP6_WEIGHT_FORMATS:
            raise ValueError(
                f"Unknown mxfp6_weight_format '{self.mxfp6_weight_format}'. "
                f"Choose from: {list(MXFP6_WEIGHT_FORMATS)}."
            )
        if self.mxfp6_weight_format != "mxfp6" and self.fp6 is None:
            raise ValueError(
                f"mxfp6_weight_format='{self.mxfp6_weight_format}' requires fp6 to be set "
                "(e.g. fp6: mxfp6); with no MXFP6 linears it has no effect."
            )
        # A6W4 narrows the weight in the forward and in dgrad. An FP8 backward replaces
        # dgrad entirely -- it re-quantizes the saved bf16 rather than reusing a packed
        # weight -- so the pair would silently degrade to forward-only W4, which is a
        # third of the already-small gain at the same accuracy cost. Reject rather than
        # leave it to be discovered from a disappointing profile.
        if self.mxfp6_weight_format == "mxfp4" and self.mxfp6_backward_precision != "mxfp6":
            raise ValueError(
                "mxfp6_weight_format='mxfp4' needs mxfp6_backward_precision='mxfp6', got "
                f"'{self.mxfp6_backward_precision}'. An FP8 backward does not consume the "
                "packed weight, so A6W4 would apply to the forward only."
            )

        # wgrad_a6w4 narrows a second operand on top of A6W4's already-narrowed
        # weight, so it is meaningless without A6W4 and would silently pack a
        # column half nothing consumes. As an environment variable this pairing
        # could not be checked; as a config key it can be.
        if self.mxfp6_wgrad_a6w4 and self.mxfp6_weight_format != "mxfp4":
            raise ValueError(
                "mxfp6_wgrad_a6w4=True requires mxfp6_weight_format='mxfp4' (A6W4), got "
                f"'{self.mxfp6_weight_format}'. It narrows the wgrad column operand on top "
                "of A6W4's weight; with an MXFP6 weight there is nothing for it to pair with."
            )

        # A4W4 backward replaces the backward GEMMs outright; the A6W4 paths would pack the
        # same operands a third way, so the two families are exclusive.
        if (self.mxfp6_bwd_fp4_dgrad or self.mxfp6_bwd_fp4_wgrad) and (
            self.mxfp6_weight_format == "mxfp4" or self.mxfp6_wgrad_a6w4
        ):
            raise ValueError(
                "mxfp6_bwd_fp4_dgrad / mxfp6_bwd_fp4_wgrad (A4W4 backward) exclude A6W4 "
                "(mxfp6_weight_format='mxfp4', mxfp6_wgrad_a6w4)."
            )
        if (
            self.mxfp6_bwd_fp4_dgrad or self.mxfp6_bwd_fp4_wgrad
        ) and self.mxfp6_backward_precision != "mxfp6":
            raise ValueError("A4W4 backward needs mxfp6_backward_precision='mxfp6'.")
        if self.mxfp6_gemm_layout not in mxfp6_gates.GEMM_LAYOUTS:
            raise ValueError(
                f"mxfp6_gemm_layout must be one of {list(mxfp6_gates.GEMM_LAYOUTS)}, got {self.mxfp6_gemm_layout!r}"
            )
        tilescale = self.mxfp6_gemm_layout == "tilescale"
        # The tilescale A4W4 backward stores each FP4 pack's scales for the GEMM that consumes it, so a gradient's two
        # directions are both A4W4 or neither.
        if tilescale and (self.mxfp6_bwd_fp4_dgrad != self.mxfp6_bwd_fp4_wgrad):
            raise ValueError("mxfp6_gemm_layout 'tilescale' needs both mxfp6_bwd_fp4_dgrad and _wgrad, or neither.")
        if (
            self.mxfp6_fwd_fp4_single_linear2
            or self.mxfp6_fwd_fp4_single_fc1
            or self.mxfp6_fwd_fp4_joint_mlp
            or self.mxfp6_fwd_fp4_joint_txt_proj
        ) and not (self.mxfp6_bwd_fp4_dgrad and self.mxfp6_bwd_fp4_wgrad and tilescale):
            raise ValueError(
                "mxfp6_fwd_fp4_single_linear2 / _single_fc1 / _joint_mlp / _joint_txt_proj need mxfp6_bwd_fp4_dgrad, "
                "_wgrad and mxfp6_gemm_layout 'tilescale'."
            )
        if tilescale and (self.mxfp6_wgrad_a6w4 or self.mxfp6_weight_format == "mxfp4"):
            raise ValueError(
                "mxfp6_gemm_layout 'tilescale' excludes the blob A6W4 paths (mxfp6_wgrad_a6w4, "
                "mxfp6_weight_format='mxfp4'); its A6W4 is mxfp6_fwd_a6w4."
            )
        if self.mxfp6_fwd_a6w4 and not tilescale:
            raise ValueError("mxfp6_fwd_a6w4 needs mxfp6_gemm_layout 'tilescale'.")
        if self.mxfp6_fwd_a6w4 and self.mxfp6_packed_param_gather:
            # The gathered weight kinds (packed_param_gather_patches.pack_kind) know only the fwd_fp4_* gates, so an
            # A6W4 weight would arrive as MXFP6 planes and the forward would refuse it at the first step.
            raise ValueError(
                "mxfp6_fwd_a6w4 is not wired into mxfp6_packed_param_gather yet (its weights gather as MXFP6 planes); "
                "set one of the two."
            )
        if self.mxfp6_fwd_bf16_joint_img_fc1 and self.mxfp6_packed_param_gather:
            # The packed gather moves only the MX packs; a bf16 forward reads the parameter itself, which only its
            # owner rank has updated, so every other rank would run that fc1 with a stale weight.
            raise ValueError(
                "mxfp6_fwd_bf16_joint_img_fc1 reads the bf16 weight, which mxfp6_packed_param_gather does not gather; "
                "set one of the two."
            )
        if self.mxfp6_packed_param_gather and not tilescale:
            raise ValueError("mxfp6_packed_param_gather needs mxfp6_gemm_layout 'tilescale'.")
        if self.mxfp6_packed_param_gather_fused_adam and not self.mxfp6_packed_param_gather:
            raise ValueError("mxfp6_packed_param_gather_fused_adam needs mxfp6_packed_param_gather.")
        if self.mxfp6_packed_param_gather_prob_bits not in (0, 2, 4):
            raise ValueError("mxfp6_packed_param_gather_prob_bits is 0 (off), 2 or 4.")
        if self.mxfp6_packed_param_gather_prob_bits and not (self.mxfp6_packed_param_gather and self.mxfp6_fp4_sr_actw):
            raise ValueError("mxfp6_packed_param_gather_prob_bits needs mxfp6_packed_param_gather and mxfp6_fp4_sr_actw.")
        if self.mxfp6_packed_param_gather_transport not in ("planes", "phased"):
            raise ValueError('mxfp6_packed_param_gather_transport is "planes" or "phased".')
        if self.mxfp6_packed_param_gather_transport != "planes" and not self.mxfp6_packed_param_gather:
            raise ValueError("mxfp6_packed_param_gather_transport needs mxfp6_packed_param_gather.")
        if self.mxfp6_packed_param_gather_neutral not in ("", "sr", "rn", "d2"):
            raise ValueError('mxfp6_packed_param_gather_neutral is "", "sr", "rn" or "d2".')
        if self.mxfp6_packed_param_gather_neutral == "d2" and not (
            self.mxfp6_packed_param_gather and self.mxfp6_bwd_fp4_dgrad and not self.mxfp6_fp4_sr_actw
        ):
            raise ValueError(
                'mxfp6_packed_param_gather_neutral "d2" needs mxfp6_packed_param_gather and mxfp6_bwd_fp4_dgrad, with the '
                "backward copies rounded to nearest (mxfp6_fp4_sr_actw off): every gathered plane is then deterministic."
            )
        if self.mxfp6_packed_param_gather_neutral in ("sr", "rn") and not (
            self.mxfp6_packed_param_gather and self.mxfp6_bwd_fp4_dgrad and self.mxfp6_fp4_hadamard_dgrad == "none"
        ):
            raise ValueError(
                "mxfp6_packed_param_gather_neutral needs mxfp6_packed_param_gather, mxfp6_bwd_fp4_dgrad and "
                'mxfp6_fp4_hadamard_dgrad "none" (the dgrad copy is made from an unrotated plane).'
            )
        if self.mxfp6_packed_param_gather_neutral == "sr" and not self.mxfp6_fp4_sr_actw:
            raise ValueError('mxfp6_packed_param_gather_neutral "sr" rounds the weight copy stochastically: set mxfp6_fp4_sr_actw.')
        if self.mxfp6_packed_param_gather_neutral and self.mxfp6_packed_param_gather_fused_adam:
            raise ValueError("mxfp6_packed_param_gather_neutral is not wired into the fused Adam + owner pack.")
        if self.mxfp6_packed_param_gather_prob_bits and self.mxfp6_packed_param_gather_fused_adam:
            raise ValueError(
                "mxfp6_packed_param_gather_prob_bits is not wired into the fused Adam + owner pack; set one of the two."
            )
        if self.mxfp6_dgrad_emit_attn_delta and not (tilescale and self.mxfp6_bwd_fp4_dgrad):
            raise ValueError("mxfp6_dgrad_emit_attn_delta needs mxfp6_gemm_layout 'tilescale' and mxfp6_bwd_fp4_dgrad.")
        if self.mxfp6_single_linear2_cat and not (tilescale and self.mxfp6_gate_mul_pack):
            raise ValueError("mxfp6_single_linear2_cat needs mxfp6_gemm_layout 'tilescale' and mxfp6_gate_mul_pack.")
        if self.mxfp6_fwd_fp4_joint_txt_proj and not self.mxfp6_joint_proj:
            raise ValueError("mxfp6_fwd_fp4_joint_txt_proj runs inside the joint out-projection pair: set mxfp6_joint_proj.")
        if self.mxfp6_attn_q_norm_rope and not self.mxfp6_strided_v:
            raise ValueError("mxfp6_attn_q_norm_rope needs mxfp6_strided_v (q leaves the QKV Function as a view, like v).")
        fp4_opts = {
            k: getattr(self, k)
            for k in (
                "mxfp6_fp4_scale_rounding_grad",
                "mxfp6_fp4_scale_rounding_actw_hp",
                "mxfp6_fp4_scale_rounding_actw_fp4fwd",
                "mxfp6_fp4_hadamard_fwd",
                "mxfp6_fp4_hadamard_dgrad",
                "mxfp6_fp4_hadamard_wgrad",
                "mxfp6_fp4_weight_2d",
                "mxfp6_fp4_sr_actw",
            )
        }
        if fp4_opts != {k: getattr(type(self), k) for k in fp4_opts}:
            if not (self.mxfp6_bwd_fp4_dgrad or self.mxfp6_bwd_fp4_wgrad):
                raise ValueError(
                    "the mxfp6_fp4_* options shape the A4W4 packs; they need mxfp6_bwd_fp4_dgrad or _wgrad."
                )
        if self.mxfp6_fwd_bf16_joint_img_fc1 and (
            self.mxfp6_wgrad_a6w4 or self.mxfp6_weight_format == "mxfp4"
        ):
            raise ValueError(
                "mxfp6_fwd_bf16_joint_img_fc1 packs its backward operands as MXFP6 or A4W4, not A6W4."
            )
        if self.mxfp6_bwd_fp4_sr and not (self.mxfp6_bwd_fp4_dgrad or self.mxfp6_bwd_fp4_wgrad):
            raise ValueError(
                "mxfp6_bwd_fp4_sr rounds the A4W4 gradient; it needs mxfp6_bwd_fp4_dgrad or _wgrad."
            )

        # Publish the fusion gates before anything builds a model. The modules
        # that consume them are imported long before this runs, which is
        # precisely why they read through `gates()` at point of use instead of
        # binding a module-level constant at import.
        #
        # Validated unconditionally, including when fp6 is None: a recipe that
        # sets a gate but forgets `fp6: mxfp6` should hear about the typo rather
        # than have the whole block silently ignored.
        mxfp6_gates.configure(self)

        # norm_rope_pin is consumed by a @triton.autotune decorator applied at
        # import, so unlike the other gates it needs an explicit narrowing step
        # once the value is known. Imported here rather than at module scope
        # because it pulls in triton, which a CPU-only config build (and much of
        # the unit-test suite) has no reason to require.
        if self.fp6 is not None and self.mxfp6_norm_rope_pin:
            try:
                from .fused_norm_rope import apply_autotune_pin

                apply_autotune_pin()
            except ImportError:
                # No triton: the fused kernels cannot run either, so whatever
                # selects them will fail with a clearer message than this would.
                pass

        if self.fp6 is None:
            requested = [
                f.name
                for f in dataclasses_fields(mxfp6_gates.Mxfp6Gates)
                if getattr(self, f"mxfp6_{f.name}") != getattr(_GATE_DEFAULTS, f.name)
            ]
            if requested:
                raise ValueError(
                    "MXFP6 fusion gates were set without fp6: "
                    f"{sorted(requested)}. Set fp6: mxfp6, or remove them -- with no "
                    "MXFP6 linears they have no effect and the run would silently be "
                    "slower than the config claims."
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
