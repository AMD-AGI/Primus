###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX dual-stream block's gated residual + second adaLN as one op each way, Primus-Turbo's
``gated_residual_adaln``.

WHY:
  After its attention, each stream of a dual-stream block computes ``x1 = x + gate_msa * a`` (``a``
  the attention's output projection) and the modulated LayerNorm ``y = LN(x1) * (1 + scale_mlp) +
  shift_mlp`` that feeds its MLP. Inductor's backward for that is a row pass that writes the
  normalized rows as an FP32 buffer and reads it back, then two more passes that re-read the
  gradients for the column sums of the scale/shift and gate gradients. The op computes the column
  sums inside the row pass, from the normalized rows it recomputes in registers.

WHAT:
  ``gated_residual_adaln(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps) ->
  (x1_img, y_img, x1_txt, y_txt)`` for both streams in one launch: ``x1 = x + g * a`` and
  ``y = LN(x1) * (1 + s) + h``. ``g``, ``s`` and ``h`` are the [B, C] chunks of the stream's
  modulation GEMM output. The block's final residual reads ``x1``, so the residual gradient comes
  back as the gradient of ``x1``, which the backward adds; the first adaLN's backward (Inductor's)
  adds the returned gradient of ``x`` to its own. The two streams may differ in length (FLUX.1-dev's
  512 text tokens); each length is a multiple of 32. Calls the op does not take (its
  ``gated_residual_adaln_shape_error``) keep diffusers' forward.

NUMERICS:
  Inductor's arithmetic: ``x1`` in FP32 (``x + g * a``), ``1 + s`` in FP32, one rounding per output;
  the row statistics combined exactly (Chan et al.) instead of by Welford's update. The LayerNorm
  output and all gradients are as close to FP64 as Inductor's or closer. ``x1`` is stored in BF16
  for the final residual, which Inductor computed from the FP32 value: the block output carries one
  more BF16 rounding step.

REQUIRES:
  gfx950 (the op runs unfused eager PyTorch arithmetic elsewhere, so the
  patch declines there) and Primus-Turbo with ``gated_residual_adaln``; diffusers'
  ``FluxTransformerBlock.forward`` (0.38.0 to 0.43.0, checked by ``pinned_source``; not wrapped by another
  patch), BF16 activations and the diffusers attention processor (two outputs). Other calls keep
  diffusers' forward.

Settings (module config, ``primus_perf:`` section):
    flux_fused_adaln_dual: false     apply this patch
"""
import functools
import logging

from primus.backends.automodel import options
from primus.backends.automodel.models.flux import pinned_source
from primus.backends.automodel.models.flux.plain_modules import is_plain_layer_norm
from primus.backends.automodel.models.flux.turbo_ops import turbo_ops

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_adaln_dual"
# pinned_source digest of diffusers' (0.38.0 to 0.43.0) FluxTransformerBlock.forward.
DUAL_FORWARD_DIGEST = "80d2db652a7562eb359529e84c6e8d58b561b0df5328dcabf95225f92954e4b3"


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_adaln_dual")


# --- the dual-stream block ---------------------------------------------------------------------


def make_forward(orig, ops):
    """diffusers' dual-stream forward with each stream's gated residual + second LayerNorm and modulation
    as ``gated_residual_adaln``; ``orig`` serves the calls it does not take."""
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    from primus_turbo.pytorch.kernels.normalization.gated_residual_adaln_impl import (
        gated_residual_adaln_shape_error,
    )

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        if (
            joint_attention_kwargs
            or not isinstance(self.attn.processor, FluxAttnProcessor)
            or not (is_plain_layer_norm(self.norm2) and is_plain_layer_norm(self.norm2_context))
            or self.norm2.eps != self.norm2_context.eps
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        # The attention outputs have their streams' shape and dtype, so the streams stand in for them
        # before the attention runs.
        if (
            gated_residual_adaln_shape_error(
                hidden_states,
                hidden_states,
                gate_msa,
                scale_mlp,
                shift_mlp,
                encoder_hidden_states,
                encoder_hidden_states,
                c_gate_msa,
                c_scale_mlp,
                c_shift_mlp,
            )
            is not None
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        x1, norm_hidden_states, c_x1, norm_encoder_hidden_states = ops.gated_residual_adaln(
            hidden_states,
            attn_output,
            gate_msa,
            scale_mlp,
            shift_mlp,
            encoder_hidden_states,
            context_attn_output,
            c_gate_msa,
            c_scale_mlp,
            c_shift_mlp,
            self.norm2.eps,
        )
        hidden_states = x1 + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)
        encoder_hidden_states = c_x1 + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_encoder_hidden_states)
        return encoder_hidden_states, hidden_states

    return forward


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

    if getattr(FluxTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxTransformerBlock.forward
    if pinned_source.digest(orig) != DUAL_FORWARD_DIGEST:
        logger.warning(
            "[PrimusFluxAdaLNDual] FluxTransformerBlock.forward is not diffusers 0.38.0's to 0.43.0's "
            "(or another patch wrapped it); not applied."
        )
        return False
    ops = turbo_ops("PrimusFluxAdaLNDual", "gated_residual_adaln", gain_on_gfx950_only=True)
    if ops is None:
        return False
    FluxTransformerBlock.forward = make_forward(orig, ops)
    setattr(FluxTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxAdaLNDual] dual-stream blocks run the gated residual + second adaLN as Primus-Turbo's "
        "gated_residual_adaln."
    )
    return True
