###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX single-stream block's adaLN (cat + LayerNorm + modulation) as one op each way,
Primus-Turbo's ``adaln_modulate``.

WHY:
  Inductor compiles the adaLN of a single-stream block into three kernels: a forward that writes
  the concatenated residual and the modulated LayerNorm output (a looped two-pass reduction, the
  rows being 3,072 wide), a LayerNorm backward that also adds the residual gradient, and a
  separate kernel for the shift and scale gradients, which re-reads the output gradient and the
  input: a whole extra pass per call. It also returns the two input gradients as slices of one
  buffer, which AOTAutograd copies to contiguous tensors at every block boundary of the backward.
  The op computes the shift and scale gradients inside the LayerNorm-backward pass and writes the
  input gradients as separate tensors.

WHAT:
  ``adaln_modulate(x_txt, x_img, scale, shift, eps) -> (r_txt, r_img, y)``: ``r_txt``/``r_img``
  are copies of the inputs for the residual add, ``y`` the modulated LayerNorm of their concat. The
  residual leaves as two tensors so that its gradient comes back as the two block-output
  gradients, which the backward reads directly: with a single concatenated residual, Inductor
  materializes their concatenation around the op and gives the fusion's gain back. The forward computes
  the block's ``AdaLayerNormZeroSingle`` itself (its modulation GEMM, then the op), so it takes only a
  stock one with an affine-free LayerNorm and no hooks. Other calls, and calls the op does not take
  (its ``adaln_modulate_shape_error``), keep ``mlp_gelu.py``'s forward.

NUMERICS:
  Inductor's arithmetic: 1 + scale rounded to BF16, FP32 inside, one rounding per output; only
  the order of the sums differs. A small fraction of outputs, input gradients and shift/scale
  gradients differ from Inductor's by one BF16 step; all are as close to an FP64 reference as
  Inductor's.

REQUIRES:
  gfx950 (the op runs unfused eager PyTorch arithmetic elsewhere, so the
  patch declines there) and Primus-Turbo with ``adaln_modulate`` and ``linear_gelu``. ``mlp_gelu.py``'s
  single-stream forward, not wrapped by another patch, which this forward re-implements with the adaLN
  replaced, and diffusers' ``AdaLayerNormZeroSingle.forward`` (0.38.0 to 0.43.0, checked by
  ``pinned_source``). Installs after it (priority 75) and before ``block_outputs.py`` (80), which wraps
  it and clones its two outputs; the clones cost nothing on this forward, which already returns two
  tensors of their own.

Settings (module config, ``primus_perf:`` section):
    flux_fused_adaln: false     apply this patch
"""
import functools
import logging

from primus.backends.automodel import options
from primus.backends.automodel.models.flux import mlp_gelu, pinned_source
from primus.backends.automodel.models.flux.plain_modules import (
    has_no_hooks,
    is_plain_layer_norm,
)
from primus.backends.automodel.models.flux.turbo_ops import turbo_ops

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_adaln"
# pinned_source digest of diffusers' (0.38.0 to 0.43.0) AdaLayerNormZeroSingle.forward.
ADA_NORM_FORWARD_DIGEST = "673055fd6685db65053d1dee96b78f098fc3d3d4b8e37a089f0c83100f4a7b5e"


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_adaln")


# --- the single-stream block -------------------------------------------------------------------


def make_forward(orig, ops):
    """``mlp_gelu.py``'s single-stream forward with cat + LayerNorm + modulation as ``adaln_modulate``;
    ``orig`` serves the calls it does not take."""
    from diffusers.models.normalization import AdaLayerNormZeroSingle
    from primus_turbo.pytorch.kernels.normalization.adaln_modulate_impl import (
        adaln_modulate_shape_error,
    )

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        norm = self.norm
        if (
            type(norm) is not AdaLayerNormZeroSingle
            or not has_no_hooks(norm)
            or not is_plain_layer_norm(norm.norm)
            or not mlp_gelu.single_block_fusable(self, hidden_states)
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        emb = norm.linear(norm.silu(temb))
        shift, scale, gate = emb.chunk(3, dim=1)
        if adaln_modulate_shape_error(encoder_hidden_states, hidden_states, scale, shift) is not None:
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        text_seq_len = encoder_hidden_states.shape[1]
        r_txt, r_img, norm_hidden_states = ops.adaln_modulate(
            encoder_hidden_states, hidden_states, scale, shift, norm.norm.eps
        )
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        gated = mlp_gelu.single_block_mlp_out(self, ops, norm_hidden_states, attn_output, gate)
        # diffusers' residual + gated, split into the text and image tokens, on the two residual halves.
        return r_txt + gated[:, :text_seq_len], r_img + gated[:, text_seq_len:]

    return forward


def install() -> bool:
    from diffusers.models.normalization import AdaLayerNormZeroSingle
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxSingleTransformerBlock.forward
    if orig is not mlp_gelu.single_forward():
        logger.warning(
            "[PrimusFluxAdaLN] needs mlp_gelu.py's single-stream forward (primus_perf.flux_fused_gelu), "
            "not wrapped by another patch; not applied."
        )
        return False
    if pinned_source.digest(AdaLayerNormZeroSingle.forward) != ADA_NORM_FORWARD_DIGEST:
        logger.warning(
            "[PrimusFluxAdaLN] AdaLayerNormZeroSingle.forward is not diffusers 0.38.0's to 0.43.0's; not applied."
        )
        return False
    ops = turbo_ops("PrimusFluxAdaLN", "adaln_modulate", "linear_gelu", gain_on_gfx950_only=True)
    if ops is None:
        return False
    FluxSingleTransformerBlock.forward = make_forward(orig, ops)
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxAdaLN] single-stream blocks run cat + LayerNorm + modulation as Primus-Turbo's "
        "adaln_modulate."
    )
    return True
