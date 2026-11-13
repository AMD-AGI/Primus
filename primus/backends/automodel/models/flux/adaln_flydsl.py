###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX single-stream block's adaLN (cat + LayerNorm + modulation) as one op each way,
Primus-Turbo's ``flux_adaln_single``.

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
  ``flux_adaln_single(x_txt, x_img, scale, shift, eps) -> (r_txt, r_img, y)``: ``r_txt``/``r_img``
  are copies of the inputs for the residual add, ``y`` the modulated LayerNorm of their concat. The
  residual leaves as two tensors so that its gradient comes back as the two block-output
  gradients, which the backward reads directly: with a single concatenated residual, Inductor
  materializes their concatenation around the op and gives the fusion's gain back. Calls the op does
  not take (its ``adaln_single_shape_error``) keep ``mlp_gelu.py``'s forward.

NUMERICS:
  Inductor's arithmetic: 1 + scale rounded to BF16, FP32 inside, one rounding per output; only
  the order of the sums differs. A small fraction of outputs, input gradients and shift/scale
  gradients differ from Inductor's by one BF16 step; all are as close to an FP64 reference as
  Inductor's.

REQUIRES:
  gfx950 and Primus-Turbo with its FLUX ops. ``mlp_gelu.py``'s single-stream forward, on diffusers
  0.40.0's (source hash), which this forward re-implements with the adaLN replaced. Installs after it
  (priority 75) and before ``block_outputs.py`` and ``last_block.py``, which wrap it; the image-only
  last block keeps Inductor's adaLN.

Settings (module config, ``primus_perf:`` section):
    flux_fused_adaln_flydsl: false     apply this patch
"""
import functools
import hashlib
import inspect
import logging

import torch

from primus.backends.automodel import options
from primus.backends.automodel.models.flux.turbo_ops import flux_ops

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_adaln_flydsl"


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_adaln_flydsl")


# --- the single-stream block -------------------------------------------------------------------


def make_forward(orig, ops):
    """``mlp_gelu.py``'s single-stream forward with cat + LayerNorm + modulation as ``flux_adaln_single``;
    ``orig`` serves the calls it does not take."""
    from primus_turbo.pytorch.kernels.flux.adaln_impl import adaln_single_shape_error

    from primus.backends.automodel.models.flux import mlp_gelu

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        act = self.act_mlp
        norm = self.norm
        if not mlp_gelu.fusable(
            hidden_states, self.proj_mlp, isinstance(act, torch.nn.GELU) and act.approximate == "tanh"
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        emb = norm.linear(norm.silu(temb))
        shift, scale, gate = emb.chunk(3, dim=1)
        if adaln_single_shape_error(encoder_hidden_states, hidden_states, scale, shift) is not None:
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        text_seq_len = encoder_hidden_states.shape[1]
        r_txt, r_img, norm_hidden_states = ops.flux_adaln_single(
            encoder_hidden_states, hidden_states, scale, shift, norm.norm.eps
        )
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        batch, seq, dim = norm_hidden_states.shape
        concat = ops.flux_gelu_linear(
            norm_hidden_states.reshape(batch * seq, dim),
            self.proj_mlp.weight,
            self.proj_mlp.bias,
            attn_output.reshape(batch * seq, -1),
        )
        gated = gate.unsqueeze(1) * self.proj_out(concat.view(batch, seq, -1))
        # diffusers' residual + gated, split into the text and image tokens, on the two residual halves.
        return r_txt + gated[:, :text_seq_len], r_img + gated[:, text_seq_len:]

    return forward


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    from primus.backends.automodel.models.flux import block_outputs, mlp_gelu

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxSingleTransformerBlock.forward
    diffusers_forward = inspect.unwrap(orig)
    digest = hashlib.sha256(inspect.getsource(diffusers_forward).encode()).hexdigest()
    if not mlp_gelu.installed() or orig is diffusers_forward or digest != mlp_gelu.SINGLE_FORWARD_SHA256:
        logger.warning(
            "[PrimusFluxAdaLN] needs mlp_gelu.py's single-stream forward (primus_perf.flux_fused_gelu) "
            "on diffusers 0.40.0's; not applied."
        )
        return False
    ops = flux_ops("PrimusFluxAdaLN", needs_gfx950=True)
    if ops is None:
        return False
    FluxSingleTransformerBlock.forward = make_forward(orig, ops)
    # Read by block_outputs.py: this forward returns two fresh tensors, nothing to clone.
    setattr(FluxSingleTransformerBlock, block_outputs.FRESH_OUTPUTS_ATTR, True)
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxAdaLN] single-stream blocks run cat + LayerNorm + modulation as Primus-Turbo's "
        "flux_adaln_single."
    )
    return True
