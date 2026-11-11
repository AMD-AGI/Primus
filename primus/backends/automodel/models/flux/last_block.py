###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The last FLUX single-stream block computes only what the model reads: its image tokens.

WHY:
  ``FluxTransformer2DModel.forward`` keeps only the image tokens of the last single-stream
  block (``norm_out`` and ``proj_out`` read ``hidden_states``; the block's
  ``encoder_hidden_states`` output is dropped). The block still runs its MLP, ``proj_out``,
  gate and residual over all tokens, and in the backward the text tokens' rows of those GEMMs
  carry exact zeros. Each block is compiled on its own, so Inductor cannot see that the text
  outputs are dead.

WHAT:
  At the model's first forward, marks the last block of ``single_transformer_blocks``. That
  block keeps the shared AdaLN norm and the attention over all tokens (its keys and values
  need the text tokens), then runs the MLP (``mlp_gelu.py``'s fused op when it is
  installed), ``proj_out``, the gate and the residual on the image tokens only. It returns the
  incoming text tokens, detached, in place of the text output: nothing reads them, and a
  detached tensor gets no FSDP2 backward hook. Every other block runs the forward installed
  before (diffusers', ``mlp_gelu.py``'s, wrapped by ``block_outputs.py``).

NUMERICS:
  The image tokens get the same operations in the same order; only the MLP and ``proj_out``
  GEMMs run on the image rows alone, which can pick other kernels, so results agree within
  BF16 rounding rather than bit for bit. The gradients the text tokens receive are the same:
  from the attention's keys and values only, as before (their MLP and ``proj_out`` gradients
  were zero).

REQUIRES:
  diffusers 0.40.0's ``FluxTransformer2DModel.forward`` (checked by source hash; otherwise the
  patch is not applied). Installs after ``mlp_gelu.py`` and ``block_outputs.py``.

Settings (module config, ``primus_perf:`` section):
    flux_last_block_image_only: false     apply this patch
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import logging

import torch

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_last_block_image_only"
_IMAGE_ONLY_ATTR = "_primus_image_only"
# sha256 of diffusers 0.40.0's FluxTransformer2DModel.forward, which drops the last
# single-stream block's encoder_hidden_states output.
_MODEL_FORWARD_SHA256 = "c99bfbbf7bf8622f7ed42c0135fbf36b2ec055f495971a74ea18871dd2a11d2c"


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_last_block_image_only")


def _fused_mlp(block, x: torch.Tensor) -> bool:
    from primus.backends.automodel.models.flux import mlp_gelu

    act = block.act_mlp
    return mlp_gelu.installed() and mlp_gelu.fusable(
        x, block.proj_mlp, isinstance(act, torch.nn.GELU) and act.approximate == "tanh"
    )


def _image_only_forward(
    self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
):
    text_seq_len = encoder_hidden_states.shape[1]
    tokens = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    norm_tokens, gate = self.norm(tokens, emb=temb)
    attn_output = self.attn(
        hidden_states=norm_tokens, image_rotary_emb=image_rotary_emb, **(joint_attention_kwargs or {})
    )
    norm_image, attn_image = norm_tokens[:, text_seq_len:], attn_output[:, text_seq_len:]
    batch, seq, dim = norm_image.shape
    mlp_input = norm_image.reshape(batch * seq, dim)
    if _fused_mlp(self, mlp_input):
        from primus.backends.automodel.models.flux.mlp_gelu import gelu_linear

        concat, _ = gelu_linear(
            mlp_input, self.proj_mlp.weight, self.proj_mlp.bias, attn_image.reshape(batch * seq, -1)
        )
        concat = concat.view(batch, seq, -1)
    else:
        concat = torch.cat([attn_image, self.act_mlp(self.proj_mlp(norm_image))], dim=2)
    hidden_states = hidden_states + gate.unsqueeze(1) * self.proj_out(concat)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states.detach(), hidden_states


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
        FluxTransformer2DModel,
    )

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    digest = hashlib.sha256(inspect.getsource(FluxTransformer2DModel.forward).encode()).hexdigest()
    if digest != _MODEL_FORWARD_SHA256:
        logger.warning(
            "[PrimusFluxLastBlock] FluxTransformer2DModel.forward is not diffusers 0.40.0's; not applied."
        )
        return False

    block_forward = FluxSingleTransformerBlock.forward

    @functools.wraps(block_forward)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        if getattr(self, _IMAGE_ONLY_ATTR, False):
            return _image_only_forward(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        return block_forward(
            self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
        )

    model_forward = FluxTransformer2DModel.forward

    @functools.wraps(model_forward)
    def model_forward_marking_last_block(self, *args, **kwargs):
        blocks = self.single_transformer_blocks
        if len(blocks) and not getattr(blocks[-1], _IMAGE_ONLY_ATTR, False):
            setattr(blocks[-1], _IMAGE_ONLY_ATTR, True)
            logger.info(
                "[PrimusFluxLastBlock] single_transformer_blocks.%d computes its image tokens only.",
                len(blocks) - 1,
            )
        return model_forward(self, *args, **kwargs)

    FluxSingleTransformerBlock.forward = forward
    FluxTransformer2DModel.forward = model_forward_marking_last_block
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxLastBlock] the last single-stream block will skip its text tokens' MLP and proj_out."
    )
    return True
