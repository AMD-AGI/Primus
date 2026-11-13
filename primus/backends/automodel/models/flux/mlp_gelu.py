###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX MLP's up-projection and GELU as one op, Primus-Turbo's ``flux_gelu_linear``.

WHY:
  Both FLUX block types run their MLP as an up-projection GEMM followed by a separate tanh-GELU
  kernel that reads the pre-activation back from memory, and the single-stream block then copies
  the attention output and the GELU output into one buffer for ``proj_out``. In the backward,
  autograd computes the GELU gradient in one pass and sums it for the bias gradient in another.

WHAT:
  ``flux_gelu_linear(x, w, b, prefix)`` returns ``cat([prefix, gelu_tanh(x @ w.T + b)])``: the GEMM
  runs as ``torch.addmm`` and one FlyDSL pass copies ``prefix`` and writes the GELU of the
  pre-activation straight into the concatenated buffer, so the single-stream block's
  ``torch.cat([attn_output, mlp_hidden_states])`` needs no pass of its own. Its backward computes the
  GELU gradient and the bias gradient in one Triton pass before the two GEMMs. The single-stream
  block's forward and the dual-stream blocks' ``FeedForward`` call it; anything that does not match
  (another activation, dtype or dropout, a swapped linear) keeps diffusers' forward.

NUMERICS:
  The GELU and its gradient are Inductor's ``gelu``/``gelu_backward`` expressions, operation for
  operation, and the bias gradient sums the BF16-rounded GELU gradient, as the unfused graph does;
  only the order of the FP32 column sums differs.

REQUIRES:
  Primus-Turbo with its FLUX ops; without them the patch is not applied. The single-stream forward
  re-implements diffusers 0.40.0's ``FluxSingleTransformerBlock.forward`` (checked by source hash; any
  other source keeps diffusers' forward there).

Settings (module config, ``primus_perf:`` section):
    flux_fused_gelu: false            apply this patch
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import logging

import torch

from primus.backends.automodel import options
from primus.backends.automodel.models.flux.plain_linear import is_plain_linear
from primus.backends.automodel.models.flux.turbo_ops import flux_ops

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_gelu"
# sha256 of diffusers 0.40.0's FluxSingleTransformerBlock.forward, which _install_single_block
# re-implements with the MLP fused; any other source keeps diffusers' forward.
SINGLE_FORWARD_SHA256 = "57350a6dab842b40e0d677b9c42794c17e45907ad45991d36dba69d89096a37b"


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_gelu")


def installed() -> bool:
    """Whether ``install()`` has run: the dual-stream ``FeedForward`` calls ``flux_gelu_linear``."""
    from diffusers.models.attention import FeedForward

    return getattr(FeedForward, _PATCHED_ATTR, False)


def fusable(x: torch.Tensor, linear: torch.nn.Linear, act_is_tanh_gelu: bool) -> bool:
    """Whether ``flux_gelu_linear`` serves ``act(linear(x))``; a swapped linear keeps its own forward."""
    return (
        act_is_tanh_gelu
        and is_plain_linear(linear)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and linear.weight.dtype == torch.bfloat16
        and linear.bias.dtype == torch.bfloat16
    )


# --- FLUX modules -------------------------------------------------------------------------------


def _install_feed_forward(ops) -> None:
    from diffusers.models.activations import GELU
    from diffusers.models.attention import FeedForward

    orig = FeedForward.forward

    @functools.wraps(orig)
    def forward(self, hidden_states, *args, **kwargs):
        net = self.net
        act = net[0] if len(net) == 3 else None
        if (
            args
            or kwargs
            or not isinstance(act, GELU)
            or not isinstance(net[1], torch.nn.Dropout)
            or (self.training and net[1].p > 0)
            or not isinstance(net[2], torch.nn.Linear)
            or not fusable(hidden_states, act.proj, act.approximate == "tanh")
        ):
            return orig(self, hidden_states, *args, **kwargs)
        lead = hidden_states.shape[:-1]
        mlp = ops.flux_gelu_linear(
            hidden_states.reshape(-1, hidden_states.shape[-1]), act.proj.weight, act.proj.bias
        )
        return net[2](mlp.view(*lead, mlp.shape[-1]))

    FeedForward.forward = forward


def _install_single_block(ops) -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    orig = FluxSingleTransformerBlock.forward
    digest = hashlib.sha256(inspect.getsource(orig).encode()).hexdigest()
    if digest != SINGLE_FORWARD_SHA256:
        logger.warning(
            "[PrimusFluxFusedGelu] FluxSingleTransformerBlock.forward is not diffusers 0.40.0's; "
            "single-stream blocks keep the unfused MLP."
        )
        return False

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        act = self.act_mlp
        if not fusable(
            hidden_states, self.proj_mlp, isinstance(act, torch.nn.GELU) and act.approximate == "tanh"
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        # diffusers 0.40.0's forward, with act_mlp(proj_mlp(x)) and the cat fused into one op.
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
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
        hidden_states = gate.unsqueeze(1) * self.proj_out(concat.view(batch, seq, -1))
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states

    FluxSingleTransformerBlock.forward = forward
    return True


def install() -> bool:
    from diffusers.models.attention import FeedForward

    if getattr(FeedForward, _PATCHED_ATTR, False):
        return True
    ops = flux_ops("PrimusFluxFusedGelu")
    if ops is None:
        return False
    _install_feed_forward(ops)
    single = _install_single_block(ops)
    setattr(FeedForward, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxFusedGelu] up-projection GEMM + GELU (+ concat) as Primus-Turbo's flux_gelu_linear: "
        f"dual-stream FeedForward{' and single-stream blocks' if single else ''}."
    )
    return True
