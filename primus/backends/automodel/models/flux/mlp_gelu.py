###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX MLP's up-projection and GELU as one op, Primus-Turbo's ``linear_gelu``.

WHY:
  Both FLUX block types run their MLP as an up-projection GEMM followed by a separate tanh-GELU
  kernel that reads the pre-activation back from memory, and the single-stream block then copies
  the attention output and the GELU output into one buffer for ``proj_out``. In the backward,
  autograd computes the GELU gradient in one pass and sums it for the bias gradient in another.

WHAT:
  ``linear_gelu(x, w, b, prefix)`` returns ``cat([prefix, gelu_tanh(x @ w.T + b)])``: the GEMM
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
  Primus-Turbo with ``linear_gelu``; without it the patch is not applied. The two forwards re-implement
  diffusers' ``FeedForward.forward`` with its ``GELU.forward`` and ``FluxSingleTransformerBlock.forward``
  (diffusers 0.38.0 to 0.43.0, checked by ``pinned_source``); where the code differs, that forward stays
  diffusers'.

Settings (module config, ``primus_perf:`` section):
    flux_fused_gelu: false            apply this patch
"""
from __future__ import annotations

import functools
import logging
from typing import Callable, Optional

import torch

from primus.backends.automodel import options
from primus.backends.automodel.models.flux import pinned_source
from primus.backends.automodel.models.flux.plain_modules import is_plain_linear
from primus.backends.automodel.models.flux.turbo_ops import turbo_ops

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_gelu"
# pinned_source digests of diffusers' (0.38.0 to 0.43.0) functions the forwards here re-implement.
SINGLE_FORWARD_DIGEST = "4147609a93fdc84e8f399b8e33808981d0240d61a9fd2daff0a42d88fa5cb3a9"
FEED_FORWARD_DIGEST = "b8fcf08644813857eaa56f64664442582d7b8cd8296275efdb102301e5eb9f93"
GELU_FORWARD_DIGEST = "07f1ace97bbeb237f9347ced72bf8913564e21854df1464e77693b16898a1b38"

_single_forward: Optional[Callable] = None


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_gelu")


def single_forward() -> Optional[Callable]:
    """The single-stream forward ``install()`` put on ``FluxSingleTransformerBlock``, None before."""
    return _single_forward


def fusable(x: torch.Tensor, linear: torch.nn.Linear, act_is_tanh_gelu: bool) -> bool:
    """Whether ``linear_gelu`` serves ``act(linear(x))``; a swapped linear keeps its own forward."""
    return (
        act_is_tanh_gelu
        and is_plain_linear(linear)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and linear.weight.dtype == torch.bfloat16
        and linear.bias.dtype == torch.bfloat16
    )


def single_block_fusable(block, hidden_states: torch.Tensor) -> bool:
    """Whether ``linear_gelu`` serves the single-stream block's ``act_mlp(proj_mlp(x))``."""
    act = block.act_mlp
    return fusable(
        hidden_states, block.proj_mlp, isinstance(act, torch.nn.GELU) and act.approximate == "tanh"
    )


def single_block_mlp_out(block, ops, norm_hidden_states, attn_output, gate) -> torch.Tensor:
    """diffusers' ``gate * proj_out(cat([attn_output, act_mlp(proj_mlp(norm_hidden_states))]))``, the
    MLP and the cat as one ``linear_gelu``."""
    batch, seq, dim = norm_hidden_states.shape
    concat = ops.linear_gelu(
        norm_hidden_states.reshape(batch * seq, dim),
        block.proj_mlp.weight,
        block.proj_mlp.bias,
        attn_output.reshape(batch * seq, -1),
    )
    return gate.unsqueeze(1) * block.proj_out(concat.view(batch, seq, -1))


def _matches(tag: str, fn, expected: str) -> bool:
    if pinned_source.digest(fn) == expected:
        return True
    logger.warning(
        "[PrimusFluxFusedGelu] %s is not diffusers 0.38.0's to 0.43.0's (or another patch wrapped it); "
        "it keeps the unfused MLP.",
        tag,
    )
    return False


# --- FLUX modules -------------------------------------------------------------------------------


def _install_feed_forward(ops) -> bool:
    from diffusers.models.activations import GELU
    from diffusers.models.attention import FeedForward

    orig = FeedForward.forward
    if not (
        _matches("FeedForward.forward", orig, FEED_FORWARD_DIGEST)
        and _matches("GELU.forward", GELU.forward, GELU_FORWARD_DIGEST)
    ):
        return False

    @functools.wraps(orig)
    def forward(self, hidden_states, *args, **kwargs):
        net = self.net
        act = net[0] if len(net) == 3 else None
        if (
            args
            or kwargs
            or type(act) is not GELU
            or not isinstance(net[1], torch.nn.Dropout)
            or (self.training and net[1].p > 0)
            or not isinstance(net[2], torch.nn.Linear)
            or not fusable(hidden_states, act.proj, act.approximate == "tanh")
        ):
            return orig(self, hidden_states, *args, **kwargs)
        lead = hidden_states.shape[:-1]
        mlp = ops.linear_gelu(
            hidden_states.reshape(-1, hidden_states.shape[-1]), act.proj.weight, act.proj.bias
        )
        return net[2](mlp.view(*lead, mlp.shape[-1]))

    FeedForward.forward = forward
    setattr(FeedForward, _PATCHED_ATTR, True)
    return True


def _install_single_block(ops) -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    global _single_forward
    orig = FluxSingleTransformerBlock.forward
    if not _matches("FluxSingleTransformerBlock.forward", orig, SINGLE_FORWARD_DIGEST):
        return False

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        if not single_block_fusable(self, hidden_states):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        # diffusers' forward, with act_mlp(proj_mlp(x)) and the cat fused into one op; fusable() admits
        # only BF16, so its FP16 clip has nothing to do.
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        hidden_states = residual + single_block_mlp_out(self, ops, norm_hidden_states, attn_output, gate)
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states

    FluxSingleTransformerBlock.forward = forward
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    _single_forward = forward
    return True


def install() -> bool:
    """Fuse the MLP in the dual-stream ``FeedForward`` and in the single-stream block, each where its
    diffusers code is the pinned one. Idempotent; True if either was fused."""
    from diffusers.models.attention import FeedForward
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    if any(getattr(cls, _PATCHED_ATTR, False) for cls in (FeedForward, FluxSingleTransformerBlock)):
        return True
    ops = turbo_ops("PrimusFluxFusedGelu", "linear_gelu")
    if ops is None:
        return False
    fused = [
        name
        for name, done in (
            ("dual-stream FeedForward", _install_feed_forward(ops)),
            ("single-stream blocks", _install_single_block(ops)),
        )
        if done
    ]
    if not fused:
        return False
    logger.info(
        "[PrimusFluxFusedGelu] up-projection GEMM + GELU (+ concat) as Primus-Turbo's linear_gelu: %s.",
        " and ".join(fused),
    )
    return True
