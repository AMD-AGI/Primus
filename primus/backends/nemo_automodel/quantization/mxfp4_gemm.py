###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP4 GEMM entry points for the linear swap.

Two of them, and both delegate the FP4 numerics to Primus-Turbo's own
``gemm_fp4`` rather than reproducing its quantization recipes:

  ``mxfp4_gemm``        pure MXFP4 forward and backward, with the token padding
                        described below.
  ``mxfp4_fwd_fp8_bwd`` MXFP4 forward, tensorwise FP8 backward.

WHY DELEGATION IS THE POINT HERE:
  Turbo sets the scaling recipes for every quantization site -- 2D-block flags,
  the random Hadamard transform on the transposed operands that feed the weight
  gradient, the scale and output shuffle flags, and the backend those flags
  imply. Holding a local copy of that recipe table is how this code fell behind
  the library: it was written against a Turbo whose flags were spelled
  differently and whose preshuffle and gradient-SR settings had not yet become
  config fields. Neither entry point below reads a Turbo internal, so neither can
  drift that way again.

=============================================================================
THE TOKEN PADDING, AND WHY IT IS UNCONDITIONAL
=============================================================================
AITER's FP4 GEMM requires the *contraction* dimension K to be a multiple of 256.
Outside that it does not raise -- it returns wrong numbers. That is the whole
difficulty: there is no error to catch and no signal in the logs, only a model
that trains badly.

A Linear does three GEMMs, and they contract over different things:

    forward   out    = x @ W^T          K = in_features
    dgrad     grad_x = grad_out @ W     K = out_features
    wgrad     grad_W = grad_out^T @ x   K = tokens

``in_features`` and ``out_features`` are static, and the swap's skip-list already
refuses any Linear where they are not 128-aligned, so the forward and the dgrad
are safe before the model ever runs. **Only the weight gradient contracts over
tokens**, and the token count is ``sequence_length x batch`` -- a runtime
property that no swap-time check can see.

For self-attention on a long sequence that count is usually a comfortable
multiple of 256 by accident. For a cross-attention key or value projection it is
the *text* sequence length times batch, which is short and set by the tokenizer,
so it lands off the multiple routinely. Those layers then get a silently wrong
weight gradient while every other layer is fine -- which is why the symptom
appears as a training problem rather than a kernel problem.

The fix is to pad the token dimension up to a multiple of 256. Padding with zeros
is exact rather than approximate, and it is worth being precise about why: the
weight gradient is ``grad_out^T @ x``, a sum over tokens, so a token whose
activation and gradient rows are both zero contributes exactly zero to every
element of the result. This is unlike the sequence padding on the FP8 attention
path, where padded keys do perturb the softmax denominator and the argument is
that the perturbation is bounded. Here there is nothing to bound.

It is unconditional because the alternative is worse. Making it opt-in means the
default is a mode that computes some weight gradients incorrectly, and the shapes
that need it cannot be identified without knowing the token count at runtime. When
the count is already aligned the pad is skipped entirely, so the aligned case pays
nothing.

Padding via composition, rather than inside a custom autograd Function, is what
keeps this honest: ``pad`` and the slice that undoes it are ordinary
differentiable operations, so the backward pass gets the padded gradient and the
un-padded input gradient for free, with no second implementation to keep in step.
HIPBLASLT is not an escape from any of this -- its FP4 solutions want the token
*output* dimension aligned on the forward and dgrad instead, so it trades one
constraint for another.
"""
from __future__ import annotations

import logging

import torch

from primus.backends.nemo_automodel.quantization import _fp4_common

logger = logging.getLogger(__name__)


def _pad_tokens(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    """Zero-pad a 2-D ``(tokens, features)`` tensor's token dim up to ``multiple``.

    Returns ``(padded, real_tokens)``, and returns ``x`` itself when it already
    conforms so the aligned case allocates nothing.
    """
    tokens = x.shape[0]
    padded_tokens = _fp4_common.pad_multiple(tokens, multiple)
    if padded_tokens == tokens:
        return x, tokens
    # F.pad fills from the last dim backwards: (feat_lo, feat_hi, tok_lo, tok_hi).
    return torch.nn.functional.pad(x, (0, 0, 0, padded_tokens - tokens)), tokens


def mxfp4_gemm(x: torch.Tensor, weight: torch.Tensor, out_dtype, config) -> torch.Tensor:
    """``x @ weight^T`` in MXFP4, with the token dimension padded to 256.

    ``x`` is ``(tokens, in_features)`` and ``weight`` is
    ``(out_features, in_features)``.

    Everything numerical here is Turbo's. The padding and the slice are ordinary
    autograd operations wrapped around its call, so the weight gradient contracts
    over a padded -- and therefore aligned -- token count without this module
    owning a backward at all.
    """
    from primus_turbo.pytorch.ops import gemm_fp4

    padded, real_tokens = _pad_tokens(x, _fp4_common.AITER_K_MULTIPLE)
    out = gemm_fp4(padded, weight, trans_a=False, trans_b=True, out_dtype=out_dtype, config=config)
    if real_tokens != padded.shape[0]:
        # Slicing is differentiable, so the backward re-pads the incoming
        # gradient with zeros -- which is exactly what the weight gradient needs.
        out = out[:real_tokens]
    return out


class _MXFP4FwdFP8Bwd(torch.autograd.Function):
    """MXFP4 forward, tensorwise FP8 backward.

    The forward value is Turbo's ``gemm_fp4`` verbatim, so this path's forward
    numerics are identical to the pure path's. Only the backward is different: it
    saves bf16 copies of the activation and weight and requantizes them to FP8 at
    backward time, using Turbo's public ``gemm_fp8``.

    The trade is explicit -- saving bf16 gives up the activation-memory benefit of
    a 4-bit forward, in exchange for a backward GEMM with more mileage on it. Both
    backward GEMMs move together; splitting them was only ever useful for
    attributing a problem to one of the two, which is a debugging activity rather
    than a training configuration.

    Autograd disables grad mode inside ``forward`` and ``backward``, so the inner
    calls to Turbo's autograd-wrapped ops record nothing and simply compute.
    """

    @staticmethod
    def forward(ctx, x, weight, out_dtype, fp4_config, fp8_config):
        from primus_turbo.pytorch.ops import gemm_fp4

        padded, real_tokens = _pad_tokens(x, _fp4_common.AITER_K_MULTIPLE)
        out = gemm_fp4(padded, weight, trans_a=False, trans_b=True, out_dtype=out_dtype, config=fp4_config)
        if real_tokens != padded.shape[0]:
            out = out[:real_tokens].contiguous()

        # bf16 x and W, because the FP8 backward requantizes from full precision
        # rather than trying to recover it from the 4-bit forward operands.
        ctx.save_for_backward(x, weight)
        ctx.out_dtype = out_dtype
        ctx.fp8_config = fp8_config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.ops import gemm_fp8

        x, weight = ctx.saved_tensors
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
        if not grad_2d.is_contiguous():
            grad_2d = grad_2d.contiguous()

        # out = x @ W^T, so:
        #   grad_x = grad_out @ W          (tokens, in_features)
        #   grad_W = grad_out^T @ x        (out_features, in_features)
        # No token padding is needed on either: grad_x contracts over
        # out_features and grad_W's FP8 GEMM has no 256 constraint.
        grad_x = gemm_fp8(
            grad_2d,
            weight,
            trans_a=False,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            config=ctx.fp8_config,
        )
        grad_weight = gemm_fp8(
            grad_2d,
            x,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            config=ctx.fp8_config,
        )
        return grad_x, grad_weight, None, None, None


def mxfp4_fwd_fp8_bwd(x, weight, out_dtype, fp4_config, fp8_config) -> torch.Tensor:
    """``x @ weight^T`` with an MXFP4 forward and a tensorwise FP8 backward."""
    return _MXFP4FwdFP8Bwd.apply(x, weight, out_dtype, fp4_config, fp8_config)
