###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FLUX attention's q/k RMSNorm, RoPE and stream concat as one op, Primus-Turbo's ``dit_qk_norm_rope``.

WHY:
  Between the fused QKV GEMM and flash attention, every FLUX block runs RMSNorm on q and k
  (separate weights for the text stream in dual-stream blocks), concatenates text and image
  along the sequence (dual-stream), and applies RoPE. Inductor splits that into many kernels
  that move the tensors well below the GPU's copy bandwidth: the forward re-reads q and k with a
  permuted index for RoPE's pair swap and reloads the cos/sin tables for every (token, head)
  row, and the backward runs RoPE's backward as separate elementwise passes before the RMSNorm
  backward.

WHAT:
  ``dit_qk_norm_rope(qkv, w_q, w_k, cos, sin, eps, layout="bshd", txt_qkv, w_txt_q, w_txt_k, bias,
  txt_bias)`` takes the QKV GEMM output(s) ``[B, S, 3 * H * D]`` and returns q, k, v as
  ``[B, S_txt + S, H, D]`` -- q and k normalized and rotated, v copied -- exactly as diffusers'
  ``FluxAttnProcessor`` computes them (RMSNorm in FP32 with one rounding to BF16, then RoPE in FP32),
  in one pass that writes attention's layout directly. Its backward undoes the rotation, computes the
  RMSNorm input gradient and writes the gradient of the QKV GEMM output (v's slice included), so
  autograd needs no concat; it also adds the QKV biases and sums that gradient over tokens, which is
  the QKV bias gradient, so the GEMMs here run without biases and autograd's separate column sum, which
  re-reads the whole gradient, goes. The weight and bias gradients are reduced from partial sums in a
  fixed order, so they are deterministic.

  ``install()`` makes ``attn_processor.PrimusFluxAttnProcessor``, a subclass of diffusers'
  ``FluxAttnProcessor`` whose ``__call__`` is diffusers' with the op in the middle, the processor
  diffusers builds every FLUX attention module with (``FluxAttention._default_processor_cls``), so it
  must run before the model is built. Nothing in diffusers is replaced, so no diffusers source is
  pinned: the processor reads only the attention module's projections, norms and head size (the same
  in diffusers 0.38.0 to 0.43.0), and the block tests compare it with diffusers' processor on the
  installed diffusers. Other calls (unfused projections, no RoPE, other dtypes, head_dim != 128,
  weightless norms) take diffusers' ``__call__``; projections that are not stock ``nn.Linear``
  modules keep their own forward and autograd sums their bias gradients.

NUMERICS:
  The QKV GEMMs run without their biases and the op adds them to the BF16 GEMM output (and rounds the sum to
  BF16 before the norm, as a separate bias add does), where diffusers' GEMM adds the bias before its one
  rounding: the outputs and gradients stay within the BF16 noise of diffusers' block, which the block tests
  check. The bias gradients sum the same BF16 values as autograd's column sum, in another order.

Settings (module config, ``primus_perf:`` section):
    flux_fused_qk_norm_rope: false    apply this patch
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from primus.backends.automodel import options
from primus.backends.automodel.models.flux.turbo_ops import dit_qk_norm_rope

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_FLAG = f"{_SECTION}.flux_fused_qk_norm_rope"
_PATCHED_ATTR = "_primus_fused_qk_norm_rope"
_HEAD_DIM = 128


def is_enabled() -> bool:
    return options.flag(_FLAG)


def _norm_eps(norms) -> Optional[float]:
    """The common eps of affine torch.nn.RMSNorm modules over head_dim, else None."""
    eps = set()
    for n in norms:
        if (
            not isinstance(n, torch.nn.RMSNorm)
            or n.weight is None
            or tuple(n.normalized_shape) != (_HEAD_DIM,)
        ):
            return None
        eps.add(n.eps)
    return eps.pop() if len(eps) == 1 and None not in eps else None


def _fusable(attn, hidden_states, encoder_hidden_states, image_rotary_emb) -> Optional[float]:
    """eps for the op if this call is one it computes exactly as diffusers does, else None."""
    if not getattr(attn, "fused_projections", False) or attn.head_dim != _HEAD_DIM:
        return None
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if not isinstance(image_rotary_emb, (tuple, list)) or len(image_rotary_emb) != 2:
        return None
    cos, sin = image_rotary_emb
    # FSDP2's mixed precision casts the block inputs, these tables included, to BF16.
    if (
        cos.dtype not in (torch.float32, torch.bfloat16)
        or sin.dtype != cos.dtype
        or cos.shape != sin.shape
        or cos.shape[-1] != _HEAD_DIM
    ):
        return None
    dual = attn.added_kv_proj_dim is not None
    if dual != (encoder_hidden_states is not None) or (dual and not hasattr(attn, "to_added_qkv")):
        return None
    norms = [attn.norm_q, attn.norm_k] + ([attn.norm_added_q, attn.norm_added_k] if dual else [])
    return _norm_eps(norms)


def install() -> bool:
    """Make every FLUX attention module built from now on use ``PrimusFluxAttnProcessor``. Idempotent;
    edits no diffusers source. Declines, changing nothing, when diffusers' default processor is not its
    own ``FluxAttnProcessor`` (another patch replaced it, or diffusers builds processors otherwise)."""
    from diffusers.models.transformers.transformer_flux import (
        FluxAttention,
        FluxAttnProcessor,
    )

    if getattr(FluxAttention, _PATCHED_ATTR, False):
        return True
    if getattr(FluxAttention, "_default_processor_cls", None) is not FluxAttnProcessor:
        logger.warning(
            "[PrimusFluxQkNormRope] FluxAttention's default processor is not diffusers' FluxAttnProcessor; "
            "not applied."
        )
        return False
    op = dit_qk_norm_rope("PrimusFluxQkNormRope")
    if op is None:
        return False
    from primus.backends.automodel.models.flux.attn_processor import (
        PrimusFluxAttnProcessor,
    )

    PrimusFluxAttnProcessor.dit_qk_norm_rope = staticmethod(op)
    FluxAttention._default_processor_cls = PrimusFluxAttnProcessor
    if PrimusFluxAttnProcessor not in FluxAttention._available_processors:
        FluxAttention._available_processors = [*FluxAttention._available_processors, PrimusFluxAttnProcessor]
    setattr(FluxAttention, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxQkNormRope] FLUX attention's q/k RMSNorm, stream concat and RoPE run as Primus-Turbo's "
        "dit_qk_norm_rope (fused QKV projections only; the op adds the QKV biases and returns their gradients)."
    )
    return True
