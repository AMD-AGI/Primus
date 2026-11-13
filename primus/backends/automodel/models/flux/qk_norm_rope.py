###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FLUX attention's q/k RMSNorm, RoPE and stream concat as one op, Primus-Turbo's ``flux_qk_norm_rope``.

WHY:
  Between the fused QKV GEMM and flash attention, every FLUX block runs RMSNorm on q and k
  (separate weights for the text stream in dual-stream blocks), concatenates text and image
  along the sequence (dual-stream), and applies RoPE. Inductor splits that into many kernels
  that move the tensors well below the GPU's copy bandwidth: the forward re-reads q and k with a
  permuted index for RoPE's pair swap and reloads the cos/sin tables for every (token, head)
  row, and the backward runs RoPE's backward as separate elementwise passes before the RMSNorm
  backward.

WHAT:
  ``flux_qk_norm_rope(qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads, bias, txt_bias)``
  takes the QKV GEMM output(s) ``[B, S, 3 * H * D]`` and returns q, k, v as ``[B, S_txt + S, H, D]``
  -- q and k normalized and rotated, v copied -- exactly as diffusers' ``FluxAttnProcessor``
  computes them (RMSNorm in FP32 with one rounding to BF16, then RoPE in FP32), in one pass that
  writes attention's layout directly. Its backward undoes the rotation, computes the RMSNorm input
  gradient and writes the gradient of the QKV GEMM output (v's slice included), so autograd needs no
  concat; it also sums that gradient over tokens, which is the QKV bias gradient, so the GEMMs here
  add detached biases and autograd's separate column sum, which re-reads the whole gradient, goes.
  The weight and bias gradients are reduced from per-program partial sums, so they are deterministic.

  ``install()`` swaps ``FluxAttnProcessor.__call__`` for the same forward with the op in the
  middle, pinned to diffusers 0.40.0 by a source hash (0.43.0's differs only in the return
  annotation). Calls the op cannot serve (unfused projections, no RoPE, other dtypes,
  head_dim != 128, weightless norms) take diffusers' path; projections that are not stock
  ``nn.Linear`` modules keep their own forward and autograd sums their bias gradients.

NUMERICS:
  Same forward; the bias gradients sum the same BF16 values as autograd's column sum, in another order.

Settings (module config, ``primus_perf:`` section):
    flux_fused_qk_norm_rope: false    apply this patch
"""
from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from primus.backends.automodel import options
from primus.backends.automodel.models.flux.plain_linear import is_plain_linear
from primus.backends.automodel.models.flux.turbo_ops import flux_ops

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_FLAG = f"{_SECTION}.flux_fused_qk_norm_rope"
_PATCHED_ATTR = "_primus_fused_qk_norm_rope"
_HEAD_DIM = 128
# sha256 of FluxAttnProcessor.__call__ in diffusers 0.40.0 (also 0.41 and 0.42) and 0.43.0, which
# differ only in the return annotation; _install_processor re-implements it.
_PROCESSOR_SHA256 = (
    "d5b36134887ebe450b6c15f200bf33f781ba8258201d25caceb16bb9bc07d2c9",
    "65d9ce34cf9616095ab895f48c09042bbb74987ea7a342714e7ce19412de8922",
)


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


def _install_processor(ops) -> bool:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

    orig = FluxAttnProcessor.__call__
    digest = hashlib.sha256(inspect.getsource(orig).encode()).hexdigest()
    if digest not in _PROCESSOR_SHA256:
        logger.warning(
            "[PrimusFluxQkNormRope] FluxAttnProcessor.__call__ is not diffusers 0.40.0's or 0.43.0's; "
            "not applied."
        )
        return False

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        eps = _fusable(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        if eps is None:
            return orig(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        # diffusers 0.40.0's forward with the q/k norms, the stream concat and RoPE in one op.
        cos, sin = image_rotary_emb
        dual = encoder_hidden_states is not None
        projections = [attn.to_qkv] + ([attn.to_added_qkv] if dual else [])
        if all(is_plain_linear(p) for p in projections):
            # The op returns the bias gradients from its backward, so the GEMMs add the biases
            # without differentiating them.
            qkv = F.linear(hidden_states, attn.to_qkv.weight, attn.to_qkv.bias.detach())
            txt_qkv = None
            if dual:
                txt_qkv = F.linear(
                    encoder_hidden_states, attn.to_added_qkv.weight, attn.to_added_qkv.bias.detach()
                )
            biases = (attn.to_qkv.bias, attn.to_added_qkv.bias if dual else None)
        else:
            qkv = attn.to_qkv(hidden_states)
            txt_qkv = attn.to_added_qkv(encoder_hidden_states) if dual else None
            biases = (None, None)
        heads = qkv.shape[-1] // (3 * attn.head_dim)
        if dual:
            query, key, value = ops.flux_qk_norm_rope(
                qkv,
                txt_qkv,
                attn.norm_q.weight,
                attn.norm_k.weight,
                attn.norm_added_q.weight,
                attn.norm_added_k.weight,
                cos,
                sin,
                eps,
                heads,
                *biases,
            )
        else:
            query, key, value = ops.flux_qk_norm_rope(
                qkv, None, attn.norm_q.weight, attn.norm_k.weight, None, None, cos, sin, eps, heads, *biases
            )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    __call__.__wrapped__ = orig
    FluxAttnProcessor.__call__ = __call__
    return True


def install() -> bool:
    """Route FLUX attention's q/k norm + RoPE through the fused op. Idempotent; edits no diffusers source."""
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

    if getattr(FluxAttnProcessor, _PATCHED_ATTR, False):
        return True
    ops = flux_ops("PrimusFluxQkNormRope")
    if ops is None or not _install_processor(ops):
        return False
    setattr(FluxAttnProcessor, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxQkNormRope] FLUX attention's q/k RMSNorm, stream concat and RoPE run as Primus-Turbo's "
        "flux_qk_norm_rope (fused QKV projections only; QKV bias gradients in its backward)."
    )
    return True
