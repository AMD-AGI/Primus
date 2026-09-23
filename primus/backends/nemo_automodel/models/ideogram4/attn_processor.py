###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Variable-length flash-attention processor for Ideogram-4.

The exact replacement for masked torch SDPA. See ``attention/varlen_utils.py`` for
why a block-diagonal mask has a variable-length equivalent and why that equivalence
is exact, and ``packing_buffer.py`` for how the packing reaches this processor.

WHAT THIS IS:
  A drop-in diffusers attention processor. It reproduces the stock
  ``Ideogram4AttnProcessor`` arithmetic verbatim -- q/k/v projection, q/k RMSNorm,
  MRoPE, output projection -- and replaces only the attention CALL. Nothing in
  diffusers or AutoModel is edited: ``install`` swaps the class default processor
  before the recipe builds the model, so every attention module is constructed
  with it.

WHICH PATH A CALL TAKES, in order:

  1. A PROVIDED PACKING. The adapter built it on the host and published it into
     the shared buffer this processor reads off its attention module. This is the
     only path that is simultaneously exact on ragged batches and safe under
     per-layer compilation with FSDP2, because nothing in it inspects tensor
     values. It is the path that matters.
  2. NO MASK, or the caller asserted equal lengths. Dense flash. Exact only when
     no row has padding.
  3. DERIVE IT FROM THE MASK. Exact, but its device-to-host reads graph-break, so
     it is unsafe under multi-rank compilation. Reached only when nothing was
     precomputed, and retained because it is the reference the tests compare
     against.

  An additive float mask is not something this model emits and has no
  variable-length equivalent, so it defers to the original dense dispatch and
  warns once rather than guessing.

Activation lives in ``_varlen_common``, which is deliberately importable without
torch so patch discovery can consult it. The gates are re-exported here because
this is where a reader looks for them.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from primus.backends.nemo_automodel.attention import varlen_utils
from primus.backends.nemo_automodel.models.ideogram4._varlen_common import (
    assume_dense_enabled,
    is_varlen_attn_enabled,
    precompute_cu_seqlens_active,
    precompute_cu_seqlens_enabled,
)
from primus.backends.nemo_automodel.models.ideogram4.packing_buffer import (
    resolve_packing,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Ideogram4VarlenAttnProcessor",
    "install",
    "assume_dense_enabled",
    "is_varlen_attn_enabled",
    "precompute_cu_seqlens_active",
    "precompute_cu_seqlens_enabled",
]

_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate-half, matching the diffusers transformer's own helper exactly."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Ideogram4VarlenAttnProcessor:
    """Ideogram-4 self-attention through variable-length flash attention."""

    # Kept for diffusers' processor discovery and set_attention_backend.
    _attention_backend = None
    _parallel_config = None

    deterministic: bool = False

    # Read once at class-definition time rather than per call. Launchers set the
    # environment before import, and this keeps the check a constant attribute
    # lookup inside the compiled graph instead of a branch on run state.
    assume_dense: bool = assume_dense_enabled()

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        image_rotary_emb: Tuple[Tensor, Tensor],
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        # The packing normally arrives on the module, published into a shared buffer.
        # The two parameters are declared by name anyway: diffusers filters forwarded
        # kwargs against this signature, so naming them is what would let a kwargs
        # route work at all -- a **kwargs-only processor would silently receive
        # nothing. An explicit argument wins over the buffer.
        cu_seqlens, max_seqlen = resolve_packing(attn, cu_seqlens, max_seqlen)

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        out = self._attention(query, key, value, attention_mask, cu_seqlens, max_seqlen)
        return attn.to_out[0](out.flatten(2, 3))

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        batch, length, heads, head_dim = query.shape

        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # An additive mask carries magnitudes rather than boundaries, so it has no
            # variable-length form. This model does not emit one, so rather than guess,
            # hand it back to the dispatch this processor replaced.
            _warn_once(
                "nonbool_mask",
                "[PrimusIdeogramVarlen] a non-boolean attention mask has no var-len "
                "equivalent, so this call falls back to the original dense dispatch. "
                "Ideogram-4 is not expected to produce one; if this appears, the model's "
                "mask construction has changed.",
            )
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            return dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        # MRoPE multiplies q and k by a float32 cos/sin, which promotes them to
        # float32. Torch SDPA is autocast-aware and downcasts at its own boundary; the
        # flash op is not, so the cast has to happen here to match what the replaced
        # path did. ``value`` skips RoPE and already carries the compute dtype.
        compute_dtype = value.dtype
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)

        if cu_seqlens is not None:
            # THE PATH THAT MATTERS. Nothing here reads a tensor value, so it is exact
            # on ragged batches and stays in one compiled graph -- which is what keeps
            # the FSDP2 per-layer collectives in order. ``attention_mask`` is
            # deliberately not consulted: the model still materializes it, and on this
            # path it is dead weight.
            #
            # The guard below compares only static shape metadata, so it costs a guard
            # and no host synchronization. It exists because the buffer outlives the
            # step that published it, so a caller that bypasses the adapter -- a
            # sampling pass, an eval loop at another batch size -- would otherwise
            # attend on a stale packing and corrupt silently.
            expected = 2 * batch + 1
            if cu_seqlens.numel() != expected:
                raise ValueError(
                    f"cu_seqlens has {cu_seqlens.numel()} entries but this batch needs "
                    f"{expected}, which is 2B+1 for B={batch}. Either the packing was "
                    "published for a different batch size and is now stale -- call "
                    "packing_buffer.clear_packing(model) before running the model outside "
                    "the adapter -- or the sequence layout no longer has exactly two "
                    "segments per row, in which case this check needs updating with it."
                )

            # The packing is passed as-is: varlen_flash_attention clones it before the
            # kernel, because the kernel mutates it and the buffer is shared. That is
            # rule 3 in packing_buffer.py, and it lives at the call site so that no
            # caller has to remember it.
            out = varlen_utils.varlen_flash_attention(
                varlen_utils.pack_for_varlen(query),
                varlen_utils.pack_for_varlen(key),
                varlen_utils.pack_for_varlen(value),
                cu_seqlens,
                length if max_seqlen is None else max_seqlen,
                deterministic=self.deterministic,
            )
            return varlen_utils.unpack_from_varlen(out, batch)

        if attention_mask is None or self.assume_dense:
            # Dense flash. No mask means there is nothing to respect; assume_dense means
            # the caller asserted every row is full, which the adapter verifies on the
            # host. Either way this is value-free and so compiles without a break.
            return varlen_utils.dense_flash_attention(query, key, value, deterministic=self.deterministic)

        # Derive the packing from the mask. Exact, but the derivation reads tensor
        # values, so it graph-breaks and is unsafe under multi-rank compilation.
        # Reached only when nothing was precomputed, and kept because it is the
        # reference the tests measure the fast path against.
        derived, derived_max, is_trivial = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(attention_mask)
        if is_trivial:
            # The mask says nothing, so packing would be busywork.
            return varlen_utils.dense_flash_attention(query, key, value, deterministic=self.deterministic)

        out = varlen_utils.varlen_flash_attention(
            varlen_utils.pack_for_varlen(query),
            varlen_utils.pack_for_varlen(key),
            varlen_utils.pack_for_varlen(value),
            derived,
            derived_max,
            deterministic=self.deterministic,
        )
        return varlen_utils.unpack_from_varlen(out, batch)


def install(model=None) -> bool:
    """Route Ideogram-4 attention through the variable-length processor.

    A no-op returning False unless ``PRIMUS_IDEOGRAM_VARLEN_ATTN`` is set. Patches
    the class default so every attention module built AFTER this call uses the
    processor, and additionally swaps the modules of an already-built ``model`` when
    one is passed. Idempotent, and edits no diffusers or AutoModel source.
    """
    if not is_varlen_attn_enabled():
        return False

    # Fail fast if the kernel is unavailable, so the run errors clearly instead of
    # quietly keeping the SDPA path it was configured to replace.
    from diffusers.models.transformers.transformer_ideogram4 import Ideogram4Attention
    from primus_turbo.pytorch.ops import flash_attn_varlen_func  # noqa: F401

    already = getattr(Ideogram4Attention, "_primus_varlen_installed", False)
    if not already:
        Ideogram4Attention._default_processor_cls = Ideogram4VarlenAttnProcessor
        if Ideogram4VarlenAttnProcessor not in Ideogram4Attention._available_processors:
            Ideogram4Attention._available_processors = [
                *Ideogram4Attention._available_processors,
                Ideogram4VarlenAttnProcessor,
            ]
        Ideogram4Attention._primus_varlen_installed = True

    swapped = 0
    if model is not None:
        for module in model.modules():
            if isinstance(module, Ideogram4Attention) and not isinstance(
                module.processor, Ideogram4VarlenAttnProcessor
            ):
                module.set_processor(Ideogram4VarlenAttnProcessor())
                swapped += 1

    if already and swapped == 0:
        return True

    logger.info(
        "[PrimusIdeogramVarlen] installed the var-len flash-attention processor for "
        "Ideogram4Attention (deterministic=%s)%s.",
        Ideogram4VarlenAttnProcessor.deterministic,
        f"; swapped {swapped} existing module(s)" if swapped else "",
    )
    return True
