###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Variable-length flash-attention processor for Ideogram-4.

The exact replacement for masked torch SDPA. See ``varlen_utils.py`` for
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

  First, an additive float mask, which this model does not emit and which has no
  variable-length equivalent, defers to the original dense dispatch and warns once
  rather than guessing. Every later path calls a Primus-Turbo flash kernel with a
  non-deterministic backward, so under torch's deterministic mode the processor
  does what torch does for its own such ops: it raises, or with ``warn_only`` it
  warns once and runs.

  1. A PROVIDED PACKING. The adapter built it on the host and published it into
     the shared buffer this processor reads off its attention module. This is the
     only path that is exact on ragged batches and reads no tensor value, so it
     compiles into one graph per layer with no recompiles. It is the path that
     matters.
  2. NO MASK. Dense flash: with no mask there is nothing to respect.
  3. DERIVE IT FROM THE MASK. Exact, but its device-to-host reads graph-break and
     its data-dependent shapes recompile whenever the number of segments changes,
     separately on each rank. Under FSDP2 that is slower, and the other ranks wait
     at the next collective while one rank compiles. Reached only when nothing was
     precomputed, and retained because it is the reference the tests compare
     against.

WHY NOT DIFFUSERS' ``flash_varlen`` BACKEND, which AutoModel uses for HunyuanVideo:
  it reduces a 4-D mask to the set of keys anything attends to, and here padding
  attends to itself, so no key would be excluded. It keeps the first ``valid_len``
  keys of each row, but this model left-pads, so its real tokens come last. It
  reads the lengths back with ``.item()``, which is what building the packing on
  the host avoids. And it is registered without context parallelism and calls the
  ``flash_attn`` package rather than Primus-Turbo.

Activation lives in ``_varlen_common``, which is deliberately importable without
torch so patch discovery can consult it. The gates are re-exported here because
this is where a reader looks for them.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from primus.backends.automodel import options
from primus.backends.automodel.models.ideogram4 import varlen_utils
from primus.backends.automodel.models.ideogram4._varlen_common import (
    is_varlen_attn_enabled,
    precompute_cu_seqlens_active,
    precompute_cu_seqlens_enabled,
)
from primus.backends.automodel.models.ideogram4.packing_buffer import (
    TOKENS_ATTR,
    resolve_packing,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Ideogram4VarlenAttnProcessor",
    "install",
    "is_varlen_attn_enabled",
    "precompute_cu_seqlens_active",
    "precompute_cu_seqlens_enabled",
]

_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


_deterministic_warned = False


@torch.compiler.disable
def _check_deterministic_mode() -> None:
    """Treat the var-len kernels as torch treats its own non-deterministic ops.

    Called on every call while deterministic mode is on, so a switch from
    ``warn_only`` to strict mode later in the same process still raises. Strict
    mode raises; ``warn_only`` warns once and lets the call run. Kept out of the
    compiled graph because Dynamo folds ``are_deterministic_algorithms_enabled``
    into a guarded constant but not the warn-only query, so the cost is one graph
    break per layer, paid only while deterministic mode is on.
    """
    global _deterministic_warned
    reason = (
        "the Ideogram-4 var-len attention backward is non-deterministic, and its "
        "deterministic variant needs far more memory than real batch sizes allow. Turn "
        "primus_ideogram4.varlen_attention off for a deterministic run; the stock "
        "attention path then follows torch's own rules."
    )
    if not torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError(f"torch.use_deterministic_algorithms(True) is set, but {reason}")
    if _deterministic_warned:
        return
    _deterministic_warned = True
    logger.warning(
        "[PrimusIdeogramVarlen] torch.use_deterministic_algorithms(True, warn_only=True) is "
        "set, so this runs anyway, but %s",
        reason,
    )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate-half, matching the diffusers transformer's own helper exactly."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Ideogram4VarlenAttnProcessor:
    """Ideogram-4 self-attention through variable-length flash attention."""

    # Kept for diffusers' processor discovery and set_attention_backend.
    _attention_backend = None
    _parallel_config = None

    # Not tied to primus_turbo.nondeterministic_attention: at image-sized segments
    # the deterministic var-len backward needs orders of magnitude more memory than
    # this one, which rules out real batch sizes.
    deterministic: bool = False

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
        packed_tokens = None if cu_seqlens is not None else getattr(attn, TOKENS_ATTR, None)
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

        out = self._attention(query, key, value, attention_mask, cu_seqlens, max_seqlen, packed_tokens)
        return attn.to_out[0](out.flatten(2, 3))

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        packed_tokens: Optional[int] = None,
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

        # The fallback above follows torch's rules inside diffusers' dispatch; every
        # path below calls a Primus-Turbo flash kernel, so the check sits here.
        if torch.are_deterministic_algorithms_enabled():
            _check_deterministic_mode()

        # MRoPE multiplies q and k by a float32 cos/sin, which promotes them to
        # float32. Torch SDPA is autocast-aware and downcasts at its own boundary; the
        # flash op is not, so the cast has to happen here to match what the replaced
        # path did. ``value`` skips RoPE and already carries the compute dtype.
        compute_dtype = value.dtype
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)

        if cu_seqlens is not None:
            # THE PATH THAT MATTERS. Nothing here reads a tensor value, so it is exact
            # on ragged batches and stays in one compiled graph, with no graph break
            # or recompile on any rank. ``attention_mask`` is
            # deliberately not consulted: the model still materializes it, and on this
            # path it is dead weight.
            #
            # The guards below compare static shapes with host integers, so they cost
            # a guard and no host synchronization. The buffer outlives the step that
            # published it, so a caller that bypasses the adapter, such as a sampling
            # pass or an eval loop, would read a stale packing. The guards turn that
            # into an error when the batch size or the sequence length differs; for
            # the same shape the caller has to call packing_buffer.clear_packing(model)
            # first.
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
            if packed_tokens is not None and packed_tokens != batch * length:
                raise ValueError(
                    f"the published packing covers {packed_tokens} tokens but this batch has "
                    f"{batch * length} ({batch} rows of {length}). The packing was published "
                    "for a different sequence length -- another resolution or text width -- "
                    "and is now stale; call packing_buffer.clear_packing(model) before "
                    "running the model outside the adapter."
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

        if attention_mask is None:
            # Dense flash. No mask means there is nothing to respect, and the call is
            # value-free, so it compiles without a break.
            return varlen_utils.dense_flash_attention(query, key, value, deterministic=self.deterministic)

        # Derive the packing from the mask. Exact, but the derivation reads tensor
        # values, so it graph-breaks and recompiles per rank as segment counts vary.
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

    A no-op returning False unless ``primus_ideogram4.varlen_attention`` is set. Patches
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
        "Ideogram4Attention (%s backward)%s.",
        "deterministic" if Ideogram4VarlenAttnProcessor.deterministic else "non-deterministic",
        f"; swapped {swapped} existing module(s)" if swapped else "",
    )
    # This processor calls the var-len kernel itself, so the attention-backend
    # overrides, which replace what model.attention_backend dispatches to, never run.
    bypassed = [
        key
        for key in ("primus_turbo.fp8_attention", "primus_turbo.nondeterministic_attention")
        if options.flag(key)
    ]
    if bypassed:
        logger.warning(
            "[PrimusIdeogramVarlen] %s has no effect with the var-len processor, which calls "
            "its own bf16 kernel rather than model.attention_backend.",
            " and ".join(bypassed),
        )
    return True
