###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared activation-checkpointing application for diffusion parallelization strategies.

WHY THIS IS SHARED:
  Every diffusion model on this path wants the same three settings -- full,
  selective, off -- applied to whatever attribute happens to hold its transformer
  blocks. Only that attribute name differs between models. Writing the branch per
  model is how the two known bugs in this area got in, and both are the kind that
  are invisible at runtime:

  - ``"selective"`` is a non-empty string and therefore truthy, so a bare
    ``if activation_checkpointing:`` followed by the full-AC branch silently makes
    ``selective`` and ``full`` the same thing.
  - ``"false"`` is *also* a non-empty string, so the same test enables
    checkpointing on a run configured to have none.

  Neither raises and neither shows up in a config echo, so they are only found by
  someone measuring. Deciding it in one place means a model added later inherits
  the fix rather than reimplementing the bug.

WHY THERE IS A STRIDE:
  Off, selective and full are three points, and at long sequence lengths none of
  them is the right size. A configuration can be a little short of memory and
  still find that full AC hands back several times what it needed, charging
  recompute for all of it -- and selective, which is meant to be the middle
  setting, frequently overshoots the same way.

  The axis that IS the right size is WHICH blocks get wrapped. Peak activation
  memory is reached at the end of the forward pass, with every block's
  activations simultaneously live, so checkpointing k of the N blocks sheds
  roughly k/N of them and costs roughly k/N of the recompute. Both sides are
  close to linear in k, which is what makes it usable: the memory a
  configuration needs can be converted into a number of blocks, and the price is
  read off the same line rather than discovered.

  Expressed as a stride rather than a count, so it does not have to be restated
  when the block count changes: a stride of n wraps indices 0, n, 2n and so on. A
  stride of 1 or 0 is plain full AC.

  It applies to full AC only. Selective AC is op-level and decides per operation
  inside every block, so a per-block stride would be composing two different
  granularities against each other.

WHAT IS NOT HERE:
  Sharding. Strategies differ genuinely in how they shard -- some delegate to an
  in-tree parent, some call the helper directly -- so that stays with each model.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple

logger = logging.getLogger(__name__)

# Spellings that mean "off" but are truthy in Python. Some CLI and config paths
# forward the flag as a raw string, so a bare truthiness test on any of these
# would enable checkpointing on a run configured to have none.
AC_OFF_VALUES = frozenset({"false", "0", "off", "no", "none", ""})

# What apply() did, for logging and for tests to assert against.
MODE_OFF = "off"
MODE_FULL = "full"
MODE_SELECTIVE = "selective"
MODE_NO_BLOCKS = "no-blocks"


def normalize(value: Any) -> Any:
    """Map false-like strings to False; leave bools and 'full'/'selective' alone.

    Deliberately does not coerce to bool: ``"selective"`` has to survive so the
    caller can still distinguish it from ``"full"``.
    """
    if isinstance(value, str) and value.strip().lower() in AC_OFF_VALUES:
        return False
    return value


def apply(
    parallelizer: Any,
    model: Any,
    block_attrs: Sequence[str],
    value: Any,
    *,
    enable_compile: bool = False,
    stride: int = 0,
    log_prefix: str,
) -> Tuple[str, int]:
    """Apply activation checkpointing to the model's transformer blocks.

    Args:
        parallelizer: AutoModel's ``distributed.parallelizer`` module, passed in
            rather than imported so a caller that already holds a reference (or a
            test that stubbed it) is the one that decides.
        block_attrs: attributes holding block lists, in a stable order. A model
            may keep its blocks in more than one list -- dual-stream and
            single-stream, say -- and all of them get wrapped.
        value: the raw ``activation_checkpointing`` setting, pre- or
            post-``normalize``.
        stride: with full AC, wrap only every nth block. 0 or 1 wraps every
            block. Indices are counted across all the block lists together, so a
            model with several of them gets an even spread rather than a stride
            that restarts at each list. Ignored for selective AC; see the module
            docstring.
        log_prefix: the caller's log tag, so a reader can tell which strategy
            emitted the line.

    Returns:
        ``(mode, count)``. ``count`` is the number of blocks wrapped, which is 0
        for every mode but full and selective.
    """
    value = normalize(value)
    if not value:
        logger.info("%s activation checkpointing OFF", log_prefix)
        return MODE_OFF, 0

    # Collected before wrapping, in the given order, so the logged count is
    # meaningful and reproducible.
    block_lists = [getattr(model, attr) for attr in block_attrs if getattr(model, attr, None) is not None]
    if not block_lists:
        logger.warning(
            "%s activation_checkpointing requested but the model has none of the block "
            "lists %s; nothing checkpointed.",
            log_prefix,
            ", ".join(block_attrs),
        )
        return MODE_NO_BLOCKS, 0

    if stride and stride < 1:
        raise ValueError(f"the activation-checkpointing stride must be >= 1, got {stride}")

    if parallelizer.is_selective_activation_checkpointing(value):
        if stride > 1:
            # Refused rather than ignored. A caller that asked for both wanted less
            # recompute than selective AC gives, and silently handing them plain
            # selective AC would look like it worked.
            raise ValueError(
                "a block stride cannot be combined with selective activation "
                "checkpointing: selective AC decides per operation inside every "
                "block, so the two are different granularities. Use full AC with a "
                "stride, or selective AC on its own."
            )
        # Op-level partial AC through the shared AutoModel machinery: keep
        # attention and half the matmuls, recompute the cheap ops. The wrapper is
        # tagged so that per-layer compile compiles it outer and the partitioner
        # honors the recompute tags.
        #
        # has_kv_sharing=False: these are diffusion transformers, with no KV cache.
        #
        # The helper replaces each block with an identity, so passing blocks from
        # several lists in one call is safe.
        layers = [block for block_list in block_lists for block in block_list]
        parallelizer.apply_selective_checkpointing_to_layers(
            model,
            layers,
            False,
            enable_compile=bool(enable_compile),
        )
        logger.info(
            "%s wrapped %d blocks with SELECTIVE (partial) activation checkpointing",
            log_prefix,
            len(layers),
        )
        return MODE_SELECTIVE, len(layers)

    # Full AC: recompute the whole block on backward. Assigned back by index
    # because these are ModuleLists and the wrapper must take the block's place.
    # NO_REENTRANT is required for torch.compile compatibility.
    #
    # The position counter runs across all the block lists rather than per list,
    # so a model with several of them gets an even spread instead of a stride that
    # restarts -- and so the total wrapped is a function of the stride alone.
    wrapped = 0
    position = 0
    total = 0
    for block_list in block_lists:
        for idx in range(len(block_list)):
            total += 1
            if stride > 1 and position % stride:
                position += 1
                continue
            position += 1
            block_list[idx] = parallelizer.checkpoint_wrapper(
                block_list[idx],
                checkpoint_impl=parallelizer.CheckpointImpl.NO_REENTRANT,
            )
            wrapped += 1

    if stride > 1:
        logger.info(
            "%s wrapped %d of %d blocks with FULL activation checkpointing (every %d)",
            log_prefix,
            wrapped,
            total,
            stride,
        )
    else:
        logger.info("%s wrapped %d blocks with FULL activation checkpointing", log_prefix, wrapped)
    return MODE_FULL, wrapped
