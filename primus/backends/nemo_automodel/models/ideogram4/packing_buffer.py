###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Transport for the variable-length packing: one shared non-persistent buffer.

The adapter builds ``cu_seqlens`` on the host, where the caption lengths are
already Python integers, and the attention processor has to receive it as a graph
INPUT. Deriving it from the mask inside the compiled region does device-to-host
reads, which graph-break and, under FSDP2, desynchronize the per-layer
collectives -- see ``attention/varlen_utils.py`` for that mechanism.

WHY A BUFFER AND NOT THE OBVIOUS CHANNEL:
  diffusers' ``attention_kwargs`` looks like the intended route and is a dead end
  for this model: the LoRA decorator keeps only ``scale``, both block call sites
  pass their arguments positionally, and the transformer block's ``forward`` has no
  ``**kwargs`` to forward. Anything put in there simply does not arrive.

  The processor is, however, handed its attention module as its first argument, so
  the packing can ride on the module. One ``int32`` tensor of fixed shape is
  registered as a non-persistent buffer on every attention module -- deliberately
  the same tensor OBJECT on all of them, so publishing is a single in-place
  ``copy_`` per step rather than one write per layer, and so a diffusers version
  bump cannot silently diverge from a copied forward.

  Two properties make this work with the compiler and the sharding: Dynamo lifts
  the buffer as a graph input rather than baking in its values, so changing the
  packing each step causes no recompilation; and FSDP2 leaves it alone, since it
  is a plain tensor rather than a parameter -- one object survives ``fully_shard``
  and it stays out of the state dict.

=============================================================================
FOUR RULES. Only the third fails loudly.
=============================================================================
1. PUBLISH BEFORE THE FORWARD, NEVER BETWEEN FORWARD AND BACKWARD.
   Per-layer compilation sits inside the checkpoint wrapper, so each block -- and
   so each read of this buffer -- runs twice per step: once in the forward and
   once recomputed during the backward. A write in between makes the recomputed
   activations disagree with the originals, and nothing raises.

2. RE-PUBLISH AFTER MATERIALIZATION.
   ``model.to_empty(device=...)``, the usual meta-device bring-up, gives every
   module its OWN buffer, quietly ending the sharing. One layer would then read
   the published packing and every other layer would read uninitialized memory.
   :func:`publish_packing` checks for this and restores sharing rather than
   trusting that it still holds.

3. THE BUFFER IS READ-ONLY ONCE PUBLISHED.
   The varlen kernel treats ``cu_seqlens`` as mutable: it saves the tensor for its
   backward and then writes to it, advancing the autograd version counter on every
   call. Handing the shared buffer straight to the kernel therefore moves its
   version once per layer while each layer's backward still expects the version it
   saved, and the step dies in ``backward()`` with a message naming a version
   counter and nothing else -- "IntTensor[5] is at version 35; expected version
   34". The clone that prevents this lives in ``varlen_flash_attention``, at the
   call site, so no caller has to remember it. Worth knowing about because a probe
   that exercises only the dense path will never see it.

4. THE BUFFER OUTLIVES THE STEP.
   Anything that runs the transformer WITHOUT going through the adapter -- a
   sampling pass, an eval loop at a different batch size -- reads whatever was
   published last. Call :func:`clear_packing` first, which returns the processor to
   deriving the packing from the mask. The processor also rejects a packing whose
   length disagrees with the batch, which is a static-shape comparison and so
   costs nothing, meaning that particular mistake at least fails loudly.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# This name appears verbatim as the graph placeholder in a TORCH_LOGS dump
# (``l_self_modules_attention_buffers_primus_cu_seqlens_``), so keep it greppable.
PACKING_ATTR = "_primus_cu_seqlens"
BOUND_ATTR = "_primus_max_seqlen"
_CONSUMERS_ATTR = "_primus_packing_consumers"

_logged: set = set()


def _log_once(key: str, level: int, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def _own_processor(module: nn.Module):
    """The processor this module itself holds, ignoring ones it merely forwards.

    A plain ``getattr`` is wrong here. The activation-checkpoint wrapper delegates
    unknown attributes to the module it wraps, so it answers ``processor`` with its
    child's and is counted as a second consumer. That doubles the module list,
    which is harmless for the ``copy_`` but leaves the one health number in the log
    -- "on N attention modules" -- unable to distinguish every block from half of
    them, which is the only thing that number is for.
    """
    own = module.__dict__.get("processor")
    if own is None:
        # A processor implemented as an nn.Module lives in _modules instead.
        own = module._modules.get("processor")
    return own


def attention_modules(model: nn.Module) -> List[nn.Module]:
    """The attention modules that can actually read the packing.

    Membership is decided by processor type, which makes the transport self-gating:
    a run without the varlen processor installed has no consumers, so no buffers
    are installed and nothing is published, instead of maintaining per-step state
    that nobody reads.
    """
    cached = getattr(model, _CONSUMERS_ATTR, None)
    if cached is not None:
        return cached

    from primus.backends.nemo_automodel.models.ideogram4.attn_processor import (
        Ideogram4VarlenAttnProcessor,
    )

    found = [
        module
        for module in model.modules()
        if isinstance(_own_processor(module), Ideogram4VarlenAttnProcessor)
    ]
    # An empty result is not cached: on an unlucky ordering this can run before the
    # processors are swapped in, and caching the empty answer would disable the
    # transport for the rest of the run.
    if found:
        setattr(model, _CONSUMERS_ATTR, found)
    return found


def _shared_buffer(modules: List[nn.Module], numel: int, device: torch.device) -> Optional[torch.Tensor]:
    """The one buffer every module shares, or None if that is no longer true.

    Returns None for all three ways the invariant lapses: never installed;
    installed with the wrong shape or device, which means the batch size changed;
    or installed and no longer shared, which means something handed each module its
    own tensor.
    """
    first = getattr(modules[0], PACKING_ATTR, None)
    if not isinstance(first, torch.Tensor):
        return None
    if first.numel() != numel or first.dtype != torch.int32 or first.device != device:
        return None
    if not all(getattr(m, PACKING_ATTR, None) is first for m in modules):
        return None
    return first


def _install(modules: List[nn.Module], numel: int, device: torch.device) -> torch.Tensor:
    """Point every module at ONE fresh buffer and return it."""
    shared = torch.zeros(numel, dtype=torch.int32, device=device)
    for module in modules:
        # register_buffer refuses a name that is already live, so replace the entry
        # directly when it exists. Either way it stays non-persistent.
        if PACKING_ATTR in module._buffers:
            module._buffers[PACKING_ATTR] = shared
        else:
            module.register_buffer(PACKING_ATTR, shared, persistent=False)
        if not hasattr(module, BOUND_ATTR):
            setattr(module, BOUND_ATTR, None)
    return shared


def _report_install(previous, numel: int, count: int, max_seqlen: Optional[int]) -> None:
    """Explain what happened, once per distinct cause."""
    if not isinstance(previous, torch.Tensor):
        # The line that shows the transport is live: the entry count should be 2B+1
        # and the module count should be every block's attention.
        _log_once(
            "install",
            logging.INFO,
            "[PrimusIdeogramPacking] installed a shared non-persistent cu_seqlens buffer "
            "(int32, %d entries, max_seqlen=%s) on %d attention module(s); the adapter "
            "publishes it with one copy_ per step.",
            numel,
            max_seqlen,
            count,
        )
    elif previous.numel() != numel:
        _log_once(
            f"reshape-{previous.numel()}-{numel}",
            logging.WARNING,
            "[PrimusIdeogramPacking] cu_seqlens length changed from %d to %d, so the batch "
            "size changed; the buffer was replaced and torch.compile will recompile. A "
            "constant shape is the entire point of the reserved pad column, so check "
            "drop_last and the eval batch size.",
            previous.numel(),
            numel,
        )
    else:
        _log_once(
            "resharing",
            logging.WARNING,
            "[PrimusIdeogramPacking] the cu_seqlens buffer was no longer shared across the "
            "%d attention module(s), so something replaced module buffers after install -- "
            "to_empty() during materialization is the usual cause. Sharing has been "
            "restored. Left alone, one layer would read the published packing and the rest "
            "would read uninitialized memory, with nothing raised.",
            count,
        )


def publish_packing(
    model: nn.Module,
    cu_seqlens: torch.Tensor,
    max_seqlen: Optional[int] = None,
    device: Optional[torch.device] = None,
    required: bool = False,
) -> Optional[torch.Tensor]:
    """Make ``cu_seqlens`` visible to every attention module for the next forward.

    Call immediately before the model call, and never between a forward and its
    backward -- rule 1 above. ``cu_seqlens`` may live on the host: the single
    ``copy_`` here is then the only host-to-device transfer, which is the point of
    having built the packing from Python integers.

    Args:
        model: the transformer, possibly FSDP2-wrapped and possibly compiled.
        cu_seqlens: ``int32 (2B+1,)`` cumulative segment starts, on any device.
        max_seqlen: the STATIC upper bound, not the batch's true maximum. A
            data-derived integer would be guarded by value and recompile.
        device: where the buffer should live, defaulting to ``cu_seqlens.device``.
            Pass the model's device explicitly when publishing a host-built tensor.
        required: raise instead of returning None when nothing consumes the
            packing. Callers that have already concluded the transport is in use
            should pass True, because a rank that publishes into the void falls
            back to the mask-derived path on every layer -- and under data
            parallelism its gradients then come from a different attention path
            than its peers' and are averaged in anyway, with nothing in the logs
            to say so once non-zero ranks are quieted.

    Returns:
        The shared buffer, or None when nothing consumes it and ``required`` is
        False.
    """
    modules = attention_modules(model)
    if not modules:
        if required:
            raise RuntimeError(
                "[PrimusIdeogramPacking] a cu_seqlens packing was precomputed but no "
                "attention module can read it: none of the model's modules carry an "
                "Ideogram4VarlenAttnProcessor. Every layer would fall back to deriving the "
                "packing from the attention mask, which host-syncs, graph-breaks under "
                "torch.compile, and -- if it happens on only some ranks -- mixes two "
                "attention paths into one gradient average. Either the varlen processor "
                "install did not run (PRIMUS_IDEOGRAM_VARLEN_ATTN) or the model was replaced "
                "afterwards. Unset PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS to take the "
                "mask-derived path deliberately."
            )
        return None

    target = torch.device(device) if device is not None else cu_seqlens.device
    numel = cu_seqlens.numel()
    shared = _shared_buffer(modules, numel, target)
    if shared is None:
        previous = getattr(modules[0], PACKING_ATTR, None)
        shared = _install(modules, numel, target)
        _report_install(previous, numel, len(modules), max_seqlen)
    shared.copy_(cu_seqlens)

    for module in modules:
        if getattr(module, BOUND_ATTR, None) != max_seqlen:
            setattr(module, BOUND_ATTR, max_seqlen)
    return shared


def resolve_packing(
    attn: nn.Module,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """Pick the packing for this call: an explicit argument first, then module state.

    An explicit argument wins so that a future kwargs route could override this
    without the processor changing. The ``getattr`` fallback is a constant attribute
    lookup, which Dynamo lifts as a graph input -- not a data-dependent read, and
    not a graph break.
    """
    if cu_seqlens is None:
        cu_seqlens = getattr(attn, PACKING_ATTR, None)
        if cu_seqlens is not None and max_seqlen is None:
            max_seqlen = getattr(attn, BOUND_ATTR, None)
    return cu_seqlens, max_seqlen


def clear_packing(model: nn.Module) -> int:
    """Drop the buffer from every consumer. See rule 4 above.

    Returns the number of modules it was removed from.
    """
    removed = 0
    for module in attention_modules(model):
        if module._buffers.pop(PACKING_ATTR, None) is not None:
            removed += 1
        if hasattr(module, BOUND_ATTR):
            delattr(module, BOUND_ATTR)
    if hasattr(model, _CONSUMERS_ATTR):
        delattr(model, _CONSUMERS_ATTR)
    return removed
