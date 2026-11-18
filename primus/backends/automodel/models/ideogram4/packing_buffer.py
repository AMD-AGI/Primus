###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Transport for the variable-length packing: one shared non-persistent buffer.

The adapter builds ``cu_seqlens`` on the host, where the caption lengths are
already Python integers, and the attention processor has to receive it as a graph
INPUT. Deriving it from the mask inside the compiled region does device-to-host
reads, which graph-break and recompile separately on each rank -- see
``varlen_utils.py`` for what that costs under FSDP2.

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
   entry count disagrees with the batch size, or whose token total -- recorded
   here as a host integer when it is published -- disagrees with the batch size
   times the sequence length. Both compare static shapes against host values, so
   they cost nothing, and a packing left over from a different batch size,
   resolution or text width fails loudly. One built for the same shape with
   different caption lengths cannot be told apart, which is what
   :func:`clear_packing` is for.
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
TOKENS_ATTR = "_primus_packed_tokens"

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

    Walked afresh on every call rather than cached, so a module that gains the
    processor after the first publish, or replaces one that had it, is found on the
    next publish. The walk is host-side and outside the compiled region, and costs
    far less than the step it precedes.
    """
    from primus.backends.automodel.models.ideogram4.attn_processor import (
        Ideogram4VarlenAttnProcessor,
    )

    return [
        module
        for module in model.modules()
        if isinstance(_own_processor(module), Ideogram4VarlenAttnProcessor)
    ]


def _concrete(device: torch.device) -> torch.device:
    """``cuda`` as ``cuda:<current>``, so it compares equal to where a buffer lives."""
    if device.type == "cpu" or device.index is not None:
        return device
    return torch.device(device.type, torch.get_device_module(device.type).current_device())


def _reinstall_cause(modules: List[nn.Module], numel: int, device: torch.device) -> Optional[str]:
    """Why the shared buffer has to be (re)installed, or None when it is fine.

    The causes are distinguished because each one points somewhere different
    when it shows up in the log.
    """
    held = [getattr(m, PACKING_ATTR, None) for m in modules]
    present = [t for t in held if isinstance(t, torch.Tensor)]
    if not present:
        return "missing"
    if len(present) != len(held):
        return "new-consumer"
    first = present[0]
    if first.numel() != numel:
        return "size"
    if first.dtype != torch.int32 or first.device != device:
        return "device"
    if not all(t is first for t in present):
        return "unshared"
    return None


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
        if not hasattr(module, TOKENS_ATTR):
            setattr(module, TOKENS_ATTR, None)
    return shared


def _report_install(cause: str, previous, numel: int, count: int, max_seqlen: Optional[int], device) -> None:
    """Explain what happened, once per distinct cause."""
    if cause == "missing":
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
    elif cause == "size":
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
    elif cause == "device":
        _log_once(
            f"device-{previous.device}-{device}",
            logging.WARNING,
            "[PrimusIdeogramPacking] the cu_seqlens buffer was on %s but this publish targets "
            "%s, so it was replaced on every attention module. Expected once after the model "
            "moves; on every step it means the caller passes a different device each time.",
            previous.device,
            device,
        )
    elif cause == "new-consumer":
        _log_once(
            "new-consumer",
            logging.WARNING,
            "[PrimusIdeogramPacking] an attention module gained the var-len processor after the "
            "first publish, so the buffer was reinstalled to include it; %d module(s) now "
            "share it.",
            count,
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
    ``copy_`` here is then its only trip to the device, and nothing reads it back,
    which is the point of having built the packing from Python integers.

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
                "attention paths into one gradient average. primus_ideogram4.varlen_attention "
                "is set, so the processor install was skipped or came too late: it is "
                "skipped when model.pipeline_spec.transformer_cls names a class other than "
                "Ideogram4Transformer2DModel, and it reaches only attention modules built "
                "after it ran, or the model passed to it. Set "
                "primus_ideogram4.precompute_cu_seqlens: false to take the mask-derived path "
                "deliberately."
            )
        return None

    target = _concrete(torch.device(device) if device is not None else cu_seqlens.device)
    numel = cu_seqlens.numel()
    cause = _reinstall_cause(modules, numel, target)
    if cause is None:
        shared = getattr(modules[0], PACKING_ATTR)
    else:
        previous = next(
            (t for t in (getattr(m, PACKING_ATTR, None) for m in modules) if isinstance(t, torch.Tensor)),
            None,
        )
        shared = _install(modules, numel, target)
        _report_install(cause, previous, numel, len(modules), max_seqlen, target)
    shared.copy_(cu_seqlens)

    # The token total, for the processor's stale-packing check. Read only from a
    # host tensor, where it is free; a device-side packing would need a sync.
    packed_tokens = int(cu_seqlens[-1]) if cu_seqlens.device.type == "cpu" else None
    for module in modules:
        if getattr(module, BOUND_ATTR, None) != max_seqlen:
            setattr(module, BOUND_ATTR, max_seqlen)
        if getattr(module, TOKENS_ATTR, None) != packed_tokens:
            setattr(module, TOKENS_ATTR, packed_tokens)
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
        for attr in (BOUND_ATTR, TOKENS_ATTR):
            if hasattr(module, attr):
                delattr(module, attr)
    return removed
