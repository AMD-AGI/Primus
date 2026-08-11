###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Transport for the Ideogram-4 var-len packing: one shared non-persistent buffer.

WHY (this is the transport that works, after the one that did not):
  The adapter builds ``cu_seqlens`` on the host, where the caption lengths are already
  Python ints, and the attention processor has to receive it as a graph INPUT -- deriving it
  from the mask inside the compiled region does device->host reads, which graph-break and,
  under FSDP2, desync the per-layer collectives. The obvious channel, diffusers'
  ``attention_kwargs``, is DEAD for this model in diffusers 0.39.0: the LoRA decorator keeps
  only ``scale``, both block call sites pass four positional arguments, and
  ``Ideogram4TransformerBlock.forward`` has no ``**kwargs`` to forward. Nothing arrived.

WHAT (no diffusers fork, nothing copied from upstream):
  The processor is already handed its ``Ideogram4Attention`` module as the first argument,
  so the packing rides on that module. One ``int32`` tensor of fixed shape is registered as a
  NON-PERSISTENT buffer on every attention module -- the same tensor OBJECT on all of them,
  so the adapter publishes with a single in-place ``copy_`` per step instead of 34 writes,
  and a diffusers version bump cannot silently diverge from a copied forward.

MEASURED (2026-08-04, torch 2.12+rocm7.14, MI355X; runbook §7.10 has commands and logs):
  Dynamo lifts the buffer as a graph input rather than baking its values in -- placeholder
  ``l_self_modules_attention_buffers_primus_cu_seqlens_``, zero recompiles across five steps
  of changing values, output and grads tracking an uncompiled twin at ~2e-07. FSDP2 leaves it
  alone -- one object through ``fully_shard``, a plain tensor rather than a DTensor shard,
  absent from the state dict, ``[1, 1]`` graphs per rank, and a reduce-scattered grad matching
  the mean-of-per-rank-packings reference at 1.8e-07.

FOUR RULES. Each was measured failing when broken, and only rule 3 fails loudly (runbook §11.7):
  1. **Publish before the forward, never between forward and backward.** Per-layer compile
     sits inside the checkpoint wrapper, so the block -- and this buffer read -- runs twice
     per step, once forward and once recomputed on backward. A write in between corrupts the
     recomputed activations with no error.
  2. **Re-publish after materialization.** ``model.to_empty(device=...)``, the usual
     meta-device bring-up, gives every module its OWN buffer. That would leave layer 0
     reading the published value and the other 33 reading uninitialized memory.
     :func:`publish_packing` detects that and restores sharing rather than trusting it.
  3. **The buffer is read-only once published.** aiter's var-len kernel treats ``cu_seqlens``
     as a mutable argument: it saves the tensor for its backward and then writes it, bumping
     the autograd version counter on every call. Passing the SHARED buffer straight in
     therefore moves its version 34 times per forward while each layer's backward still
     expects the version it saved, and the step dies in ``.backward()`` (measured 2026-08-04:
     "IntTensor[5] is at version 35; expected version 34"). The processor clones before the
     kernel call for that reason -- the probe in §7.10 missed it because it exercised the
     DENSE path, where no kernel ever sees the packing.
  4. **The buffer outlives the step.** Anything that runs the transformer WITHOUT going
     through the adapter -- a sampling pass, an eval loop at a different batch size -- would
     read whatever was published last. Call :func:`clear_packing` first, which returns the
     processor to its mask-derived path. The processor also rejects a packing whose length
     does not match the batch, which is a static-shape comparison and therefore free, so this
     mistake fails loudly rather than silently.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# The buffer name appears verbatim as the graph placeholder in a TORCH_LOGS dump
# (``l_self_modules_attention_buffers_primus_cu_seqlens_``), so keep it greppable.
PACKING_ATTR = "_primus_cu_seqlens"
BOUND_ATTR = "_primus_max_seqlen"
_CONSUMERS_ATTR = "_primus_packing_consumers"

_logged: set = set()


def _log_once(key: str, level: int, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def _own_processor(module: nn.Module) -> object:
    """The processor this module itself holds, ignoring attributes it merely forwards.

    Plain ``getattr`` is wrong here: the activation-checkpoint wrapper delegates unknown
    attributes to the module it wraps, so it answers ``processor`` with its child's and gets
    counted as a second consumer. That doubled the module list (68 for 34 blocks), which is
    harmless for the copy_ but makes the one health number in the log -- "on N attention
    module(s)" -- unable to distinguish "all 34 blocks" from "half of them".
    """
    own = module.__dict__.get("processor")
    if own is None:
        # A processor implemented as an nn.Module would live in _modules instead.
        own = module._modules.get("processor")
    return own


def attention_modules(model: nn.Module) -> List[nn.Module]:
    """The attention modules that can actually READ the packing.

    Membership is decided by the processor type, which makes the whole transport
    self-gating: a run without ``PRIMUS_IDEOGRAM_VARLEN_ATTN`` has no var-len processors, so
    no buffers are installed and nothing is published, rather than maintaining per-step state
    that nobody reads. Cached on the model -- the module set is fixed once the recipe has
    built and parallelized it.
    """
    cached = getattr(model, _CONSUMERS_ATTR, None)
    if cached is not None:
        return cached

    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        Ideogram4VarlenAttnProcessor,
    )

    found = [
        module
        for module in model.modules()
        if isinstance(_own_processor(module), Ideogram4VarlenAttnProcessor)
    ]
    # Do not cache an empty result: on an odd ordering this could run before the processors
    # are swapped in, and caching [] would disable the transport for the whole run.
    if found:
        setattr(model, _CONSUMERS_ATTR, found)
    return found


def _shared_buffer(modules: List[nn.Module], numel: int, device: torch.device) -> Optional[torch.Tensor]:
    """The one buffer object every module shares, or None if that is no longer true.

    Returns None for all three ways the invariant can lapse: never installed, installed but
    the wrong shape/device (a batch-size change), or installed and no longer shared (a
    materialization pass handed each module its own tensor).
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
        # register_buffer refuses to re-register a live name, so replace in ``_buffers``
        # directly when it is already there. Either way it stays non-persistent.
        if PACKING_ATTR in module._buffers:
            module._buffers[PACKING_ATTR] = shared
        else:
            module.register_buffer(PACKING_ATTR, shared, persistent=False)
        if not hasattr(module, BOUND_ATTR):
            setattr(module, BOUND_ATTR, None)
    return shared


def _report_install(previous: object, numel: int, count: int, max_seqlen: Optional[int]) -> None:
    if not isinstance(previous, torch.Tensor):
        # The one line that proves the transport is live in a real run: the entry count is
        # 2B+1 and the module count should be every block's attention (34 for Ideogram-4).
        _log_once(
            "install",
            logging.INFO,
            "[PrimusIdeogramPacking] Installed a shared non-persistent cu_seqlens buffer "
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
            "[PrimusIdeogramPacking] cu_seqlens length changed %d -> %d (batch size changed), so "
            "the buffer was replaced and torch.compile will recompile. Constant shapes are the "
            "whole point of the reserved pad column; check drop_last and the eval batch size.",
            previous.numel(),
            numel,
        )
    else:
        _log_once(
            "resharing",
            logging.WARNING,
            "[PrimusIdeogramPacking] The cu_seqlens buffer was no longer shared across the %d "
            "attention module(s) -- something replaced module buffers after install "
            "(to_empty()/materialization is the usual cause). Sharing has been restored. Left "
            "unrepaired, one layer would read the published packing and the rest would read "
            "uninitialized memory, with no error raised.",
            count,
        )


def publish_packing(
    model: nn.Module,
    cu_seqlens: torch.Tensor,
    max_seqlen: Optional[int] = None,
    device: Optional[torch.device] = None,
    required: bool = False,
) -> Optional[torch.Tensor]:
    """Make ``cu_seqlens`` visible to every attention module for the forward that follows.

    Call this immediately before the model call and never between a forward and its backward
    (rule 1 in the module docstring). ``cu_seqlens`` may live on the host: the single
    ``copy_`` below is then the only host->device transfer, which is the point of building the
    packing from Python ints in the first place.

    Args:
        model: the (possibly FSDP2-wrapped, possibly compiled) transformer.
        cu_seqlens: ``int32 (2B+1,)`` cumulative segment starts, on any device.
        max_seqlen: the STATIC upper bound ``S``, not the batch's true maximum -- a
            data-derived int would be guarded by value.
        device: where the buffer should live. Defaults to ``cu_seqlens.device``, so pass the
            model's device explicitly when publishing from a host-built tensor.
        required: raise instead of returning None when nothing consumes the packing. Callers
            that have already decided the transport is in use should pass True: a rank that
            publishes into the void falls back to the mask-derived path for every layer, and
            under data parallelism its gradients then come from a different attention path than
            its peers' and are averaged in regardless. Nothing in the logs would say so --
            Primus quiets non-zero ranks once training starts.

    Returns:
        The shared buffer, or None when nothing in ``model`` consumes it and ``required`` is
        False.
    """
    modules = attention_modules(model)
    if not modules:
        if required:
            raise RuntimeError(
                "[PrimusIdeogramPacking] a cu_seqlens packing was precomputed but no attention "
                "module can read it: none of the model's modules carry an "
                "Ideogram4VarlenAttnProcessor. Every layer would silently fall back to deriving "
                "the packing from the attention mask, which host-syncs, graph-breaks under "
                "torch.compile, and -- if this happens on only some ranks -- mixes two "
                "attention paths into one gradient average. Either the var-len processor install "
                "did not run (PRIMUS_IDEOGRAM_VARLEN_ATTN) or the model was replaced after it "
                "did; unset PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS to run the legacy path on "
                "purpose."
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
    """Pick the packing for this attention call: explicit argument first, then module state.

    An explicit argument wins so that a future kwargs route (patching the diffusers block
    forward) can override this without touching the processor. The ``getattr`` fallback is a
    constant attribute lookup that Dynamo lifts as a graph input -- not a data-dependent read,
    and not a graph break.
    """
    if cu_seqlens is None:
        cu_seqlens = getattr(attn, PACKING_ATTR, None)
        if cu_seqlens is not None and max_seqlen is None:
            max_seqlen = getattr(attn, BOUND_ATTR, None)
    return cu_seqlens, max_seqlen


def clear_packing(model: nn.Module) -> int:
    """Drop the buffer from every consumer. For tests and for switching transports mid-process."""
    removed = 0
    for module in attention_modules(model):
        if module._buffers.pop(PACKING_ATTR, None) is not None:
            removed += 1
        if hasattr(module, BOUND_ATTR):
            delattr(module, BOUND_ATTR)
    if hasattr(model, _CONSUMERS_ATTR):
        delattr(model, _CONSUMERS_ATTR)
    return removed
