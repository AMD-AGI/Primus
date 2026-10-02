###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""ZeRO-1 optimizer sharding for DDP runs.

WHY:
  Pure DDP replicates parameters and gradients, so there is no per-layer
  all-gather -- but it also replicates the optimizer state, which for AdamW is
  two more float32 copies of the parameters. ZeRO-1 shards only that state across
  the data-parallel ranks, keeping DDP's compute profile.

  Per-layer compilation is wired only into the FSDP2 path, so on a single node
  DDP with ZeRO-1 gives up compilation for collectives that were a small part of
  the step anyway. It is the multi-node lever; on one node, hybrid sharding (a
  data-parallel replicate dimension above one) sheds traffic and keeps compile.

WHAT (NO AutoModel fork):
  Wraps the optimizers the recipe builds in torch's ZeroRedundancyOptimizer. The
  subtle part is which classes to patch; see ``_optimizer_config_classes``.

  Activation checkpointing on the DDP path is not handled here: the Ideogram-4
  sidecar checkpoints blocks on every strategy, and the DDP flag repair forwards
  the setting upstream's parser drops.

Opt in with ``primus_ideogram4.zero1: true`` in the module config.
"""
from __future__ import annotations

import functools
import inspect
import logging

from primus.backends.nemo_automodel import options

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusIdeogramZeRO1]"

# AutoModel saves the optimizer through its state_dict, which an unconsolidated
# ZeroRedundancyOptimizer refuses on every rank, so the first save would crash.
_checkpointing_enabled = False


def is_zero1_enabled() -> bool:
    """Whether to shard the optimizer state."""
    return options.flag("primus_ideogram4.zero1")


def _params_are_dtensor(params) -> bool:
    """Whether these parameters are already sharded, i.e. this is an FSDP run."""
    return any(type(p).__name__ == "DTensor" or hasattr(p, "_local_tensor") for p in params)


def _constructor_defaults(optimizer_cls, defaults):
    """Filter ``defaults`` to what the optimizer's own constructor accepts.

    ZeRO forwards these to rebuild a per-rank optimizer, and an optimizer's
    ``defaults`` can hold keys its ``__init__`` does not take: AdamW carries
    ``decoupled_weight_decay``, set internally by its parent, which AdamW's own
    signature has no parameter for. Passing it back in is a TypeError from inside
    ZeRO, which reads as ZeRO being broken rather than as this.

    A constructor taking ``**kwargs`` is left alone, since it may accept keys that
    are not named in its signature.
    """
    try:
        signature = inspect.signature(optimizer_cls.__init__)
    except (ValueError, TypeError):
        # Signature unavailable, as for a C implementation. Pass through unchanged
        # rather than guess at what to drop.
        return defaults

    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return defaults

    accepted = set(signature.parameters)
    dropped = [key for key in defaults if key not in accepted]
    if dropped:
        logger.info(
            "%s dropped optimizer defaults %s, which %s does not accept in its " "constructor",
            _LOG_PREFIX,
            dropped,
            optimizer_cls.__name__,
        )
    return {key: value for key, value in defaults.items() if key in accepted}


def _wrap_in_zero1(base):
    """Rebuild ``base`` as a ZeroRedundancyOptimizer, or return it unchanged.

    Returns it unchanged in the two cases where ZeRO-1 does not APPLY, each with a
    warning: parameters that are already sharded, and a single rank. Both mean
    there is no replicated optimizer state to shard, so nothing is lost.

    A failure to build one, by contrast, raises. Nothing else in the run needs
    ZeRO-1 to be there, so falling back would let training start -- and then either
    run out of memory or quietly use the replicated optimizer state this was turned
    on to avoid.
    """
    if type(base).__name__ == "ZeroRedundancyOptimizer":
        # A subclass build() that chains to super() would otherwise double-wrap.
        return base

    params = [p for group in base.param_groups for p in group["params"]]
    if _params_are_dtensor(params):
        logger.warning(
            "%s primus_ideogram4.zero1 is set but the parameters are already sharded, "
            "so this is an FSDP run. FSDP shards the optimizer state itself, so there "
            "is nothing for ZeRO-1 to do; keeping the plain optimizer.",
            _LOG_PREFIX,
        )
        return base

    import torch.distributed as dist
    from torch.distributed.optim import ZeroRedundancyOptimizer

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        logger.warning(
            "%s primus_ideogram4.zero1 is set but there is only one rank, so there is "
            "nothing to shard the optimizer state across; keeping the plain optimizer.",
            _LOG_PREFIX,
        )
        return base

    if _checkpointing_enabled:
        raise RuntimeError(
            f"{_LOG_PREFIX} primus_ideogram4.zero1 cannot be combined with checkpoint.enabled: "
            "AutoModel saves the optimizer through state_dict(), which ZeroRedundancyOptimizer "
            "refuses until consolidated, so the first save would fail. Turn one of them off."
        )

    optimizer_cls = type(base)
    defaults = dict(base.defaults)
    learning_rate = defaults.pop("lr", None)

    optimizer = ZeroRedundancyOptimizer(
        # The groups, not a flat list: AutoModel's param_group_overrides (lr_mult,
        # wd_mult) live in them.
        [dict(group) for group in base.param_groups],
        optimizer_class=optimizer_cls,
        lr=learning_rate,
        # The overlapping mode ties the optimizer step to DDP's gradient buckets
        # and does not support changing the learning rate after construction,
        # which every schedule here does.
        overlap_with_ddp=False,
        **_constructor_defaults(optimizer_cls, defaults),
    )
    logger.info(
        "%s wrapped %s in ZeroRedundancyOptimizer across %d ranks; the optimizer "
        "state is sharded while DDP keeps parameters and gradients replicated.",
        _LOG_PREFIX,
        optimizer_cls.__name__,
        dist.get_world_size(),
    )
    return optimizer


def _optimizer_config_classes():
    """Every class in the optimizer-config hierarchy that defines its OWN ``build``.

    THIS IS THE PART THAT IS EASY TO GET WRONG, and it fails silently.

    Patching the base class alone is not enough. A config naming a plain torch
    optimizer resolves to a torch class, not a config subclass, so the recipe wraps
    it in a factory config -- and that class overrides ``build`` without ever
    chaining to ``super()``. A base-class-only patch is therefore never called: no
    sharding, no warning, and a run that looks correct while the optimizer state
    stays replicated. At least one other subclass overrides ``build`` the same way.

    So the hierarchy is walked rather than those classes being named. Naming them
    would work today and re-open the same hole the first time someone adds another
    override.
    """
    from nemo_automodel.components.optim.optimizer import OptimizerConfig

    seen, stack, found = set(), [OptimizerConfig], []
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        if "build" in vars(cls):
            found.append(cls)
        stack.extend(cls.__subclasses__())
    return found


def _install_optimizer_patch() -> bool:
    """Wrap every optimizer the recipe builds."""
    patched = []
    for cls in _optimizer_config_classes():
        existing = vars(cls)["build"]
        if getattr(existing, "_primus_zero1_patched", False):
            patched.append(cls.__name__)
            continue

        def _wrap(original):
            @functools.wraps(original)
            def build(self, *args, **kwargs):
                optimizers = original(self, *args, **kwargs)
                # Re-checked at call time rather than captured at install time, so
                # the patch is inert if the switch is turned off between the two.
                if not is_zero1_enabled():
                    return optimizers
                return [_wrap_in_zero1(opt) for opt in optimizers]

            build._primus_zero1_patched = True
            return build

        cls.build = _wrap(existing)
        patched.append(cls.__name__)

    logger.info("%s optimizer sharding installed on %s", _LOG_PREFIX, patched)
    return bool(patched)


def install(checkpoint_enabled: bool = True) -> bool:
    """Install the optimizer patch.

    A no-op returning False unless ``primus_ideogram4.zero1`` is set. Idempotent,
    and edits no upstream source. Inert on FSDP2 runs, whose parameters are
    already sharded. With ``checkpoint_enabled`` (AutoModel's
    ``checkpoint.enabled``, which defaults to true) the optimizer build raises
    wherever ZeRO-1 would apply.
    """
    global _checkpointing_enabled
    if not is_zero1_enabled():
        return False
    _checkpointing_enabled = bool(checkpoint_enabled)
    return _install_optimizer_patch()
