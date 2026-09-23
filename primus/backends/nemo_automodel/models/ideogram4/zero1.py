###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DDP and ZeRO-1 for Ideogram-4.

WHY THIS PATH EXISTS:
  On a device with memory to spare at the batch size the model is compute-bound
  at, one way to spend that memory is to shard LESS and cut collectives. Pure DDP
  replicates parameters and gradients, so there is no per-layer all-gather at all
  -- but it also replicates the optimizer state, which for AdamW is two more
  copies of the parameters in float32. ZeRO-1 shards ONLY that optimizer state
  across the data-parallel ranks, so the compute profile stays DDP's and the
  optimizer memory comes back.

WHERE IT IS AND IS NOT THE RIGHT CHOICE, stated up front because the answer is
not the one the memory arithmetic alone suggests:
  Per-layer compilation is wired only into the FSDP2 path; the DDP manager never
  compiles. So on a single node, where the exposed FSDP communication is a small
  part of the step anyway, DDP with ZeRO-1 gives up compilation and cannot make
  that back from the collectives it saved. This is the MULTI-NODE lever, where
  all-gather crosses the slower interconnect and there is real traffic to remove.
  The single-node way to shard less while KEEPING compilation is hybrid sharding
  (a data-parallel replicate dimension greater than one).

  It is also a reference point: DDP alone is the baseline that makes the ZeRO-1
  optimizer-memory recovery measurable, which is why the two are separately
  switchable.

WHAT IT DOES -- two patches, neither editing any upstream source:

  1. THE OPTIMIZER. Wraps the optimizers the recipe builds in torch's
     ZeroRedundancyOptimizer. The subtle part is WHICH classes to patch; see
     ``_optimizer_config_classes``, where patching only the obvious one is a
     silent no-op.

  2. ACTIVATION CHECKPOINTING ON THE DDP PATH. The DDP manager applies
     checkpointing by submodule attribute name -- attributes a single-stream
     diffusion block does not have -- so on this path
     ``activation_checkpointing`` is a SILENT NO-OP, the same failure the FSDP
     strategy was written to fix. Without real checkpointing the model runs out of
     memory at any useful batch size, so DDP is not merely slower without this
     patch, it does not run.

     The wrapping is delegated to the shared activation-checkpointing helper, the
     same one the FSDP strategy uses. That is deliberate: it means the two paths
     agree on what "selective" and "false" mean, rather than each having its own
     copy of a branch with two known silent failure modes.

Activation, by environment, with no config schema change:
  PRIMUS_IDEOGRAM_DDP=1     real activation checkpointing on the DDP path. The
                            pure-DDP baseline.
  PRIMUS_IDEOGRAM_ZERO1=1   the above, plus sharding the optimizer state.

PRIMUS_IDEOGRAM_AC_EVERY, the block stride, is read from the FSDP strategy's
parser and applies here too, since it describes how much of the model's
activation memory to trade for recompute and has nothing to do with which
sharding strategy is in use.
"""
from __future__ import annotations

import functools
import inspect
import logging

from primus.backends.nemo_automodel._env import env_flag
from primus.backends.nemo_automodel.distributed import activation_checkpointing as ac
from primus.backends.nemo_automodel.models.ideogram4 import (
    parallelize as ideogram_parallelize,
)

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusIdeogramZeRO1]"
_MODEL_NAME = "Ideogram4Transformer2DModel"
_BLOCK_ATTRS = ("layers",)


def is_zero1_enabled() -> bool:
    """Whether to shard the optimizer state as well as fixing DDP checkpointing."""
    return env_flag("PRIMUS_IDEOGRAM_ZERO1")


def is_ddp_enabled() -> bool:
    """Whether to fix activation checkpointing on the DDP path.

    Implied by ZeRO-1, which cannot run without it, but separately switchable so
    that pure DDP is available as the baseline the optimizer saving is measured
    against.
    """
    return env_flag("PRIMUS_IDEOGRAM_DDP")


def _looks_like_ideogram(model) -> bool:
    """Matched by class name, the way AutoModel dispatches strategies.

    Compared by name rather than isinstance so this module never has to import
    diffusers, which keeps it importable for the patch condition.
    """
    return type(model).__name__ == _MODEL_NAME and any(hasattr(model, attr) for attr in _BLOCK_ATTRS)


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
            "%s PRIMUS_IDEOGRAM_ZERO1 is set but the parameters are already sharded, "
            "so this is an FSDP run. FSDP shards the optimizer state itself, so there "
            "is nothing for ZeRO-1 to do; keeping the plain optimizer.",
            _LOG_PREFIX,
        )
        return base

    import torch.distributed as dist
    from torch.distributed.optim import ZeroRedundancyOptimizer

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        logger.warning(
            "%s PRIMUS_IDEOGRAM_ZERO1 is set but there is only one rank, so there is "
            "nothing to shard the optimizer state across; keeping the plain optimizer.",
            _LOG_PREFIX,
        )
        return base

    optimizer_cls = type(base)
    defaults = dict(base.defaults)
    learning_rate = defaults.pop("lr", None)

    optimizer = ZeroRedundancyOptimizer(
        params,
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


def _install_ddp_ac_patch() -> bool:
    """Make activation checkpointing take effect on the DDP path."""
    import nemo_automodel.components.distributed.ddp as ddp_module
    import nemo_automodel.components.distributed.parallelizer as P

    if getattr(ddp_module, "_primus_ideogram_ddp_ac_patched", False):
        return True

    original = ddp_module.DDPManager.parallelize

    def parallelize(self, model):
        requested = getattr(self, "activation_checkpointing", False)
        if not (ac.normalize(requested) and _looks_like_ideogram(model)):
            return original(self, model)

        # Through the shared helper, so this path and the FSDP path agree on what
        # each setting means -- including the block stride, which describes the
        # model's memory profile and not the sharding strategy, so it would be
        # wrong for it to apply to only one of the two. Before the DDP wrap, since
        # the module structure has to be final before anything indexes the
        # parameters.
        ac.apply(
            P,
            model,
            _BLOCK_ATTRS,
            requested,
            stride=ideogram_parallelize.ac_stride(),
            log_prefix=_LOG_PREFIX,
        )

        # Suppress the manager's own checkpointing for this call. It wraps by
        # submodule attribute name and would find nothing, but leaving it enabled
        # means a second traversal whose "wrapped 0" is indistinguishable from the
        # bug this patch fixes.
        self.activation_checkpointing = False
        try:
            return original(self, model)
        finally:
            self.activation_checkpointing = requested

    ddp_module.DDPManager.parallelize = parallelize
    ddp_module._primus_ideogram_ddp_ac_patched = True
    return True


def install() -> bool:
    """Install the DDP and ZeRO-1 patches.

    A no-op returning False unless one of the two switches is set. Idempotent, and
    edits no upstream source.

    The checkpointing patch touches only the DDP manager, so it is inert on the
    FSDP2 path, where that manager is never used.
    """
    if not (is_zero1_enabled() or is_ddp_enabled()):
        return False

    _install_ddp_ac_patch()
    if is_zero1_enabled():
        _install_optimizer_patch()
    return True
