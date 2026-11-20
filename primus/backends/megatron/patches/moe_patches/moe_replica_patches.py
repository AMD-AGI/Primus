###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Wire dynamic expert replication (``moe_replica_planner``) into Megatron's model build and step.

With replication on, replicas of hot experts compute some of their rows on other EP ranks
(``primus.backends.megatron.core.transformer.moe.replica``). Three hooks, all off without ``moe_replica_planner``:

- ``megatron.moe.replica_attach``: once a ``MoELayer`` is built, its token dispatcher's scheduler is handed to the
  layer's two expert linears (no global lookup at run time).
- ``megatron.moe.replica_slots``: after ``get_model`` (DDP wrapping gives the expert weights their ``main_grad``),
  the replica slots are allocated, so their memory is taken and reported at startup, and each model chunk records
  its schedulers.
- ``megatron.moe.replica_finalize``: the slot gradients must be in the owners' ``main_grad`` before
  ``finalize_model_grads`` reduces the expert gradients and the optimizer steps, so the wrapper returns them first.

``training.py`` imports ``get_model`` and ``finalize_model_grads`` by name and the pipeline schedules call
``config.finalize_model_grads_func``, which ``train()`` sets from that name, so the module attributes and every alias
are rebound.
"""

import functools
import importlib

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_ATTACH_KEY = "megatron.moe.replica_attach"
_SLOTS_KEY = "megatron.moe.replica_slots"
_FINALIZE_KEY = "megatron.moe.replica_finalize"


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_replica_planner", None))


@register_patch(
    _ATTACH_KEY,
    backend="megatron",
    phase="before_train",
    description="Hand each MoE layer's replica scheduler to its expert linears once the layer is built",
    condition=_enabled,
)
def patch_moe_replica_attach(ctx: PatchContext):
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.core.transformer.moe.replica.scheduler import attach

    cls = moe_layer.MoELayer
    if is_patched(cls, _ATTACH_KEY):
        return
    original = cls.__init__

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        attach(self)

    cls.__init__ = __init__
    mark_patched(cls, _ATTACH_KEY)
    log_rank_0(
        f"[Patch:{_ATTACH_KEY}] MoELayer.__init__ attaches the replica scheduler to the expert linears"
    )


@register_patch(
    _SLOTS_KEY,
    backend="megatron",
    phase="before_train",
    description="Allocate the MoE replica slots after get_model",
    condition=_enabled,
    # after the default-priority patches that replace get_model, so this wraps their version
    priority=60,
)
def patch_moe_replica_slots(ctx: PatchContext):
    import megatron.training.training as training
    import torch

    from primus.backends.megatron.core.transformer.moe.replica.scheduler import (
        allocate_slots,
    )

    if is_patched(training, _SLOTS_KEY):
        return
    original = training.get_model

    @functools.wraps(original)
    def _get_model(*args, **kwargs):
        model = original(*args, **kwargs)
        chunks = model if isinstance(model, (list, tuple)) else [model]
        before = torch.cuda.memory_allocated()
        scheds = allocate_slots(chunks)
        if scheds:
            gib = (torch.cuda.memory_allocated() - before) / 2**30
            log_rank_0(
                f"[moe_replica] {len(scheds)} MoE layers replicate experts; replica slots: {gib:.2f} GiB"
            )
        return model

    rebound = rebind_everywhere(training, "get_model", _get_model)
    mark_patched(training, _SLOTS_KEY)
    log_rank_0(f"[Patch:{_SLOTS_KEY}] wrapped get_model in {', '.join(rebound)}")


@register_patch(
    _FINALIZE_KEY,
    backend="megatron",
    phase="before_train",
    description="Return the MoE replica slot gradients into the owners' main_grad before finalize_model_grads",
    condition=_enabled,
    # after the default-priority patches that replace finalize_model_grads, so this wraps their version
    priority=60,
)
def patch_moe_replica_finalize(ctx: PatchContext):
    # the package re-exports a function of the same name, which hides the module as an attribute
    finalize_model_grads = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    if is_patched(finalize_model_grads, _FINALIZE_KEY):
        return

    from primus.backends.megatron.core.transformer.moe.replica.scheduler import (
        end_step_all,
    )

    original = finalize_model_grads.finalize_model_grads

    @functools.wraps(original)
    def _finalize_model_grads(model, *args, **kwargs):
        end_step_all(model)
        return original(model, *args, **kwargs)

    rebound = rebind_everywhere(finalize_model_grads, "finalize_model_grads", _finalize_model_grads)
    mark_patched(finalize_model_grads, _FINALIZE_KEY)
    log_rank_0(f"[Patch:{_FINALIZE_KEY}] wrapped finalize_model_grads in {', '.join(rebound)}")
