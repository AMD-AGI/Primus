###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Wire dynamic expert replication (``moe_replica_planner``) into Megatron's MoE layer and DDP.

With replication on, replicas of hot experts compute some of their rows on other EP ranks
(``primus.backends.megatron.core.transformer.moe.replica``). Two hooks, off without ``moe_replica_planner``:

- ``megatron.moe.replica_attach``: once a ``MoELayer`` is built, its token dispatcher's scheduler is handed to the
  layer's two expert linears, which allocate their replica slots (``replica.scheduler.attach``). Subclasses of
  ``MoELayer`` build their own experts or forward and are rejected; ``attach`` also rejects a layer whose
  dispatcher has no replica scheduler (replication would silently not run).
- ``megatron.moe.replica_ddp``: DDP's backward hook of a replica owner's weights (``moe_replica_owner``) does not
  mark them ready for the gradient reduction on the last micro-batch: with ``overlap_grad_reduce`` their bucket
  would be scaled and reduced before the replica gradients reach ``main_grad``. ``end_step`` marks them ready
  after returning the replica gradients (``moe_replica_grad_ready``), once per step, if the hook ran in the last
  micro-batch; Megatron's pipeline schedules (with ``align_grad_reduce``) may run every backward of a stage in
  ``no_sync`` and start a model chunk's reduction after its last backward, and then there is nothing to mark.
  With ``overlap_param_gather`` the replica weights are pushed before the expert linears' forward pre-hook waits
  for their all-gather, so the push first waits for it the same way (``moe_replica_wait_param_gather``).
"""

import functools
import weakref

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_ATTACH_KEY = "megatron.moe.replica_attach"
_DDP_KEY = "megatron.moe.replica_ddp"


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
    margs = get_args(ctx)
    overlap = getattr(margs, "overlap_grad_reduce", False) or getattr(margs, "overlap_param_gather", False)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if type(self) is not cls:
            raise NotImplementedError(
                f"moe_replica_planner supports Megatron's MoELayer only, not its subclass {type(self).__name__}"
            )
        if overlap:
            from megatron.core.distributed.distributed_data_parallel import (
                DistributedDataParallel,
            )

            # the patch runner logs a failed patch and goes on: without this one the replica gradients would miss
            # the overlapped reduction
            if not is_patched(DistributedDataParallel, _DDP_KEY):
                raise RuntimeError(
                    f"moe_replica_planner with overlap_grad_reduce or overlap_param_gather needs the {_DDP_KEY} "
                    "patch, which was not applied (see its error in the patch log)"
                )
        attach(self)

    cls.__init__ = __init__
    mark_patched(cls, _ATTACH_KEY)
    log_rank_0(
        f"[Patch:{_ATTACH_KEY}] MoELayer.__init__ attaches the replica scheduler to the expert linears"
    )


def defer_owner_grad_ready(ddp, param, hook):
    """The DDP backward hook ``hook`` of a replica owner's ``param``, without its grad-ready registration on the
    last micro-batch; puts ``moe_replica_grad_ready`` and ``moe_replica_wait_param_gather`` on ``param``."""
    ddp_ref = weakref.ref(ddp)
    ran = pending = False

    def deferred_hook(*args):
        nonlocal ran, pending
        group = ddp.param_to_bucket_group.get(param)
        if group is None or not ddp.ddp_config.overlap_grad_reduce:
            return hook(*args)
        # the original hook still adds the gradient into main_grad; only its register_grad_ready is skipped
        last, group.is_last_microbatch = group.is_last_microbatch, False
        try:
            hook(*args)
        finally:
            group.is_last_microbatch = last
        ran, pending = True, pending or last

    def grad_ready():
        nonlocal ran, pending
        ddp = ddp_ref()
        if ddp is None or not ddp.ddp_config.overlap_grad_reduce or param not in ddp.param_to_bucket_group:
            return
        if not ran:
            raise RuntimeError(
                "moe replica: the DDP backward hook of a replica owner's weights did not run before the step ended"
            )
        # the hook ran in no_sync only (pipeline schedules with align_grad_reduce): DDP's hook would not have
        # marked the weights either, the schedule starts the reduction itself
        if pending:
            ddp.param_to_bucket_group[param].register_grad_ready(param, ddp.force_all_reduce)
        ran = pending = False

    def wait_param_gather():
        ddp = ddp_ref()
        if (
            ddp is None
            or not ddp.ddp_config.overlap_param_gather
            or param not in ddp.param_to_bucket_group
            # the forward pre-hooks are off in the first iteration, evaluation and checkpointing, which gather
            # the parameters synchronously; a wait here would dispatch an all-gather nobody waits for
            or not (ddp.use_forward_hook and len(ddp.remove_forward_pre_hook_handles) > 0)
        ):
            return
        # as DDP's forward pre-hook does
        skip_next_bucket_dispatch = (
            ddp.ddp_config.align_param_gather or ddp.overlap_param_gather_with_optimizer_step
        )
        ddp.param_to_bucket_group[param].finish_param_sync(
            skip_next_bucket_dispatch=skip_next_bucket_dispatch
        )

    param.moe_replica_grad_ready = grad_ready
    param.moe_replica_wait_param_gather = wait_param_gather
    return deferred_hook


@register_patch(
    _DDP_KEY,
    backend="megatron",
    phase="before_train",
    description="Mark replica owners' weights ready for DDP's gradient reduction after their replica gradients",
    condition=_enabled,
)
def patch_moe_replica_ddp(ctx: PatchContext):
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    cls = DistributedDataParallel
    if is_patched(cls, _DDP_KEY):
        return
    original = cls._make_backward_post_hook

    @functools.wraps(original)
    def _make_backward_post_hook(self, param):
        hook = original(self, param)
        if not getattr(param, "moe_replica_owner", False):
            return hook
        return defer_owner_grad_ready(self, param, hook)

    cls._make_backward_post_hook = _make_backward_post_hook
    mark_patched(cls, _DDP_KEY)
    log_rank_0(f"[Patch:{_DDP_KEY}] DDP marks replica owners' weights ready after their replica gradients")
