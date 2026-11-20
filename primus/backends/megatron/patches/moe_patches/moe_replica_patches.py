###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Wire dynamic expert replication (``moe_replica_planner``) into Megatron's MoE layer.

With replication on, replicas of hot experts compute some of their rows on other EP ranks
(``primus.backends.megatron.core.transformer.moe.replica``). One hook, off without ``moe_replica_planner``:

- ``megatron.moe.replica_attach``: once a ``MoELayer`` is built, its token dispatcher's scheduler is handed to the
  layer's two expert linears, which allocate their replica slots, and the step ends in the backward of the layer's
  input (``replica.scheduler.attach``). Subclasses of ``MoELayer`` build their own experts or forward and are
  rejected; ``attach`` also rejects a layer whose dispatcher has no replica scheduler (replication would silently
  not run).
"""

import functools

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_ATTACH_KEY = "megatron.moe.replica_attach"


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
        if type(self) is not cls:
            raise NotImplementedError(
                f"moe_replica_planner supports Megatron's MoELayer only, not its subclass {type(self).__name__}"
            )
        attach(self)

    cls.__init__ = __init__
    mark_patched(cls, _ATTACH_KEY)
    log_rank_0(
        f"[Patch:{_ATTACH_KEY}] MoELayer.__init__ attaches the replica scheduler to the expert linears"
    )
