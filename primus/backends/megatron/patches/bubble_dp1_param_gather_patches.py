###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Skip the distributed-optimizer param all-gather when the data-parallel group has one rank
(opt-in: ``PRIMUS_BUBBLE_SKIP_DP1_AG=1``; default off).

With one DP rank the all-gather of each bucket is ``all_gather_into_tensor(param_data,
shard(param_data, 1)[0])``: input and output are the same buffer, so it is an identity. RCCL
launches no kernel for it, but every bucket still costs a host-side coalesced collective
(~17 us of host time each) and a cross-stream event handshake between the compute stream and
the NCCL stream. On the DeepSeek-V4 proxy (EP=1, 32 buckets) that is ~0.6 ms of host time and
a ~180 us GPU gap after the optimizer (the next eager kernel waits for the NCCL stream).

The patch keeps Megatron's bookkeeping (``param_gather_handle``, ``param_gather_dispatched``,
``_post_param_sync`` on a forced sync with overlap) and only drops the collective. Layer-wise
optimizers (``use_distributed_optimizer=False`` path) already return early upstream.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled() -> bool:
    return os.environ.get("PRIMUS_BUBBLE_SKIP_DP1_AG", "0") == "1"


@register_patch(
    "megatron.bubble_skip_dp1_param_gather",
    backend="megatron",
    phase="before_train",
    description="Skip identity param all-gathers on a 1-rank DP group (PRIMUS_BUBBLE_SKIP_DP1_AG=1)",
    priority=60,
    condition=lambda ctx: _enabled(),
)
def patch_skip_dp1_param_gather(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    orig = _ParamAndGradBucketGroup.start_param_sync
    if getattr(orig, "_primus_bubble_dp1", False):
        return
    n = {"skipped": 0}

    def start_param_sync(self, force_sync: bool = False):
        if (
            self.ddp_config.use_distributed_optimizer
            and getattr(self, "intra_distributed_optimizer_instance_size", None) == 1
            and self.param_gather_handle is None
        ):
            if n["skipped"] == 0:
                log_rank_0("[Patch:bubble_skip_dp1_param_gather] DP size 1: param all-gather skipped (identity)")
            n["skipped"] += 1
            if force_sync and self.ddp_config.overlap_param_gather:
                self._post_param_sync()
            self.param_gather_dispatched = True
            return
        return orig(self, force_sync=force_sync)

    start_param_sync._primus_bubble_dp1 = True
    _ParamAndGradBucketGroup.start_param_sync = start_param_sync
    log_rank_0("[Patch:bubble_skip_dp1_param_gather] installed (PRIMUS_BUBBLE_SKIP_DP1_AG=1)")
