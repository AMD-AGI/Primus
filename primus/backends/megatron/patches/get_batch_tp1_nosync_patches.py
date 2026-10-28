###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Skip the TP broadcasts in ``get_batch_on_this_tp_rank`` when TP=1 (``get_batch_skip_tp1_broadcast``).

With a single-rank tensor-parallel group every broadcast there is a no-op, but
``_broadcast_cu_seqlens`` still builds ``torch.tensor(n, device=cuda)`` from a host
int: a synchronous pageable H2D copy that waits for the whole GPU queue. It runs
once per microbatch, so the GPU drains and then waits for the CPU to launch the next
forward (about 38 ms of GPU idle per step on the DeepSeek-V3 proxy, EP4 on
4x MI455X). With TP=1 this patch moves the batch to the GPU exactly like the TP
rank-0 path (same keys, same order, ``non_blocking=True``) and does nothing else.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_KEYS = (
    "tokens",
    "labels",
    "loss_mask",
    "attention_mask",
    "position_ids",
    "cu_seqlens",
    "max_seqlen",
    "local_cp_size",
)


def _tp1(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return bool(
        args is not None
        and getattr(args, "get_batch_skip_tp1_broadcast", False)
        and getattr(args, "tensor_model_parallel_size", 1) == 1
        and not getattr(args, "hybrid_context_parallel", False)
    )


@register_patch(
    "megatron.training.get_batch_tp1_nosync",
    backend="megatron",
    phase="before_train",
    description="TP=1: get_batch_on_this_tp_rank without the no-op broadcasts and their H2D sync",
    condition=_tp1,
)
def patch_get_batch_tp1_nosync(ctx: PatchContext) -> None:
    import megatron.training.utils as mutils
    from megatron.core import parallel_state as mpu

    from primus.backends.megatron.patches._rebind import rebind_everywhere

    original = mutils.get_batch_on_this_tp_rank

    def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):
        if mpu.get_tensor_model_parallel_world_size() != 1:
            return original(data_iterator, mtp_on_this_rank=mtp_on_this_rank)
        assert data_iterator is not None
        data = next(data_iterator)
        return {k: (data[k].cuda(non_blocking=True) if k in data else None) for k in _KEYS}

    rebound = rebind_everywhere(mutils, "get_batch_on_this_tp_rank", get_batch_on_this_tp_rank)
    log_rank_0(f"[Patch:megatron.training.get_batch_tp1_nosync] rebound in {rebound}")
