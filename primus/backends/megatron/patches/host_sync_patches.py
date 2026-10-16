###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Remove two blocking host->device copies per micro-batch (opt-in, ``PRIMUS_NO_HOST_SYNC=1``).

1. ``megatron.training.utils.get_batch_on_this_tp_rank`` builds ``torch.tensor(n, device=cuda)``
   for the cu_seqlens broadcast even with TP=1, where every broadcast is a no-op. That pageable
   copy blocks the CPU until the previous micro-batch's backward has drained, so the GPU idles
   while the CPU then runs get_batch and the embedding. With TP=1 (no hybrid CP) the batch is
   just the micro-batch moved with ``non_blocking=True``.
2. TE's cross-entropy backward checks ``torch.equal(grad_output, torch.tensor(1.0, device=...))``
   (another blocking copy) unless ``is_cg_capturable=True``; with per-token grad_output the check
   never matches, so passing ``is_cg_capturable=True`` changes nothing but the sync.
Outputs are identical.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_NO_HOST_SYNC", "0").strip().lower() in ("1", "true", "yes", "on")


@register_patch(
    "megatron.no_host_sync",
    backend="megatron",
    phase="before_train",
    description="TP=1 get_batch without the blocking cu_seqlens copy; TE CE backward without its sync",
    condition=_enabled,
)
def patch_no_host_sync(ctx: PatchContext):
    from megatron.core import parallel_state as mpu
    from megatron.training import get_args
    from megatron.training import utils as tutils

    from primus.backends.megatron.patches._rebind import rebind_everywhere

    orig_get_batch = tutils.get_batch_on_this_tp_rank

    def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):
        args = get_args()
        if (
            mpu.get_tensor_model_parallel_world_size() != 1
            or getattr(args, "hybrid_context_parallel", False)
            or not (args.pipeline_model_parallel_size == 1 or mtp_on_this_rank)
        ):
            return orig_get_batch(data_iterator, mtp_on_this_rank)
        data = next(data_iterator)
        keys = (
            "tokens",
            "labels",
            "loss_mask",
            "attention_mask",
            "position_ids",
            "cu_seqlens",
            "max_seqlen",
            "local_cp_size",
        )
        return {
            k: (data[k].cuda(non_blocking=True) if k in data and data[k] is not None else None) for k in keys
        }

    names = rebind_everywhere(tutils, "get_batch_on_this_tp_rank", get_batch_on_this_tp_rank)

    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.models.common.language_module import language_module as lm

    orig_ce = te_ext.te_parallel_cross_entropy
    if orig_ce is not None and getattr(te_ext, "_TE_SUPPORTS_CG_CAPTURABLE", False):

        def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable: bool = False):
            return orig_ce(logits, labels, tp_group, True)

        rebind_everywhere(te_ext, "te_parallel_cross_entropy", te_parallel_cross_entropy)
        lm.te_parallel_cross_entropy = te_parallel_cross_entropy
        ce = "and TE cross-entropy backward"
    else:
        ce = "(TE cross-entropy unchanged: no is_cg_capturable support)"
    log_rank_0(f"[Patch:megatron.no_host_sync] get_batch_on_this_tp_rank rebound in {names} {ce}")
