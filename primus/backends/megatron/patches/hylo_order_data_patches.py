###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
HyLo-Order Dataset Patch
========================

With ``hylo_order_data_dir`` set, the hybrid-model dataset provider (``pretrain_mamba``'s, or
``pretrain_hybrid``'s from Megatron-core 0.19) serves ``HyLoOrderDataset``: the batches a reference
HyLo (TRL) run recorded, in its order, in place of Megatron's ``GPTDataset``. Megatron's random
sampler would reorder them, so the patch also sets ``dataloader_type`` to ``single``. Replay builds
no validation or test set.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.training.hylo_utils import hybrid_pretrain_module, unset
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.mamba.hylo_order_dataset"


def check_replay_args(args) -> None:
    """Replay serves one recorded micro-batch per data-parallel rank per optimizer step."""
    model_type = getattr(args, "model_type", "gpt")
    if model_type != "mamba":
        raise ValueError(
            f"hylo_order_data_dir replays through the hybrid (mamba) model only, not {model_type!r}"
        )
    if int(getattr(args, "context_parallel_size", 1) or 1) != 1:
        raise ValueError("hylo_order_data_dir does not support context_parallel_size > 1")
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    data_parallel = int(args.world_size) // (tp * pp)
    if args.global_batch_size != args.micro_batch_size * data_parallel:
        raise ValueError(
            "hylo_order_data_dir replays one recorded micro-batch per rank per step; "
            f"global_batch_size={args.global_batch_size} must be micro_batch_size x data-parallel size "
            f"({args.micro_batch_size} x {data_parallel})"
        )


def _install_hylo_order_data_patch() -> None:
    module = hybrid_pretrain_module()
    if is_patched(module, _PATCH_KEY):
        return
    original_provider = module.train_valid_test_datasets_provider

    def patched_provider(train_val_test_num_samples, vp_stage=None):
        from megatron.core import parallel_state
        from megatron.training import get_args as _get_args
        from megatron.training import print_rank_0

        from primus.backends.megatron.training.hylo_order_dataset import (
            HyLoOrderDataset,
        )

        args = _get_args()
        if unset(getattr(args, "hylo_order_data_dir", None)):
            return original_provider(train_val_test_num_samples, vp_stage=vp_stage)
        train_ds = HyLoOrderDataset(
            args.hylo_order_data_dir,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        needed = args.train_iters * args.global_batch_size
        if len(train_ds) < needed:
            raise ValueError(
                f"{args.hylo_order_data_dir} has {train_ds.steps} recorded steps, "
                f"fewer than train_iters={args.train_iters}"
            )
        print_rank_0(f"> HyLo-order dataset: {train_ds.steps} steps from {args.hylo_order_data_dir}")
        return train_ds, None, None

    module.train_valid_test_datasets_provider = patched_provider
    mark_patched(module, _PATCH_KEY)


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Serve a reference HyLo run's recorded batches (HyLoOrderDataset) in its order.",
    priority=50,
    condition=lambda ctx: not unset(getattr(get_args(ctx), "hylo_order_data_dir", None)),
)
def patch_hylo_order_data(ctx: PatchContext) -> None:
    args = get_args(ctx)
    check_replay_args(args)
    if args.dataloader_type != "single":
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] dataloader_type {args.dataloader_type} -> single, to keep the recorded order"
        )
        args.dataloader_type = "single"
    _install_hylo_order_data_patch()
    log_rank_0(f"[Patch:{_PATCH_KEY}] training reads {args.hylo_order_data_dir}")
