###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Validation Data Sampling Patches

Patches Megatron's validation data loading to use a fixed, reproducible
subset of validation samples (eval_iters * global_batch_size) starting
from offset 0, matching MLPerf evaluation protocol.

Without this patch, Megatron:
  1. Over-allocates validation samples proportional to total training steps
  2. Advances through validation data using consumed_valid_samples offset,
     making eval loss non-reproducible across runs
"""

import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_mlperf_enabled(ctx):
    return os.getenv("PRIMUS_MLPERF", "0") == "1" and getattr(get_args(ctx), "eval_iters", 0) > 0


@register_patch(
    "megatron.training.validation_num_samples",
    backend="megatron",
    phase="before_train",
    description=(
        "Fix validation sample count to eval_iters * global_batch_size "
        "instead of scaling with train_iters"
    ),
    condition=_is_mlperf_enabled,
)
def patch_validation_num_samples(ctx: PatchContext):
    """
    Patch get_train_valid_test_num_samples to allocate only
    eval_iters * global_batch_size validation samples instead of
    (train_iters // eval_interval + 1) * eval_iters * global_batch_size.
    """
    import megatron.training.training as training_module

    def patched_get_train_valid_test_num_samples():
        from megatron.training import get_args

        args = get_args()

        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size

        if args.full_validation:
            eval_samples = None
        else:
            eval_samples = args.eval_iters * args.global_batch_size

        test_samples = args.eval_iters * args.global_batch_size

        if hasattr(args, "phase_transition_iterations") and args.phase_transition_iterations:
            phase_transition_samples = (
                [0]
                + [t * args.global_batch_size for t in args.phase_transition_iterations]
                + [args.train_samples]
            )
            current_sample = args.iteration * args.global_batch_size
            for i in range(len(phase_transition_samples) - 1):
                if phase_transition_samples[i] <= current_sample < phase_transition_samples[i + 1]:
                    train_samples = phase_transition_samples[i + 1] - phase_transition_samples[i]
                    break

        return (train_samples, eval_samples, test_samples)

    training_module.get_train_valid_test_num_samples = patched_get_train_valid_test_num_samples
    log_rank_0(
        "[Patch:megatron.training.validation_num_samples] "
        "Patched get_train_valid_test_num_samples: eval_samples = eval_iters * gbs"
    )


@register_patch(
    "megatron.training.validation_data_loader",
    backend="megatron",
    phase="before_train",
    description=(
        "Always build validation dataloader with consumed_samples=0 and "
        "cap total_samples at eval_iters * gbs for reproducible evaluation"
    ),
    condition=_is_mlperf_enabled,
)
def patch_validation_data_loader(ctx: PatchContext):
    """
    Patch build_pretraining_data_loader to detect validation datasets and
    force consumed_samples=0 with total_samples capped at eval_iters * gbs.

    Also patches build_train_valid_test_data_loaders to always pass
    consumed_samples=0 for validation dataloaders (instead of
    args.consumed_valid_samples).
    """
    import torch.utils.data

    from megatron.core import mpu
    from megatron.core.datasets.utils import Split

    try:
        from megatron.training.datasets import data_samplers as samplers_module
        from megatron.training.datasets.data_samplers import MegatronPretrainingSampler
    except ImportError:
        from megatron.legacy.data import data_samplers as samplers_module
        from megatron.legacy.data.data_samplers import MegatronPretrainingSampler

    orig_build_loader = samplers_module.build_pretraining_data_loader

    def patched_build_pretraining_data_loader(dataset, consumed_samples, name=""):
        """Replacement that forces validation to always start from sample 0
        with a fixed sample count."""
        if dataset is None:
            return None

        from megatron.training import get_args

        args = get_args()

        if hasattr(dataset, "split"):
            split = dataset.split
        elif hasattr(dataset, "index_split"):
            split = dataset.index_split
        else:
            split = None

        is_validation = (split == Split.valid) or (name == "validation")

        if is_validation:
            eval_samples = args.eval_iters * args.global_batch_size
            total_samples = min(len(dataset), eval_samples)
            batch_sampler = MegatronPretrainingSampler(
                total_samples=total_samples,
                consumed_samples=0,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
            )

        return orig_build_loader(dataset, consumed_samples)

    # Replace in both the data_samplers module and training module so all
    # call sites (including build_train_valid_test_data_loaders) use our version
    samplers_module.build_pretraining_data_loader = patched_build_pretraining_data_loader

    import megatron.training.training as training_module

    training_module.build_pretraining_data_loader = patched_build_pretraining_data_loader

    log_rank_0(
        "[Patch:megatron.training.validation_data_loader] "
        "Patched build_pretraining_data_loader: validation always uses "
        "consumed_samples=0, total_samples capped at eval_iters * gbs"
    )
