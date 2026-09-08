###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Derive the evaluation budget from a sample count instead of an iteration count.

``eval_iters`` only expresses "cover the validation set" at one particular
global batch size, so it silently becomes wrong when the batch size changes.
``eval_samples`` says what the evaluation is actually meant to measure.

This runs in the ``build_args`` phase, before ``pretrain()`` builds the Energon
validation dataloader, so both the provider and the evaluation loop observe the
corrected ``eval_iters``.
"""

from primus.backends.megatron.training.eval_budget import (
    get_eval_global_batch_size,
    assert_val_worker_divisibility,
    get_val_num_workers,
    read_energon_split_sample_count,
    resolve_eval_iters,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_kv_rank_0

# Keys Megatron's parser does not define. MegatronArgBuilder drops anything
# absent from that parser ("Non-Megatron parameters are silently ignored"), and
# the runtime merges the leftover Primus-only params into backend_args only
# *after* the build_args phase. So this patch cannot read them off args yet and
# must take them from the module config, or it would see every one as unset and
# quietly leave eval_iters alone.
PRIMUS_ONLY_EVAL_KEYS = (
    "eval_samples",
    "val_num_workers",
    "eval_global_batch_size",
    "eval_micro_batch_size",
)


def _hydrate_primus_only_keys(args, module_config):
    """Copy Primus-only eval keys from the module config onto args."""
    params = getattr(module_config, "params", None)
    if params is None:
        return
    for key in PRIMUS_ONLY_EVAL_KEYS:
        if not hasattr(args, key) and hasattr(params, key):
            setattr(args, key, getattr(params, key))


@register_patch(
    "megatron.args.eval_samples",
    backend="megatron",
    phase="build_args",
    description="Derive eval_iters from eval_samples and assert validation coverage is exact",
)
def patch_eval_samples(ctx: PatchContext):
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    _hydrate_primus_only_keys(args, ctx.extra.get("module_config"))

    # full_validation has never done anything on the Energon path: the
    # validation LimitDataset is always sized from eval_iters. Give it a
    # meaning by reading the split's true size out of the dataset index.
    if getattr(args, "full_validation", False) and getattr(args, "eval_samples", None) is None:
        dataset_samples = read_energon_split_sample_count(getattr(args, "data_path", None))
        if dataset_samples is None:
            raise ValueError(
                "full_validation is set but the validation split size could not be read "
                "from the dataset index (.nv-meta/.info.json). Set eval_samples explicitly."
            )
        args.eval_samples = dataset_samples
        args.eval_iters = 0
        log_kv_rank_0(
            "[Patch:megatron.args.eval_samples] -full_validation",
            f"eval_samples={dataset_samples} (read from dataset index)",
        )

    derived = resolve_eval_iters(args)
    if derived is not None:
        previous = getattr(args, "eval_iters", None)
        args.eval_iters = derived
        # Name the value being replaced: eval_iters is set in trainer_base.yaml
        # for every module, so this almost always overrides something, and a
        # silent override is how the budget drifted from the intent before.
        replaced = "" if previous in (None, derived) else f", was {previous}"
        eval_batch_size = get_eval_global_batch_size(args)
        batch = (
            f"global_batch_size={eval_batch_size}"
            if eval_batch_size == args.global_batch_size
            else f"eval_global_batch_size={eval_batch_size} "
            f"(training global_batch_size={args.global_batch_size})"
        )
        log_kv_rank_0(
            "[Patch:megatron.args.eval_samples] -eval_iters",
            f"{derived} (from eval_samples={args.eval_samples}, {batch}{replaced})",
        )
        return

    # No eval_samples given: leave eval_iters alone, but still check that the
    # budget it implies can actually be read. Without this, only configs that
    # opt into eval_samples get the coverage guarantee.
    eval_iters = getattr(args, "eval_iters", 0) or 0
    if eval_iters > 0:
        eval_samples = eval_iters * get_eval_global_batch_size(args)
        assert_val_worker_divisibility(args, eval_samples)
        log_kv_rank_0(
            "[Patch:megatron.args.eval_samples] -val_num_workers",
            f"{get_val_num_workers(args)} (coverage verified for {eval_samples} samples)",
        )
