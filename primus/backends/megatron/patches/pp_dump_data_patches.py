###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
pp_dump_data patches: schedule wrapper and dump for Megatron / Primus Pipeline / ZeroBubble.

before_train:
  - Primus pipeline or ZeroBubble: wrap get_forward_backward_func with schedule_wrapper only
    (handlers already use fwd_bwd_wrapper for forward/backward, do not patch them).
  - Megatron native: wrap get_forward_backward_func with schedule_wrapper AND patch
    forward_step/backward_step via set_dump_pp_data_patch.
  - Patch schedule_wrapper to avoid double-wrap (legacy MegatronTrainer flow).

after_train:
  - Call dump_pp_data to write JSON for PP visualization.
"""

import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _make_wrapped_get_forward_backward_func(original_get_forward_backward_func):
    """Wrap get_forward_backward_func to add schedule_wrapper when dump_pp_data."""

    def wrapped_get_forward_backward_func():
        from megatron.training import get_args as get_megatron_args

        from primus.modules.trainer.megatron.utils import schedule_wrapper

        func = original_get_forward_backward_func()
        args = get_megatron_args()
        if getattr(args, "dump_pp_data", False):
            func = schedule_wrapper(func)
            func._pp_dump_schedule_wrapped = True  # noqa: B010
        return func

    return wrapped_get_forward_backward_func


def _make_guarded_schedule_wrapper(original_schedule_wrapper):
    """Avoid double-wrap when func is already wrapped by get_forward_backward_func patch."""

    def guarded_schedule_wrapper(func):
        if getattr(func, "_pp_dump_schedule_wrapped", False):
            return func
        return original_schedule_wrapper(func)

    return guarded_schedule_wrapper


def _make_guarded_set_dump_pp_data_patch(original_set_dump_pp_data_patch, utils_module):
    """Avoid applying forward/backward patch multiple times (e.g. legacy MegatronTrainer calls it from both patch and trainer)."""

    def guarded_set_dump_pp_data_patch():
        if getattr(utils_module, "_pp_dump_data_patch_applied", False):
            return
        original_set_dump_pp_data_patch()
        utils_module._pp_dump_data_patch_applied = True  # noqa: B010

    return guarded_set_dump_pp_data_patch


@register_patch(
    "megatron.pp.dump_pp_data.before_train",
    backend="megatron",
    phase="before_train",
    priority=100,  # Run after schedule/zero_bubble patches so we wrap their get_forward_backward_func
    description="Wrap schedule and patch forward/backward for pp visualization",
    condition=lambda ctx: (
        getattr(get_args(ctx), "dump_pp_data", False)
        and getattr(get_args(ctx), "pipeline_model_parallel_size", 1) > 1
    ),
)
def patch_pp_dump_data_before_train(ctx: PatchContext):
    """Apply schedule_wrapper to get_forward_backward_func; for Megatron native only, patch forward_step/backward_step."""
    import megatron.core.pipeline_parallel as pp_module
    import megatron.training.training as training_module

    import primus.modules.trainer.megatron.utils as utils

    args = get_args(ctx)
    is_primus_pipeline = getattr(args, "patch_primus_pipeline", False)
    is_zero_bubble = getattr(args, "patch_zero_bubble", False)

    # 1. Patch schedule_wrapper to avoid double-wrap (for legacy MegatronTrainer flow)
    utils.schedule_wrapper = _make_guarded_schedule_wrapper(utils.schedule_wrapper)

    # 2. Always wrap get_forward_backward_func with schedule_wrapper when dump_pp_data
    #    Patch both pipeline_parallel and training.training (training.py uses direct import)
    original_get_fwd_bwd = pp_module.get_forward_backward_func
    wrapped = _make_wrapped_get_forward_backward_func(original_get_fwd_bwd)
    pp_module.get_forward_backward_func = wrapped
    training_module.get_forward_backward_func = wrapped

    # 3. Megatron native only: patch forward_step and backward_step (Primus pipeline / ZeroBubble handlers already use fwd_bwd_wrapper)
    #    Replace with guarded version to avoid double-patch when legacy MegatronTrainer also calls it
    if not is_primus_pipeline and not is_zero_bubble:
        utils.set_dump_pp_data_patch = _make_guarded_set_dump_pp_data_patch(
            utils.set_dump_pp_data_patch, utils
        )
        utils.set_dump_pp_data_patch()

    log_rank_0(
        "[Patch:megatron.pp.dump_pp_data]   Applied schedule wrapper"
        + (
            " and forward/backward patch (Megatron native)"
            if not is_primus_pipeline and not is_zero_bubble
            else " (Primus pipeline/ZeroBubble)"
        )
    )


@register_patch(
    "megatron.pp.dump_pp_data.after_train",
    backend="megatron",
    phase="after_train",
    description="Dump pp schedule data to JSON for visualization",
    condition=lambda ctx: (
        getattr(get_args(ctx), "dump_pp_data", False)
        and getattr(get_args(ctx), "pipeline_model_parallel_size", 1) > 1
    ),
)
def patch_pp_dump_data_after_train(ctx: PatchContext):
    """Call dump_pp_data after training (for core runtime where trainer does not call it)."""
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        from megatron.training import get_args as get_megatron_args

        from primus.modules.trainer.megatron.utils import dump_pp_data

        args = get_megatron_args()
        pp_data_dir = os.environ.get("DUMP_PP_DIR", "output/pp_data")
        dump_pp_data(args, get_num_microbatches(), pp_data_dir)
        log_rank_0(f"[Patch:megatron.pp.dump_pp_data]   pp schedule data dumped to {pp_data_dir}")
    except Exception as e:
        log_rank_0(f"[Patch:megatron.pp.dump_pp_data]   WARNING: failed to dump pp data: {e}")
