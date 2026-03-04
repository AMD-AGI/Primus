###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Torch Profiler Patches

Patches torch.profiler.profile to apply Primus-specific options when called
from Megatron's training.train(). Logic mirrors trainer.py L1277-1298.
"""

import inspect

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _is_called_from_training_train() -> bool:
    """Check if the current call stack originates from megatron.training.training.train."""
    for frame_info in inspect.stack():
        filename = frame_info.filename or ""
        function = frame_info.function or ""
        # Require both "megatron" and "training" to avoid false positives from other projects
        if "megatron" in filename and "training" in filename and function == "train":
            return True
    return False


def _create_primus_prof(args, exp_name: str, original_profile):
    """
    Create torch profiler with Primus options.

    Logic from primus/modules/trainer/megatron/trainer.py L1277-1298.
    """
    import torch

    activities = [torch.profiler.ProfilerActivity.CUDA]
    if not getattr(args, "disable_profiler_activity_cpu", False):
        activities.append(torch.profiler.ProfilerActivity.CPU)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    worker_name = f"primus-megatron-exp[{exp_name}]-rank[{rank}]"

    return original_profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=max(getattr(args, "profile_step_start", 10) - 1, 0),
            warmup=1 if getattr(args, "profile_step_start", 10) > 0 else 0,
            active=getattr(args, "profile_step_end", 12) - getattr(args, "profile_step_start", 10),
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=worker_name,
            use_gzip=getattr(args, "torch_profiler_use_gzip", False),
        ),
        record_shapes=getattr(args, "torch_profiler_record_shapes", True),
        with_stack=getattr(args, "torch_profiler_with_stack", True),
    )


@register_patch(
    "megatron.torch_profiler",
    backend="megatron",
    phase="before_train",
    description="Patch torch.profiler.profile with Primus profiler options (trainer.py L1277-1298)",
)
def patch_torch_profiler(ctx: PatchContext) -> None:
    """
    Wrap torch.profiler.profile to intercept calls from megatron.training.training.train
    and create the profiler with Primus options from trainer.py L1277-1298.
    """
    try:
        import torch
    except ImportError as e:
        log_rank_0(f"[Patch:megatron.torch_profiler] Skip patch (PyTorch not available): {e}")
        return

    if getattr(torch.profiler.profile, "_primus_torch_profiler_patched", False):
        return

    exp_name = "default"
    primus_config = ctx.extra.get("primus_config")
    if primus_config and getattr(primus_config, "exp_meta_info", None):
        exp_meta = primus_config.exp_meta_info
        if isinstance(exp_meta, dict):
            exp_name = exp_meta.get("exp_name", "default")

    original_profile = torch.profiler.profile

    def _patched_profile(*args, **kwargs):
        if _is_called_from_training_train():
            try:
                from megatron.training.global_vars import get_args

                megatron_args = get_args()
                return _create_primus_prof(megatron_args, exp_name, original_profile)
            except Exception as e:
                log_rank_0(f"[Patch:megatron.torch_profiler] Fallback to original: {e}")
                return original_profile(*args, **kwargs)
        return original_profile(*args, **kwargs)

    _patched_profile._primus_torch_profiler_patched = True  # type: ignore[attr-defined]
    torch.profiler.profile = _patched_profile
    log_rank_0("[Patch:megatron.torch_profiler] Patched torch.profiler.profile with Primus options.")
