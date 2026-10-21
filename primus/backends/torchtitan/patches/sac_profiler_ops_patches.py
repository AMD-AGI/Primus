###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
torch 2.14 SAC requires each op to run the same number of times in forward and
recompute. Under torch.compile, FSDP2's ``record_function`` markers do not, which
fails backward with "invocation index N ... not found in storage". They are
profiler-only, so let SAC run them uncached.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "torchtitan.torch.sac_ignore_profiler_ops",
    backend="torchtitan",
    phase="setup",
    description="Let selective activation checkpointing ignore profiler record_function ops",
)
def patch_sac_ignore_profiler_ops(ctx: PatchContext) -> None:  # noqa: ARG001
    import torch
    import torch.utils.checkpoint as checkpoint

    ignored = getattr(checkpoint, "SAC_IGNORED_OPS", None)
    if ignored is None:
        return
    ops = [
        getattr(getattr(torch.ops.profiler, name, None), overload, None)
        for name, overload in (
            ("_record_function_enter_new", "default"),
            ("_record_function_exit", "_RecordFunction"),
        )
    ]
    ignored.update(op for op in ops if op is not None)
    log_rank_0(
        "[Patch:torchtitan.torch.sac_ignore_profiler_ops] Added profiler record_function ops to SAC_IGNORED_OPS"
    )
