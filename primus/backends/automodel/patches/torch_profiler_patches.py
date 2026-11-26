###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the diffusion train-loop ``torch.profiler`` wrapper.

Gated on ``primus_profiler.enabled``. The gate is expressed as the patch's
``condition`` rather than as an early return inside the body, so a disabled
profiler is reported as "skipped (condition not met)" in the run log instead of
being invisible -- which matters when someone is wondering why they got no trace.
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

options.declare(
    "primus_profiler",
    "enabled",
    "output_dir",
    "tag",
    "wait",
    "warmup",
    "active",
    "with_stack",
    "with_modules",
    "record_shapes",
    "profile_memory",
)


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.profiling import torch_profiler

    return torch_profiler.is_enabled()


@register_patch(
    "automodel.profiling.torch_profiler",
    backend="automodel",
    phase="before_train",
    description="Wrap the diffusion train loop in torch.profiler (primus_profiler.enabled)",
    condition=_enabled,
    # Runs late: it wraps the train loop, so anything that replaces methods on the
    # recipe should already have done so.
    priority=90,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.profiling import torch_profiler

    torch_profiler.install(exp_root=getattr(ctx.extra.get("primus_config"), "exp_root_path", None))
