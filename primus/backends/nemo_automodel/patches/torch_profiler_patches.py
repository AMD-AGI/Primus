###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the diffusion train-loop ``torch.profiler`` wrapper.

Gated on ``PRIMUS_DIFFUSION_PROFILE``. The gate is expressed as the patch's
``condition`` rather than as an early return inside the body, so a disabled
profiler is reported as "skipped (condition not met)" in the run log instead of
being invisible -- which matters when someone is wondering why they got no trace.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.profiling import torch_profiler

    return torch_profiler.is_enabled()


@register_patch(
    "nemo_automodel.profiling.torch_profiler",
    backend="nemo_automodel",
    phase="before_train",
    description="Wrap the diffusion train loop in torch.profiler (PRIMUS_DIFFUSION_PROFILE)",
    condition=_enabled,
    # Runs late: it wraps the train loop, so anything that replaces methods on the
    # recipe should already have done so.
    priority=90,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.profiling import torch_profiler

    torch_profiler.install()
