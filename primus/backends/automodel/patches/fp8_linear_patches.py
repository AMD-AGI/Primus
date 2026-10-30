###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Primus-Turbo FP8 linear swap.

The condition is ``_common.is_active(...)``, not merely "was FP8 requested".
Exactly one precision can own AutoModel's swap symbol, so the condition has to
account for a narrower format having been requested at the same time; the
selector decides, and a losing request is logged rather than silently dropped.
So are traceable FP8 settings left on when the swap is not applied.

Like every before_train patch it runs before the transformer is built and
sharded, which is when the swap has to be in place.
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, get_param, register_patch

options.declare(
    "primus_turbo",
    "fp8_linear",
    "fp8_linear_granularity",
    "fp8_linear_format",
    "fp8_linear_block_size",
    "fp8_linear_traceable",
    "fp8_linear_delayed_scaling",
    "fp8_linear_amax_warmup_steps",
    "fp8_linear_amax_refresh_interval",
    "fp8_linear_inductor_cast",
    "fp8_linear_shared_input_scales",
    "fp8_linear_gemm_backend",
)


def _active(ctx: PatchContext) -> bool:
    from primus.backends.automodel.quantization import _common, fp8_linear

    if _common.is_active(fp8_linear.BACKEND_NAME):
        return True
    fp8_linear.warn_if_traceable_settings_unused()
    return False


@register_patch(
    "automodel.quantization.fp8_linear",
    backend="automodel",
    phase="before_train",
    description="Swap nn.Linear for Primus-Turbo Float8Linear (primus_turbo.fp8_linear)",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.quantization import fp8_linear

    fp8_linear.install(get_param(ctx, "model"))
