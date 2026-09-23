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

Runs before the per-model parallelization strategies, because the swap has to be
in place before the transformer is built and sharded.
"""

from primus.core.patches import PatchContext, register_patch


def _active(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.quantization import _common, fp8_linear

    return _common.is_active(fp8_linear.BACKEND_NAME)


@register_patch(
    "nemo_automodel.quantization.fp8_linear",
    backend="nemo_automodel",
    phase="before_train",
    description="Swap nn.Linear for Primus-Turbo Float8Linear (PRIMUS_TURBO_FP8)",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.quantization import fp8_linear

    fp8_linear.install()
