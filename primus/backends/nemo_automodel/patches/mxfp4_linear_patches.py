###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Primus-Turbo MXFP4 linear swap.

Gated on ``PRIMUS_TURBO_FP4``, and conditioned on ``_common.is_active`` rather
than on the env var alone: only one precision can own AutoModel's swap symbol, so
the selector decides and a losing request is logged instead of silently dropped.
MXFP4 outranks FP8, so asking for both gives four bits.

Shares a priority with the FP8 swap because the two are mutually exclusive by
construction -- at most one can be active, so their relative order never comes
up. What matters is that both run before the per-model strategies, since the swap
has to happen before the transformer is built and sharded.
"""

from primus.core.patches import PatchContext, register_patch


def _active(ctx: PatchContext) -> bool:
    # _fp4_common, not mxfp4_linear: the latter needs torch to define Float4Linear,
    # and a patch condition should be answerable without importing a kernel stack.
    from primus.backends.nemo_automodel.quantization import _common, _fp4_common

    return _common.is_active(_fp4_common.BACKEND_NAME)


@register_patch(
    "nemo_automodel.quantization.mxfp4_linear",
    backend="nemo_automodel",
    phase="before_train",
    description="Swap nn.Linear for Primus-Turbo MXFP4 Float4Linear (PRIMUS_TURBO_FP4)",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.quantization import mxfp4_linear

    mxfp4_linear.install()
