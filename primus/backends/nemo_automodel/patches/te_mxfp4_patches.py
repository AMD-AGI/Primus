###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Transformer Engine native MXFP4 linear swap.

Gated on ``primus_te.mxfp4_linear``, and conditioned on ``_common.is_active``
rather than the setting alone: only one precision can own AutoModel's swap symbol. This
one has the highest precedence of the three, because naming both a precision and
an implementation is the most specific request and should not lose to a broader
one.

Same priority as the other two swaps, since at most one can be active and their
relative order therefore never arises. Like every before_train patch it runs
before the transformer is built, sharded and checkpoint-wrapped.
"""

from primus.core.patches import PatchContext, get_param, register_patch


def _active(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.quantization import _common, te_mxfp4_linear

    return _common.is_active(te_mxfp4_linear.BACKEND_NAME)


@register_patch(
    "nemo_automodel.quantization.te_mxfp4_linear",
    backend="nemo_automodel",
    phase="before_train",
    description="Swap nn.Linear for TE Linear under an MXFP4 autocast (primus_te.mxfp4_linear)",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.quantization import te_mxfp4_linear

    te_mxfp4_linear.install(get_param(ctx, "model"))
