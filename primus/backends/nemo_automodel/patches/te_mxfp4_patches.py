###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Transformer Engine native MXFP4 linear swap.

Gated on ``PRIMUS_TE_MXFP4``, and conditioned on ``_common.is_active`` rather
than the env var alone: only one precision can own AutoModel's swap symbol. This
one has the highest precedence of the three, because naming both a precision and
an implementation is the most specific request and should not lose to a broader
one.

Same priority as the other two swaps, since at most one can be active and their
relative order therefore never arises. What matters is that it runs before the
per-model strategies -- the swap and its autocast wrapping have to be in place
before the transformer is built and sharded, and the autocast has to be wrapped
before activation checkpointing wraps the blocks.
"""

from primus.core.patches import PatchContext, register_patch


def _active(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.quantization import _common, te_mxfp4_linear

    return _common.is_active(te_mxfp4_linear.BACKEND_NAME)


@register_patch(
    "nemo_automodel.quantization.te_mxfp4_linear",
    backend="nemo_automodel",
    phase="before_train",
    description="Swap nn.Linear for TE Linear under an MXFP4 autocast (PRIMUS_TE_MXFP4)",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.quantization import te_mxfp4_linear

    te_mxfp4_linear.install()
