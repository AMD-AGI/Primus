###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the non-deterministic bf16 attention override.

Gated on ``PRIMUS_ATTN_NONDETERMINISTIC``. Runs after the FP8 attention override,
which excludes it -- see ``attention_fp8_patches`` for why FP8 wins. With both env
gates set this one is refused and says so at WARNING, rather than wrapping the
FP8 wrapper and changing numerics silently.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.attention import nondeterministic

    return nondeterministic.is_enabled()


@register_patch(
    "nemo_automodel.attention.nondeterministic",
    backend="nemo_automodel",
    phase="before_train",
    description="Run the flash-attention backward non-deterministically " "(PRIMUS_ATTN_NONDETERMINISTIC)",
    condition=_enabled,
    # After the FP8 attention override (30), which takes precedence.
    priority=31,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.attention import nondeterministic

    nondeterministic.install()
