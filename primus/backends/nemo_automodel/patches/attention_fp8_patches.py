###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Primus-Turbo FP8 attention override.

Gated on ``PRIMUS_TURBO_FP8_ATTN``. Runs before the non-deterministic override
because the two claim the same diffusers registry entries and only one can be
active: whichever installs first owns them, and the registry refuses the second
rather than layering it on top. Ordering it by priority here makes the winner a
stated decision rather than a consequence of module discovery order.

FP8 wins because it replaces the bf16 kernel outright, whereas the
non-deterministic override is a refinement of the kernel FP8 is replacing.
Composing the two -- passing ``deterministic=False`` through the FP8 kernel, which
its signature does accept -- is a reasonable follow-up, but it is a behaviour
change worth landing on its own evidence.

Runs before the per-model strategies, since the override has to be in place
before the transformer's first forward.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.attention import fp8

    return fp8.is_enabled()


@register_patch(
    "nemo_automodel.attention.fp8",
    backend="nemo_automodel",
    phase="before_train",
    description="Override diffusers attention with Primus-Turbo FP8 flash attention "
    "(PRIMUS_TURBO_FP8_ATTN)",
    condition=_enabled,
    # Before the non-deterministic override (31), which it excludes.
    priority=30,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.attention import fp8

    fp8.install()
