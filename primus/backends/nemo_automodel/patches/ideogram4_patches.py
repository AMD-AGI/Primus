###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Ideogram-4 adapter and the var-len attention processor.

Two patches at two priorities, because the ordering between them is not arbitrary.
Both sit below the per-model parallelization strategies, since each has to be in
place before the transformer is built: the adapter before the recipe resolves
``adapter_type``, and the processor before any attention module is constructed --
afterwards they all hold the stock processor and only an explicit walk reaches
them.

Neither condition imports the module it would install. Discovery runs before
anything has decided a run needs torch or diffusers, so a condition that reached
for the implementation would make an unrelated diffusion job fail at startup on an
import it never needed. The gates live in ``_varlen_common`` for exactly this, the
same split as ``_fp4_common`` and the linear swap.
"""

from primus.core.patches import PatchContext, register_patch


def _ideogram4_configured(ctx: PatchContext) -> bool:
    """Whether this run asked for the Ideogram-4 adapter.

    The wrapper is additive -- it only adds a route, and every other adapter type
    is untouched -- so registering it unconditionally would be harmless. It is
    gated anyway, because a FLUX or Wan run has no reason to have its adapter
    factory replaced, and a patch that only ever appears in the runs that use it is
    a much easier thing to reason about when something goes wrong.

    Defaults to REGISTERING when the config cannot be read. The two ways to be
    wrong are not symmetric: registering for a run that does not want it changes
    nothing, while skipping for a run that does want it fails at adapter
    resolution, well after startup, with an error about an unknown adapter type
    that says nothing about this patch.
    """
    config = (ctx.extra or {}).get("config") if ctx is not None else None
    if config is None:
        return True
    flow_matching = getattr(config, "flow_matching", None)
    if flow_matching is None:
        return True
    adapter_type = getattr(flow_matching, "adapter_type", None)
    if adapter_type is None:
        return True
    return adapter_type == "ideogram4"


@register_patch(
    "nemo_automodel.models.ideogram4.adapter",
    backend="nemo_automodel",
    phase="before_train",
    description="Register the 'ideogram4' flow-matching adapter with AutoModel",
    condition=_ideogram4_configured,
    priority=5,
)
def apply_adapter(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import adapter

    adapter.install()


def _varlen_requested(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.models.ideogram4 import _varlen_common

    return _varlen_common.is_varlen_attn_enabled() and _ideogram4_configured(ctx)


@register_patch(
    "nemo_automodel.models.ideogram4.varlen_attn",
    backend="nemo_automodel",
    phase="before_train",
    description=(
        "Route Ideogram-4 attention through var-len flash attention " "(PRIMUS_IDEOGRAM_VARLEN_ATTN)"
    ),
    condition=_varlen_requested,
    priority=6,
)
def apply_varlen_attn(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import attn_processor

    attn_processor.install()
