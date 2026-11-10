###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register the MiniMax-H3 adapter, its transformer-only training pipeline, its sidecar, and
the FP32-accumulating AdaLN modulation gather.

All three have to be in place before the recipe builds the model: the pipeline class
before ``DiffusionPipeline.from_pretrained`` resolves ``model_index.json``, the
adapter before the recipe resolves ``adapter_type``, and the sidecar before the model
is parallelized. Neither condition imports the
module it would install; discovery runs before anything has decided a run needs
torch or diffusers, so a condition that reached for the implementation would make
an unrelated diffusion job fail at startup on an import it never needed.

The wrappers are additive: registering them for a FLUX or Wan run would change
nothing, but they are gated anyway, so a patch only ever appears in the runs that
use it and is easier to reason about when something goes wrong.
"""

from primus.core.patches import PatchContext, get_param, register_patch

_ADAPTER_TYPE = "minimax_h3"


def _minimax_h3_configured(ctx: PatchContext) -> bool:
    """Whether this run asked for the MiniMax-H3 adapter.

    Defaults to REGISTERING when the config cannot be read. The two ways to be wrong
    are not symmetric: registering for a run that does not want it changes nothing,
    while skipping for a run that does want it fails at adapter or model resolution,
    well after startup, with an error that says nothing about this patch.
    """
    adapter_type = get_param(ctx, "flow_matching.adapter_type") if ctx is not None else None
    if adapter_type is None:
        return True
    return adapter_type == _ADAPTER_TYPE


@register_patch(
    "automodel.models.minimax_h3.pipeline",
    backend="automodel",
    phase="before_train",
    description="Register the transformer-only MiniMax-H3 training pipeline with diffusers",
    condition=_minimax_h3_configured,
    priority=4,
)
def apply_pipeline(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.minimax_h3 import pipeline

    pipeline.install()


@register_patch(
    "automodel.models.minimax_h3.adapter",
    backend="automodel",
    phase="before_train",
    description="Register the 'minimax_h3' flow-matching adapter with AutoModel",
    condition=_minimax_h3_configured,
    priority=5,
)
def apply_adapter(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.minimax_h3 import adapter

    adapter.install()


@register_patch(
    "automodel.models.minimax_h3.parallelize",
    backend="automodel",
    phase="before_train",
    description="Compute MiniMax-H3's keep-in-FP32 modules in FP32 under FSDP2",
    condition=_minimax_h3_configured,
    priority=50,
)
def apply_parallelize(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.minimax_h3 import parallelize

    parallelize.install()


@register_patch(
    "automodel.models.minimax_h3.modulation",
    backend="automodel",
    phase="before_train",
    description="Accumulate MiniMax-H3's AdaLN modulation gradients in FP32",
    condition=_minimax_h3_configured,
    priority=6,
)
def apply_modulation(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.minimax_h3 import modulation

    modulation.install()
