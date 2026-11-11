###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the FLUX single-stream blocks' QKV and MLP up-projections as one linear1.

Gated on ``primus_perf.flux_fold_linear1`` for FLUX runs; see ``models/flux/linear1_fold.py``. Runs after the
fused GELU and q/k norm (priority 70) and the FlyDSL adaLN (75), whose ops it uses, and before
the un-aliased block outputs (80) and the image-only last block (85), which wrap its forward.
Folds the weights when AutoModel fuses the QKV projections, before FSDP2 wraps the model.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "flux_fold_linear1")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import linear1_fold

    return _is_flux(ctx) and linear1_fold.is_enabled()


@register_patch(
    "automodel.models.flux.fold_linear1",
    backend="automodel",
    phase="before_train",
    description="FLUX single-stream to_qkv + proj_mlp as one linear1 GEMM (primus_perf.flux_fold_linear1)",
    condition=_enabled,
    priority=76,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import linear1_fold

    linear1_fold.install()
