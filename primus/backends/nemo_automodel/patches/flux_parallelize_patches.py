###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register the FLUX block-checkpointing sidecar.

Ungated: it changes nothing unless ``fsdp.activation_checkpointing`` is set, and
then makes it cover every FLUX block as configured.
"""

from primus.backends.nemo_automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch


@register_patch(
    "nemo_automodel.models.flux.parallelize",
    backend="nemo_automodel",
    phase="before_train",
    description="Checkpoint both FLUX block lists when activation_checkpointing is set",
    condition=transformer_is("FluxTransformer2DModel"),
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.flux import parallelize

    parallelize.install()
