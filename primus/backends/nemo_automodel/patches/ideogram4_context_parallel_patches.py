###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Ideogram-4 context-parallel plan.

Not gated on a setting of its own. The plan is a class attribute that upstream reads only when a config asks for a context-parallel
degree above one, so installing it on every Ideogram-4 run costs an attribute
assignment and changes nothing else. A flag here would only add a way to configure
CP and have it refused for a reason the error message would not mention.

It is gated on the run training the Ideogram-4 transformer, because installing it
means importing that transformer from diffusers -- worth avoiding on a run that has
nothing to do with this model.

Runs after the adapter and attention patches that have to be in place before the
model is built; CP itself is switched on later, when the model is parallelized.
"""

from primus.backends.nemo_automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch


@register_patch(
    "nemo_automodel.models.ideogram4.context_parallel",
    backend="nemo_automodel",
    phase="before_train",
    description="Attach a context-parallel (Ulysses) plan to Ideogram-4",
    condition=transformer_is("Ideogram4Transformer2DModel"),
    priority=7,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import context_parallel

    context_parallel.install()
