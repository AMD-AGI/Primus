###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Ideogram-4 context-parallel plan.

Not gated on an environment flag of its own, unlike its siblings. The plan is a
class attribute that upstream reads only when a config asks for a context-parallel
degree above one, so installing it on every Ideogram-4 run costs an attribute
assignment and changes nothing else. A flag here would only add a way to configure
CP and have it refused for a reason the error message would not mention.

It is gated on the run being an Ideogram-4 one, reusing the adapter patch's
config gate, because installing it means importing the transformer from diffusers
-- worth avoiding on a run that has nothing to do with this model.

Runs before the parallelization strategies, which is where CP is actually switched
on, and after the adapter and attention patches that have to be in place before
the model is built.
"""

from primus.backends.nemo_automodel.patches.ideogram4_patches import (
    _ideogram4_configured,
)
from primus.core.patches import PatchContext, register_patch


@register_patch(
    "nemo_automodel.models.ideogram4.context_parallel",
    backend="nemo_automodel",
    phase="before_train",
    description="Attach a context-parallel (Ulysses) plan to Ideogram-4",
    condition=_ideogram4_configured,
    priority=7,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import context_parallel

    context_parallel.install()
