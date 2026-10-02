###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register the Ideogram-4 block-checkpointing sidecar.

Applies to Ideogram-4 runs. It changes nothing unless ``activation_checkpointing``
is set, and then checkpoints whole blocks, every ``primus_ideogram4.ac_every``-th
one if that is set.
"""

from primus.backends.nemo_automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

_is_ideogram4 = transformer_is("Ideogram4Transformer2DModel")


def _ideogram4_with_valid_stride(ctx: PatchContext) -> bool:
    # The stride is validated here rather than in apply(): the patch runner logs
    # and skips whatever apply() raises, which would leave a run that asked for a
    # partial stride training with stock checkpointing. Conditions run outside
    # that isolation, so a malformed value stops the run instead.
    if not _is_ideogram4(ctx):
        return False
    from primus.backends.nemo_automodel.models.ideogram4 import parallelize

    parallelize.ac_stride()
    return True


@register_patch(
    "nemo_automodel.models.ideogram4.parallelize",
    backend="nemo_automodel",
    phase="before_train",
    description="Checkpoint whole Ideogram-4 blocks when activation_checkpointing is set",
    condition=_ideogram4_with_valid_stride,
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import parallelize

    parallelize.install()
