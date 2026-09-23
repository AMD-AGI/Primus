###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the real-AC Ideogram-4 parallelization strategy.

Gated on ``PRIMUS_IDEOGRAM_REAL_AC``. Runs after the FSDP2 reshard repair, which
it depends on: the strategy forwards ``reshard_after_forward`` faithfully, but
that setting never reaches it unless the repair is in place. The strategy logs an
error rather than raising if it finds the repair missing, because by then a
training run is under way and failing it would be the worse outcome.

Registered at the same priority as the sibling model strategies, since only one
of them can match a given model class and they do not interact.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    # parallelize imports the shared AC helper and the environment helper, neither
    # of which pulls in torch or AutoModel, so this stays answerable during
    # discovery.
    from primus.backends.nemo_automodel.models.ideogram4 import parallelize

    if not parallelize.is_enabled():
        return False

    # Validate the block stride HERE, not in apply(). ac_stride() rejects a
    # malformed value, but the patch runner catches whatever apply() raises, logs
    # it and carries on -- so raising from apply() would leave the run training
    # with the stock strategy after the user asked for a partial stride, with one
    # ERROR line as the only evidence. Conditions are evaluated during filtering,
    # outside that try, so a typo stops the run here instead.
    parallelize.ac_stride()
    return True


@register_patch(
    "nemo_automodel.models.ideogram4.parallelize",
    backend="nemo_automodel",
    phase="before_train",
    description=(
        "Make fsdp.activation_checkpointing take effect for Ideogram-4 " "(PRIMUS_IDEOGRAM_REAL_AC)"
    ),
    condition=_enabled,
    # After the reshard repair at priority 10, which this depends on.
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import parallelize

    parallelize.install()
