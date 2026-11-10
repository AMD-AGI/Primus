###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register ZeRO-1 optimizer sharding for Ideogram-4 DDP runs.

Opt-in, gated on ``primus_ideogram4.zero1``, so a default run and every FSDP run
are untouched. The optimizer patch is process-wide, so it is also gated on the
run training Ideogram-4: the setting in another model's config changes nothing.
"""

import ast
from typing import Any

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, get_param, register_patch

options.declare("primus_ideogram4", "zero1")

_is_ideogram4 = transformer_is("Ideogram4Transformer2DModel")

# The spellings AutoModel's config loader turns into constants before falling back
# to literal evaluation (its translate_value).
_AUTOMODEL_CONSTANTS = {
    "none": None,
    "None": None,
    "true": True,
    "True": True,
    "false": False,
    "False": False,
}


def as_automodel_reads_it(value: Any) -> Any:
    """``value`` as AutoModel's config loader will hand it to the recipe."""
    if not isinstance(value, str):
        return value
    if value in _AUTOMODEL_CONSTANTS:
        return _AUTOMODEL_CONSTANTS[value]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def checkpoint_enabled(ctx: PatchContext) -> bool:
    """Whether AutoModel will save checkpoints, which defaults to yes.

    Read the way AutoModel reads it, not with Primus's own true/false spellings:
    "false" and "0" turn saving off, but "off" or "no" stay non-empty strings,
    which AutoModel treats as on -- so they must count as on here too, or ZeRO-1
    would be let through to fail at the first save.
    """
    return bool(as_automodel_reads_it(get_param(ctx, "checkpoint.enabled", True)))


def _enabled(ctx: PatchContext) -> bool:
    if not _is_ideogram4(ctx):
        return False
    # zero1 imports only the settings module, so this stays answerable during
    # discovery without torch or AutoModel.
    from primus.backends.automodel.models.ideogram4 import zero1

    return zero1.is_zero1_enabled()


@register_patch(
    "automodel.models.ideogram4.zero1",
    backend="automodel",
    phase="before_train",
    description="ZeRO-1 optimizer sharding on the DDP path (primus_ideogram4.zero1)",
    condition=_enabled,
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.ideogram4 import zero1

    zero1.install(checkpoint_enabled=checkpoint_enabled(ctx))
