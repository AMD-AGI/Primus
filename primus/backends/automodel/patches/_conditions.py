###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared patch conditions.

Not a ``*_patches`` module, so auto-discovery does not import it for its own
sake; patch modules import what they need from it.
"""

import ast
from typing import Any, Callable

from primus.core.patches import PatchContext, get_param

TRANSFORMER_CLS_KEY = "model.pipeline_spec.transformer_cls"

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


def transformer_is(*class_names: str) -> Callable[[PatchContext], bool]:
    """Condition: the run trains one of ``class_names``.

    Declines only when the config names a different transformer class. A config
    that names none (a fine-tune that loads the model with ``from_pretrained``,
    say) still gets the patch: every model patch is a no-op for a model it does
    not recognise, while skipping one the run needed changes training silently.
    """

    def condition(ctx: PatchContext) -> bool:
        configured = get_param(ctx, TRANSFORMER_CLS_KEY) if ctx is not None else None
        return configured is None or str(configured) in class_names

    condition.__name__ = f"transformer_is({', '.join(class_names)})"
    return condition


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
    which AutoModel treats as on -- so they must count as on here too, or a
    patch that refuses or warns about saving would be skipped exactly when
    saving happens.
    """
    return bool(as_automodel_reads_it(get_param(ctx, "checkpoint.enabled", True)))
