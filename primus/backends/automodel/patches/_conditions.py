###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared patch conditions.

Not a ``*_patches`` module, so auto-discovery does not import it for its own
sake; patch modules import what they need from it.
"""

from typing import Callable

from primus.core.patches import PatchContext, get_param

TRANSFORMER_CLS_KEY = "model.pipeline_spec.transformer_cls"


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
