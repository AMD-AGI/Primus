###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Rebind a monkeypatched symbol everywhere it was already imported by name.

Setting ``module.attr = new`` only reaches callers that look the symbol up
through the module (``arguments.core_transformer_config_from_args(...)``).
Megatron's entrypoints mostly do the opposite -- ``gpt_builders.py`` has
``from megatron.training.arguments import core_transformer_config_from_args``
at import time -- so the name is already copied into *their* namespace and the
module-attribute assignment never reaches them.

Naming one such importer explicitly (the way the LFM2 patch force-imports
``gpt_builders``) fixes the one caller you thought of and forces an import the
process may not otherwise need. Scanning ``sys.modules`` instead covers every
module that holds a reference *right now*, and needs no import: a module that
imports the symbol *after* the patch reads the already-patched module
attribute and is correct for free.
"""

import sys
from typing import Any, List

__all__ = ["rebind_everywhere"]


def rebind_everywhere(module: Any, name: str, new_value: Any) -> List[str]:
    """Point ``module.<name>`` and every already-imported alias at ``new_value``.

    Args:
        module: the module that defines the symbol.
        name: attribute name to rebind.
        new_value: replacement callable/object.

    Returns:
        The names of every module rebound, ``module`` first, in the order they
        were found -- suitable for a log line.
    """
    old_value = getattr(module, name)
    setattr(module, name, new_value)
    rebound = [getattr(module, "__name__", repr(module))]

    if old_value is new_value:
        return rebound

    # list(...) because importing inside the loop would mutate sys.modules.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is module:
            continue
        try:
            if getattr(mod, name, None) is old_value:
                setattr(mod, name, new_value)
                rebound.append(mod_name)
        except Exception:  # noqa: BLE001 - lazy/proxy modules can raise on getattr
            continue

    return rebound
