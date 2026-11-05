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

Only the namespaces of real module objects are searched, without running any of
their attribute hooks. So a wrapper that forwards attribute access to a module
outside ``sys.modules``, or a non-module object registered there, keeps whatever
it holds.
"""

import sys
from types import ModuleType
from typing import Any, List

from primus.core.utils.module_utils import warning_rank_0

__all__ = ["rebind_everywhere"]

# A module's own namespace, read through ModuleType's C-level ``__dict__`` slot.
# Every other way to reach it can run code owned by the entry: getattr() calls
# ``__getattr__`` on a miss, where apex's JIT op proxies (amp_C, syncbn, ...) build
# their extension and transformers' lazy modules print an alias warning; any
# attribute access executes an importlib LazyLoader module; and a subclass's
# ``__dict__`` property (apipkg's ApiModule) imports every lazy export.
_module_dict = ModuleType.__dict__["__dict__"].__get__


def _is_module(obj: Any) -> bool:
    """True for real module objects. type() rather than isinstance(), which falls
    back to reading ``obj.__class__`` through the object's own hooks (cffi's Lib
    reports ModuleType there)."""
    return issubclass(type(obj), ModuleType)


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

    # None (an optional import's placeholder) identifies nothing: it would match
    # every module that lacks the name.
    if old_value is None or old_value is new_value:
        return rebound

    # list(...) because importing inside the loop would mutate sys.modules.
    # `from x import name` binds the name in the importer's module namespace. Other
    # entries are objects standing in for a module (apex's op proxies, cffi's Lib);
    # they are left untouched.
    for mod_name, mod in list(sys.modules.items()):
        if mod is module or not _is_module(mod) or _module_dict(mod).get(name) is not old_value:
            continue
        try:
            setattr(mod, name, new_value)
        except Exception as exc:  # noqa: BLE001 - e.g. a module whose __setattr__ refuses the name
            warning_rank_0(f"[rebind] {mod_name}.{name} still holds the original: setattr failed: {exc!r}")
            continue
        rebound.append(mod_name)

    return rebound
