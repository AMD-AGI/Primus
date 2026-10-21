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

Only a module's own namespace is searched, never its ``__getattr__``, so a
wrapper that forwards attribute access to a module outside ``sys.modules``
keeps whatever that module holds.
"""

import sys
from types import ModuleType
from typing import Any, List, Mapping

__all__ = ["rebind_everywhere"]

_module_dict = ModuleType.__dict__["__dict__"].__get__


def _namespace(obj: Any) -> Mapping[str, Any]:
    """``obj``'s own attribute dict, read without running its attribute hooks.

    getattr() runs an entry's ``__getattr__`` on a miss, where apex's JIT op proxies
    (amp_C, syncbn, ...) build their extension and transformers' lazy modules print
    an alias warning, and any attribute access executes an importlib LazyLoader
    module. object.__getattribute__ skips both but still calls a ``__dict__``
    property defined on the class, and apipkg's ApiModule has one that imports
    every lazy export. ModuleType's own ``__dict__`` descriptor skips that too;
    entries that are not modules (apex's proxies are plain objects) fall back to
    object.__getattribute__.
    """
    if isinstance(obj, ModuleType):
        return _module_dict(obj)
    return object.__getattribute__(obj, "__dict__")


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
    # `from x import name` binds the name in the importer's own namespace.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is module:
            continue
        try:
            if _namespace(mod).get(name) is old_value:
                setattr(mod, name, new_value)
                rebound.append(mod_name)
        except Exception:  # noqa: BLE001 - no __dict__ (__slots__), or setattr refused
            continue

    return rebound
