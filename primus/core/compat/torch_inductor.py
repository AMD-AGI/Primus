###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Workaround for PyTorch inductor TritonTemplate duplicate name assertion.

When transformer_engine is imported on ROCm, its use of ``torch.compile``
triggers the ``torch._inductor`` import chain:

    transformer_engine  →  torch.compile  →  torch._inductor.compile_fx
    →  distributed_autotune  →  select_algorithm  →  lowering
    →  kernel/flex_attention.py

``flex_attention.py`` registers ``TritonTemplate("flex_attention", ...)`` at
module level, but under certain import orderings the name is already present
in ``TritonTemplate.all_templates``, causing:

    AssertionError: duplicate template name

This module pre-imports ``select_algorithm`` with the assertion replaced by
an idempotent guard, so subsequent imports from transformer_engine see the
already-cached (and safe) module in ``sys.modules``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import warnings

log = logging.getLogger(__name__)

_APPLIED = False

_TARGET_MODULE = "torch._inductor.select_algorithm"
_ASSERT_NEEDLE = 'assert name not in self.all_templates, "duplicate template name"'
_ASSERT_REPLACEMENT = (
    "if name in self.all_templates:\n"
    "                    self.__dict__.update(self.all_templates[name].__dict__); return"
)


def apply_inductor_compat() -> bool:
    """
    Pre-import ``torch._inductor.select_algorithm`` with the duplicate-name
    assertion patched out.  Returns True if the workaround was applied (or was
    unnecessary), False on failure.

    Safe to call multiple times; only the first call has any effect.
    """
    global _APPLIED
    if _APPLIED:
        return True

    if _TARGET_MODULE in sys.modules:
        _APPLIED = True
        return True

    try:
        spec = importlib.util.find_spec(_TARGET_MODULE)
    except (ModuleNotFoundError, ValueError):
        _APPLIED = True
        return True

    if spec is None or spec.loader is None:
        _APPLIED = True
        return True

    try:
        source = spec.loader.get_source(_TARGET_MODULE)
    except Exception:
        _APPLIED = True
        return True

    if source is None or _ASSERT_NEEDLE not in source:
        _APPLIED = True
        return True

    patched_source = source.replace(_ASSERT_NEEDLE, _ASSERT_REPLACEMENT)

    module = importlib.util.module_from_spec(spec)
    module.__file__ = spec.origin
    module.__loader__ = spec.loader
    module.__package__ = spec.parent
    module.__spec__ = spec
    if spec.submodule_search_locations is not None:
        module.__path__ = list(spec.submodule_search_locations)

    sys.modules[_TARGET_MODULE] = module

    parent = sys.modules.get(spec.parent)
    if parent is not None:
        setattr(parent, "select_algorithm", module)

    try:
        code = compile(patched_source, spec.origin, "exec")
        exec(code, module.__dict__)  # noqa: S102
    except Exception as exc:
        sys.modules.pop(_TARGET_MODULE, None)
        if parent is not None:
            try:
                delattr(parent, "select_algorithm")
            except AttributeError:
                pass
        warnings.warn(
            f"[Primus] torch inductor compat: patched import failed ({exc}); "
            "falling back to unpatched import",
            stacklevel=2,
        )
        _APPLIED = True
        return False

    _APPLIED = True
    log.info("[Primus:compat] Applied torch._inductor TritonTemplate duplicate-name workaround")
    return True
