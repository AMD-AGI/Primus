###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Registry and discovery of :class:`OpModel` subclasses.

Discovery imports every module under ``op_models/generated`` and every
``*.py`` in the directories listed on ``PRIMUS_OP_MODEL_DIRS`` (comma or
``os.pathsep`` separated).  A module that fails to import is skipped and its
error kept in :func:`discovery_errors`: one broken generated op must not take
the projection down with it.  ``PRIMUS_OP_MODELS=0`` disables every op model,
for A/B against the built-in profilers.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type

from .base import SLOTS, OpModel

_REGISTRY: Dict[str, Type[OpModel]] = {}
_DISCOVERED = False
_ERRORS: Dict[str, str] = {}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def register_op_model(cls: Type[OpModel]) -> Type[OpModel]:
    """Class decorator: add *cls* to the registry (last registration wins)."""
    if not isinstance(cls, type) or not issubclass(cls, OpModel):
        raise TypeError(f"{cls!r} is not an OpModel subclass")
    if not _NAME_RE.match(cls.name or ""):
        raise ValueError(f"{cls.__name__}.name must be snake_case, got {cls.name!r}")
    if cls.slot not in SLOTS:
        raise ValueError(f"{cls.__name__}.slot must be one of {SLOTS}, got {cls.slot!r}")
    _REGISTRY[cls.name] = cls
    return cls


def unregister_op_model(name: str) -> None:
    _REGISTRY.pop(name, None)


def op_models_enabled() -> bool:
    return os.getenv("PRIMUS_OP_MODELS", "1").strip().lower() not in ("0", "false", "no", "off")


def load_op_model_file(path) -> List[Type[OpModel]]:
    """Import one op-model file by path; return the classes it registered."""
    path = Path(path).resolve()
    before = dict(_REGISTRY)
    mod_name = f"_primus_op_model_{path.stem}_{abs(hash(str(path))) & 0xFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import op model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return [cls for name, cls in _REGISTRY.items() if before.get(name) is not cls]


def _extra_dirs() -> List[Path]:
    raw = os.getenv("PRIMUS_OP_MODEL_DIRS", "")
    parts = [p.strip() for p in re.split(r"[,%s]" % re.escape(os.pathsep), raw) if p.strip()]
    return [Path(p) for p in parts if Path(p).is_dir()]


def discover_op_models(force: bool = False) -> Dict[str, Type[OpModel]]:
    global _DISCOVERED
    if _DISCOVERED and not force:
        return dict(_REGISTRY)
    _DISCOVERED = True
    from . import generated

    for info in pkgutil.iter_modules(generated.__path__):
        if info.name.startswith("_"):
            continue
        full = f"{generated.__name__}.{info.name}"
        try:
            if force and full in sys.modules:
                importlib.reload(sys.modules[full])
            else:
                importlib.import_module(full)
        except Exception as exc:  # keep the projection running
            _ERRORS[full] = f"{type(exc).__name__}: {exc}"
    for d in _extra_dirs():
        for path in sorted(d.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                load_op_model_file(path)
            except Exception as exc:
                _ERRORS[str(path)] = f"{type(exc).__name__}: {exc}"
    return dict(_REGISTRY)


def discovery_errors() -> Dict[str, str]:
    return dict(_ERRORS)


def registered_op_models() -> List[Type[OpModel]]:
    discover_op_models()
    return list(_REGISTRY.values())


def active_op_models(model_config, slots: Optional[Sequence[str]] = None) -> List[OpModel]:
    """Instances of every registered op present in *model_config*, by slot."""
    if model_config is None or not op_models_enabled():
        return []
    out = []
    for cls in registered_op_models():
        if slots is not None and cls.slot not in slots:
            continue
        try:
            if cls.matches(model_config):
                out.append(cls())
        except Exception as exc:
            _ERRORS[cls.name] = f"matches() raised {type(exc).__name__}: {exc}"
    return out
