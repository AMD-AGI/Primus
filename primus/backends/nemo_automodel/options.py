###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-side settings for this backend, read from the module config.

Settings that AutoModel does not know about live in top-level sections of the
module params whose names start with ``primus_`` -- ``primus_profiler:``,
``primus_turbo:`` and so on, the same shape as the TorchTitan backend's
``primus_turbo:`` section. ``strip_primus_keys`` removes them before the config
reaches AutoModel, and the trainer hands the params to ``load`` before any patch
runs.

Patch conditions could read these through ``get_param(ctx, ...)``, but several
settings are read later, by code AutoModel calls: the linear-swap selector, the
attention processor, the parallelization plan. Those have no patch context, so
the values are kept here, once per process, and every reader goes through the
same accessors with the default stated at the read site.

Keep this module dependency-free -- no torch, no AutoModel -- so that patch
conditions can be evaluated, and unit tests can run, without importing either.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Optional

from primus.backends.nemo_automodel.argument_builder import PRIMUS_SECTION_PREFIX

# Accepted spellings when a value arrives as a string, e.g. from a CLI override.
# Generous on purpose: rejecting "True" would silently disable a feature.
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})

_sections: dict = {}


def _as_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _as_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, Mapping):
        return {k: _as_dict(v) for k, v in obj.items()}
    return obj


def load(params: Any) -> None:
    """Keep the ``primus_*`` sections of ``params`` (a namespace or a mapping).

    Replaces whatever was loaded before, so a second job in the same process
    does not inherit the first one's settings.
    """
    global _sections
    params = _as_dict(params) or {}
    _sections = {k: v for k, v in params.items() if k.startswith(PRIMUS_SECTION_PREFIX)}
    for name, section in _sections.items():
        if section is not None and not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping of settings, got {section!r}")


def get(path: str, default: Any = None) -> Any:
    """The value at ``section.key`` (dotted), or ``default`` when it is unset."""
    current: Any = _sections
    for part in path.split("."):
        if not isinstance(current, dict) or current.get(part) is None:
            return default
        current = current[part]
    return current


def flag(path: str, default: bool = False) -> bool:
    """A boolean setting. Raises on a value that is neither true nor false."""
    value = get(path)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{path} must be true or false, got {value!r}")


def integer(path: str, default: Optional[int]) -> Optional[int]:
    """An integer setting. Raises rather than falling back on a malformed value."""
    value = get(path)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer, got {value!r}")
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{path} must be an integer, got {value!r}") from exc


def string(path: str, default: str) -> str:
    """A string setting, with empty treated as unset."""
    value = get(path)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()
