###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Declarative backend environment registry.

This module is the *mechanism* behind ``BackendAdapter.env_defaults()`` /
``BackendAdapter.apply_env_defaults()``. Each backend adapter declares the
performance/architecture environment it needs as a list of :class:`EnvVar`
entries; the base adapter applies them (once, before the backend imports its
GPU libraries) via :func:`apply_env_defaults`.

Design goals:
  - **Single source of truth.** A backend's env contract lives in one Python
    list, not scattered across shell launchers and prepare hooks.
  - **Layered precedence** (highest wins)::

        per-config ``env:``  >  outer/shell env  >  these backend defaults  >  image-baked

    Achieved with ``os.environ.setdefault`` for ordinary vars so we NEVER clobber
    something the user/shell/YAML already set.
  - **Architecture awareness.** An entry may be gated to a specific GPU arch
    (e.g. ``gfx950`` only, ``gfx942`` only). Non-matching entries are skipped.
  - **XLA_FLAGS merge.** ``XLA_FLAGS`` is special: Docker images bake it (often
    with a value we must override, e.g. ``--xla_gpu_autotune_level=0``), so plain
    ``setdefault`` would be a no-op. Entries with ``mode="xla_merge"`` are merged
    into any existing ``XLA_FLAGS`` at the individual ``--flag`` granularity, with
    the managed knobs winning while unrelated baked flags are preserved.

Backends with no special env (Megatron, TorchTitan, ...) simply return ``[]``
from ``env_defaults()`` and this module is a no-op for them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

# Supported arch gates. "all" applies everywhere; the others are matched against
# the detected GPU architecture string from rocminfo.
ARCH_ALL = "all"
ARCH_GFX950 = "gfx950"  # MI350X / MI355X
ARCH_GFX942 = "gfx942"  # MI300X / MI325X

# Application modes.
MODE_SETDEFAULT = "setdefault"  # os.environ.setdefault (respect anything already set)
MODE_XLA_MERGE = "xla_merge"  # per-flag merge into XLA_FLAGS, managed knobs win


@dataclass(frozen=True)
class EnvVar:
    """A single declarative environment default.

    Args:
        name: Environment variable name.
        value: Desired value (string).
        arch: Arch gate — ``"all"`` (default), ``"gfx950"``, or ``"gfx942"``.
        mode: ``"setdefault"`` (default) or ``"xla_merge"`` (for ``XLA_FLAGS``).
        note: Optional human-readable rationale (for logs / maintainers).
    """

    name: str
    value: str
    arch: str = ARCH_ALL
    mode: str = MODE_SETDEFAULT
    note: str = ""


_ARCH_CACHE: Optional[str] = None


def detect_gpu_arch() -> str:
    """Best-effort GPU arch detection via ``rocminfo``.

    Returns ``"gfx950"``, ``"gfx942"``, or ``"unknown"``. Result is cached for the
    process. Never raises — detection must never abort a training run.
    """
    global _ARCH_CACHE
    if _ARCH_CACHE is not None:
        return _ARCH_CACHE

    arch = "unknown"
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    try:
        out = subprocess.run([rocminfo], capture_output=True, text=True, timeout=15).stdout
        for cand in (ARCH_GFX950, ARCH_GFX942):
            if cand in out:
                arch = cand
                break
    except Exception:  # noqa: BLE001 - detection must never abort a run
        pass

    _ARCH_CACHE = arch
    return arch


def _parse_xla_flags(flags: str) -> "OrderedDict[str, str]":
    """Parse an ``XLA_FLAGS`` string into an ordered ``flag-key -> full-token`` map.

    Tokens look like ``--name=value``, ``--name=''`` or bare ``--name``; none of
    the values contain spaces, so a whitespace split is sufficient. The key is the
    portion before ``=`` so we can override on a per-flag basis.
    """
    parsed: "OrderedDict[str, str]" = OrderedDict()
    for tok in flags.split():
        key = tok.split("=", 1)[0] if "=" in tok else tok
        parsed[key] = tok
    return parsed


def merge_xla_flags(existing: str, managed: str) -> str:
    """Merge ``managed`` XLA flags over ``existing`` at ``--flag`` granularity.

    Managed knobs win; flags present only in ``existing`` (e.g. image-baked flags
    we don't manage) are preserved in their original position.
    """
    merged = _parse_xla_flags(existing)
    for key, tok in _parse_xla_flags(managed).items():
        merged[key] = tok
    return " ".join(merged.values())


def apply_env_defaults(
    entries: Iterable[EnvVar],
    framework: str,
    logger: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Apply a backend's declarative env defaults into ``os.environ``.

    - Arch-gated entries are skipped unless the detected arch matches (rocminfo is
      only queried if at least one arch-gated entry is present).
    - ``setdefault`` entries never override an already-set value.
    - ``xla_merge`` entries merge into ``XLA_FLAGS`` (managed knobs win).

    Returns the list of variable names that were actually applied (useful for
    diagnostics / parity checks).
    """
    entries = list(entries)
    if not entries:
        return []

    log = logger or (lambda _msg: None)

    needs_arch = any(e.arch != ARCH_ALL for e in entries)
    arch = detect_gpu_arch() if needs_arch else ARCH_ALL

    applied: List[str] = []
    for e in entries:
        if e.arch != ARCH_ALL and e.arch != arch:
            continue

        if e.mode == MODE_XLA_MERGE:
            before = os.environ.get(e.name, "")
            after = merge_xla_flags(before, e.value)
            os.environ[e.name] = after
            if after != before:
                applied.append(e.name)
                log(f"[Primus:{framework}] {e.name} merged (managed XLA knobs win)")
        else:
            # setdefault semantics: only set (and count) when currently unset, so
            # outer/shell/YAML `env:` values always take precedence.
            if e.name not in os.environ:
                os.environ[e.name] = e.value
                applied.append(e.name)
                suffix = f" ({e.note})" if e.note else ""
                log(f"[Primus:{framework}] {e.name}={e.value} (default){suffix}")

    return applied
