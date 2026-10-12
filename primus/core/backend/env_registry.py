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

        XLA_FLAGS_APPEND  >  per-config ``env:``  >  these backend defaults  >  inherited

    "Inherited" is whatever the process started with: an image-baked ``ENV`` or an
    outer ``export`` / ``--env``. Those two are indistinguishable from inside the
    process, so they necessarily share one layer.

    Ordinary vars use ``os.environ.setdefault``, so anything already set wins.
  - **Architecture awareness.** An entry may be gated to a specific GPU arch
    (e.g. ``gfx950`` only, ``gfx942`` only). Non-matching entries are skipped.
  - **XLA_FLAGS append.** ``XLA_FLAGS`` packs many settings into one string, so
    ``setdefault`` cannot express "override one of them": images bake a value we
    must correct (notably ``--xla_gpu_autotune_level=0``, which NaNs fp8 MoE runs),
    and ``setdefault`` against a baked value is a no-op that would leave *every*
    managed knob unapplied. XLA honours the LAST occurrence of a repeated flag, so
    ``mode="xla_append"`` entries are appended rather than parsed and merged.

    A config whose ``env:`` block sets ``XLA_FLAGS`` owns the variable outright:
    the managed defaults are skipped instead of being appended after it, so the
    precedence above actually holds. To override individual flags while keeping the
    managed defaults, use ``XLA_FLAGS_APPEND`` — it is applied last, from either the
    shell or an ``env:`` block.

Backends with no special env (Megatron, TorchTitan, ...) simply return ``[]``
from ``env_defaults()`` and this module is a no-op for them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

# Supported arch gates. "all" applies everywhere; the others are matched against
# the detected GPU architecture string from rocminfo.
ARCH_ALL = "all"
ARCH_GFX950 = "gfx950"  # MI350X / MI355X
ARCH_GFX942 = "gfx942"  # MI300X / MI325X

# Application modes.
MODE_SETDEFAULT = "setdefault"  # os.environ.setdefault (respect anything already set)
MODE_XLA_APPEND = "xla_append"  # append to XLA_FLAGS; XLA honours the last occurrence

# Highest-priority XLA flags, appended after everything else. Settable from the
# shell or from a config ``env:`` block.
XLA_FLAGS_APPEND = "XLA_FLAGS_APPEND"

# Env vars supplied by the experiment YAML's top-level ``env:`` block. Tracked so
# append-mode entries can step aside instead of overriding an explicit user value
# (see module docstring). TrainRuntime populates this via :func:`mark_config_owned`.
_CONFIG_OWNED: set = set()


def mark_config_owned(*names: str) -> None:
    """Record env vars that came from the experiment YAML's ``env:`` block."""
    _CONFIG_OWNED.update(names)


def clear_config_owned() -> None:
    """Forget config ownership. A process trains one config, so this is a test hook."""
    _CONFIG_OWNED.clear()


@dataclass(frozen=True)
class EnvVar:
    """A single declarative environment default.

    Args:
        name: Environment variable name.
        value: Desired value (string).
        arch: Arch gate — ``"all"`` (default), ``"gfx950"``, or ``"gfx942"``.
        mode: ``"setdefault"`` (default) or ``"xla_append"`` (for ``XLA_FLAGS``).
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


def append_xla_flags(existing: str, addition: str) -> str:
    """Append ``addition`` after ``existing``, dropping empty operands.

    XLA honours the last occurrence of a repeated flag, so appending overrides
    whatever came before without parsing it. That also means flags we do not manage
    survive untouched, and values may contain anything (the old per-flag merge had
    to assume no value ever contained a space).
    """
    return " ".join(part for part in (existing.strip(), addition.strip()) if part)


def apply_env_defaults(
    entries: Iterable[EnvVar],
    framework: str,
    logger: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Apply a backend's declarative env defaults into ``os.environ``.

    - Arch-gated entries are skipped unless the detected arch matches (rocminfo is
      only queried if at least one arch-gated entry is present).
    - ``setdefault`` entries never override an already-set value.
    - ``xla_append`` entries append to ``XLA_FLAGS`` (last occurrence wins), unless
      the config's ``env:`` block owns the variable, in which case they step aside.

    Does NOT apply ``XLA_FLAGS_APPEND``; that is :func:`apply_xla_flags_append`,
    which must run for every backend including those declaring no defaults here.

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

        if e.mode == MODE_XLA_APPEND:
            if e.name in _CONFIG_OWNED:
                log(
                    f"[Primus:{framework}] {e.name} comes from the config `env:` block; "
                    f"managed defaults skipped. Set {XLA_FLAGS_APPEND} instead to override "
                    f"individual flags on top of them."
                )
                continue
            before = os.environ.get(e.name, "")
            after = append_xla_flags(before, e.value)
            if after != before:
                os.environ[e.name] = after
                applied.append(e.name)
                log(f"[Primus:{framework}] {e.name} appended (managed defaults override inherited)")
        else:
            # setdefault semantics: only set (and count) when currently unset, so
            # outer/shell/YAML `env:` values always take precedence.
            if e.name not in os.environ:
                os.environ[e.name] = e.value
                applied.append(e.name)
                suffix = f" ({e.note})" if e.note else ""
                log(f"[Primus:{framework}] {e.name}={e.value} (default){suffix}")

    return applied


def apply_xla_flags_append(logger: Optional[Callable[[str], None]] = None) -> bool:
    """Append ``XLA_FLAGS_APPEND`` onto ``XLA_FLAGS`` as the final, winning layer.

    This is the supported way to override an individual managed flag (or add one)
    without taking ownership of the whole ``XLA_FLAGS`` string, and it works
    identically from the shell and from a config ``env:`` block.

    Runs for every backend — including those that declare no ``env_defaults()`` —
    so it is deliberately separate from :func:`apply_env_defaults`. The source
    variable is consumed, making repeat calls no-ops.

    Returns whether anything was appended.
    """
    addition = os.environ.pop(XLA_FLAGS_APPEND, "").strip()
    if not addition:
        return False

    os.environ["XLA_FLAGS"] = append_xla_flags(os.environ.get("XLA_FLAGS", ""), addition)
    (logger or (lambda _msg: None))(f"[Primus] {XLA_FLAGS_APPEND} appended (wins): {addition}")
    return True
