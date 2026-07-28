###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Idempotency guard for monkeypatches that *wrap* (compose) a target callable.

The core patch runner (``primus.core.patches``) has no built-in re-apply guard:
running the same phase twice re-invokes every patch handler. For patches that
replace a class method or module attribute outright that is harmless (the
assignment is idempotent), but patches that *wrap* an existing callable -- e.g.
the stacked ``train_step`` wrappers (FP8 cache refresh, delayed-scaling
preamble, wall-clock timer) -- would wrap again on a second run and silently
double their side effects.

This helper records applied patch keys in a sentinel set attached to the
*object that owns the wrapped attribute* (typically a module). Keying on the
owner object -- which is stable across re-runs -- rather than on the wrapped
callable makes the guard robust even when several wrappers compose on the same
attribute.

This lives under ``primus.backends.megatron.patches`` (feature-owned) on
purpose: the shared ``primus.core.patches`` framework is inherited and must not
grow feature-specific behavior.
"""

from typing import Any

_SENTINEL_ATTR = "_primus_applied_patch_keys"


def is_patched(target: Any, key: str) -> bool:
    """Return True if ``key`` has already been applied to ``target``."""
    applied = getattr(target, _SENTINEL_ATTR, None)
    return applied is not None and key in applied


def mark_patched(target: Any, key: str) -> None:
    """Record that ``key`` has been applied to ``target``."""
    applied = getattr(target, _SENTINEL_ATTR, None)
    if applied is None:
        applied = set()
        setattr(target, _SENTINEL_ATTR, applied)
    applied.add(key)
