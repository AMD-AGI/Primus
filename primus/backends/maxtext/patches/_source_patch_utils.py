###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for MaxText patches.

Most of the ROCm/maxtext ``release/v26.10`` changes insert a few lines in the
*middle* of a MaxText function, where a wrapper cannot reach. For those the
patches fall back to the "source-string rewrite" style also used by the
Megatron backend: ``inspect.getsource()`` the original, replace exact anchor
fragments, recompile, and rebind the result onto the owning module/class.

Every anchor must match exactly the expected number of times. A mismatch
raises, so a MaxText bump that moves the code fails at startup instead of
silently training without the change.

This module is MaxText-owned on purpose: importing the Megatron helper would
import ``primus.backends.megatron.patches`` (and torch) as a side effect.
"""

import inspect
import textwrap
from typing import Any, Callable, Iterable, Sequence, Tuple, Union

# (anchor, replacement) or (anchor, replacement, expected_count)
Replacement = Union[Tuple[str, str], Tuple[str, str, int]]


_SOURCE_ATTR = "__primus_source__"

# ``run_patches`` only logs a failing patch, so anchor mismatches are also recorded here
# and :func:`raise_on_source_patch_failures` turns them into a hard error before training.
SOURCE_PATCH_FAILURES: list = []


def raise_on_source_patch_failures() -> None:
    """Raise if any MaxText source patch failed to find its anchor."""
    if SOURCE_PATCH_FAILURES:
        details = "\n  ".join(SOURCE_PATCH_FAILURES)
        raise RuntimeError(
            "MaxText source patches could not be applied, so this MaxText would train without them:\n  "
            f"{details}\nUpdate the anchors in primus/backends/maxtext/patches/ for this MaxText, or leave "
            "the failing patch ids out of PRIMUS_PATCHES."
        )


def get_source(owner: Any, name: str) -> str:
    """Return the current source of ``owner.<name>``, including earlier rewrites by :func:`rewrite_source`."""
    fn = inspect.unwrap(getattr(owner, name))
    return getattr(fn, _SOURCE_ATTR, None) or inspect.getsource(fn)


def rewrite_source(owner: Any, name: str, replacements: Sequence[Replacement]) -> Callable:
    """Apply ``replacements`` to the source of ``owner.<name>`` and rebind the result.

    Args:
        owner: The module (for a function) or class (for a method) that owns ``name``.
        name: Attribute name of the function/method to rewrite.
        replacements: ``(anchor, replacement[, expected_count])`` tuples applied in
            order. Anchors are written at the upstream file's absolute indentation;
            ``expected_count`` defaults to 1.

    Returns:
        The newly compiled function that was set on ``owner``.

    Raises:
        RuntimeError: If an anchor does not match exactly ``expected_count`` times.
    """
    original = inspect.unwrap(getattr(owner, name))
    source = get_source(owner, name)
    label = f"{getattr(owner, '__qualname__', getattr(owner, '__name__', owner))}.{name}"
    for item in replacements:
        anchor, new_code = item[0], item[1]
        expected = item[2] if len(item) > 2 else 1
        found = source.count(anchor)
        if found != expected:
            message = (
                f"[SourcePatch] Anchor matched {found} times in {label}, expected {expected}; "
                f"upstream MaxText may have changed. Anchor: {anchor!r}"
            )
            SOURCE_PATCH_FAILURES.append(message)
            raise RuntimeError(message)
        source = source.replace(anchor, new_code)
    # Dedent after replacing so anchors/replacements can use the file's absolute indentation.
    modified = textwrap.dedent(source)

    is_method = isinstance(owner, type)
    if is_method:
        # Compile inside a throwaway class so zero-arg ``super()`` keeps its ``__class__`` cell,
        # then retarget that cell to the real owner.
        modified = "class _PrimusPatchWrapper:\n" + textwrap.indent(modified, "    ")
    namespace: dict = {}
    code = compile(modified, f"<primus source patch: {label}>", "exec")
    exec(code, original.__globals__, namespace)  # noqa: S102
    if is_method:
        wrapper_cls = namespace["_PrimusPatchWrapper"]
        new_func = wrapper_cls.__dict__[name]
        for cell in new_func.__closure__ or ():
            if cell.cell_contents is wrapper_cls:
                cell.cell_contents = owner
        new_func.__qualname__ = f"{owner.__qualname__}.{name}"
    else:
        new_func = namespace[name]
    # exec'd code has no file to read back, so keep the (un-dedented) source for later
    # rewrites of the same function and for "already applied" checks.
    setattr(new_func, _SOURCE_ATTR, source)
    setattr(owner, name, new_func)
    return new_func


def any_in_source(owner: Any, name: str, markers: Iterable[str]) -> bool:
    """True if any marker already appears in ``owner.<name>``'s source (i.e. the change is upstream)."""
    source = get_source(owner, name)
    return any(marker in source for marker in markers)
