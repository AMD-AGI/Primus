###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helper for "source-string rewrite" style patches.

Some upstream Megatron-LM functions/methods need a small code fragment
inserted in the *middle* of their body (not just wrapped before/after), where
a plain function-wrapping monkey-patch cannot reach. For those cases Primus
falls back to: ``inspect.getsource()`` the original, apply a targeted
``str.replace()`` on a unique anchor fragment, recompile with ``exec()``, and
rebind the result onto the owning module/class.

This lives under ``primus.backends.megatron.patches`` (feature-owned) on
purpose: the shared ``primus.core.patches`` framework is inherited and must
not grow feature-specific behavior.
"""

import inspect
import textwrap
from typing import Any, Callable


def patch_method_source(
    cls: Any,
    method_name: str,
    ori_code: str,
    new_code: str,
) -> Callable:
    """Rewrite a fragment of ``cls.<method_name>``'s source and rebind it.

    Args:
        cls: The class owning the method (e.g. ``MambaModel``).
        method_name: Name of the method to rewrite (e.g. ``"__init__"``).
        ori_code: Unique anchor substring to locate within the method source.
        new_code: Replacement text for ``ori_code`` (typically ``ori_code``
            plus extra inserted lines, or vice versa).

    Returns:
        The newly compiled function that was set on ``cls``.

    Raises:
        AssertionError: If ``ori_code`` is not found in the method's source
            (e.g. upstream Megatron-LM changed the function unexpectedly).
    """
    # IMPORTANT: replace on the *raw* (non-dedented) source -- ori_code/new_code
    # are written using the upstream file's absolute column indentation (i.e.
    # what you see reading the file directly). Dedent must happen AFTER the
    # replacement, once, so the whole (possibly now-longer) body shifts by a
    # single consistent amount; dedenting first and writing new_code at
    # post-dedent indentation is a common source of IndentationError.
    original = getattr(cls, method_name)
    source = inspect.getsource(original)
    assert ori_code in source, (
        f"[SourcePatch] Anchor not found in {cls.__name__}.{method_name}; "
        f"upstream source may have changed. Anchor: {ori_code!r}"
    )
    modified_source = textwrap.dedent(source.replace(ori_code, new_code))
    namespace: dict = {}
    exec(modified_source, original.__globals__, namespace)  # noqa: S102
    new_func = namespace[original.__name__]
    setattr(cls, method_name, new_func)
    return new_func


def patch_function_source(
    module: Any,
    function_name: str,
    ori_code: str,
    new_code: str,
) -> Callable:
    """Same as :func:`patch_method_source`, but for a module-level function."""
    original = getattr(module, function_name)
    source = inspect.getsource(original)
    assert ori_code in source, (
        f"[SourcePatch] Anchor not found in {module.__name__}.{function_name}; "
        f"upstream source may have changed. Anchor: {ori_code!r}"
    )
    modified_source = textwrap.dedent(source.replace(ori_code, new_code))
    namespace: dict = {}
    exec(modified_source, original.__globals__, namespace)  # noqa: S102
    new_func = namespace[original.__name__]
    setattr(module, function_name, new_func)
    return new_func
