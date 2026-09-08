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
from typing import Any, Callable, List, Tuple


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
    return patch_method_source_multi(cls, method_name, [(ori_code, new_code)])


def patch_method_source_multi(
    cls: Any,
    method_name: str,
    replacements: List[Tuple[str, str]],
) -> Callable:
    """Rewrite several fragments of ``cls.<method_name>``'s source in one pass.

    Calling :func:`patch_method_source` twice on the same method does not work:
    the function it leaves behind was produced by ``exec``, so its code object
    has no source file and ``inspect.getsource`` raises on the second call. A
    method needing more than one edit therefore has to make all of them before
    the single recompile, which is what this does.

    Args:
        cls: The class owning the method.
        method_name: Name of the method to rewrite.
        replacements: ``(anchor, replacement)`` pairs, applied in order. Each
            anchor must appear in the source, so a pair that upstream has
            already made itself is an error rather than a silent no-op.

    Returns:
        The newly compiled function that was set on ``cls``.

    Raises:
        AssertionError: If any anchor is not found (e.g. upstream changed).
    """
    # IMPORTANT: replace on the *raw* (non-dedented) source -- ori_code/new_code
    # are written using the upstream file's absolute column indentation (i.e.
    # what you see reading the file directly). Dedent must happen AFTER the
    # replacement, once, so the whole (possibly now-longer) body shifts by a
    # single consistent amount; dedenting first and writing new_code at
    # post-dedent indentation is a common source of IndentationError.
    original = getattr(cls, method_name)
    source = inspect.getsource(original)
    for ori_code, new_code in replacements:
        assert ori_code in source, (
            f"[SourcePatch] Anchor not found in {cls.__name__}.{method_name}; "
            f"upstream source may have changed. Anchor: {ori_code!r}"
        )
        source = source.replace(ori_code, new_code)
    modified_source = textwrap.dedent(source)

    # IMPORTANT: exec'ing `modified_source` as a bare top-level `def` (not
    # nested in a class body) silently loses the implicit `__class__` closure
    # cell that zero-arg `super()` depends on: the function still compiles
    # and defines fine, but crashes at *call* time with
    # `RuntimeError: super(): __class__ cell not found` the first time the
    # method runs. Compiling inside a throwaway wrapper class makes the
    # compiler wire up that closure normally; we then retarget the cell from
    # the throwaway class to the real owning `cls` (required for `super()`'s
    # MRO walk to actually find `cls` in `type(self).__mro__`).
    wrapper_source = "class _PrimusPatchWrapper:\n" + textwrap.indent(modified_source, "    ")
    namespace: dict = {}
    exec(wrapper_source, original.__globals__, namespace)  # noqa: S102
    wrapper_cls = namespace["_PrimusPatchWrapper"]
    new_func = wrapper_cls.__dict__[original.__name__]
    if new_func.__closure__:
        for cell in new_func.__closure__:
            if cell.cell_contents is wrapper_cls:
                cell.cell_contents = cls
    new_func.__qualname__ = f"{cls.__qualname__}.{method_name}"
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
