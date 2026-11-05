###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Source-string rewrite helper for patches that must edit inside a method body.

Most patches in this backend wrap or replace a whole function. A few need to
change a line in the *middle* of a long upstream method (the AutoModel train
loop, say), where wrapping cannot reach and copying the whole method would
fork it. For those, the method source is read
with ``inspect.getsource``, each anchor fragment is replaced exactly once, and
the result is recompiled against the original module globals and rebound.

This is the same technique as ``primus.backends.megatron.patches._source_patch_utils``.
It is duplicated rather than imported because importing anything under
``primus.backends.megatron`` pulls in the Megatron adapter and registers every
Megatron patch.

Keep this module free of torch and AutoModel imports so unit tests can use it
on stub classes.
"""
import __future__

import functools
import inspect
import linecache
import operator
import textwrap
from typing import Any, Callable, Sequence, Tuple

_FUTURE_FLAGS = functools.reduce(
    operator.or_, (getattr(__future__, name).compiler_flag for name in __future__.all_feature_names)
)
_REBUILT_FLAG = "_primus_source_patched"


def patch_method_source(
    cls: Any,
    method_name: str,
    replacements: Sequence[Tuple[str, str]],
) -> Callable:
    """Apply ``(anchor, replacement)`` rewrites to ``cls.<method_name>`` and rebind it.

    Anchors are written at the upstream file's absolute indentation, i.e. exactly
    as they appear when reading the file. Each must match exactly once: a missing
    anchor means upstream changed, and an ambiguous one would inject the change
    somewhere it was never meant to go.

    A method can be source-patched only once: batch all rewrites of one method
    into a single call. The rebuilt method keeps the ``from __future__`` imports
    of the module it came from. Tracebacks and debuggers show its rewritten
    source, under a name that gives the upstream file, at the upstream line
    numbers up to the first rewrite that adds or removes lines.

    Raises:
        RuntimeError: If ``cls`` does not define the method itself as a plain
            function, the method reads variables of an enclosing function, it was
            source-patched already, its source cannot be read, or an anchor does
            not match exactly once.
    """
    original = cls.__dict__.get(method_name)
    if original is None:
        raise RuntimeError(
            f"[SourcePatch] {method_name} is not defined on {cls.__name__} itself; "
            "patch the class that defines it"
        )
    if not inspect.isfunction(original):
        raise RuntimeError(
            f"[SourcePatch] {cls.__name__}.{method_name} is a {type(original).__name__}; "
            "only plain functions can be source-patched"
        )
    if getattr(original, _REBUILT_FLAG, False):
        raise RuntimeError(
            f"[SourcePatch] {cls.__name__}.{method_name} was source-patched already; "
            "a method can be source-patched only once"
        )
    free_vars = sorted(set(original.__code__.co_freevars) - {"__class__"})
    if free_vars:
        raise RuntimeError(
            f"[SourcePatch] {cls.__name__}.{method_name} reads variables of an enclosing function "
            f"({', '.join(free_vars)}), which the rebuilt method could not see"
        )
    try:
        source = inspect.getsource(original)
    except OSError as exc:
        raise RuntimeError(f"[SourcePatch] cannot read the source of {cls.__name__}.{method_name}") from exc
    for ori_code, new_code in replacements:
        found = source.count(ori_code)
        if found != 1:
            raise RuntimeError(
                f"[SourcePatch] Anchor matched {found} times in {cls.__name__}.{method_name}, "
                f"expected exactly 1; upstream source may have changed. Anchor: {ori_code!r}"
            )
        source = source.replace(ori_code, new_code, 1)
    # Dedent only after replacing, so the whole body shifts by one consistent amount.
    modified_source = textwrap.dedent(source)

    # Compiling inside a throwaway class keeps the implicit ``__class__`` cell that
    # zero-argument ``super()`` needs; the cell is then retargeted to ``cls``. The
    # leading blank lines put the method at its upstream line numbers.
    wrapper_source = (
        "\n" * max(original.__code__.co_firstlineno - 2, 0)
        + "class _PrimusPatchWrapper:\n"
        + textwrap.indent(modified_source, "    ")
    )
    filename = f"<{cls.__qualname__}.{method_name}, source-patched from {original.__code__.co_filename}>"
    # dont_inherit: the rebuilt method takes the ``from __future__`` flags of the
    # module it came from, never those of the module calling compile().
    code = compile(
        wrapper_source,
        filename,
        "exec",
        flags=original.__code__.co_flags & _FUTURE_FLAGS,
        dont_inherit=True,
    )
    namespace: dict = {}
    exec(code, original.__globals__, namespace)  # noqa: S102
    # No modification time, so linecache.checkcache() never drops the entry.
    linecache.cache[filename] = (
        len(wrapper_source),
        None,
        wrapper_source.splitlines(keepends=True),
        filename,
    )
    wrapper_cls = namespace["_PrimusPatchWrapper"]
    new_func = wrapper_cls.__dict__[original.__name__]
    if new_func.__closure__:
        for cell in new_func.__closure__:
            if cell.cell_contents is wrapper_cls:
                cell.cell_contents = cls
    new_func.__qualname__ = f"{cls.__qualname__}.{method_name}"
    new_func.__module__ = original.__module__
    new_func.__doc__ = original.__doc__
    setattr(new_func, _REBUILT_FLAG, True)
    setattr(cls, method_name, new_func)
    return new_func
