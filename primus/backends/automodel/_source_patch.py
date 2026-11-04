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
import operator
import textwrap
from typing import Any, Callable, Sequence, Tuple

_FUTURE_FLAGS = functools.reduce(
    operator.or_, (getattr(__future__, name).compiler_flag for name in __future__.all_feature_names)
)


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

    A method can be source-patched only once, because ``inspect.getsource``
    cannot read a function rebuilt from a string. Batch all rewrites of one
    method into a single call. The rebuilt method keeps the ``from __future__``
    imports of the module it came from.

    Raises:
        RuntimeError: If ``cls`` does not define the method itself, its source
            cannot be read (as after an earlier source patch), or an anchor does
            not match exactly once.
    """
    original = cls.__dict__.get(method_name)
    if original is None:
        raise RuntimeError(
            f"[SourcePatch] {method_name} is not defined on {cls.__name__} itself; "
            "patch the class that defines it"
        )
    try:
        source = inspect.getsource(original)
    except OSError as exc:
        raise RuntimeError(
            f"[SourcePatch] cannot read the source of {cls.__name__}.{method_name}; "
            "a method can be source-patched only once"
        ) from exc
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
    # zero-argument ``super()`` needs; the cell is then retargeted to ``cls``.
    wrapper_source = "class _PrimusPatchWrapper:\n" + textwrap.indent(modified_source, "    ")
    # dont_inherit: the rebuilt method takes the ``from __future__`` flags of the
    # module it came from, never those of the module calling compile().
    code = compile(
        wrapper_source,
        f"<source patch of {cls.__qualname__}.{method_name}>",
        "exec",
        flags=original.__code__.co_flags & _FUTURE_FLAGS,
        dont_inherit=True,
    )
    namespace: dict = {}
    exec(code, original.__globals__, namespace)  # noqa: S102
    wrapper_cls = namespace["_PrimusPatchWrapper"]
    new_func = wrapper_cls.__dict__[original.__name__]
    if new_func.__closure__:
        for cell in new_func.__closure__:
            if cell.cell_contents is wrapper_cls:
                cell.cell_contents = cls
    new_func.__qualname__ = f"{cls.__qualname__}.{method_name}"
    new_func.__module__ = original.__module__
    new_func.__doc__ = original.__doc__
    setattr(cls, method_name, new_func)
    return new_func
