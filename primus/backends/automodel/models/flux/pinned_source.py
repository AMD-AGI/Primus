###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Whether a diffusers function is the code a FLUX patch re-implements.

A patch that re-implements a diffusers function stands in for it only while that function's code is
the code it was written against, compared as a digest of its syntax tree without the docstring and
the type annotations, so that a release that only reformats it, re-annotates it or edits its comments
still matches and one that changes a statement does not. A function another patch has wrapped does
not match either: standing in for it would skip the wrapper.
"""
import ast
import hashlib
import inspect
import textwrap
from typing import Optional


def _canonical(node) -> str:
    """The syntax tree as text, leaving out empty fields: ``ast.dump`` and ``ast.unparse`` differ between
    Python versions (newer ones add fields such as ``type_params``, or parenthesize tuples otherwise)."""
    if isinstance(node, ast.AST):
        fields = (
            f"{name}={_canonical(value)}" for name, value in ast.iter_fields(node) if value not in (None, [])
        )
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return f"[{', '.join(_canonical(value) for value in node)}]"
    return repr(node)


def digest(fn) -> Optional[str]:
    """sha256 of ``fn``'s code without its docstring and annotations; None for a wrapped function or
    one whose source cannot be read."""
    if getattr(fn, "__wrapped__", None) is not None:
        return None
    try:
        function = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    function.returns = None
    arguments = function.args
    for argument in (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ):
        if argument is not None:
            argument.annotation = None
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        function.body = body[1:]
    return hashlib.sha256(_canonical(function).encode()).hexdigest()
