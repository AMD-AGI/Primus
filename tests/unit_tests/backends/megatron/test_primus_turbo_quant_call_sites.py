###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""AST-level guard for ``_maybe_create_quantized_weight_buffers`` call sites.

The FP4 ``Linear`` weight-buffer path once called this helper with a
non-existent keyword (``need_cache_colwise=``) while the parameter is
``disable_parameter_transpose_cache``. That is a runtime-only ``TypeError`` on
the first FP4 microbatch, which CI never hits because MXFP4 runs on MI355X
(gfx950) while CI runs on MI300X (gfx942).

This test parses the source statically (no import, no GPU, no primus_turbo
required) and asserts every call site of the helper binds to the helper's
signature. It catches the exact class of the original regression (wrong kwarg
name / missing required arg). It does NOT verify the *value* passed is
semantically correct -- that needs real FP4 execution on gfx950.
"""

import ast
import inspect
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[4] / "primus/backends/megatron/core/extensions/primus_turbo.py"
_FUNC_NAME = "_maybe_create_quantized_weight_buffers"

# Placeholder standing in for argument values/defaults; only names, kinds and
# arity are relevant when validating that a call binds to the signature.
_SENTINEL = object()


def _signature_from_ast(func_def: ast.FunctionDef) -> inspect.Signature:
    """Reconstruct an ``inspect.Signature`` from an ``ast.FunctionDef``."""
    a = func_def.args
    P = inspect.Parameter
    positional = a.posonlyargs + a.args
    # Trailing ``len(defaults)`` positional params are the ones with defaults.
    first_default = len(positional) - len(a.defaults)

    params: list[inspect.Parameter] = []
    for i, arg in enumerate(positional):
        kind = P.POSITIONAL_ONLY if i < len(a.posonlyargs) else P.POSITIONAL_OR_KEYWORD
        default = _SENTINEL if i >= first_default else P.empty
        params.append(P(arg.arg, kind, default=default))
    if a.vararg is not None:
        params.append(P(a.vararg.arg, P.VAR_POSITIONAL))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        params.append(P(arg.arg, P.KEYWORD_ONLY, default=P.empty if default is None else _SENTINEL))
    if a.kwarg is not None:
        params.append(P(a.kwarg.arg, P.VAR_KEYWORD))

    return inspect.Signature(params)


def _load_def_and_calls() -> tuple[ast.FunctionDef, list[ast.Call]]:
    """Return the ``_FUNC_NAME`` definition and all direct calls to it."""
    tree = ast.parse(_TARGET.read_text(), filename=str(_TARGET))
    func_def = None
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _FUNC_NAME:
            func_def = node
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _FUNC_NAME:
            calls.append(node)
    return func_def, calls


def _binds(sig: inspect.Signature, call: ast.Call) -> tuple[bool, str]:
    """Check whether ``call`` binds to ``sig`` using name/arity only.

    ``*args`` / ``**kwargs`` spreads at the call site supply an unknown number
    of arguments, so such calls cannot be checked statically and are treated as
    valid. Returns ``(ok, error_message)``.
    """
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        return True, ""
    args = [_SENTINEL] * len(call.args)
    kwargs = {kw.arg: _SENTINEL for kw in call.keywords}
    try:
        sig.bind(*args, **kwargs)
    except TypeError as e:
        return False, str(e)
    return True, ""


def test_maybe_create_quantized_weight_buffers_call_sites_are_valid():
    """Every call site must bind to ``_maybe_create_quantized_weight_buffers``."""
    func_def, calls = _load_def_and_calls()
    assert func_def is not None, f"could not find `def {_FUNC_NAME}` in {_TARGET}"
    assert calls, f"expected at least one call to `{_FUNC_NAME}` in {_TARGET}"

    sig = _signature_from_ast(func_def)
    for call in calls:
        ok, err = _binds(sig, call)
        assert ok, (
            f"{_TARGET.name}:{call.lineno}: `{_FUNC_NAME}{sig}` call does not bind "
            f"({err}); valid parameters are {sorted(sig.parameters)}"
        )
