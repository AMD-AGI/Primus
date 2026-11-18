###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-layer ``*_freq`` patterns (``moe_layer_freq``, Muse Spark's ``window_attn_skip_freq`` /
``no_rope_freq``) as ints or 0/1 lists.

Pure Python (no torch / Megatron), so FLOPs patches can import it at ``before_train``.
Used by the Muse Spark FLOPs patch and the Muse Spark layer patterns. Kimi-K3's FLOPs
patch keeps its own, more lenient parser (``kimi_k3_flops_patches.py``: an int string means
all-MoE, list entries are coerced with ``int()``); moving it here would change Kimi-K3's
results, so the two parsers stay separate.

Strings are parsed without ``eval``: only int literals (with unary minus), int arithmetic
(``+ - *``, e.g. a repeat count ``(60+2)``), flat int list literals, ``list * int``,
``int * list`` and ``list + list`` are accepted (e.g. ``"([0]*1+[1]*7)"``,
``"([1]*3+[0])*6"`` or ``"[1]*(60+2)"``). This is a subset of what Megatron's ``eval``
accepts: a list comprehension or a name is refused with a clear error (write the list out),
never evaluated. A tuple literal is refused: Megatron's ``eval`` raises TypeError on
``list + tuple``. A list may hold at most ``MAX_PATTERN_LEN``
entries (checked before a repetition is built), a string at most ``MAX_SOURCE_LEN``
characters. Any other input (a name, a call, bad syntax, deep nesting) raises ValueError.
As in Megatron's ``moe_freq_type``, a string without ``[`` must be a plain int (``"2"``);
``"1+1"`` or ``"(1,0)"`` are refused.
"""

from __future__ import annotations

import ast
from typing import Any, List

__all__ = [
    "MAX_PATTERN_LEN",
    "MAX_SOURCE_LEN",
    "moe_layer_pattern",
    "parse_layer_pattern",
]

# Upper bound on a pattern list (far above any real depth); stops e.g. "[1]*1000000000".
MAX_PATTERN_LEN = 4096
# Upper bound on a pattern string (a 4096-entry literal list is ~12k characters).
MAX_SOURCE_LEN = 4 * MAX_PATTERN_LEN
_SHOWN_PATTERN_CHARS = 80  # how much of a bad pattern string an error shows
_SHOWN_BAD_ENTRIES = 5  # how many bad entries of a pattern list an error shows
_TOO_LONG = f"layer pattern longer than {MAX_PATTERN_LEN}"


def _repeat(lst, n):
    # n itself is bounded too: an empty list times a huge n would pass the length check
    if n < 0 or n > MAX_PATTERN_LEN or len(lst) * n > MAX_PATTERN_LEN:
        raise ValueError(f"layer-pattern repetition {len(lst)}*{n} is outside [0, {MAX_PATTERN_LEN}]")
    return lst * n


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_node(node.operand)
        if not isinstance(operand, int):
            raise ValueError("unary minus applies to an int only")
        return -operand
    if isinstance(node, ast.List):
        out = [_eval_node(e) for e in node.elts]
        if any(isinstance(v, list) for v in out):
            raise ValueError("nested lists are not a layer pattern")
        if len(out) > MAX_PATTERN_LEN:
            raise ValueError(_TOO_LONG)
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(left, int) and isinstance(right, int):  # a count, e.g. [1]*(60+2)
            if isinstance(node.op, ast.Mult):
                value = left * right
            else:
                value = left + right if isinstance(node.op, ast.Add) else left - right
            if abs(value) > MAX_PATTERN_LEN:
                raise ValueError(
                    f"layer-pattern count {value} is outside [-{MAX_PATTERN_LEN}, {MAX_PATTERN_LEN}]"
                )
            return value
        if isinstance(node.op, ast.Add) and isinstance(left, list) and isinstance(right, list):
            if len(left) + len(right) > MAX_PATTERN_LEN:
                raise ValueError(_TOO_LONG)
            return left + right
        if isinstance(node.op, ast.Mult) and isinstance(left, list) and isinstance(right, int):
            return _repeat(left, right)
        if isinstance(node.op, ast.Mult) and isinstance(left, int) and isinstance(right, list):
            return _repeat(right, left)
    raise ValueError(
        f"unsupported layer-pattern expression {ast.unparse(node)[:_SHOWN_PATTERN_CHARS]!r}: use int / "
        "list literals with +, -, * (no names, calls or comprehensions; write the list out)"
    )


def parse_layer_pattern(value):
    """``value`` as an int, a flat list of ints, or None (strings are parsed safely).

    Raises ValueError for anything else (bools, nested lists, bad syntax, too long).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"layer pattern must be an int or a list, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        src = value.strip()
        if len(src) > MAX_SOURCE_LEN:
            raise ValueError(f"layer-pattern string longer than {MAX_SOURCE_LEN} characters")
        if "[" not in src:
            # Megatron's moe_freq_type reads a string without "[" as int(x): no arithmetic
            # or tuple there, so a pattern Megatron reads differently is refused here too.
            try:
                return int(src)
            except ValueError:
                raise ValueError(
                    f"layer pattern {value[:_SHOWN_PATTERN_CHARS]!r}: a string without a list must be "
                    "an int (as Megatron reads it); for a pattern, write the list out with [...]"
                ) from None
        try:
            return _eval_node(ast.parse(src, mode="eval"))
        except SyntaxError as e:
            raise ValueError(
                f"cannot parse layer pattern {value[:_SHOWN_PATTERN_CHARS]!r}: {e.msg}"
            ) from None
        except (MemoryError, RecursionError) as e:  # deeply nested input
            raise ValueError(f"layer pattern {value[:_SHOWN_PATTERN_CHARS]!r} is nested too deeply") from e
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_PATTERN_LEN:
            raise ValueError(_TOO_LONG)
        out = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"layer-pattern entries must be ints, got {v!r}")
            out.append(v)
        return out
    raise ValueError(f"layer pattern must be an int, a list or a string, got {type(value).__name__}")


def moe_layer_pattern(raw: Any, num_layers: int) -> List[int]:
    """``1`` = MoE MLP, ``0`` = dense MLP, per layer.

    The pattern is the one Megatron builds, or a ValueError (a TypeError for an unsupported type):

    * ``None`` (unset) is upstream's default ``moe_layer_freq=1``: every layer is MoE.
    * An int ``N >= 1`` means MoE every ``N``-th layer starting at 0 (upstream's
      ``i % N == 0``). ``N <= 0`` is a ValueError: upstream divides by zero for 0, and for
      a negative ``N`` builds MoE on every ``|N|``-th layer, which no config means.
    * An int string (``"2"``, a CLI / env override) means the same as the int.
    * A list (or list expression) must have ``num_layers`` entries, each 0 or 1 (all 0s
      is how a config asks for no MoE layer).
    * A bool or a blank string is a ValueError.
    * Any other type (float, dict, ...) is a TypeError.
    """
    if raw is None:
        raw = 1
    if isinstance(raw, bool) or (isinstance(raw, str) and not raw.strip()):
        raise ValueError(f"moe_layer_freq: {raw!r} is not a layer pattern")
    if isinstance(raw, (str, list, tuple)):
        try:
            raw = parse_layer_pattern(raw)  # parsed once: an int or a list of ints from here on
        except ValueError as e:
            raise ValueError(f"moe_layer_freq: {e}") from None
    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError(
                f"moe_layer_freq={raw!r}: an int must be >= 1 (MoE every N-th layer); "
                "give a 0/1 list for other layouts"
            )
        return [1 if (i % raw == 0) else 0 for i in range(num_layers)]
    if not isinstance(raw, list):
        raise TypeError(f"Unsupported per-layer pattern type: {type(raw).__name__}")
    bad = sorted({v for v in raw if v not in (0, 1)})
    if bad:
        raise ValueError(
            f"moe_layer_freq entries must be 0 (dense) or 1 (MoE), got {bad[:_SHOWN_BAD_ENTRIES]}"
        )
    if len(raw) != num_layers:
        raise ValueError(f"moe_layer_freq has {len(raw)} entries, expected num_layers={num_layers}")
    return raw
