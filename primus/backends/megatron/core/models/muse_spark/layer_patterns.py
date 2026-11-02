###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark per-layer 0/1 patterns (``window_attn_skip_freq`` / ``no_rope_freq``).

One parser for the config and the ``build_args`` patch. It reads an int, a list,
or a Python-list expression string such as ``"([1,0,1,1]*15+[1,0])"`` without
``eval``: only int literals (with unary minus), flat int list / tuple literals,
``list * int``, ``int * list`` and ``list + list`` are accepted. A list may hold at
most ``MAX_PATTERN_LEN`` entries (checked before a repetition is built), a string at
most ``MAX_SOURCE_LEN`` characters, and every malformed input (including unary minus
on a list and deeply nested expressions) raises ValueError. An int means "every N-th layer" (Megatron's
meaning) and must be >= 1.

A longer list is cut to ``num_layers`` on purpose: a depth-reduced build (proxy,
trainer test) keeps the first layers of the released model, so it keeps the first
pattern entries. The cut is reported through the ``warn`` callback (the ``build_args``
patch logs it on rank 0), so a wrong ``num_layers`` (e.g. 60 for the 62-layer release)
is visible in the log; the checkpoint converter derives ``num_layers`` from the HF
config, so its patterns always match exactly. A shorter list is an error, and an int
must be in ``[1, num_layers]`` (a larger N skips no layer). Every error names the
argument. Pure Python (no torch / Megatron).
"""

import ast
import re

# re-exported: the one implementation lives in the shared light helper
from primus.backends.megatron.core.attention_pairs import causal_visible_pairs

__all__ = [
    "MODEL_TYPE",
    "MODEL_TYPES",
    "parse_layer_pattern",
    "normalize_layer_pattern",
    "check_all_moe_layers",
    "parse_window_size",
    "hf_sliding_window",
    "causal_visible_pairs",
    "MAKE_VOCAB_SIZE_DIVISIBLE_BY",
    "vocab_size_with_padding",
]

# The Primus model_type (one value: the trainer, get_model_provider and the patch condition
# compare against it; ``muse_spark_text`` is only the HF text_config's model_type, which the
# checkpoint hook maps to this family).
MODEL_TYPE = "muse_spark"
MODEL_TYPES = (MODEL_TYPE,)

# The presets' make_vocab_size_divisible_by (202048 = 64 * 3157), used when the args set none
# (the converter's own default; Megatron's argument default is 128).
MAKE_VOCAB_SIZE_DIVISIBLE_BY = 64


def vocab_size_with_padding(vocab_size, divisible_by=MAKE_VOCAB_SIZE_DIVISIBLE_BY, tp=1):
    """The embedding / output-layer rows: ``vocab_size`` rounded up to ``divisible_by``.

    The training ``build_args`` patch and the checkpoint converter both use this, so a
    converted checkpoint has the training shape. Unlike Megatron's own padding
    (``divisible_by * TP``), TP is not a factor: the rows depend only on the model config,
    so one checkpoint loads at every TP. The vocab-parallel layers need the rows to divide
    by TP; that is checked here (202048 rows divide by every TP that 128 heads / 16 query
    groups allow) and an error names the fix.
    """
    multiple = int(divisible_by)
    if multiple < 1:
        raise ValueError(f"make_vocab_size_divisible_by={divisible_by!r} must be >= 1")
    rows = -(-int(vocab_size) // multiple) * multiple
    tp = int(tp or 1)
    if rows % tp:
        raise ValueError(
            f"padded vocab {rows} (vocab_size {vocab_size} rounded up to make_vocab_size_divisible_by="
            f"{multiple}) does not divide by tensor_model_parallel_size={tp}: set "
            f"make_vocab_size_divisible_by to a multiple of {tp}"
        )
    return rows


# Upper bound on a pattern list (far above any real depth); stops e.g. "[1]*1000000000".
MAX_PATTERN_LEN = 4096
# Upper bound on a pattern string (a 4096-entry literal list is ~12k characters).
MAX_SOURCE_LEN = 4 * MAX_PATTERN_LEN
_SHOWN_PATTERN_CHARS = 80  # how much of a bad pattern string an error shows


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
    if isinstance(node, (ast.List, ast.Tuple)):
        out = [_eval_node(e) for e in node.elts]
        if any(isinstance(v, list) for v in out):
            raise ValueError("nested lists are not a layer pattern")
        if len(out) > MAX_PATTERN_LEN:
            raise ValueError(f"layer pattern longer than {MAX_PATTERN_LEN}")
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add) and isinstance(left, list) and isinstance(right, list):
            if len(left) + len(right) > MAX_PATTERN_LEN:
                raise ValueError(f"layer pattern longer than {MAX_PATTERN_LEN}")
            return left + right
        if isinstance(node.op, ast.Mult) and isinstance(left, list) and isinstance(right, int):
            return _repeat(left, right)
        if isinstance(node.op, ast.Mult) and isinstance(left, int) and isinstance(right, list):
            return _repeat(right, left)
    raise ValueError(f"unsupported layer-pattern expression: {ast.dump(node)}")


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
            raise ValueError(f"layer pattern longer than {MAX_PATTERN_LEN}")
        out = []
        for v in value:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"layer-pattern entries must be ints, got {v!r}")
            out.append(v)
        return out
    raise ValueError(f"layer pattern must be an int, a list or a string, got {type(value).__name__}")


def normalize_layer_pattern(value, num_layers, name="layer pattern", warn=None):
    """Parse ``value``, validate the whole pattern, then cut a list to ``num_layers`` entries.

    An int (every N-th layer) must be in ``[1, num_layers]`` (``num_layers`` unknown: >= 1).
    A list must be non-empty, every entry (also those past ``num_layers``) must be 0 or 1,
    and it must have at least ``num_layers`` entries; ``num_layers`` must be a positive int
    when ``value`` is a list. A cut calls ``warn(message)`` when ``warn`` is given. Every
    ValueError names ``name``.
    """
    source = value
    try:
        value = parse_layer_pattern(value)
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None
    layers_ok = not isinstance(num_layers, bool) and isinstance(num_layers, int) and num_layers >= 1
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name}: an int means every N-th layer and must be >= 1, got {value}")
        if layers_ok and value > num_layers:
            raise ValueError(
                f"{name}: every {value}-th layer is outside num_layers={num_layers} "
                f"(the int must be in [1, {num_layers}])"
            )
    if isinstance(value, list):
        if not value:
            shown = f" ({source!r} has 0 entries)" if isinstance(source, str) else ""
            raise ValueError(f"{name} is an empty list{shown}; it needs num_layers={num_layers} entries")
        if any(v not in (0, 1) for v in value):
            raise ValueError(f"{name} entries must be 0 or 1, got {sorted(set(value))}")
        if not layers_ok:
            raise ValueError(f"{name}: a per-layer list needs num_layers >= 1, got {num_layers!r}")
        if len(value) < num_layers:
            raise ValueError(f"{name} has {len(value)} entries, fewer than num_layers={num_layers}")
        if len(value) > num_layers:
            if warn is not None:
                warn(
                    f"{name}: the pattern has {len(value)} entries; num_layers={num_layers} keeps "
                    f"the first {num_layers} (a depth-reduced build). Check num_layers if this "
                    f"is not intended."
                )
            value = value[:num_layers]
    return value


def check_all_moe_layers(moe_layer_freq, num_layers):
    """Muse Spark has a MoE MLP in every layer: ``moe_layer_freq`` must mean "all layers".

    The layer spec builds every layer as MoE (the released model has no dense layer), so
    any other value would be ignored by the model while FLOPs / memory estimates used it.
    Accepts None (unset), 1, or a list (or list string) of all ones. Raises ValueError otherwise.
    """
    value = normalize_layer_pattern(moe_layer_freq, num_layers, "moe_layer_freq")
    if value is None or value == 1 or (isinstance(value, list) and all(v == 1 for v in value)):
        return
    raise ValueError(
        f"Muse Spark builds a MoE MLP in every layer: moe_layer_freq must be 1 "
        f"(or all ones), got {moe_layer_freq!r}"
    )


_WINDOW_RE = re.compile(
    r"\s*(?:\(\s*({p})\s*\)|\[\s*({p})\s*\]|({p}))\s*".format(p=r"[-+]?\d+\s*,\s*[-+]?\d+")
)


def parse_window_size(value):
    """``window_size`` as a causal pair ``(left, 0)`` of ints, or None.

    Accepts None, a 2-element list / tuple of ints, or a string of exactly one pair:
    ``"(2048, 0)"``, ``"[2048,0]"`` or ``"2048,0"`` (yaml / CLI). Checked: ``left`` is
    -1 (no limit) or >= 0, and ``right`` is 0 (Muse Spark attention is causal). Anything
    else (nesting, unbalanced brackets, empty or extra items, floats, bools) raises a
    ValueError that names ``window_size``. The config and the FLOPs patch both use it.
    """
    if value is None:
        return None
    if isinstance(value, str):
        m = _WINDOW_RE.fullmatch(value)
        if not m:
            raise ValueError(f"window_size must be one pair '(left, right)' of ints, got {value!r}")
        parts = [int(p) for p in next(g for g in m.groups() if g).split(",")]
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(p, int) and not isinstance(p, bool) for p in value)
    ):
        parts = list(value)
    else:
        raise ValueError(f"window_size must be a pair (left, right) of ints, got {value!r}")
    left, right = parts
    if left < -1:
        raise ValueError(f"window_size[0] must be -1 (no limit) or >= 0, got {left}")
    if right != 0:
        raise ValueError(f"Muse Spark attention is causal: window_size[1] must be 0, got {right}")
    return left, right


def hf_sliding_window(value):
    """HF ``sliding_window`` (keys incl. the query) from a ``window_size`` in the presets'
    HF convention: the parsed ``left``, or None for no window (None / ``left == -1``).
    ``left == 0`` raises (an HF window counts the query, so it is >= 1)."""
    ws = parse_window_size(value)
    if ws is None or ws[0] == -1:
        return None
    if ws[0] < 1:
        raise ValueError(f"window_size[0] is an HF sliding_window and must be >= 1 (or -1), got {ws[0]}")
    return ws[0]
