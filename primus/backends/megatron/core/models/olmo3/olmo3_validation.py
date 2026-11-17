###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 settings validation, shared by the args patch, ``Olmo3TransformerConfig`` and the builder.

One module, so the three entry points refuse the same settings and read the window the same way.

Window contract
---------------
The window W is the HF ``sliding_window``: keys a query sees, the current token included (4096 for
OLMo-3). ``olmo3_sliding_window`` holds W and is required. Next to it, ``window_size`` is either an
int, W itself (the model preset writes ``window_size: 4096``, as ``config.json`` states
``sliding_window``), or a pair, the Megatron / TE ``(W - 1, 0)``; the shape says which, so no value
means two windows. ``window_size`` without ``olmo3_sliding_window`` is refused. Every reader returns
``(W - 1, 0)``. Both keys take integers only (``4096.0`` is refused for both, as Megatron's
``type=int``). For ``window_size`` the YAML null spellings a CLI override passes (``null``, ``~``,
``none``) mean unset.

``window_attn_skip_freq``
-------------------------
An int ``k`` with ``2 <= k <= num_layers`` (every k-th layer is full attention), or a 0/1 list of
``num_layers`` entries (1 = sliding) with at least one 0 and one 1. String forms from the CLI or a
quoted YAML string (``"4"``, ``"([1]*3 + [0])*8"``) use the grammar of Megatron's
``--window-attn-skip-freq`` (int and list literals, ``+``, ``*``, parentheses; spaces allowed). They
are read by :func:`_skip_freq_value`: Megatron's character check, then an ``ast`` evaluator that
checks the length of every ``+`` / ``*`` result before it builds it (a list literal is bounded by the input text and checked after parsing) (Megatron's ``eval`` has no bound, so
``[1]*10**10`` would exhaust memory).

Other string values are read with Primus's shared CLI reader (``arg_utils.parse_cli_overrides``,
"modern" mode), then checked strictly. Only None, an empty or blank string, and the string "None"
are unset (see :func:`is_unset`).

Values may come from the YAML (typed) or from ``--key value`` CLI overrides (strings), so the checks
accept both.

Refused settings (:data:`OLMO3_UNSUPPORTED`) include CUDA graphs: ``enable_cuda_graph``,
``external_cuda_graph`` and any ``cuda_graph_impl`` other than ``none`` (the OLMo-3 attention is not
tested under graph capture).
"""

import ast
import math
import os
import re

from primus.core.utils.arg_utils import parse_cli_overrides

__all__ = [
    "OLMO3_UNSUPPORTED",
    "YARN_BETA_FAST_DEFAULT",
    "YARN_BETA_SLOW_DEFAULT",
    "parse_olmo3_skip_freq",
    "olmo3_window_from_settings",
    "olmo3_window_keys",
    "olmo3_te_window",
    "check_olmo3_unsupported",
    "check_olmo3_cp",
    "check_olmo3_norm_epsilon",
    "check_olmo3_checkpoint_layout",
]

# HF's YaRN defaults (transformers ``_compute_yarn_parameters``) for a config that omits them; used
# by the converter and as the Olmo3TransformerConfig defaults.
YARN_BETA_FAST_DEFAULT = 32.0
YARN_BETA_SLOW_DEFAULT = 1.0


# ---------------------------------------------------------------------------
# value helpers. YAML values arrive typed; a string (a quoted YAML value, a CLI override, an
# environment substitution) is read with Primus's shared CLI reader (arg_utils.parse_cli_overrides,
# "modern" mode, as the CLI overrides themselves are read), then checked strictly.
# ---------------------------------------------------------------------------
def _read(v):
    """``v`` as Primus reads a CLI override value (strings only; other values are already typed)."""
    if isinstance(v, str):
        return parse_cli_overrides([f"value={v.strip()}"], type_mode="modern").get("value", "")
    return v


def is_unset(v):
    """None, an empty / blank string (an empty override), or a string the shared reader reads as
    None (``"None"``). Other strings (``none``, ``null``) are values, as in the shared reader."""
    return v is None or (isinstance(v, str) and (not v.strip() or _read(v) is None))


# every integer up to this magnitude is exact as a float; beyond it a float may not be the int meant
_MAX_EXACT_FLOAT_INT = 2**53


def _as_int(v, what):
    """An int, or an integral float (``4.0``), after :func:`_read`; anything else (a bool,
    ``4.5``, ``"two"``, an unresolved ``"${TP:1}"``) raises ValueError naming ``what``."""
    r = _read(v)
    if isinstance(r, float) and r.is_integer() and abs(r) <= _MAX_EXACT_FLOAT_INT:
        r = int(r)
    if isinstance(r, bool) or not isinstance(r, int):
        raise ValueError(f"OLMo-3 {what} must be an integer, got {v!r}")
    return r


def _strict_int(v, what):
    """An int (Megatron's ``type=int`` / ``tuple_type`` int), after :func:`_read`: floats (``4.0``),
    bools and empty or other strings raise ValueError naming ``what``."""
    r = _read(v)
    if isinstance(r, bool) or not isinstance(r, int):
        raise ValueError(f"OLMo-3 {what} must be an integer, got {v!r}")
    return r


def _truthy(v):
    """A switch after :func:`_read`: unset is off, else a bool or the int 0 / 1. Anything else
    (a typo such as ``"ture"``) raises ValueError, which check_olmo3_unsupported reports as the
    setting's error instead of reading it as on (``bool("ture")``) or off."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return False
    r = _read(v)  # parsed once; "None" reads as None (unset)
    if r is None:
        return False
    if isinstance(r, bool):
        return r
    if isinstance(r, int) and r in (0, 1):
        return bool(r)
    raise ValueError(f"switch must be true or false, got {v!r}")


def _as_float(v):
    r = _read(v)
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        raise ValueError
    return float(r)


# ---------------------------------------------------------------------------
# window_attn_skip_freq
# ---------------------------------------------------------------------------
# Upper bound for an int skip frequency, and for a list pattern's length when num_layers is
# unknown (no model has more layers).
_MAX_PATTERN = 4096
# Megatron's --window-attn-skip-freq character set (arguments._eval_pattern), spaces allowed
_PATTERN_CHARS = re.compile(r"[,\d\[\]\(\)\+\*\s]*")


def _eval_pattern(node, max_len):
    """Evaluate one node of a ``window_attn_skip_freq`` expression: int literals, lists of int
    literals, ``+`` and ``*`` (parentheses are only grouping in the AST). ``+`` and ``*`` results
    are length-checked against ``max_len`` before they are built, from their operands' lengths; a
    list literal is bounded by the input text and is checked after it is parsed."""
    if isinstance(node, ast.Expression):
        return _eval_pattern(node.body, max_len)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.List):
        items = [_eval_pattern(e, max_len) for e in node.elts]
        if any(type(i) is not int for i in items):
            raise ValueError("list entries must be ints")
        if len(items) > max_len:
            raise ValueError("too large")
        return items
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        a, b = _eval_pattern(node.left, max_len), _eval_pattern(node.right, max_len)
        if isinstance(node.op, ast.Add):
            if type(a) is not type(b):
                raise ValueError("cannot add a list and an int")
            if isinstance(a, list) and len(a) + len(b) > max_len:
                raise ValueError("too large")
            return a + b
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        lst, n = (a, b) if isinstance(a, list) else (b, a)
        if not isinstance(lst, list) or not isinstance(n, int):
            raise ValueError("cannot multiply two lists")
        if len(lst) * max(n, 0) > max_len:
            raise ValueError("too large")
        return lst * n
    raise ValueError(f"unsupported element {type(node).__name__}")


def _skip_freq_value(text, max_len=_MAX_PATTERN):
    """A ``window_attn_skip_freq`` string, as Megatron's ``--window-attn-skip-freq`` reads it
    (``moe_freq_type``): ``"4"``, or a list expression of int / list literals with ``+``, ``*`` and
    parentheses (spaces allowed). Megatron checks the characters and runs ``eval``. Here the
    expression is parsed with ``ast`` and evaluated by :func:`_eval_pattern`, which checks every
    ``+`` / ``*`` result's length against ``max_len`` before it builds it (a list literal is bounded by the input text and checked after parsing) (Megatron's parser has no bound, so
    ``[1]*10**10`` would exhaust memory). The caller checks the result (an int's bound, a list's
    length ``num_layers`` and 0 / 1 entries)."""
    s = text.strip()
    if "[" not in s:
        value = _read(s)  # Megatron reads it with int(): "4.0" is refused, as a float is
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"OLMo-3 window_attn_skip_freq: cannot parse {text!r} (not an int)")
        return value  # the caller (parse_olmo3_skip_freq) bounds it
    if not _PATTERN_CHARS.fullmatch(s) or "**" in s:
        raise ValueError(f"OLMo-3 window_attn_skip_freq: cannot parse {text!r}")
    try:
        value = _eval_pattern(ast.parse(s, mode="eval"), max_len)
    except (SyntaxError, RecursionError, MemoryError) as e:
        raise ValueError(
            f"OLMo-3 window_attn_skip_freq: cannot parse {text!r} ({type(e).__name__})"
        ) from None
    except ValueError as e:
        if str(e) == "too large":
            raise ValueError(
                f"OLMo-3 window_attn_skip_freq: {text!r} is too large (more than {max_len} entries)"
            ) from None
        raise ValueError(f"OLMo-3 window_attn_skip_freq: cannot parse {text!r} ({e})") from None
    # an expression with a list literal evaluates to a list (+ needs equal types, list * int is a list)
    return value


def parse_olmo3_skip_freq(skip_freq, num_layers=None):
    """``window_attn_skip_freq`` as an int ``k`` (``2 <= k <= num_layers``) or a 0/1 list.

    Accepts Megatron's ``--window-attn-skip-freq`` forms: an int, ``"4"``, or a list expression
    such as ``"([1]*3 + [0])*8"`` (evaluated by :func:`_skip_freq_value`, bounded by ``num_layers``). Floats are refused everywhere, both as a
    scalar (``4.0``) and as list entries (``[1.0, 0.0]``): Megatron's option is ``type=int`` or a list of ints. A list needs ``num_layers`` entries
    (when ``num_layers`` is known), at least one sliding layer (1) and one full layer (0).
    """
    if is_unset(skip_freq):
        raise ValueError(
            "OLMo-3 needs window_attn_skip_freq (4: every 4th layer full attention); without it "
            "every layer is sliding."
        )
    if num_layers is not None:
        num_layers = _as_int(num_layers, "num_layers")
    bound = num_layers if num_layers is not None else _MAX_PATTERN
    if isinstance(skip_freq, str):
        skip_freq = _skip_freq_value(skip_freq, bound)
    if isinstance(skip_freq, int) and not isinstance(skip_freq, bool):
        if skip_freq < 2:
            raise ValueError(
                f"OLMo-3 window_attn_skip_freq must be >= 2 (1 would make every layer full attention), "
                f"got {skip_freq}"
            )
        if skip_freq > bound:
            raise ValueError(
                f"OLMo-3 window_attn_skip_freq={skip_freq} > "
                + (
                    f"num_layers={num_layers}: no layer would be full attention"
                    if num_layers is not None
                    else f"{_MAX_PATTERN} (the bound when num_layers is unknown)"
                )
            )
        return skip_freq
    if isinstance(skip_freq, (list, tuple)):
        pattern = list(skip_freq)
        if num_layers is not None and len(pattern) != num_layers:
            raise ValueError(
                f"OLMo-3 window_attn_skip_freq list has {len(pattern)} entries for {num_layers} layers"
            )
        if any(isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1) for v in pattern):
            raise ValueError(f"OLMo-3 window_attn_skip_freq list must hold only 0 and 1, got {pattern}")
        if 1 not in pattern or 0 not in pattern:
            raise ValueError(
                "OLMo-3 window_attn_skip_freq list needs at least one sliding layer (1) and one full "
                f"attention layer (0), got {pattern}"
            )
        return pattern
    raise ValueError(f"OLMo-3 window_attn_skip_freq must be an int or a 0/1 list, got {skip_freq!r}")


# ---------------------------------------------------------------------------
# window_size / olmo3_sliding_window
# ---------------------------------------------------------------------------
_WINDOW_STR = re.compile(r"\s*(?:\(([^()\[\]]*)\)|\[([^()\[\]]*)\]|([^()\[\]]*))\s*")


def _parse_window(ws):
    if isinstance(ws, str):
        # "W, 0" bare (Megatron's tuple_type form) or inside one balanced pair of () or []
        m = _WINDOW_STR.fullmatch(ws)
        if m is None:
            raise ValueError(f"OLMo-3 window_size must be a (left, 0) pair, got {ws!r}")
        inner = next(g for g in m.groups() if g is not None)
        ws = inner.split(",")  # an empty entry ("4096,,0", "(4096,0,)") is refused below
    if not isinstance(ws, (list, tuple)):
        raise ValueError(f"OLMo-3 window_size must be a (left, 0) pair, got {ws!r}")
    if len(ws) != 2:
        raise ValueError(
            f"OLMo-3 window_size must be a (left, 0) pair, i.e. (W - 1, 0), got {tuple(ws)!r} "
            f"({len(ws)} entries). Set the window W in olmo3_sliding_window (window_size: null)."
        )
    ws = tuple(_strict_int(v, "window_size entry") for v in ws)
    if ws[1] != 0:
        raise ValueError(
            f"OLMo-3 window_size must be a (left, 0) pair, i.e. (W - 1, 0) (causal, no future keys), got {ws}"
        )
    return ws


def olmo3_window_keys(sliding_window) -> int:
    """The HF ``sliding_window`` W as a checked int (> 1: keys incl. the current token)."""
    if is_unset(sliding_window):
        raise ValueError("OLMo-3 needs olmo3_sliding_window (the HF sliding_window W)")
    w = _strict_int(sliding_window, "olmo3_sliding_window")
    if w <= 1:
        raise ValueError(f"OLMo-3 olmo3_sliding_window must be > 1 keys (current token included), got {w}")
    return w


def olmo3_te_window(sliding_window):
    """The Megatron / TE ``window_size`` ``(W - 1, 0)`` for the HF ``sliding_window`` W."""
    return (olmo3_window_keys(sliding_window) - 1, 0)


def _window_value(ws):
    """``window_size`` as given: an int (or a bare-int string) is W, the HF window in keys; a pair
    (list, tuple or "(a, b)" string) is the Megatron / TE ``(left, right)`` form. Megatron itself
    only ever holds a pair, so the two shapes never mean the same thing."""
    if isinstance(ws, bool):
        raise ValueError(f"OLMo-3 window_size must be W (int) or a Megatron/TE (left, 0) pair, got {ws!r}")
    if isinstance(ws, int):
        return ("keys", ws)
    if isinstance(ws, str) and re.fullmatch(r"\s*-?\d+\s*", ws):
        return ("keys", int(ws))
    if isinstance(ws, float) or (isinstance(ws, str) and isinstance(_read(ws), float)):
        # a scalar that is not an integer W ("4096.0", 4096.5): say so, not "must be a pair"
        raise ValueError(f"OLMo-3 window_size W must be an integer, got {ws!r}")
    return ("te", _parse_window(ws))


def olmo3_window_from_settings(window_size, sliding_window):
    """``(megatron_window, W)`` from ``window_size`` and ``olmo3_sliding_window`` (W).

    The window is ``olmo3_sliding_window`` W (HF form: keys incl. the current token); it is
    required. ``window_size`` is only checked to agree with it, in one of two forms told apart by
    shape, never by value: an int is W itself (the model preset writes ``window_size: 4096``, as
    ``config.json`` states ``sliding_window``), a pair is the Megatron / TE ``(W - 1, 0)`` (what
    the args patch stores and TE receives); unset is fine. Anything else is refused, e.g. the
    preset's ``4096`` next to a changed W. ``window_size`` alone is refused. The result is always
    ``(W - 1, 0)``.
    """
    if is_unset(sliding_window):
        alone = "" if is_unset(window_size) else f"; got window_size={window_size!r} without it"
        raise ValueError(
            "OLMo-3 needs olmo3_sliding_window: W (HF sliding_window, keys incl. the current token; "
            f"4096 for OLMo-3, the model preset sets it){alone}. Set olmo3_sliding_window."
        )
    w = olmo3_window_keys(sliding_window)
    if isinstance(window_size, str) and window_size.strip().lower() in ("null", "none", "~"):
        window_size = None  # YAML's null spellings, as a CLI override passes them
    if not is_unset(window_size):
        form, value = _window_value(window_size)
        if (form == "keys" and value != w) or (form == "te" and value != (w - 1, 0)):
            raise ValueError(
                f"OLMo-3 window_size {window_size!r} disagrees with olmo3_sliding_window={w}: it must be "
                f"{w} (W, as the preset writes it), the Megatron/TE pair {[w - 1, 0]}, or unset. To "
                "change the window, set olmo3_sliding_window and window_size: null (or the new W)."
            )
    return (w - 1, 0), w


# ---------------------------------------------------------------------------
# settings the OLMo-3 model does not implement (one table for args and config)
# ---------------------------------------------------------------------------
def _off(v):
    return not _truthy(v)


def _one(v):
    return is_unset(v) or _as_float(v) == 1.0


def _is(*allowed):
    return lambda v: v in allowed


def _unset_or(*allowed):
    """``None`` (not set), or exactly one of ``allowed``. Megatron compares these enum options
    exactly (e.g. ``normalization == "RMSNorm"``), so any other value is refused, including the
    strings ``"none"`` / ``"null"`` / ``"~"``: for these keys they are not "unset" (e.g.
    ``position_embedding_type: none`` is Megatron's "no position embedding"), and Megatron sees the
    string, not None. So ``is_unset`` is not used here."""
    return lambda v: v is None or v in allowed


def _no_rope_off(v):
    """``no_rope_freq`` that skips RoPE on no layer: unset, 0, or an all-zero list (also as a
    string, e.g. ``"[0,0,0,0]"``); read once with :func:`_read`, then branched on the parsed value."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return True
    r = _read(v)  # parsed once; "None" reads as None (unset)
    if r is None:
        return True
    if isinstance(r, (list, tuple)):
        return not any(_as_float(x) for x in r)
    return _as_float(r) == 0


def _silu(v):
    import torch.nn.functional as F

    return v is F.silu


# (name, allowed(value) -> bool, why). A name is checked on every object that has it (args and/or
# TransformerConfig); a missing name is skipped. Values may be YAML values or CLI strings.
OLMO3_UNSUPPORTED = (
    # CUDA graphs: the OLMo-3 attention (TP all-reduce inside the full-width q/k norm, per-layer
    # RoPE table selection) is not tested under graph capture
    ("enable_cuda_graph", _off, "OLMo-3 is not tested with CUDA graphs; enable_cuda_graph must be false"),
    ("external_cuda_graph", _off, "OLMo-3 is not tested with CUDA graphs; external_cuda_graph must be false"),
    (
        "cuda_graph_impl",
        _unset_or("none"),
        "OLMo-3 is not tested with CUDA graphs; cuda_graph_impl must be none",
    ),
    # attention
    ("qk_layernorm", _off, "OLMo-3 owns full-width q/k norms; keep the per-head qk_layernorm false"),
    ("qk_l2_norm", _off, "OLMo-3 q/k norms are full-width RMSNorm, not L2 norm; qk_l2_norm must be false"),
    ("multi_latent_attention", _off, "OLMo-3 must keep multi_latent_attention false"),
    ("softmax_type", _unset_or("vanilla"), "OLMo-3 uses the plain softmax; softmax_type must be vanilla"),
    ("fused_single_qkv_rope", _off, "OLMo-3 normalises q/k before RoPE; fused_single_qkv_rope must be false"),
    ("attention_output_gate", _off, "OLMo-3 has no gated attention output"),
    ("flash_decode", _off, "OLMo-3 per-layer RoPE tables have no flash_decode path"),
    (
        "transformer_impl",
        _unset_or("transformer_engine"),
        "OLMo-3 layers are built on Transformer Engine; transformer_impl must be transformer_engine",
    ),
    (
        "use_turbo_attention",
        _off,
        "OLMo-3 uses sliding-window attention, which Primus-Turbo attention does not support; "
        "set use_turbo_attention: false (enable_primus_turbo may stay on)",
    ),
    # position embedding / RoPE
    ("position_embedding_type", _unset_or("rope"), "OLMo-3 uses RoPE; position_embedding_type must be rope"),
    (
        "rotary_percent",
        _one,
        "OLMo-3 applies RoPE (and YaRN) to the full head dim; rotary_percent must be 1.0",
    ),
    ("rotary_interleaved", _off, "OLMo-3 uses rotate-half RoPE; rotary_interleaved must be false"),
    (
        "rotary_seq_len_interpolation_factor",
        _one,
        "OLMo-3 scales RoPE by YaRN only; unset rotary_seq_len_interpolation_factor",
    ),
    ("use_rope_scaling", _off, "OLMo-3 scales RoPE by YaRN only; use_rope_scaling must be false"),
    ("no_rope_freq", _no_rope_off, "OLMo-3 applies RoPE on every layer; unset no_rope_freq"),
    (
        "apply_rope_fusion",
        _off,
        "OLMo-3 RoPE runs unfused (the path the HF parity tests cover; the model preset sets "
        "apply_rope_fusion: false); set apply_rope_fusion: false",
    ),
    (
        "olmo3_rope_layout",
        _is("per_type", "yarn_all"),
        "olmo3_rope_layout must be per_type (the trained model) or yarn_all; auto depends on the "
        "installed transformers (reference builds resolve it first, as derive_megatron_fields does) "
        "and None leaves the layout implicit",
    ),
    # norms / MLP / embeddings
    ("normalization", _unset_or("RMSNorm"), "OLMo-3 uses RMSNorm; normalization must be RMSNorm"),
    ("layernorm_zero_centered_gamma", _off, "OLMo-3 RMSNorm weights are not zero-centred"),
    (
        "apply_residual_connection_post_layernorm",
        _off,
        "OLMo-3 adds the residual to the un-normalised input; apply_residual_connection_post_layernorm must be false",
    ),
    ("swiglu", _truthy, "OLMo-3 MLP is SwiGLU; swiglu must be true"),
    ("gated_linear_unit", _truthy, "OLMo-3 MLP is SwiGLU; gated_linear_unit must be true"),
    ("activation_func", _silu, "OLMo-3 MLP is SwiGLU; activation_func must be F.silu"),
    (
        "glu_linear_offset",
        lambda v: is_unset(v) or _as_float(v) == 0.0,
        "OLMo-3 SwiGLU is silu(x0) * x1; glu_linear_offset must be 0",
    ),
    (
        "activation_func_clamp_value",
        is_unset,
        "OLMo-3 SwiGLU does not clamp linear_fc1; unset activation_func_clamp_value",
    ),
    (
        "use_te_activation_func",
        _off,
        "OLMo-3 MLP runs the eager SwiGLU the parity tests cover; use_te_activation_func must be false",
    ),
    ("add_bias_linear", _off, "OLMo-3 linear layers have no bias; add_bias_linear must be false"),
    ("add_qkv_bias", _off, "OLMo-3 q/k/v projections have no bias; add_qkv_bias must be false"),
    (
        "untie_embeddings_and_output_weights",
        _truthy,
        "OLMo-3 has separate input and output embeddings; untie_embeddings_and_output_weights must be true",
    ),
    # dense
    (
        "num_experts",
        lambda v: is_unset(v) or _as_float(v) == 0,
        "OLMo-3 is dense (Olmo3MLP); unset num_experts",
    ),
    (
        "num_moe_experts",
        lambda v: is_unset(v) or _as_float(v) == 0,
        "OLMo-3 is dense (Olmo3MLP); unset num_moe_experts",
    ),
)

_MISSING = object()


def check_olmo3_unsupported(obj):
    """Raise ``ValueError`` for any :data:`OLMO3_UNSUPPORTED` setting ``obj`` (args or config) has,
    and for the checks that look at several settings together: context parallelism with the sliding
    window (:func:`check_olmo3_cp`), KV heads per TP rank for the full-width k_norm
    (:func:`check_olmo3_kv_heads_per_tp`), selective layernorm recompute
    (:func:`check_olmo3_recompute_modules`) and ``norm_epsilon`` vs ``layernorm_epsilon``
    (:func:`check_olmo3_norm_epsilon`)."""
    for name, ok, why in OLMO3_UNSUPPORTED:
        value = getattr(obj, name, _MISSING)
        if value is _MISSING:
            continue
        try:
            good = ok(value)
        except (TypeError, ValueError):
            good = False
        if not good:
            raise ValueError(f"{why} (got {name}={value!r})")
    check_olmo3_cp(obj)
    check_olmo3_kv_heads_per_tp(obj)
    check_olmo3_recompute_modules(obj)
    check_olmo3_norm_epsilon(obj)


def check_olmo3_norm_epsilon(obj):
    """The OLMo-3 model reads ``layernorm_epsilon`` (Megatron's dest for ``--norm-epsilon``). A
    ``norm_epsilon`` that differs from it would be silently ignored, so it is refused."""
    a, b = getattr(obj, "layernorm_epsilon", None), getattr(obj, "norm_epsilon", None)
    if is_unset(a) or is_unset(b):
        return
    try:
        same = math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError(
            f"OLMo-3 reads layernorm_epsilon (got {a!r}); norm_epsilon={b!r} differs and would be "
            "ignored. Set both to the same value (the HF rms_norm_eps)."
        )


def check_olmo3_recompute_modules(obj):
    """Selective recompute of ``layernorm`` would recompute nothing for OLMo-3: upstream
    TransformerLayer recomputes only input_layernorm / pre_mlp_layernorm, which are IdentityOp
    here (OLMo-3's norms are post-norms inside the attention and the MLP). Refused, so a user does
    not silently lose the memory saving; ``core_attn`` and ``mlp`` work as for any model."""
    if getattr(obj, "recompute_granularity", None) != "selective":
        return
    modules = getattr(obj, "recompute_modules", None)
    if is_unset(modules):
        return
    if isinstance(modules, str):
        parsed = _read(modules)  # "['layernorm']" (a quoted list) parses to a list
        if isinstance(parsed, (list, tuple)):
            modules = parsed
        else:  # "layernorm mlp", "core_attn,layernorm", "[layernorm,mlp]" (unquoted list)
            modules = modules.strip().strip("[]()").replace(",", " ").split()
    if "layernorm" in [str(m).strip().strip("'\"") for m in modules]:
        raise ValueError(
            "OLMo-3 has no input / pre-MLP layernorm (its norms are post-norms), so "
            "recompute_modules: [layernorm] would recompute nothing; remove 'layernorm' "
            "(use core_attn and/or mlp, or recompute_granularity: full)"
        )


def check_olmo3_kv_heads_per_tp(obj):
    """The full-width k_norm spans all KV heads, sharded over TP. With num_query_groups < TP,
    Megatron replicates each KV head on TP / num_query_groups ranks, so the TP all-reduce of the
    k sum of squares would count each head several times and the k_norm weight shards would not
    be TP-disjoint. That layout is refused (the released OLMo-3 7B has 32 KV heads: TP must divide 32, i.e. 1, 2, 4, 8, 16 or 32).
    """
    tp = getattr(obj, "tensor_model_parallel_size", 1)
    gqa = getattr(obj, "group_query_attention", _MISSING)
    ng = getattr(obj, "num_query_groups", None)
    # args: num_query_groups only applies with group_query_attention (Megatron's
    # core_transformer_config_from_args uses num_attention_heads otherwise, whatever
    # --num-query-groups holds); a TransformerConfig has no such switch and holds the real value
    if is_unset(ng) or (gqa is not _MISSING and not _truthy(gqa)):
        ng = getattr(obj, "num_attention_heads", None)
    if is_unset(tp) or is_unset(ng):
        return
    tp, ng = _as_int(tp, "tensor_model_parallel_size"), _as_int(ng, "num_query_groups")
    if tp <= 1:
        return
    if ng % tp:
        raise ValueError(
            f"OLMo-3: tensor_model_parallel_size ({tp}) must divide num_query_groups ({ng}): Megatron "
            "shards whole KV heads over TP, and the full-width k_norm statistic needs each KV head on "
            "exactly one TP rank (no replication when num_query_groups < TP)"
        )


def check_olmo3_cp(obj):
    """TE's sliding-window attention under context parallelism needs cp_comm_type a2a or all_gather."""
    cp = getattr(obj, "context_parallel_size", 1)
    cp = 1 if is_unset(cp) else _as_int(cp, "context_parallel_size")
    cp_comm = getattr(obj, "cp_comm_type", None)
    cp_comm = "p2p" if is_unset(cp_comm) else cp_comm
    cp_comm = list(cp_comm) if isinstance(cp_comm, (list, tuple)) else [cp_comm]
    if cp > 1 and any(c not in ("a2a", "all_gather") for c in cp_comm):
        raise ValueError(
            f"OLMo-3 sliding-window attention with context_parallel_size={cp} needs "
            f"cp_comm_type a2a or all_gather, got {cp_comm}"
        )


# the tracker file Megatron writes next to its iter_* directories (checkpointing.py:
# get_checkpoint_tracker_filename)
# The same name as native_convert_common.TRACKER_FILENAME (a test keeps them equal); kept here so
# this module stays light (no torch import for the args patch).
CHECKPOINT_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def _single_shard_legacy_checkpoint(path, ckpt_step=None):
    """True if the iteration Megatron would load from ``path`` (``ckpt_step`` if set, else the
    tracker's) is a legacy ``torch`` checkpoint with exactly one shard, ``mp_rank_00``: the layout
    the native OLMo-3 converter writes (TP = PP = 1). A lone ``mp_rank_00_000`` is not read as
    single-shard: Megatron adds the ``_{pp:03d}`` suffix only for PP > 1, so it may be the first
    shard of a PP save. Megatron's own load reports any other mismatch."""
    tracker = os.path.join(path, CHECKPOINT_TRACKER_FILENAME)
    if not os.path.isfile(tracker):
        return False
    if not is_unset(ckpt_step) and _read(ckpt_step):
        step = f"iter_{_as_int(ckpt_step, 'ckpt_step'):07d}"
    else:
        with open(tracker) as f:
            it = f.read().strip()
        step = "release" if it == "release" else f"iter_{int(it):07d}" if it.isdigit() else None
    step_dir = os.path.join(path, step) if step else None
    if step_dir is None or not os.path.isdir(step_dir):
        return False
    shards = sorted(d for d in os.listdir(step_dir) if d.startswith("mp_rank_"))
    return shards == ["mp_rank_00"]


def _checkpoint_megatron_loads(args):
    """``(key, path)`` of the checkpoint Megatron's ``load_checkpoint`` reads: ``load`` when it holds
    a tracker file, else ``pretrained_checkpoint`` (checkpointing.py: pretrained_checkpoint is read
    only when ``checkpoint_exists(load)`` is false); ``(None, None)`` when neither is set."""
    load = getattr(args, "load", None)
    if isinstance(load, str) and load and os.path.isfile(os.path.join(load, CHECKPOINT_TRACKER_FILENAME)):
        return "load", load
    pretrained = getattr(args, "pretrained_checkpoint", None)
    if isinstance(pretrained, str) and pretrained:
        return "pretrained_checkpoint", pretrained
    return None, None


def check_olmo3_checkpoint_layout(args):
    """Fail fast before the model is built: with ``ckpt_format: torch`` the native OLMo-3 converter
    (posttrain hook) writes a single-shard legacy ``torch`` checkpoint (``mp_rank_00``, TP = PP = 1;
    its default ``torch_dist`` is resharded on load and passes). Megatron cannot reshard
    a legacy checkpoint, so loading one with tensor or pipeline parallel size > 1 is refused here
    with the fix, instead of failing later on a missing ``mp_rank_*`` file. Only the checkpoint
    Megatron will actually load is inspected (:func:`_checkpoint_megatron_loads`)."""
    sizes = []
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size"):
        value = getattr(args, key, None)
        size = 1 if is_unset(value) else _as_int(value, key)  # an explicit 0 is not read as 1
        if size < 1:
            raise ValueError(f"OLMo-3 {key} must be >= 1, got {value!r}")
        sizes.append(size)
    tp, pp = sizes
    if tp == 1 and pp == 1:
        return
    key, path = _checkpoint_megatron_loads(args)
    if path and _single_shard_legacy_checkpoint(path, getattr(args, "ckpt_step", None)):
        raise ValueError(
            f"OLMo-3: {key}={path} is a single-shard legacy torch checkpoint (mp_rank_00 only, as the "
            f"native converter writes), but this run uses tensor_model_parallel_size={tp}, "
            f"pipeline_model_parallel_size={pp}. Fine-tune it with TP=PP=1, or first load it at "
            "TP=PP=1 and re-save it with ckpt_format: torch_dist, then load that checkpoint."
        )
