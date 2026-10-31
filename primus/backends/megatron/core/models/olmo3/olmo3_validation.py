###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 settings validation, shared by the args patch, ``Olmo3TransformerConfig`` and the builder.

One module, so the three entry points refuse the same settings and read the window the same way.

Window contract
---------------
Two conventions exist for the sliding window:

* HF ``sliding_window`` W: keys a query sees, the current token included (4096 for OLMo-3).
* Megatron / TE ``window_size = (left, right)`` = ``(W - 1, 0)``.

Two settings say which form ``window_size`` holds; neither is ever guessed:

* ``olmo3_window_size_form``: ``hf`` (``window_size = (W, 0)``, the model preset) or ``megatron``
  (``(W - 1, 0)``). The args patch converts an ``hf`` window once and then sets ``megatron``.
* ``olmo3_sliding_window = W``: the HF window itself (the converter, the args patch output and
  saved checkpoint args carry it; the model preset does not set it, so the preset states the window
  once, as ``window_size: [W, 0]`` with form ``hf``). With it, ``window_size`` must be
  ``(W - 1, 0)`` (or ``(W, 0)`` when the form says ``hf``), or unset (then it is derived). A
  ``window_size`` that disagrees is refused, never re-read in the other form.

A ``window_size`` with neither setting is refused: reading it as HF would silently shorten a
converted window by one key when both settings are lost (they are not argparse options, so
``use_checkpoint_args`` or a CLI ``null`` can drop them). The converter, the args patch output and
saved checkpoint args carry both settings. ``Olmo3TransformerConfig`` requires
``olmo3_sliding_window`` (W); the form decides how ``window_size`` is read.

``window_attn_skip_freq``
-------------------------
An int ``k`` with ``2 <= k <= num_layers`` (every k-th layer is full attention), or a 0/1 list of
``num_layers`` entries (1 = sliding) with at least one 0 and one 1. String forms from the CLI
(``"4"``, ``"([1]*3+[0])*8"``) are parsed by a small grammar (integers, lists, ``+``, ``*``,
parentheses), never by ``eval``, with caps on the input and result sizes.

Values may come from the YAML (typed) or from ``--key value`` CLI overrides (strings), so the checks
accept both.
"""

import ast
import math

from primus.core.utils.arg_utils import as_int, is_unset, parse_optional_switch

__all__ = [
    "OLMO3_UNSUPPORTED",
    "YARN_BETA_FAST_DEFAULT",
    "YARN_BETA_SLOW_DEFAULT",
    "parse_olmo3_skip_freq",
    "olmo3_window_from_settings",
    "check_olmo3_unsupported",
    "check_olmo3_cp",
    "check_olmo3_norm_epsilon",
]

# HF's YaRN defaults (transformers ``_compute_yarn_parameters``) for a config that omits them; used
# by the converter and as the Olmo3TransformerConfig defaults.
YARN_BETA_FAST_DEFAULT = 32.0
YARN_BETA_SLOW_DEFAULT = 1.0


_MAX_EXPR_CHARS = 4096
_MAX_LIST_LEN = 65536
_MAX_INT = 1 << 20
_MAX_DEPTH = 32


# ---------------------------------------------------------------------------
# value helpers (YAML values or CLI strings)
# ---------------------------------------------------------------------------
def _as_int(v, what):
    """The shared ``as_int`` with the OLMo-3 prefix in the message."""
    return as_int(v, f"OLMo-3 {what}")


def _truthy(v):
    """A switch value: unset (:func:`is_unset`) is False, else the shared ``parse_switch`` (a typo
    such as "ture" raises ValueError, which check_olmo3_unsupported reports as the setting's error
    instead of reading it as on or off)."""
    return parse_optional_switch(v, "switch")


def _as_float(v):
    if isinstance(v, bool):
        raise ValueError
    return float(v)


# ---------------------------------------------------------------------------
# window_attn_skip_freq: safe parser
# ---------------------------------------------------------------------------
def _skip_freq_value(text):
    """Evaluate a ``window_attn_skip_freq`` string (``"4"``, ``"([1]*3+[0])*8"``, ``"[1,0,]*16"``)
    without ``eval``: Python's own parser, then only int / list literals (a trailing comma is
    allowed), unary minus, ``+`` (add ints, join lists) and ``*`` (multiply ints, repeat a list).
    Floats are refused here as in a list, so every string form has one rule. Sizes are checked
    before a list is built."""

    def error(why):
        return ValueError(f"OLMo-3 window_attn_skip_freq: cannot parse {text!r} ({why})")

    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, RecursionError, MemoryError, ValueError) as e:
        raise error(getattr(e, "msg", None) or type(e).__name__) from None

    def ev(node, depth=0):
        if depth > _MAX_DEPTH:
            raise error("nested too deeply")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > _MAX_INT:
                raise error("value too large")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = ev(node.operand, depth + 1)
            if isinstance(v, int):
                return -v
        elif isinstance(node, ast.List):
            items = [ev(e, depth + 1) for e in node.elts]
            if all(isinstance(v, int) for v in items):
                return items
            raise error("list entries must be integers")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            a, b = ev(node.left, depth + 1), ev(node.right, depth + 1)
            if isinstance(node.op, ast.Add) and type(a) is type(b):
                if isinstance(a, list) and len(a) + len(b) > _MAX_LIST_LEN:
                    raise error(f"the list would have more than {_MAX_LIST_LEN} entries")
                return a + b
            if isinstance(node.op, ast.Mult) and not (isinstance(a, list) and isinstance(b, list)):
                if isinstance(a, int) and isinstance(b, int):
                    if abs(a * b) > _MAX_INT:
                        raise error("value too large")
                    return a * b
                items, n = (a, b) if isinstance(a, list) else (b, a)
                if len(items) * max(n, 0) > _MAX_LIST_LEN:
                    raise error(f"the list would have more than {_MAX_LIST_LEN} entries")
                return items * n
            raise error(f"'{'+' if isinstance(node.op, ast.Add) else '*'}' cannot combine these values")
        raise error(f"only integers, lists, unary -, + and * are allowed, got {ast.unparse(node)!r}")

    return ev(tree.body)


def parse_olmo3_skip_freq(skip_freq, num_layers=None):
    """``window_attn_skip_freq`` as an int ``k`` (``2 <= k <= num_layers``) or a 0/1 list.

    Accepts Megatron's ``--window-attn-skip-freq`` forms: an int, ``"4"``, or a list expression
    such as ``"([1]*3+[0])*8"`` (parsed without ``eval``). Floats are refused everywhere (``4.0``
    as in ``[1.0, 0.0]``): Megatron's option is ``type=int`` or a list of ints. A list needs ``num_layers`` entries
    (when ``num_layers`` is known), at least one sliding layer (1) and one full layer (0).
    """
    if is_unset(skip_freq):
        raise ValueError(
            "OLMo-3 needs window_attn_skip_freq (4: every 4th layer full attention); without it "
            "every layer is sliding."
        )
    if num_layers is not None:
        num_layers = _as_int(num_layers, "num_layers")
    if isinstance(skip_freq, str):
        if len(skip_freq) > _MAX_EXPR_CHARS:
            raise ValueError(f"OLMo-3 window_attn_skip_freq string longer than {_MAX_EXPR_CHARS} characters")
        skip_freq = _skip_freq_value(skip_freq)
    if isinstance(skip_freq, int) and not isinstance(skip_freq, bool):
        if skip_freq < 2:
            raise ValueError(
                f"OLMo-3 window_attn_skip_freq must be >= 2 (1 would make every layer full attention), "
                f"got {skip_freq}"
            )
        if num_layers is not None and skip_freq > num_layers:
            raise ValueError(
                f"OLMo-3 window_attn_skip_freq={skip_freq} > num_layers={num_layers}: no layer would be "
                "full attention"
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
def _parse_window(ws):
    if isinstance(ws, str):
        ws = [v for v in ws.strip().strip("()[]").split(",") if v.strip()]
    if not isinstance(ws, (list, tuple)):
        raise ValueError(f"OLMo-3 window_size must be a (left, 0) pair, got {ws!r}")
    ws = tuple(_as_int(v, "window_size entry") for v in ws)
    if len(ws) != 2 or ws[1] != 0:
        raise ValueError(f"OLMo-3 window_size must be (W, 0) (causal, no future keys), got {ws}")
    return ws


_WINDOW_FORMS = ("hf", "megatron")


def olmo3_window_from_settings(window_size, sliding_window, form=None):
    """``(megatron_window, W)`` from ``window_size``, ``olmo3_sliding_window`` (W) and
    ``olmo3_window_size_form``.

    * ``olmo3_sliding_window`` W set: ``window_size`` must agree with it in its declared form: ``(W - 1, 0)``
      (``form`` unset or ``megatron``) or ``(W, 0)`` (``form: hf``), or be unset.
    * W unset: ``form`` says how to read ``window_size``. ``"hf"`` (the model preset):
      ``(W, 0)``; ``"megatron"``: ``(W - 1, 0)``. In the HF form an odd W is refused: it is how a
      Megatron/TE value ``(W - 1, 0)`` of an even W looks (``[4095, 0]`` for 4096), and a silent
      W - 1 would cost one key; an odd W is accepted when ``olmo3_sliding_window`` names it. Without ``form`` the window is ambiguous
      (a lost W would turn a converted ``(W - 1, 0)`` into ``(W - 2, 0)``) and is refused.

    ``megatron_window`` is ``(W - 1, 0)``.
    """
    ws = None if is_unset(window_size) else _parse_window(window_size)
    form = None if is_unset(form) else str(form).strip().lower()
    if form is not None and form not in _WINDOW_FORMS:
        raise ValueError(f"OLMo-3 olmo3_window_size_form must be one of {_WINDOW_FORMS}, got {form!r}")
    if is_unset(sliding_window):
        if ws is None:
            raise ValueError(
                "OLMo-3 needs window_size: [sliding_window, 0] (HF convention, e.g. [4096, 0]) with "
                "olmo3_window_size_form: hf; without it every layer runs full attention, which is a "
                "different model."
            )
        if form is None:
            raise ValueError(
                f"OLMo-3 window_size {ws} without olmo3_sliding_window or olmo3_window_size_form is "
                "ambiguous (HF (W, 0) or Megatron/TE (W - 1, 0)). Set olmo3_window_size_form: hf for "
                "the HF form (the model preset does), or olmo3_sliding_window: W for the Megatron form."
            )
        if form == "hf" and ws[0] % 2 == 1:
            raise ValueError(
                f"OLMo-3 window_size {ws} is read in the HF form (olmo3_window_size_form: hf, as the "
                f"model preset sets), so W = {ws[0]}, an odd window. That is what the Megatron/TE form "
                f"(W - 1, 0) of W = {ws[0] + 1} looks like, so it is not guessed: for W = {ws[0] + 1} "
                f"override window_size: [{ws[0] + 1}, 0] (HF form), or set olmo3_window_size_form: "
                f"megatron; for an odd W = {ws[0]} confirm it with olmo3_sliding_window: {ws[0]}."
            )
        w = ws[0] if form == "hf" else ws[0] + 1
    else:
        w = _as_int(sliding_window, "olmo3_sliding_window")
        allowed = {w} if form == "hf" else {w - 1}
        if ws is not None and ws[0] not in allowed:
            raise ValueError(
                f"OLMo-3 window_size {ws} disagrees with olmo3_sliding_window={w} "
                f"(olmo3_window_size_form={form!r}): window_size must be "
                f"{(w, 0) if form == 'hf' else (w - 1, 0)} in that form. The two come from different "
                "sources (e.g. an override changed only one of them); the window is not guessed. To "
                "change the window, set olmo3_sliding_window: W and window_size: null (or both, "
                "consistently); with the model preset, override window_size: [W, 0]."
            )
    if w <= 1:
        raise ValueError(f"OLMo-3 sliding_window must be > 1 keys (current token included), got {w}")
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


def _unset_or_text(*allowed):
    """Unset, or a string in ``allowed`` after strip / lower. Used only for the Primus option
    olmo3_window_size_form, which olmo3_window_from_settings reads after strip / lower; the rule of
    each check is the rule of the code that reads the value."""
    return lambda v: is_unset(v) or (isinstance(v, str) and v.strip().lower() in allowed)


def _silu(v):
    import torch.nn.functional as F

    return v is F.silu


# (name, allowed(value) -> bool, why). A name is checked on every object that has it (args and/or
# TransformerConfig); a missing name is skipped. Values may be YAML values or CLI strings.
OLMO3_UNSUPPORTED = (
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
    (
        "no_rope_freq",
        lambda v: is_unset(v)
        or (not isinstance(v, (list, tuple)) and _as_float(v) == 0)
        or (isinstance(v, (list, tuple)) and not any(v)),
        "OLMo-3 applies RoPE on every layer; unset no_rope_freq",
    ),
    (
        "apply_rope_fusion",
        _off,
        "OLMo-3 RoPE runs unfused (the path the HF parity tests cover; the model preset sets "
        "apply_rope_fusion: false); set apply_rope_fusion: false",
    ),
    (
        "olmo3_window_size_form",
        _unset_or_text(*_WINDOW_FORMS),
        "olmo3_window_size_form must be hf (window_size = (W, 0)) or megatron ((W - 1, 0))",
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
    """Raise ``ValueError`` for any :data:`OLMO3_UNSUPPORTED` setting ``obj`` (args or config) has."""
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
