###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for SmolLM3
(``model_type: smollm3`` / ``SmolLM3ForCausalLM``).

SmolLM3's weight layout is exactly Llama's (q/k/v/o, gate/up/down, two RMSNorms
per layer, ``model.norm``, ``model.embed_tokens``; tied embeddings, so there is
no ``lm_head.weight``). The one architectural difference is NoPE: layers where
``no_rope_layers[i] == 0`` skip RoPE. In HF,

    no_rope_layers[i] = int((i + 1) % no_rope_layer_interval != 0)   # 1 = RoPE

Megatron's ``TransformerConfig.no_rope_freq`` uses the opposite convention
(1 = skip RoPE), so we translate it here and reuse the Llama mapping.
"""

import sys
import warnings

from primus.backends.megatron.checkpoint import native_loader_llama as _llama

# HF SmolLM3Config defaults (transformers 4.57 / 5.x). A config.json that OMITS a key gets
# these, so the converter builds the model HF builds. They differ from the Llama
# converter's fallbacks (rope_theta 1e4, untied). An explicit null is not simply
# replaced by a default: it is resolved as HF resolves it (bias null -> False, kv-heads
# null -> MHA) or rejected; see with_hf_defaults().
HF_SMOLLM3_DEFAULTS = {
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 2000000.0,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "no_rope_layer_interval": 4,
}
# Keys whose HF default is a model size or a value the SmolLM3-3B release overrides: a
# config.json without them is warned about (the bias / activation defaults are not).
_WARN_IF_MISSING = tuple(
    k for k in HF_SMOLLM3_DEFAULTS if k not in ("attention_bias", "mlp_bias", "hidden_act")
)
HF_DEFAULT_NO_ROPE_LAYER_INTERVAL = HF_SMOLLM3_DEFAULTS["no_rope_layer_interval"]
# HF default for a MISSING num_key_value_heads (an explicit null means plain MHA, so it
# is not in HF_SMOLLM3_DEFAULTS, whose null handling differs; see with_hf_defaults).
HF_DEFAULT_NUM_KEY_VALUE_HEADS = 4
# Keys where HF treats an explicit null like false (nn.Linear(bias=None) has no bias).
_NULL_MEANS_FALSE = ("attention_bias", "mlp_bias")
# Keys a null is allowed for, because something else gives the value (see with_hf_defaults).
_NULL_RESOLVED_ELSEWHERE = ("rope_theta", "no_rope_layer_interval")

LOG_NAME = "smollm3"  # log prefix in native_loader_llama.convert


def _binary(values, what):
    """Return ``values`` as a list of ints; every entry must be exactly 0 or 1 (int or bool)."""
    out = []
    for v in values:
        if isinstance(v, bool) or (isinstance(v, int) and v in (0, 1)):
            out.append(int(v))
        else:
            raise ValueError(f"{what} must hold only 0/1 integers, got {v!r}")
    return out


def nope_pattern(cfg):
    """Return Megatron's per-layer ``no_rope_freq`` list (1 = NoPE layer), or None (all RoPE).

    Follows HF ``SmolLM3Config``: an explicit ``no_rope_layers`` (1 = RoPE) is used as
    given, and ``no_rope_layer_interval`` is then ignored (HF always saves the interval,
    so it is not checked against a custom list). Only when ``no_rope_layers`` is
    missing or null is the list built from the interval: a missing interval takes HF's
    default (4); an explicit null interval is rejected, as HF itself fails on it
    (``(i + 1) % None``).
    Megatron uses the opposite polarity (1 = skip RoPE), so the result is inverted.
    """
    n = int(cfg["num_hidden_layers"])
    rope_layers = cfg.get("no_rope_layers")
    if rope_layers is None:
        interval = cfg.get("no_rope_layer_interval", HF_DEFAULT_NO_ROPE_LAYER_INTERVAL)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(f"no_rope_layer_interval must be a positive int, got {interval!r}")
        rope_layers = [int((i + 1) % interval != 0) for i in range(n)]
    else:
        rope_layers = list(rope_layers)
        if len(rope_layers) != n:
            raise ValueError(f"no_rope_layers has {len(rope_layers)} entries for num_hidden_layers={n}")
        rope_layers = _binary(rope_layers, "no_rope_layers")
    pattern = [1 - r for r in rope_layers]
    return pattern if any(pattern) else None


def megatron_no_rope_freq(pattern):
    """Value for Megatron's ``no_rope_freq`` (1 = NoPE layer).

    Returns the period N when the pattern is ``[0]*(N-1) + [1]`` repeated at
    least twice (SmolLM3: N = 4 over 36 layers; all-NoPE is N = 1). Megatron
    expands the int form in ``TransformerConfig.__post_init__`` (it asserts
    ``num_layers % N == 0``). Any other pattern, including one period that spans
    the whole depth, is returned as the explicit list (Megatron asserts
    ``len == num_layers``). Both forms give the same per-layer placement.
    """
    if not pattern:
        return None
    pattern = _binary(pattern, "no_rope_freq pattern")
    n = len(pattern)
    for k in range(1, n // 2 + 1):
        if n % k == 0 and pattern == ([0] * (k - 1) + [1]) * (n // k):
            return k
    return pattern


def expand_no_rope_freq(value, num_layers):
    """Megatron's per-layer expansion of ``no_rope_freq`` (int N or list), or None.

    Follows ``TransformerConfig.__post_init__`` for every value Megatron gives a
    meaning to: a falsy value (``None``, ``0``, ``[]``) means no NoPE layer; an int
    N >= 1 must divide ``num_layers``; a list must have ``num_layers`` entries.
    It is stricter than Megatron on purpose, because it is used to check a built
    model (``check_model_nope``): list entries must be 0 or 1, and bools, negative
    ints, floats and strings raise ValueError, where Megatron would accept some of
    them without a check. The length and entries of a non-empty list are checked
    before the all-zero case, so a wrong-length all-zero list is rejected too.
    """
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return None  # falsy: Megatron skips its no_rope_freq handling
    if isinstance(value, bool):
        raise ValueError(f"no_rope_freq must be an int or a list, got bool {value!r}")
    if isinstance(value, int):
        if value == 0:
            return None
        if value < 0:
            raise ValueError(f"no_rope_freq must be a positive int, got {value}")
        if num_layers % value != 0:
            raise ValueError(f"no_rope_freq={value} does not divide num_layers={num_layers}")
        return ([0] * (value - 1) + [1]) * (num_layers // value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"no_rope_freq must be an int or a list, got {type(value).__name__} {value!r}")
    value = _binary(list(value), "no_rope_freq")
    if len(value) != num_layers:
        raise ValueError(f"no_rope_freq has {len(value)} entries for num_layers={num_layers}")
    return value if any(value) else None


def check_supported(cfg):
    """Raise NotImplementedError for SmolLM3 settings this converter does not build, and
    ValueError for an inconsistent or invalid config.json.

    Returns ``(resolved, pattern)``: ``with_hf_defaults(cfg)`` and its ``nope_pattern``,
    so callers do not resolve them a second time.

    Sliding-window attention follows HF ``SmolLM3Config``/``SmolLM3Attention``:
    an explicit ``layer_types`` is used as given (HF then ignores
    ``use_sliding_window`` for choosing layer types), and any ``sliding_attention``
    entry is rejected (HF builds a sliding-window mask for it). Without
    ``layer_types``, HF makes the NoPE layers sliding only when
    ``use_sliding_window`` is true AND ``sliding_window`` is not null; that case is
    rejected only when there is at least one NoPE layer, as otherwise HF builds
    all full attention. RoPE scaling must be absent or explicitly
    ``rope_type``/``type`` "default"; a dict without a type is rejected, not guessed.
    """
    # Explicit nulls and missing keys are resolved as HF does first (a null that HF
    # cannot build raises ValueError here), so the checks below see real values.
    resolved = with_hf_defaults(cfg)
    layer_types = cfg.get("layer_types")
    if layer_types is not None:
        layer_types = list(layer_types)
        n = int(resolved["num_hidden_layers"])
        if len(layer_types) != n:
            raise ValueError(f"SmolLM3 layer_types has {len(layer_types)} entries for num_hidden_layers={n}")
        if "sliding_attention" in layer_types:
            raise NotImplementedError("SmolLM3 layer_types with 'sliding_attention' is not supported")
        unknown = sorted(set(layer_types) - {"full_attention"})
        if unknown:
            raise NotImplementedError(f"SmolLM3 layer_types {unknown} are not supported")
    pattern = nope_pattern(resolved)
    # HF: without layer_types, the NoPE layers become sliding_attention layers.
    if (
        pattern
        and layer_types is None
        and cfg.get("use_sliding_window")
        and cfg.get("sliding_window") is not None
    ):
        raise NotImplementedError(
            "SmolLM3 sliding-window attention is not supported by this converter "
            "(use_sliding_window with a sliding_window makes HF's NoPE layers sliding)"
        )
    if cfg.get("mlp_bias"):
        raise NotImplementedError("SmolLM3 mlp_bias=true is not supported by this converter")
    if cfg.get("attention_bias"):
        # HF adds a bias to q/k/v AND o_proj; the reused Llama mapping copies weights only.
        raise NotImplementedError(
            "SmolLM3 attention_bias=true (biases on q/k/v/o_proj) is not supported by this converter"
        )
    act = resolved["hidden_act"]
    if act != "silu":
        raise NotImplementedError(
            f"SmolLM3 hidden_act={act!r} is not supported: the converter builds a SwiGLU (silu) MLP"
        )
    for key in ("rope_scaling", "rope_parameters"):
        rope = cfg.get(key)
        if rope is None:
            continue
        if not isinstance(rope, dict):
            raise NotImplementedError(f"SmolLM3 {key}={rope!r} is not supported")
        rope_type = rope.get("rope_type", rope.get("type"))
        if rope_type is None and key == "rope_parameters" and set(rope) <= {"rope_theta"}:
            rope_type = "default"  # transformers 5.x: rope_parameters may carry only the base
        if rope_type != "default":
            raise NotImplementedError(
                f"SmolLM3 {key}={rope!r}: only rope_type 'default' is supported "
                "(YaRN/linear/dynamic scaling or a dict without a type is not)"
            )
        extra = sorted(set(rope) - {"rope_type", "type", "rope_theta"})
        if extra:
            raise NotImplementedError(
                f"SmolLM3 {key} keys {extra} are not supported with rope_type 'default'"
            )
    # Every rope_theta given in config.json must be the one the converter builds with
    # (resolved: the top-level value, else the first nested one, else HF's default).
    for key, theta in _nested_thetas(cfg):
        if float(theta) != float(resolved["rope_theta"]):
            raise ValueError(
                f"{key}.rope_theta={theta} disagrees with the resolved rope_theta={resolved['rope_theta']}"
            )
    return resolved, pattern


def _nested_thetas(cfg):
    """``[(key, theta)]`` for each of ``rope_parameters`` / ``rope_scaling`` (in that
    order) that is a dict with a non-null ``rope_theta``."""
    out = []
    for key in ("rope_parameters", "rope_scaling"):
        rope = cfg.get(key)
        if isinstance(rope, dict) and rope.get("rope_theta") is not None:
            out.append((key, rope["rope_theta"]))
    return out


def with_hf_defaults(cfg):
    """``cfg`` with HF SmolLM3Config defaults filled in for MISSING keys; a warning names
    each key that fell back to a default.

    An explicit null is resolved the way HF resolves it, where this converter supports
    that, and rejected otherwise:
      * ``attention_bias`` / ``mlp_bias``: null means no bias (as in HF), so False;
      * ``num_key_value_heads``: null means plain MHA (HF sets it to the head count);
      * ``rope_theta``: missing or null is taken from ``rope_parameters.rope_theta`` or
        ``rope_scaling.rope_theta`` if one of them is set (``check_supported`` then
        rejects a nested value that disagrees with the resolved one);
      * ``no_rope_layer_interval``: kept as null; ``nope_pattern`` rejects it unless an
        explicit ``no_rope_layers`` list is given (HF fails on ``(i + 1) % None``);
      * any other null key raises ValueError: this converter does not support it (HF may
        store the null, but the Megatron model needs a value).
    """
    out = dict(cfg)
    defaults = {**HF_SMOLLM3_DEFAULTS, "num_key_value_heads": HF_DEFAULT_NUM_KEY_VALUE_HEADS}
    nested = _nested_thetas(cfg)
    # rope_theta taken from rope_parameters / rope_scaling is not a fallback.
    filled = sorted(
        k
        for k in _WARN_IF_MISSING + ("num_key_value_heads",)
        if k not in cfg and not (k == "rope_theta" and nested)
    )
    if filled:
        # Shape keys and rope_theta / max_position_embeddings are normally in config.json;
        # say which ones fell back (the HF defaults are not the SmolLM3-3B preset values,
        # e.g. rope_theta 2e6 vs 5e6).
        warnings.warn(
            "SmolLM3 config.json omits "
            + ", ".join(f"{k} (using HF default {defaults[k]!r})" for k in filled),
            stacklevel=2,
        )
    if out.get("rope_theta") is None and nested:
        out["rope_theta"] = nested[0][1]
    for k, v in HF_SMOLLM3_DEFAULTS.items():
        if k not in out:
            out[k] = v
        elif out[k] is None:
            if k in _NULL_MEANS_FALSE:
                out[k] = False
            elif k not in _NULL_RESOLVED_ELSEWHERE:
                raise ValueError(
                    f"SmolLM3 config.json sets {k}=null; this converter does not support a null {k}"
                )
    if out["rope_theta"] is None:
        raise ValueError(
            "SmolLM3 config.json sets rope_theta=null and no rope_parameters/rope_scaling rope_theta"
        )
    # HF: a missing key takes the default 4; an explicit null means plain MHA.
    if "num_key_value_heads" not in cfg:
        out["num_key_value_heads"] = HF_DEFAULT_NUM_KEY_VALUE_HEADS
    elif cfg["num_key_value_heads"] is None:
        out["num_key_value_heads"] = out["num_attention_heads"]
    return out


def derive_megatron_fields(cfg):
    cfg, pattern = check_supported(cfg)
    fields = _llama.derive_megatron_fields(cfg)
    fields["no_rope_freq"] = pattern
    # config.json's depth, kept apart from ``num_layers`` (which callers such as the
    # layer-wise verifier lower to build a cut model); check_model_nope uses it.
    fields["hf_num_layers"] = int(cfg["num_hidden_layers"])
    return fields


def extra_log_fields(cfg_fields):
    """Fields that ``native_loader_llama.convert`` adds to its arch log line."""
    return {"no_rope_freq": cfg_fields["no_rope_freq"]} if cfg_fields.get("no_rope_freq") else {}


def build_margs(cfg_fields, args):
    # no_rope_freq is a TransformerConfig field that validate_args does not check;
    # TransformerConfig.__post_init__ checks it against num_layers when the model is built.
    # A caller may lower cfg_fields["num_layers"] to build a cut model (the first N
    # layers, as the layer-wise verifier does): give Megatron the pattern of those N
    # layers only, so it has num_layers entries (or an int period that divides N).
    # The shared Llama build_margs sets layernorm_epsilon from config.json's rms_norm_eps
    # (1e-6 for SmolLM3, as the preset trains with).
    m = _llama.build_margs(cfg_fields, args)
    m.no_rope_freq = megatron_no_rope_freq(nope_pattern_for_depth(cfg_fields, int(cfg_fields["num_layers"])))
    return m


def _hf_depth(cfg_fields):
    """config.json's depth, ``hf_num_layers`` (set by ``derive_megatron_fields``).
    Raises ValueError when it is missing, or when the ``no_rope_freq`` pattern's length
    is not that depth."""
    if cfg_fields.get("hf_num_layers") is None:
        raise ValueError("SmolLM3: cfg_fields has no hf_num_layers (use derive_megatron_fields)")
    depth = int(cfg_fields["hf_num_layers"])
    full = cfg_fields.get("no_rope_freq")
    if full and len(full) != depth:
        raise ValueError(f"no_rope_freq pattern has {len(full)} entries, config.json has {depth} layers")
    return depth


def nope_pattern_for_depth(cfg_fields, num_layers):
    """The config.json NoPE pattern (1 = NoPE) of the first ``num_layers`` layers, or
    None when none of them is NoPE. Raises ValueError for a depth deeper than
    config.json (see ``_hf_depth``)."""
    hf_layers = _hf_depth(cfg_fields)
    if num_layers > hf_layers:
        raise ValueError(
            f"num_layers={num_layers} is deeper than config.json's {hf_layers} layers "
            f"(built model has {num_layers} layers, config.json has {hf_layers})"
        )
    head = list(cfg_fields.get("no_rope_freq") or [])[:num_layers]
    return head if any(head) else None


def check_model_nope(model, cfg_fields):
    """Raise if the built model's per-layer NoPE pattern is not the one derived from
    config.json (a dropped ``no_rope_freq`` would give every layer RoPE while the
    weights still load cleanly).

    ``convert()`` builds the full depth, so the model has exactly as many layers as
    config.json. A model with fewer layers is accepted only as a cut to its first
    ``num_layers`` layers (the layer-wise verifier builds 4-layer models); it is
    then checked against those layers. A model deeper than config.json raises; the
    depth is ``cfg_fields["hf_num_layers"]`` (config.json's ``num_hidden_layers``),
    so the check works also when config.json has no NoPE layer.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("SmolLM3 converter: built model has no .config to check no_rope_freq on")
    num_layers = int(config.num_layers)
    try:
        expected = nope_pattern_for_depth(cfg_fields, num_layers)
    except ValueError as e:
        raise RuntimeError(f"SmolLM3 converter: {e}") from None
    try:
        got = expand_no_rope_freq(getattr(config, "no_rope_freq", None), num_layers)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"SmolLM3 converter: invalid model.config.no_rope_freq: {e}") from e
    if got != expected:
        raise RuntimeError(
            f"SmolLM3 converter: model.config.no_rope_freq expands to {got}, "
            f"but config.json gives {expected} for the first {num_layers} layers (1 = NoPE layer)"
        )
    return got


def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    log(f"  margs.no_rope_freq (1 = NoPE; int N = every N-th layer) = {margs.no_rope_freq}")
    got = check_model_nope(model, cfg_fields)
    nope_layers = [i for i, v in enumerate(got or []) if v]
    log(f"  model.config.no_rope_freq matches config.json; NoPE layers = {nope_layers}")
    return _llama.build_mcore_state_dict(model, store, margs, cfg_fields, log)


def convert(hf_dir, save_dir, **kwargs):
    """Convert a SmolLM3 HF checkpoint to a legacy Megatron torch checkpoint: the Llama
    conversion sequence (``native_loader_llama.convert``) with this module's field
    functions. Same keyword arguments."""
    return _llama.convert(hf_dir, save_dir, family=sys.modules[__name__], **kwargs)
