###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which Llama / Mistral HF configs the native Llama converter supports, and the
Megatron fields it derives from them.

Standard library only: the posttrain conversion hook loads this file by path
(without importing ``primus.backends.megatron``, torch or Megatron) to decide
between the native converter and Megatron-Bridge, and ``native_loader_llama``
uses the same functions, so the hook and the converter cannot disagree.

Config values are read as HF's ``LlamaConfig`` / ``MistralConfig`` read them,
with one exception: an absent Llama ``rms_norm_eps`` is 1e-5
(``LLAMA_FALLBACK_RMS_NORM_EPS``; Llama 2/3 configs set it), not LlamaConfig's
1e-6. In addition:

* the model shape (``hidden_size``, ``num_attention_heads``,
  ``num_hidden_layers``, ``intermediate_size``) must be set: the converter does
  not take the class defaults for the shape of the weights it copies;
* any other absent key takes that class's default;
* null is accepted where the class maps it to a value or to "none":
  ``num_key_value_heads`` (MHA), ``head_dim`` (hidden_size / heads),
  ``sliding_window`` (no window), ``rope_scaling`` / ``rope_parameters`` (no
  scaling), ``attention_bias`` / ``mlp_bias`` (no bias), and ``architectures`` /
  ``model_type`` (read as absent; the other one names the family); any other null
  is rejected, ``rope_type: null`` included;
* a value must have the class's type (``attention_bias``, ``mlp_bias`` and
  ``tie_word_embeddings`` true or false, sizes positive integers);
* RoPE is read from transformers 5.x ``rope_parameters`` or from the legacy
  top-level ``rope_theta`` / ``rope_scaling``; a config with both must have
  the same theta and the same scaling;
* ``llama3`` scaling (Llama only) needs every HF llama3 key, a ``factor`` of
  at least 1 (as HF), and the low / high frequency factors and original context
  Megatron fixes (1 / 4 / 8192, ``MEGATRON_LLAMA3_ROPE_FIXED``);
* every ``architectures`` entry must be ``LlamaForCausalLM`` or
  ``MistralForCausalLM`` (one family), and ``model_type``, when set, must name
  the same family; without ``architectures`` the family comes from ``model_type``;
* a Mistral config needs ``sliding_window: null``. A window is not modelled:
  Megatron trains with full causal attention, also past the window when
  seq_length is longer.
"""

import math

MISTRAL_NAMES = ("mistral", "mistralforcausallm")
LLAMA_NAMES = ("llama", "llamaforcausallm")

# Defaults for a config without the key. Mistral uses MistralConfig's defaults;
# Llama keeps 1e-5 (Llama 2/3 configs set it), not LlamaConfig's 1e-6.
LLAMA_FALLBACK_RMS_NORM_EPS = 1e-5
HF_DEFAULT_ROPE_THETA = 10000.0
HF_DEFAULT_MISTRAL_RMS_NORM_EPS = 1e-6
HF_DEFAULT_MISTRAL_SLIDING_WINDOW = 4096
HF_DEFAULT_MISTRAL_KV_HEADS = 8
HF_DEFAULT_MISTRAL_MAX_POSITION_EMBEDDINGS = 131072
HF_DEFAULT_LLAMA_MAX_POSITION_EMBEDDINGS = 2048

# RoPE types the converter accepts. The conversion only copies weights; RoPE
# scaling changes no weight. "llama3" (Llama 3.1/3.2/3.3) is accepted for Llama
# configs, but the training config must apply it (use_rope_scaling and
# rope_scaling_factor; Megatron fixes the other llama3 parameters to
# MEGATRON_LLAMA3_ROPE_FIXED): the conversion hook passes HF's factor to the
# training run (llama3_rope_scaling). Mistral accepts only no scaling.
LLAMA_ROPE_TYPES = ("default", "llama3")
MISTRAL_ROPE_TYPES = ("default",)
# The llama3 RoPE parameters Megatron does not read from the training config
# (megatron/core/models/common/embeddings/rotary_pos_embedding.py, _apply_scaling).
# They are also HF's values for Llama 3.1 / 3.2 / 3.3.
MEGATRON_LLAMA3_ROPE_FIXED = {
    "low_freq_factor": 1,
    "high_freq_factor": 4,
    "original_max_position_embeddings": 8192,
}


class _Unsupported(Exception):
    pass


def config_names(cfg):
    """The lower-cased ``architectures`` entries and ``model_type`` of an HF config.
    ``ValueError`` when ``architectures`` is not a string or a list of strings."""
    architectures = cfg.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, list) or not all(isinstance(a, str) for a in architectures):
        raise ValueError(f"architectures {architectures!r} (a list of strings is required)")
    return [a.lower() for a in architectures] + [str(cfg.get("model_type", "")).lower()]


def _family_of(name):
    return "mistral" if name in MISTRAL_NAMES else "llama" if name in LLAMA_NAMES else None


def config_family(cfg):
    """``"llama"`` | ``"mistral"`` | None. Every ``architectures`` entry must be a
    Llama or Mistral causal LM of one family, and ``model_type`` (when set) must
    name that family; without ``architectures`` the family is ``model_type``'s.
    ``_Unsupported`` when the names conflict (for example ``Qwen2ForCausalLM``
    with ``model_type: llama``, or ``LlamaForSequenceClassification``); None
    when nothing names Llama or Mistral."""
    try:
        names = config_names(cfg)
    except ValueError as e:
        raise _Unsupported(str(e)) from None
    architectures = names[:-1]
    model_type = names[-1] if names[-1] not in ("", "none") else None
    if not architectures:
        return _family_of(model_type)
    arch_families = {_family_of(n) for n in architectures}
    if arch_families == {None} and _family_of(model_type) is None:
        return None  # another model: not this converter's
    if None in arch_families or len(arch_families) > 1:
        raise _Unsupported(
            f"architectures {architectures} (only LlamaForCausalLM or MistralForCausalLM: "
            "not other models, heads, or both)"
        )
    family = arch_families.pop()
    if model_type is not None and _family_of(model_type) != family:
        raise _Unsupported(f"architectures/model_type {names} (they name different models)")
    return family


def llama3_rope_scaling(cfg):
    """The ``llama3`` RoPE scaling parameters of a config (``rope_parameters`` or
    ``rope_scaling``; ``_rope`` requires the two to agree), or None. The converter
    copies weights only; the conversion hook passes the scaling to the training
    run."""
    for key in ("rope_parameters", "rope_scaling"):
        params = cfg.get(key)
        if isinstance(params, dict) and params.get("rope_type", params.get("type")) == "llama3":
            return params
    return None


def llama3_rope_problem(params):
    """Why Megatron's llama3 RoPE scaling cannot reproduce these ``llama3``
    parameters, or None. As HF (``_validate_llama3_parameters``), every key is
    required and ``factor`` is at least 1; Megatron has an option only for
    ``factor`` and fixes the others (``MEGATRON_LLAMA3_ROPE_FIXED``)."""
    factor = params.get("factor")
    try:
        ok = (
            not isinstance(factor, bool)
            and isinstance(factor, (int, float))
            and 1 <= float(factor) < math.inf
        )
    except OverflowError:  # an int too large for a float
        ok = False
    if not ok:
        return f"factor {factor!r} (a finite number of at least 1 is required)"
    for name, fixed in MEGATRON_LLAMA3_ROPE_FIXED.items():
        if name not in params:
            return f"{name} missing (HF requires it; Megatron's llama3 RoPE scaling uses {fixed})"
        value = params[name]
        if isinstance(value, bool) or value != fixed:
            return f"{name} {value!r} (Megatron's llama3 RoPE scaling uses {fixed})"
    return None


def _int(key, value):
    """A positive integer; an integral float (JSON ``4096.0``) is accepted."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise _Unsupported(f"{key} {value!r} (a positive integer is required)")


def _number(key, value):
    """A positive, finite number (also as a float: an integer too large for a float is rejected)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if finite and value > 0:
            return value
    raise _Unsupported(f"{key} {value!r} (a positive finite number is required)")


def _present(cfg, key, default):
    """The value for ``key``: ``default`` when the key is absent; null is rejected."""
    if key not in cfg:
        return default
    if cfg[key] is None:
        raise _Unsupported(f"{key} null (HF has no value for it)")
    return cfg[key]


def _rope(cfg, family, rope_types):
    """``rope_theta`` from ``rope_parameters`` (transformers 5.x) or the legacy keys."""
    rope_parameters = cfg.get("rope_parameters")
    if rope_parameters is not None and not isinstance(rope_parameters, dict):
        raise _Unsupported(f"rope_parameters {rope_parameters!r} (a JSON object is required)")
    rope_scaling = cfg.get("rope_scaling")
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise _Unsupported(f"rope_scaling {rope_scaling!r} (a JSON object is required)")
    scalings = {}
    for key, params in (("rope_parameters", rope_parameters), ("rope_scaling", rope_scaling)):
        if params:
            rope_type = params.get("rope_type", params.get("type", "default"))
            if rope_type not in rope_types:
                raise _Unsupported(f"{family} {key} {params!r} (RoPE type {rope_type!r})")
            if rope_type == "llama3":
                llama3_problem = llama3_rope_problem(params)
                if llama3_problem:
                    raise _Unsupported(f"{key}.{llama3_problem}")
            scalings[key] = (
                rope_type,
                {k: v for k, v in params.items() if k not in ("rope_theta", "type", "rope_type")},
            )
    if len(scalings) == 2 and scalings["rope_parameters"] != scalings["rope_scaling"]:
        raise _Unsupported(f"rope_parameters {rope_parameters!r} and rope_scaling {rope_scaling!r} differ")
    thetas = {
        float(_number(label, d["rope_theta"]))
        for label, d in (("rope_theta", cfg), ("rope_parameters.rope_theta", rope_parameters or {}))
        if "rope_theta" in d
    }
    if len(thetas) > 1:
        raise _Unsupported(
            f"rope_theta {cfg.get('rope_theta')!r} and rope_parameters {rope_parameters!r} differ"
        )
    return thetas.pop() if thetas else HF_DEFAULT_ROPE_THETA


def _fields(cfg):
    if not isinstance(cfg, dict):
        raise _Unsupported("config.json is not a JSON object")
    family = config_family(cfg)
    if family is None:
        raise _Unsupported(f"architectures/model_type {config_names(cfg)} (not a Llama or Mistral config)")
    mistral = family == "mistral"
    label = family.capitalize()  # "Llama" / "Mistral", for messages
    experts = [k for k in ("num_local_experts", "num_experts", "n_routed_experts") if cfg.get(k)]
    if experts:
        raise _Unsupported(f"{experts} (the converter maps dense models only)")
    for key in ("attention_bias", "mlp_bias"):
        bias = cfg.get(key)
        if bias is not None and not isinstance(bias, bool):
            raise _Unsupported(f"{key} {bias!r} (true or false is required)")
        if bias:
            raise _Unsupported(f"{key} true (the converter maps no bias tensors)")

    shape = ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size")
    missing = [k for k in shape if k not in cfg]
    if missing:
        raise _Unsupported(f"{missing} missing (the converter needs the model shape set, not class defaults)")
    hidden, heads, layers, ffn = (_int(k, cfg[k]) for k in shape)
    max_pos = _int(
        "max_position_embeddings",
        _present(
            cfg,
            "max_position_embeddings",
            (
                HF_DEFAULT_MISTRAL_MAX_POSITION_EMBEDDINGS
                if mistral
                else HF_DEFAULT_LLAMA_MAX_POSITION_EMBEDDINGS
            ),
        ),
    )
    # num_key_value_heads: null is MHA for both families (as in HF); an absent key
    # is MistralConfig's 8 for Mistral and MHA for Llama.
    if mistral and "num_key_value_heads" not in cfg:
        kv_heads = HF_DEFAULT_MISTRAL_KV_HEADS
    else:
        kv_heads = cfg.get("num_key_value_heads")
        kv_heads = heads if kv_heads is None else _int("num_key_value_heads", kv_heads)
    if heads % kv_heads:
        raise _Unsupported(f"num_key_value_heads {kv_heads} does not divide num_attention_heads {heads}")
    # head_dim: absent or null is hidden_size / num_attention_heads (as in HF).
    if cfg.get("head_dim") is None:
        if hidden % heads:
            raise _Unsupported(
                f"hidden_size {hidden} is not a multiple of num_attention_heads {heads} (and no head_dim)"
            )
        head_dim = hidden // heads
    else:
        head_dim = _int("head_dim", cfg["head_dim"])
    eps = _number(
        "rms_norm_eps",
        _present(
            cfg, "rms_norm_eps", HF_DEFAULT_MISTRAL_RMS_NORM_EPS if mistral else LLAMA_FALLBACK_RMS_NORM_EPS
        ),
    )
    act = _present(cfg, "hidden_act", "silu")
    if act not in ("silu", "swish"):
        raise _Unsupported(f"{label} hidden_act {act!r} (only SwiGLU with silu)")
    tie = _present(cfg, "tie_word_embeddings", False)
    if not isinstance(tie, bool):
        raise _Unsupported(f"tie_word_embeddings {tie!r} (true or false is required)")
    theta = _rope(cfg, label, MISTRAL_ROPE_TYPES if mistral else LLAMA_ROPE_TYPES)
    window = cfg.get("sliding_window", HF_DEFAULT_MISTRAL_SLIDING_WINDOW)
    if mistral and window is not None:
        # HF's MistralConfig reads an absent key as 4096. Megatron trains with full
        # causal attention, also past a window that covers max_position_embeddings
        # when seq_length is longer, so only "no window" is accepted. (LlamaConfig
        # has no sliding window, and HF's Llama model ignores a sliding_window key.)
        raise _Unsupported(
            f"Mistral sliding_window {window!r} (only full causal attention: sliding_window: null)"
        )

    return dict(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_layers=layers,
        max_position_embeddings=max_pos,
        norm_epsilon=eps,
        rotary_base=theta,
        untie_embeddings_and_output_weights=not tie,
        ffn_hidden_size=ffn,
        # GQA (num_query_groups == num_attention_heads is plain MHA); it drives the
        # fused-QKV layout.
        num_query_groups=kv_heads,
        kv_channels=head_dim,
    )


def unsupported_reason(cfg):
    """Why the native Llama converter cannot convert this HF config, or None when it can."""
    try:
        _fields(cfg)
    except _Unsupported as e:
        return str(e)
    return None


def derive_megatron_fields(cfg):
    """Megatron arg values for a supported Llama (2/3/3.x) or dense Mistral HF
    config; ``ValueError`` with ``unsupported_reason`` otherwise."""
    try:
        return _fields(cfg)
    except _Unsupported as e:
        raise ValueError(f"native Llama converter: config not supported: {e}") from None
