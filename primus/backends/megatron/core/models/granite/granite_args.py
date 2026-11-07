###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Argument checks for the Granite builder (pure Python, no ``megatron.training`` import)."""

__all__ = [
    "GRANITE_ATTENTION_OFF",
    "GRANITE_LAYOUT",
    "UNSUPPORTED_CHECKS",
    "attention_problems",
    "builder_problems",
    "layout_problems",
    "reject_unsupported_args",
    "unsupported_problems",
]

# The Llama layout of HF Granite (and of the native converter), plus the norm / residual
# options that would change its math: {arg: (required value, Megatron argparse default)}.
# A missing arg takes the default. ``key: null`` matches a required False (Megatron reads
# None as off there) and no other required value. The builder (reject_unsupported_args)
# and the converter's preset check (native_loader_granite.preset_mismatches) both read
# this one table.
GRANITE_LAYOUT = {
    "normalization": ("RMSNorm", "LayerNorm"),
    "swiglu": (True, False),
    "add_bias_linear": (False, True),
    "add_qkv_bias": (False, False),
    "qk_layernorm": (False, False),
    # Granite uses RoPE. "learned_absolute" (the default) adds learned position embeddings;
    # None / "none" builds no positional encoding at all.
    "position_embedding_type": ("rope", "learned_absolute"),
    # Granite 4.1 rotates the full head (HF partial_rotary_factor 1).
    "rotary_percent": (1.0, 1.0),
    # Zero-centred norm gain (the norm computes with 1 + w): HF Granite RMSNorm weights
    # are the gain itself, so every converted norm would be off by +1.
    "apply_layernorm_1p": (False, False),
    # HF Granite adds the scaled branch to the block input, not to the normed input.
    "apply_residual_connection_post_layernorm": (False, False),
    # L2-normalised Q/K: HF Granite has no QK norm.
    "qk_l2_norm": (False, False),
}

# Upstream attention options that change the attention math relative to HF Granite. Each
# must be unset or off (see ``_is_on``): {arg: reason}. The builder and the converter's
# preset check both read this table.
GRANITE_ATTENTION_OFF = {
    # An explicit scale wins over MuP (TransformerConfig only fills softmax_scale from MuP when
    # it is None), so it would replace HF attention_multiplier. Set the scale through
    # mup_base_head_dim / kv_channels instead.
    "softmax_scale": "overrides the MuP scale that equals HF attention_multiplier",
    "window_size": "sliding-window attention; HF Granite attends to the full causal context",
    "attention_output_gate": "gated attention output; HF Granite has no gate",
    "no_rope_freq": "NoPE layers; HF Granite applies RoPE on every layer",
    "apply_query_key_layer_scaling": "scales QK^T by the layer number; HF Granite does not",
    "use_sink_attention": "learned attention sinks; HF Granite has none",
    # TransformerConfig.softmax_type: 'off-by-one' adds 1 to the softmax denominator and
    # 'learnable' adds a learned sink logit per head; HF Granite uses the plain softmax.
    "softmax_type": "non-vanilla softmax (off-by-one / learnable sink); HF Granite uses the plain softmax",
}


# Per-option off values, besides None, False, a number equal to 0 and an empty list.
# TE / flash-attn take window_size [-1, -1] (no window) and [-1, 0] (causal, no window) as
# "no sliding window"; any other pair is a window. A no_rope_freq list of zeros applies
# RoPE on every layer.
_OFF_SENTINELS = {
    "window_size": lambda v: isinstance(v, (list, tuple))
    and len(v) == 2
    and all(_is_int(x) for x in v)
    and v[0] == -1
    and v[1] in (-1, 0),
    "no_rope_freq": lambda v: isinstance(v, (list, tuple)) and all(_is_int(x) and x == 0 for x in v),
    # Upstream's default softmax.
    "softmax_type": lambda v: v == "vanilla",
}

# Options for which only None is off. TransformerConfig fills softmax_scale from MuP only
# when it is None, so an explicit 0 would be a real scale (uniform attention).
_ONLY_NONE_IS_OFF = frozenset({"softmax_scale"})


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_on(key, value):
    """Whether the ``GRANITE_ATTENTION_OFF`` option ``key`` is on.

    Off: None. For ``softmax_scale`` nothing else (``_ONLY_NONE_IS_OFF``). For the other
    options also False, a number equal to 0, an empty list / tuple, and the option's own off
    sentinel (``_OFF_SENTINELS``: ``window_size: [-1, -1]`` or ``[-1, 0]``,
    ``no_rope_freq: [0, ...]``, ``softmax_type: 'vanilla'``). Anything else is on. A string is on, also ``'false'`` /
    ``'0'``: Megatron reads the value as given (a non-empty string is true), so a quoted
    ``'false'`` would enable the option; write it unquoted.
    """
    if value is None:
        return False
    if key in _ONLY_NONE_IS_OFF:
        return True
    if value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return False
    if isinstance(value, (list, tuple)) and not value:
        return False
    sentinel = _OFF_SENTINELS.get(key)
    return not (sentinel and sentinel(value))


def attention_problems(get):
    """``[(key, reason, value)]`` for each ``GRANITE_ATTENTION_OFF`` arg that is set;
    ``get(key, default)`` reads one arg (``getattr`` on args, ``dict.get`` on a preset)."""
    out = []
    for key, reason in GRANITE_ATTENTION_OFF.items():
        value = get(key, None)
        if _is_on(key, value):
            out.append((key, reason, value))
    return out


def builder_problems(get):
    """Labels of the set options the dense Granite builder does not implement (the list
    below and MuP width scaling), without the ``GRANITE_ATTENTION_OFF`` and
    ``GRANITE_LAYOUT`` checks (``attention_problems``, ``layout_problems``).

    ``get(key, default)`` reads one arg (``getattr`` on the args, ``dict.get`` on a model
    preset). The converter's preset check (``native_loader_granite.preset_mismatches``)
    calls the three functions itself, to word each problem; the builder calls them through
    ``unsupported_problems``. So both refuse the same options.
    """
    unsupported = {
        "use_legacy_models": bool(get("use_legacy_models", False)),
        # ModelOpt swaps in its own builder (no Granite scaled residual add). The model
        # provider refuses it first; this covers a direct caller of granite_builder.
        "modelopt_enabled": bool(get("modelopt_enabled", False)),
        "yaml_cfg": get("yaml_cfg", None) is not None,
        "experimental_attention_variant": get("experimental_attention_variant", None) is not None,
        "heterogeneous_layers_config_path": get("heterogeneous_layers_config_path", None) is not None,
        "transformer_impl=inference_optimized": get("transformer_impl", None) == "inference_optimized",
        "num_experts": bool(get("num_experts", None)),
        "mtp_num_layers": bool(get("mtp_num_layers", None)),  # None / 0: no MTP
        "multi_latent_attention": bool(get("multi_latent_attention", False)),
        # Granite 4.1 uses plain RoPE (HF rope_scaling: null; the converter refuses any
        # rope_scaling). Upstream's use_rope_scaling is Llama-3.x scaling (factor 8.0 by
        # default) and would make the RoPE differ from HF without an error.
        "use_rope_scaling": bool(get("use_rope_scaling", False)),
        # HF Granite rotates the two halves of each head (rotate_half). Interleaved RoPE
        # pairs neighbouring channels instead, so a converted checkpoint would get wrong
        # positions without an error.
        "rotary_interleaved": bool(get("rotary_interleaved", False)),
        # Linear position interpolation (HF rope_scaling "linear"); Granite 4.1 has none.
        # The builder does not pass it to GPTModel, so it is refused, not silently ignored.
        "rotary_seq_len_interpolation_factor": get("rotary_seq_len_interpolation_factor", None) is not None,
        # NVIDIA Kitchen swaps the linear / attention modules of the layer spec for its own
        # quantized ones; not tested with the Granite scaled residual add.
        "use_kitchen": bool(get("use_kitchen", False)),
    }
    bad = [name for name, is_set in unsupported.items() if is_set]
    # MuP width scaling rescales init, LR and logits away from the checkpoint's fixed
    # multipliers (GraniteTransformerConfig refuses it too, at build time).
    base_hidden = get("mup_base_hidden_size", None)
    if base_hidden is not None and base_hidden != get("hidden_size", None):
        bad.append("mup_base_hidden_size != hidden_size (MuP width scaling)")
    return bad


def _attention_labels(get):
    return [key for key, _, _ in attention_problems(get)]


def _layout_labels(get):
    return [label for _, label, _, _ in layout_problems(get)]


# The checks ``unsupported_problems`` runs, in order: ``{name: labels(get)}``. The converter's
# preset check (``native_loader_granite.preset_mismatches``) runs the underlying functions
# itself, to give a message with the HF value; a test refuses one option per entry here
# (parametrized from this mapping) through both paths, so a check added here without a case,
# or added to the builder only, fails it.
UNSUPPORTED_CHECKS = {
    "builder_problems": builder_problems,
    "attention_problems": _attention_labels,
    "layout_problems": _layout_labels,
}


def unsupported_problems(get):
    """Labels of every set option the dense Granite model does not support: the labels of
    each check in ``UNSUPPORTED_CHECKS``. ``get`` as in ``builder_problems``."""
    return [label for check in UNSUPPORTED_CHECKS.values() for label in check(get)]


def reject_unsupported_args(args):
    """Refuse upstream ``gpt_builder`` options the dense Granite builder does not implement.

    Raises ``ValueError`` naming every option ``unsupported_problems`` reports. Primus Turbo
    attention (``enable_primus_turbo`` + ``use_turbo_attention``) is allowed: it receives
    ``config.softmax_scale`` like TE attention.
    """
    bad = unsupported_problems(lambda key, default: getattr(args, key, default))
    if bad:
        raise ValueError(f"the dense Granite builder does not support: {', '.join(bad)}")


def _layout_matches(value, want):
    """``value`` is the required ``want``: the same bool (None counts as False, which is how
    Megatron reads an unset flag), the same string, or (for a number) a real int / float
    equal to it (not a bool, a string or None)."""
    if want is False and value is None:
        return True
    if isinstance(want, (bool, str)):
        return type(value) is type(want) and value == want
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == want


def _layout_label(key, want):
    if want is False:
        return key
    return f"{key}!={want:g}" if isinstance(want, float) else f"{key}!={want}"


def layout_problems(get):
    """``[(key, label, value, want)]`` for each ``GRANITE_LAYOUT`` arg that is not the Granite
    value; ``get(key, default)`` reads one arg (``getattr`` on args, ``dict.get`` on a preset)."""
    out = []
    for key, (want, default) in GRANITE_LAYOUT.items():
        value = get(key, default)
        if not _layout_matches(value, want):
            out.append((key, _layout_label(key, want), value, want))
    return out
