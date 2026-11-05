###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Config checks of the native Mistral converter (``native_loader_mistral``), for the
dense Mistral checkpoints with full causal attention (Mistral-7B-v0.2 / v0.3)
whose ``config.json`` Llama's converter translates to Mistral's model:

* ``architectures`` ``[MistralForCausalLM]`` and ``model_type`` ``mistral``. A
  Mistral checkpoint with another head (sequence / token classification, also
  ``model_type: mistral``) is not a causal LM.
* ``sliding_window`` must be present and ``null``. HF Mistral reads an absent key
  as 4096 and applies the window whatever ``use_sliding_window`` says, while
  Megatron here trains with full causal attention, so any window is rejected.
* no RoPE scaling: ``rope_scaling`` / ``rope_parameters`` (the newer transformers
  form; the two are aliases) must be null, empty or say type ``default`` (or, with no
  type, hold only ``rope_theta``) and have no other key but ``rope_theta``; no
  ``partial_rotary_factor`` other than 1 (null or absent is 1).
* ``rope_theta``: a positive finite number, the same wherever it is given (top
  level, ``rope_parameters`` or ``rope_scaling``; a null value is not given);
  10000 when none is given. A base
  given only in the nested form (as transformers v5 re-saves Mistral-7B-v0.3) is
  used: ``native_loader_mistral`` passes the one checked base on as ``rope_theta``.
* SiLU activation (``silu`` or its HF alias ``swish``), no attention / MLP biases.
* untied embeddings: ``tie_word_embeddings`` false or absent (HF Mistral's default);
  no ``quantization_config`` (only unquantized weights are read).
* the shape: positive integer sizes; ``num_key_value_heads`` and ``rms_norm_eps``
  present and not null (Llama's translation and HF Mistral fill an absent or null
  value differently); heads a multiple of the KV heads; ``hidden_size`` a multiple
  of the heads; ``head_dim`` the integer ``hidden_size // num_attention_heads``, null
  (as recent transformers write it) or absent (HF Mistral and Llama's translation
  both read null / absent as ``hidden_size // num_attention_heads``; another head
  size is not verified here).

Boolean keys must be JSON booleans (or null): ``0`` / ``1`` are rejected.

Standard library only, so the posttrain conversion hook can run it before torch or
Megatron is importable. ``check_config`` raises ``ValueError`` naming the failed check.
"""

import math

# Sizes that must be positive integers.
_SIZES = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "vocab_size",
)
# Same order as native_convert_common.hf_rope_theta reads them.
_ROPE_KEYS = ("rope_parameters", "rope_scaling")
# Hugging Face's RoPE base when a config gives none (native_convert_common's
# HF_DEFAULT_ROPE_THETA; not imported: this module is standard library only).
_HF_DEFAULT_ROPE_THETA = 10000.0


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value):
    """A positive finite number that fits a float (an int above float range is not)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except OverflowError:
        return False


def checked_rope_theta(cfg):
    """The one RoPE base of ``cfg``, checked: no scaling in ``rope_scaling`` /
    ``rope_parameters`` and every ``rope_theta`` given (top level, in either dict)
    the same positive number. HF's default, 10000, when none is given."""
    thetas = {}
    if cfg.get("rope_theta") is not None:  # null is no base given, as hf_rope_theta reads it
        thetas["rope_theta"] = cfg["rope_theta"]
    for name in _ROPE_KEYS:
        rope = cfg.get(name)
        if rope is None or rope == {}:  # HF reads an empty dict as no scaling
            continue
        ok = isinstance(rope, dict) and not set(rope) - {"rope_type", "type", "rope_theta"}
        if ok:
            kinds = [rope[k] for k in ("rope_type", "type") if k in rope]
            # a type stated is "default"; with none stated, the dict holds only rope_theta
            ok = all(k == "default" for k in kinds) and (kinds or "rope_theta" in rope)
        if not ok:
            raise ValueError(
                f"Mistral {name}={rope!r} is not supported: no RoPE scaling (null, or a dict with "
                "type 'default' and no key but rope_type / type / rope_theta)"
            )
        if "rope_theta" in rope:
            thetas[f"{name}.rope_theta"] = rope["rope_theta"]
    for where, theta in thetas.items():
        if not _positive_number(theta):
            raise ValueError(f"Mistral {where}={theta!r} must be a positive finite number")
    if len(set(thetas.values())) > 1:
        raise ValueError(f"Mistral RoPE base given differently: {thetas}")
    return float(next(iter(thetas.values()), _HF_DEFAULT_ROPE_THETA))


def check_config(cfg):
    """Raise ``ValueError`` (naming the failed check) for a config that the Llama
    converter would not translate to Mistral's model; else return the checked RoPE
    base (``rope_theta``)."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Mistral config must be a JSON object, not {type(cfg).__name__}")
    archs = cfg.get("architectures")
    if archs != ["MistralForCausalLM"]:
        raise ValueError(
            f'Mistral architectures={archs!r} is not supported: only the list ["MistralForCausalLM"] '
            "(one Mistral causal-LM class; other or additional classes are not converted)"
        )
    if cfg.get("model_type") != "mistral":
        raise ValueError(f"Mistral model_type={cfg.get('model_type')!r} is not supported: only 'mistral'")
    if "sliding_window" not in cfg:
        raise ValueError(
            "Mistral config has no sliding_window key (HF reads it as 4096); this converter "
            "supports only full causal attention (sliding_window: null)"
        )
    if cfg["sliding_window"] is not None:
        raise ValueError(
            f"Mistral sliding_window={cfg['sliding_window']!r} is not supported (also with "
            "use_sliding_window: false, which HF Mistral ignores); only sliding_window: null"
        )
    theta = checked_rope_theta(cfg)  # checks the RoPE settings (no scaling, one base)
    factor = cfg.get("partial_rotary_factor")
    if factor is not None and not (_positive_number(factor) and factor == 1):
        raise ValueError(
            f"Mistral partial_rotary_factor={factor!r} is not supported (RoPE on the full head only)"
        )
    act = cfg.get("hidden_act", "silu")
    if act not in ("silu", "swish"):
        raise ValueError(f"Mistral hidden_act={act!r} is not supported (SwiGLU needs silu)")
    # identity, not ==: 0, 0.0, 1 and 1.0 are not booleans
    for key in ("attention_bias", "mlp_bias"):
        value = cfg.get(key)
        if value is not None and value is not False:
            raise ValueError(f"Mistral {key}={value!r} is not supported (no bias tensors are mapped)")
    tied = cfg.get("tie_word_embeddings")
    if tied is not None and tied is not False:
        raise ValueError(
            f"Mistral tie_word_embeddings={tied!r} is not supported: only untied embeddings "
            "(false or absent, as in Mistral-7B-v0.2 / v0.3)"
        )
    if cfg.get("quantization_config") is not None:
        raise ValueError(
            "Mistral quantization_config is not supported: the converter reads unquantized "
            "weights only (GPTQ / AWQ / bitsandbytes / FP8 exports hold packed tensors)"
        )
    for key, ours, hf in (
        ("num_key_value_heads", "num_attention_heads", "8"),
        ("rms_norm_eps", "1e-5", "1e-6"),
    ):
        if key not in cfg:
            raise ValueError(
                f"Mistral config has no {key}: the converter would use its own fallback ({ours}), "
                f"not HF Mistral's default ({hf}); set it"
            )
    if cfg["num_key_value_heads"] is None:
        raise ValueError(
            "Mistral num_key_value_heads=None is not supported (HF Mistral then uses "
            "num_attention_heads, i.e. no GQA); set it to the number of KV heads"
        )
    for key in _SIZES:
        if not _positive_int(cfg.get(key)):
            raise ValueError(f"Mistral {key}={cfg.get(key)!r} must be a positive integer")
    hidden, heads, groups = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    if heads % groups:
        raise ValueError(
            f"Mistral num_attention_heads={heads} is not a multiple of num_key_value_heads={groups}"
        )
    if hidden % heads:
        raise ValueError(f"Mistral hidden_size={hidden} is not a multiple of num_attention_heads={heads}")
    derived_head_dim = hidden // heads
    # head_dim: null (as recent transformers write it) is hidden_size // heads, as in
    # HF Mistral; Llama's translation passes it on as kv_channels None, which
    # Megatron's validate_args sets to hidden_size // num_attention_heads
    head_dim = cfg.get("head_dim")
    if head_dim is not None and not (_positive_int(head_dim) and head_dim == derived_head_dim):
        raise ValueError(
            f"Mistral head_dim={head_dim!r} is not supported: only hidden_size // "
            f"num_attention_heads ({derived_head_dim}) or null"
        )
    if not _positive_number(cfg["rms_norm_eps"]):
        raise ValueError(f"Mistral rms_norm_eps={cfg['rms_norm_eps']!r} must be a positive finite number")
    return theta
