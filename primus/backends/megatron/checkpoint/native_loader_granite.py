###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for IBM Granite dense
(``GraniteForCausalLM``, e.g. ``ibm-granite/granite-4.1-3b``).

Granite has the Llama weight layout (GQA, RoPE, SwiGLU, RMSNorm, no biases), so the
weight mapping is the Llama one (``native_loader_llama.build_mcore_state_dict``).
The differences are four scalar multipliers in ``config.json``. They are model
settings, not weight transforms, and no weight is rescaled here:

  * ``embedding_multiplier`` -> ``use_mup`` + ``mup_embedding_mult``
  * ``logits_scaling``       -> ``mup_output_mult = 1 / logits_scaling``
  * ``attention_multiplier`` -> ``GraniteTransformerConfig.attention_multiplier``,
    which sets ``softmax_scale`` explicitly (HF uses it directly as the scaling).
  * ``residual_multiplier``  -> ``GraniteTransformerConfig.residual_multiplier``,
    applied in ``GraniteTransformerLayer`` (scaled bias-dropout-add).

``build_model`` builds through the Granite builder (the same one training uses),
so the converted model has the Granite layer and config classes.
"""

from types import SimpleNamespace

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint import native_loader_llama as llama
from primus.backends.megatron.core.models.granite.granite_validation import (
    positive_finite,
)

# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
_WHERE = "Granite config.json: "


def _positive_number(key, value):
    """``value`` as a positive finite float (shared ``positive_finite``): only JSON numbers
    (int/float, not bool) are accepted."""
    return positive_finite(key, value, _WHERE)


def _positive_number_or_default(cfg, key, default):
    """``cfg[key]`` as a positive finite float; ``default`` (the HF ``GraniteConfig``
    default) when the key is missing. An explicit null is an error."""
    if key not in cfg:
        return float(default)
    return _positive_number(key, cfg[key])


# Plain RoPE as newer transformers serialize it; anything else changes the rotary math.
_PLAIN_ROPE_TYPES = (None, "default")
_ROPE_PLAIN_KEYS = {"rope_type", "type", "rope_theta"}
_ROPE_DICT_KEYS = ("rope_scaling", "rope_parameters")
# HF GraniteConfig defaults (transformers GraniteConfig.__init__).
_DEFAULT_ROPE_THETA = 10000.0
_DEFAULT_RMS_NORM_EPS = 1e-6


def _rms_norm_eps(cfg):
    """``rms_norm_eps`` as a positive finite float. A missing key takes the HF
    ``GraniteConfig`` default (1e-6, not Llama's 1e-5); null or a non-number is an error."""
    return _positive_number_or_default(cfg, "rms_norm_eps", _DEFAULT_RMS_NORM_EPS)


def _check_top_level_config(cfg):
    """Refuse a non-dict config or a wrapper config. Only a top-level dense Granite config is accepted:
    a ``text_config`` means a wrapper (e.g. a vision-language model whose weights are
    prefixed ``language_model.*``), which the Llama weight mapping cannot read."""
    if not isinstance(cfg, dict):
        raise ValueError(f"{_WHERE}must be a JSON object, got {type(cfg).__name__}")
    if "text_config" in cfg:
        raise NotImplementedError(
            "Granite config with a text_config (a wrapper model, "
            f"architectures={cfg.get('architectures')!r}) is not supported; only GraniteForCausalLM"
        )


def _check_model_type(cfg):
    """Dense Granite only: ``model_type`` must be 'granite' and ``architectures``, when
    given, must be exactly ``["GraniteForCausalLM"]``. granitemoe / granitemoehybrid and
    task heads share the multiplier keys but not the layout."""
    model_type = cfg.get("model_type")
    architectures = cfg.get("architectures")
    if model_type != "granite":
        raise ValueError(f"not a dense Granite config: model_type={model_type!r} (expected 'granite')")
    if architectures is not None and architectures != ["GraniteForCausalLM"]:
        raise ValueError(f"not a dense Granite config: architectures={architectures!r}")


def _positive_int(cfg, key):
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{_WHERE}{key}={value!r} must be a positive integer")
    return value


def _check_llama_layout(cfg):
    """The Llama mapping's assumptions: SiLU-gated MLP (SwiGLU), full rotary, integer
    shapes with GQA groups dividing the heads, and a positive integer head_dim.

    An explicit ``head_dim`` is used as given (as HF ``GraniteAttention`` does); it is not
    required to equal ``hidden_size / num_attention_heads``. Without one, ``hidden_size``
    must be a multiple of ``num_attention_heads``."""
    act = cfg.get("hidden_act", "silu")
    if act not in ("silu", "swish"):
        raise NotImplementedError(f"Granite hidden_act={act!r} is not supported (SwiGLU/silu only)")
    for key in (
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    ):
        _positive_int(cfg, key)
    heads = cfg["num_attention_heads"]
    if cfg.get("num_key_value_heads") is not None:  # absent or null: MHA, as in HF
        kv_heads = _positive_int(cfg, "num_key_value_heads")
        if heads % kv_heads:
            raise ValueError(
                f"Granite num_key_value_heads={kv_heads} does not divide num_attention_heads={heads}"
            )
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        if cfg["hidden_size"] % heads:
            raise ValueError(
                "Granite hidden_size is not a multiple of num_attention_heads and no head_dim given"
            )
    else:
        _positive_int(cfg, "head_dim")
    partial = cfg.get("partial_rotary_factor", 1.0)
    if isinstance(partial, bool) or not isinstance(partial, (int, float)) or partial != 1.0:
        raise NotImplementedError(
            f"Granite partial_rotary_factor={partial!r} is not supported (full RoPE only)"
        )
    if not isinstance(cfg.get("tie_word_embeddings", False), bool):
        raise ValueError(f"{_WHERE}tie_word_embeddings={cfg.get('tie_word_embeddings')!r} must be a bool")


def _check_no_bias(cfg):
    """The weight mapping is the bias-free Llama one. HF ``attention_bias`` adds q/k/v and
    o_proj biases, ``mlp_bias`` adds MLP biases; both are refused before any model is built."""
    for key in ("attention_bias", "mlp_bias"):
        value = cfg.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{_WHERE}{key}={value!r} must be a bool")
        if value:
            raise NotImplementedError(f"Granite checkpoints with {key}=true are not supported")


def _check_plain_rope(cfg):
    """Accept no RoPE scaling (absent, null, ``{}``, or a dict whose ``rope_type`` and
    ``type`` are each absent, null or 'default'); refuse everything else. Both type keys
    are checked on their own, so ``{"rope_type": "default", "type": "linear"}`` is refused."""
    for key in _ROPE_DICT_KEYS:
        rope = cfg.get(key)
        if rope is None:
            continue
        if not isinstance(rope, dict):
            raise NotImplementedError(f"Granite {key}={rope!r} is not supported by this converter")
        types = [rope.get(type_key) for type_key in ("rope_type", "type")]
        if any(t not in _PLAIN_ROPE_TYPES for t in types) or set(rope) - _ROPE_PLAIN_KEYS:
            raise NotImplementedError(
                f"Granite {key} {rope!r} is not supported (only plain RoPE, rope_type 'default')"
            )


def _rope_theta(cfg):
    """RoPE base from top-level ``rope_theta`` or from ``rope_scaling`` / ``rope_parameters``
    (newer transformers). All the values given must agree. A null is treated as not given;
    if no value is given at all the HF default (10000) is used, unless a key was explicitly
    null, which is an error."""
    found = {}
    explicit_null = "rope_theta" in cfg and cfg["rope_theta"] is None
    if cfg.get("rope_theta") is not None:
        found["rope_theta"] = cfg["rope_theta"]
    for key in _ROPE_DICT_KEYS:
        rope = cfg.get(key)
        if isinstance(rope, dict) and "rope_theta" in rope:
            if rope["rope_theta"] is None:
                explicit_null = True
            else:
                found[f"{key}.rope_theta"] = rope["rope_theta"]
    if not found:
        if explicit_null:
            raise ValueError(f"{_WHERE}rope_theta is null and no other RoPE base is given")
        return _DEFAULT_ROPE_THETA
    values = [_positive_number(key, value) for key, value in found.items()]
    if len(set(values)) != 1:
        raise ValueError(f"{_WHERE}conflicting RoPE base values {found}")
    return values[0]


_MULTIPLIER_KEYS = ("embedding_multiplier", "logits_scaling", "attention_multiplier", "residual_multiplier")


def check_supported(cfg):
    """Raise if this converter cannot convert the HF config ``cfg`` exactly; else return
    a normalised copy of ``cfg`` (resolved ``rope_theta``, ``rms_norm_eps`` and the four
    multipliers, and a null
    ``head_dim`` replaced by ``hidden_size // num_attention_heads`` and a null
    ``num_key_value_heads`` by ``num_attention_heads``).

    Used by ``derive_megatron_fields`` and by the SFT checkpoint hook. The hook stops with
    this message for every Granite config it refuses: when ``native_ckpt_convert`` is unset
    (the default) or ``true``, because the pinned Megatron-Bridge has no Granite bridge to
    fall back to (and it refuses ``native_ckpt_convert: false`` for Granite up front).
    """
    _check_top_level_config(cfg)
    _check_model_type(cfg)
    _check_llama_layout(cfg)
    _check_no_bias(cfg)
    _check_plain_rope(cfg)
    resolved = {"rope_theta": _rope_theta(cfg), "rms_norm_eps": _rms_norm_eps(cfg)}
    if cfg.get("head_dim") is None:
        # Absent or explicit null: derive like the Llama layout (hidden / heads; the
        # divisibility is checked in _check_llama_layout), never pass None on.
        resolved["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
    if cfg.get("num_key_value_heads") is None:
        # Absent or explicit null: multi-head attention, as HF GraniteConfig does.
        resolved["num_key_value_heads"] = cfg["num_attention_heads"]
    for key in _MULTIPLIER_KEYS:
        # HF GraniteConfig default for every multiplier is 1.0.
        resolved[key] = _positive_number_or_default(cfg, key, 1.0)
    return {**cfg, **resolved}


def derive_megatron_fields(cfg):
    """Translate a Granite HF config dict into Megatron arg values.

    Missing multipliers take the HF ``GraniteConfig`` defaults (all 1.0, including
    ``attention_multiplier``, which HF uses directly as the attention scaling); a missing
    ``rms_norm_eps`` takes the GraniteConfig default 1e-6 and a missing ``rope_theta`` 1e4.
    """
    cfg = check_supported(cfg)
    fields = llama.derive_megatron_fields(cfg)
    fields.update(
        use_mup=True,
        mup_embedding_mult=cfg["embedding_multiplier"],
        mup_output_mult=1.0 / cfg["logits_scaling"],
        # The softmax scale is set explicitly from attention_multiplier
        # (GraniteTransformerConfig.attention_multiplier -> softmax_scale), so the MuP
        # attention-scale fields stay at their defaults. No MuP width scaling: the
        # multipliers are fixed constants of the checkpoint.
        mup_base_hidden_size=None,
        mup_base_head_dim=None,
        mup_attn_scale_power=1.0,
        residual_multiplier=cfg["residual_multiplier"],
        attention_multiplier=cfg["attention_multiplier"],
    )
    return fields


GRANITE_KEYS = (
    "use_mup",
    "mup_embedding_mult",
    "mup_output_mult",
    "mup_base_head_dim",
    "mup_attn_scale_power",
    "residual_multiplier",
    "attention_multiplier",
    "mup_base_hidden_size",
)
# The only keys Granite may add to the Llama args (llama.build_margs allowlist).
# residual_multiplier / attention_multiplier are not Megatron arguments; they are
# checked by GraniteTransformerConfig (built from these args).
GRANITE_ALLOWED_FIELDS = frozenset(GRANITE_KEYS) | {"model_type"}


def build_margs(cfg_fields, args):
    """Llama Megatron args plus the Granite multipliers and ``model_type: granite``.

    The Granite fields are set before ``validate_args`` runs (``extra_fields``). The MuP
    fields are Megatron arguments; ``residual_multiplier`` / ``attention_multiplier`` are
    not, and are checked when ``GraniteTransformerConfig`` is built from these args.
    """
    extra = {k: cfg_fields[k] for k in GRANITE_KEYS}
    extra["model_type"] = "granite"
    return llama.build_margs(cfg_fields, args, extra_fields=extra, allowed_fields=GRANITE_ALLOWED_FIELDS)


def build_model(margs):
    """Build the Granite ``GPTModel`` through Primus' model-provider resolver.

    The provider builds from Megatron's global args, so ``margs`` must be that object
    (``init_megatron_single_process`` installs it); this is checked, not assumed.
    """
    from megatron.training import get_args

    from primus.core.utils.import_utils import get_model_provider

    if get_args() is not margs:
        raise RuntimeError(
            "build_model: margs is not Megatron's global args; call init_megatron_single_process first"
        )
    # Same resolver as training, so parameter names match what training builds.
    model_provider = get_model_provider(model_type="granite")
    model = model_provider(pre_process=True, post_process=True)
    return model.to(margs.params_dtype)


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping (identical to Llama; no weight is rescaled)
# ---------------------------------------------------------------------------
def build_mcore_state_dict(model, store, margs, cfg_fields, log):
    # derive_megatron_fields already refuses biased checkpoints; this guards direct callers.
    if margs.add_bias_linear or margs.add_qkv_bias:
        raise NotImplementedError("Granite checkpoints with attention/MLP biases are not supported")
    return llama.build_mcore_state_dict(model, store, margs, cfg_fields, log)


# ---------------------------------------------------------------------------
# In-process entrypoint
# ---------------------------------------------------------------------------
def convert(
    hf_dir,
    save_dir,
    *,
    dtype="bf16",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    seq_length=4096,
    make_vocab_size_divisible_by=128,
    master_port=29563,
    log=None,
):
    """Convert a Granite HF checkpoint to a legacy Megatron torch checkpoint (single process, CPU)."""
    if log is None:

        def log(*a):
            print("[granite-convert]", *a, flush=True)

    common.ensure_megatron_on_path()
    common.assert_bridge_free("entry")
    args = SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=save_dir,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        seq_length=seq_length,
        make_vocab_size_divisible_by=make_vocab_size_divisible_by,
    )
    fields = derive_megatron_fields(common.read_hf_config(hf_dir))
    log(f"granite fields: {fields}")
    margs = build_margs(fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    model = build_model(margs)
    store = common.SafetensorsStore(hf_dir)
    sd, consumed = build_mcore_state_dict(model, store, margs, fields, log)
    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    common.assert_bridge_free("post-save")
    return save_dir
