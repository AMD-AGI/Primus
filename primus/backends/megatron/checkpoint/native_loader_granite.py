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
  * ``attention_multiplier`` -> upstream ``softmax_scale`` through MuP: with ``use_mup``
    Megatron sets ``softmax_scale = sqrt(mup_base_head_dim or 1) / kv_channels``. For
    ``attention_multiplier == 1 / head_dim`` (granite-4.1-3b: 1/64) ``mup_base_head_dim``
    stays null; otherwise it is ``(attention_multiplier * head_dim) ** 2``.
  * ``residual_multiplier``  -> ``GraniteTransformerConfig.residual_multiplier``,
    applied by the layer spec's scaled bias-dropout-add.

``build_model`` builds through Primus' model provider with ``model_type="granite"`` (the
Granite builder training uses), so the converted model has the Granite config class and
layer spec.
"""

import json
import math
import os
import socket
from pathlib import Path
from types import SimpleNamespace

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint import native_loader_llama as llama
from primus.backends.megatron.core.models.granite.granite_args import (
    attention_problems,
    builder_problems,
    layout_problems,
)
from primus.core.utils.validation import positive_finite

# ---------------------------------------------------------------------------
# HF config.json -> Megatron args
# ---------------------------------------------------------------------------
_WHERE = "Granite config.json: "


def _positive_number_or_default(cfg, key, default):
    """``cfg[key]`` as a positive finite float; ``default`` (the HF ``GraniteConfig``
    default) when the key is missing. An explicit null is an error."""
    if key not in cfg:
        return float(default)
    return positive_finite(key, cfg[key], _WHERE)


# Plain RoPE as newer transformers serialize it; anything else changes the rotary math.
_PLAIN_ROPE_TYPES = (None, "default")
_ROPE_PLAIN_KEYS = {"rope_type", "type", "rope_theta"}
_ROPE_DICT_KEYS = ("rope_scaling", "rope_parameters")
# HF GraniteConfig defaults (transformers GraniteConfig.__init__).
_DEFAULT_ROPE_THETA = 10000.0
_DEFAULT_RMS_NORM_EPS = 1e-6


def _rms_norm_eps(cfg):
    """``rms_norm_eps`` as a positive finite float. A missing key takes the HF
    ``GraniteConfig`` default (1e-6, not Megatron's ``norm_epsilon`` default 1e-5); null or a
    non-number is an error."""
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
                f"Granite hidden_size={cfg['hidden_size']} is not a multiple of "
                f"num_attention_heads={heads} and no head_dim given"
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
    """Accept no RoPE scaling: absent, null, ``{}`` or a dict with only ``rope_type`` /
    ``type`` (each absent, null or 'default') and ``rope_theta``; refuse everything else.
    Both type keys are checked on their own, so ``{"rope_type": "default", "type": "linear"}``
    is refused."""
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
    values = [positive_finite(key, value, _WHERE) for key, value in found.items()]
    if len(set(values)) != 1:
        raise ValueError(f"{_WHERE}conflicting RoPE base values {found}")
    return values[0]


_MULTIPLIER_KEYS = ("embedding_multiplier", "logits_scaling", "attention_multiplier", "residual_multiplier")


def check_supported(cfg):
    """Raise if this converter cannot convert the HF config ``cfg`` exactly; else return
    a normalised copy of ``cfg``: resolved ``rope_theta``, ``rms_norm_eps`` and the four
    multipliers; a null ``head_dim`` becomes ``hidden_size // num_attention_heads`` and a
    null ``num_key_value_heads`` becomes ``num_attention_heads``.

    Used by ``derive_megatron_fields`` (so the SFT checkpoint hook stops with this message
    for a Granite config it cannot convert) and by ``preset_mismatches``, which the
    ``build_args`` patch (``patches/granite_checkpoint_args_patches.py``) calls through
    ``check_args_against_checkpoint`` before a run that loads a Granite checkpoint builds
    its trainer and model.
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
    return _fields_from_checked(check_supported(cfg))


def _fields_from_checked(cfg):
    """``derive_megatron_fields`` for a config ``check_supported`` has already returned."""
    fields = llama.derive_megatron_fields(cfg)
    fields.update(
        use_mup=True,
        mup_embedding_mult=cfg["embedding_multiplier"],
        mup_output_mult=1.0 / cfg["logits_scaling"],
        # attention_multiplier becomes upstream's MuP softmax scale (see the module doc).
        # No MuP width scaling: the multipliers are fixed constants of the checkpoint.
        mup_base_hidden_size=None,
        mup_base_head_dim=mup_base_head_dim_for(cfg["attention_multiplier"], fields["kv_channels"]),
        mup_attn_scale_power=1.0,
        residual_multiplier=cfg["residual_multiplier"],
    )
    return fields


# Float tolerances. Both compare values read from JSON / YAML (decimal literals) with
# values derived from them by one or two float operations, so the error is a few ulps.
# ``_SCALE_REL_TOL`` decides whether attention_multiplier is exactly 1 / head_dim (then no
# mup_base_head_dim is needed); ``_PRESET_REL_TOL`` is looser because the preset's softmax
# scale goes through sqrt and a power before it is compared.
_SCALE_REL_TOL = 1e-12
_PRESET_REL_TOL = 1e-9


def mup_base_head_dim_for(attention_multiplier, kv_channels):
    """``mup_base_head_dim`` that makes upstream's MuP softmax scale equal HF's
    ``attention_multiplier`` (None when it is already ``1 / kv_channels``)."""
    if math.isclose(attention_multiplier * kv_channels, 1.0, rel_tol=_SCALE_REL_TOL, abs_tol=0.0):
        return None
    # Upstream: softmax_scale = sqrt(mup_base_head_dim) / kv_channels (mup_attn_scale_power 1).
    return (attention_multiplier * kv_channels) ** 2


# Model-preset keys that must equal the converter's values for this checkpoint
# (derive_megatron_fields), with the Megatron default used when the preset does not set
# the key (``_REQUIRED``: no default, the preset must set it).
_REQUIRED = object()
PRESET_KEYS = {
    "hidden_size": _REQUIRED,
    "num_attention_heads": _REQUIRED,
    "num_layers": _REQUIRED,
    "ffn_hidden_size": _REQUIRED,
    "num_query_groups": _REQUIRED,  # compared as built: num_attention_heads without GQA (unset ok)
    "kv_channels": None,  # Megatron default: hidden_size // num_attention_heads
    # Not compared: max_position_embeddings. With RoPE (no learned position table) it only
    # sizes the rotary cache and does not change the model math; the converter itself builds
    # with max(config.max_position_embeddings, seq_length) (llama.build_margs), and SFT and
    # long-context runs set it to their own sequence length.
    "norm_epsilon": 1e-5,
    "rotary_base": 10000,
    "untie_embeddings_and_output_weights": False,
    "use_mup": False,
    "mup_embedding_mult": 1.0,
    "mup_output_mult": 1.0,
    "residual_multiplier": _REQUIRED,
}


def _same(a, b):
    # Strings compare only with strings: a quoted number ('0.22', '1') is not accepted here,
    # because the builder refuses it at build time.
    if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return type(a) is type(b) and a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=_PRESET_REL_TOL, abs_tol=0.0)
    except (TypeError, ValueError):
        return a == b


def _head_dim(values):
    """Megatron's default ``kv_channels`` (hidden / heads), or None when not derivable."""
    try:
        return int(values["hidden_size"]) // int(values["num_attention_heads"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def preset_mismatches(hf_cfg, preset):
    """Where the model preset ``preset`` (a dict of Megatron args, e.g. the post_trainer
    params) disagrees with the HF Granite config ``hf_cfg``; an empty list when it matches.

    * Each ``PRESET_KEYS`` value is compared with ``derive_megatron_fields(hf_cfg)``; a key
      the preset does not set takes the Megatron default (a missing ``mup_embedding_mult``
      is 1.0, not the checkpoint's embedding_multiplier; a missing ``kv_channels`` is hidden / heads). The
      ``_REQUIRED`` keys have no usable default and are reported when missing.
      ``num_query_groups`` is compared as Megatron builds it: without
      ``group_query_attention`` it is ``num_attention_heads`` (plain MHA).
    * ``granite_args.GRANITE_LAYOUT`` (shared with the builder) must be the Llama layout the
      converter builds (RMSNorm, SwiGLU, no
      linear or QKV biases, full RoPE, no QK norm, no 1p norm gain, pre-norm residual).
    * ``vocab_size``, when the preset sets it, must be config.json's (else the tokenizer
      gives it).
    * Every other option the Granite builder refuses (``granite_args.builder_problems``,
      which the builder calls through ``unsupported_problems``): MoE, MTP, MLA, RoPE scaling / interleaving /
      interpolation, MuP width scaling (``mup_base_hidden_size`` != ``hidden_size``), ...
    * The ``granite_args.GRANITE_ATTENTION_OFF`` options must be unset: an explicit
      ``softmax_scale`` (which wins over MuP), sliding window, output gate, NoPE layers,
      QK layer scaling, attention sinks.
    * The softmax scale upstream MuP derives from the preset
      (``sqrt(mup_base_head_dim or 1) / kv_channels ** mup_attn_scale_power``) must be HF
      ``attention_multiplier``.

    Training builds from the preset, not from config.json, so a mismatch trains a
    different model on the converted weights. Raises ``ValueError`` / ``NotImplementedError``
    (from ``check_supported``) when ``hf_cfg`` itself is not a supported Granite config."""
    resolved = check_supported(hf_cfg)
    fields = _fields_from_checked(resolved)
    out = []
    values = {}
    gqa = preset.get("group_query_attention")
    gqa = False if gqa is None else gqa
    for key, default in PRESET_KEYS.items():
        if key == "num_query_groups":
            if not isinstance(gqa, bool):
                out.append(f"group_query_attention: preset {gqa!r}, must be true or false (a boolean)")
                continue
            if not gqa:
                # Without GQA Megatron builds one group per head, whatever num_query_groups
                # says (also when it is not set): compare that.
                value = values.get("num_attention_heads")
                values[key] = value
                if not _same(value, fields[key]):
                    out.append(
                        f"num_query_groups: group_query_attention is not true in the preset, so "
                        f"Megatron builds {value!r} groups (one per head); config.json gives {fields[key]!r}"
                    )
                continue
        value = preset.get(key)
        if value is None:
            if default is _REQUIRED:
                out.append(f"{key}: not set in the preset, config.json gives {fields[key]!r}")
                continue
            # Unset or explicit null: Megatron uses its default (an unset store_true flag
            # such as untie_embeddings_and_output_weights is False), as granite_args does.
            value = _head_dim(values) if key == "kv_channels" else default
        values[key] = value
        if not _same(value, fields[key]):
            out.append(f"{key}: preset {value!r}, config.json gives {fields[key]!r}")
    layout = layout_problems(preset.get)
    for key, _, value, want in layout:
        out.append(f"{key}: preset {value!r}, the Granite (Llama) layout needs {want!r}")
    attention = attention_problems(preset.get)
    for key, reason, value in attention:
        out.append(f"{key}: preset {value!r}, must be unset or off ({reason})")
    # The other options the Granite builder refuses (the builder's own list).
    for label in builder_problems(preset.get):
        out.append(f"{label}: set in the preset, the Granite builder refuses it")
    if preset.get("vocab_size") is not None and not _same(preset["vocab_size"], resolved.get("vocab_size")):
        out.append(
            f"vocab_size: preset {preset['vocab_size']!r}, config.json gives {resolved.get('vocab_size')!r}"
        )
    kv = values.get("kv_channels")
    # mup_base_head_dim / mup_attn_scale_power only matter through the softmax scale they
    # give, compared with attention_multiplier (a null power is Megatron's 1.0).
    base, power = preset.get("mup_base_head_dim"), preset.get("mup_attn_scale_power")
    power = 1.0 if power is None else power
    if values.get("use_mup") and kv:
        try:
            scale = math.sqrt(1.0 if base is None else float(base)) / float(kv) ** float(power)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            scale = None
        if not _same(scale, resolved["attention_multiplier"]):
            out.append(
                f"softmax scale: the preset gives {scale!r} (sqrt(mup_base_head_dim or 1) / "
                f"kv_channels ** mup_attn_scale_power), config.json attention_multiplier is "
                f"{resolved['attention_multiplier']!r}"
            )
    return out


def check_preset(hf_cfg, preset, where="the model preset"):
    """Raise ``ValueError`` listing every ``preset_mismatches`` entry."""
    bad = preset_mismatches(hf_cfg, preset)
    if bad:
        raise ValueError(
            f"{where} does not match the Granite checkpoint's config.json ({len(bad)} value(s)): "
            + "; ".join(bad)
        )


# The HF config.json the converter saves into the converted checkpoint's iteration directory,
# so a run that loads it can check its model args without the HF directory (hf_path may be a
# hub id).
HF_CONFIG_COPY = "granite_hf_config.json"


# Megatron's tracker file: a directory holds a checkpoint when it has one
# (megatron.training.checkpointing.checkpoint_exists).
TRACKER_FILE = "latest_checkpointed_iteration.txt"


def _holds_checkpoint(directory):
    return isinstance(directory, str) and bool(directory) and (Path(directory) / TRACKER_FILE).is_file()


def checkpoint_to_load(args):
    """The directory Megatron's ``load_checkpoint`` will load, or None: ``load`` when it holds
    a checkpoint (a resume), else ``pretrained_checkpoint`` when it holds one (fine-tuning),
    else nothing (training from scratch)."""
    load = getattr(args, "load", None)
    if _holds_checkpoint(load):
        return load
    pretrained = getattr(args, "pretrained_checkpoint", None)
    return pretrained if _holds_checkpoint(pretrained) else None


def _iter_dir_name(iteration, release=False):
    """Megatron's checkpoint directory name (``checkpointing.get_checkpoint_name(...,
    return_base_dir=True)``): ``release``, or ``iter_{:07d}``. Megatron's own function is not
    imported here because ``megatron.training.checkpointing`` imports torch and probes the
    GPU, and this module runs before distributed init (build_args) and in CPU-only hooks."""
    return "release" if release else "iter_{:07d}".format(int(iteration))


def iteration_dir(checkpoint_dir, ckpt_step=None):
    """The directory of the one checkpoint Megatron loads from ``checkpoint_dir``:
    ``ckpt_step``'s when set, else the tracker file's iteration (``release`` when the tracker
    says so). None when the tracker cannot be read."""
    if ckpt_step:
        return Path(checkpoint_dir) / _iter_dir_name(ckpt_step)
    try:
        meta = (Path(checkpoint_dir) / TRACKER_FILE).read_text().strip()
    except OSError:
        return None
    if meta == "release":
        return Path(checkpoint_dir) / _iter_dir_name(0, release=True)
    try:
        return Path(checkpoint_dir) / _iter_dir_name(meta)
    except ValueError:
        return None


def _read_hf_config_copy(copy):
    try:
        saved = json.loads(copy.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(
            f"cannot read the Granite HF config copy {copy} (saved with the checkpoint to "
            f"check the model args): {e}. Restore it from the HF checkpoint's config.json "
            "or delete it"
        ) from e
    if not isinstance(saved, dict):
        raise ValueError(f"the Granite HF config copy {copy} is not a JSON object")
    return saved


def _copy_in(checkpoint_dir, ckpt_step=None):
    """``HF_CONFIG_COPY`` in the iteration Megatron loads from ``checkpoint_dir``, or None."""
    it_dir = iteration_dir(checkpoint_dir, ckpt_step)
    copy = it_dir / HF_CONFIG_COPY if it_dir is not None else None
    return copy if copy is not None and copy.is_file() else None


def config_copy_for(args):
    """``(copy, how)``: the ``HF_CONFIG_COPY`` the args are checked against, or ``(None, why)``.

    * the copy in the iteration Megatron loads (``checkpoint_to_load``; ``ckpt_step`` or the
      tracker's iteration), which ``convert`` writes;
    * for a resume (``load`` holds a checkpoint without a copy, e.g. one saved by an earlier
      fine-tuning run) with ``pretrained_checkpoint`` set: the copy of the converted checkpoint
      there (its tracker's iteration). The resumed weights were trained from it with the args
      that were checked when that run started, so a resume with other Granite multipliers is
      refused too;
    * else none: Megatron loads the checkpoint as usual and the check is skipped."""
    source = checkpoint_to_load(args)
    if source is None:
        return None, None
    copy = _copy_in(source, getattr(args, "ckpt_step", None))
    if copy is not None:
        return copy, "the loaded checkpoint"
    pretrained = getattr(args, "pretrained_checkpoint", None)
    if source != pretrained and _holds_checkpoint(pretrained):
        copy = _copy_in(pretrained)
        if copy is not None:
            return copy, f"pretrained_checkpoint (the resumed checkpoint {source} was trained from it)"
    return None, f"the checkpoint loaded from {source}"


def check_args_against_checkpoint(args, log=print):
    """Raise ``ValueError`` when the model args differ from the HF config of the checkpoint
    this run loads (``preset_mismatches``); return that config, or None when there is nothing
    to check.

    Called once per run by the ``build_args`` patch
    ``patches/granite_checkpoint_args_patches.py``, before the trainer and the model are
    built. Training builds the model from the args, not from config.json, so without this
    check a preset with, for example, a wrong ``residual_multiplier`` would train a
    different model on the loaded weights without an error.

    The config is the ``HF_CONFIG_COPY`` file ``config_copy_for`` finds: the one in the
    loaded iteration's directory (``convert`` writes it), or, for a resume of a fine-tuning
    run, the one of its ``pretrained_checkpoint``. A local ``hf_path`` / ``tokenizer_model``
    directory may hold another model or revision and is never used. When there is no copy
    (a resume without ``pretrained_checkpoint``, a Megatron-Bridge conversion, a checkpoint
    from another tool), Megatron loads the checkpoint as usual: the check is skipped with a
    warning, and nothing is refused. An unreadable copy raises."""
    copy, how = config_copy_for(args)
    if copy is None:
        if how is not None:
            log(
                f"[Primus:Granite] WARNING: {how} has no {HF_CONFIG_COPY} in its iteration directory "
                "(and no pretrained_checkpoint with one), so the Granite model args (residual_multiplier "
                "and the MuP multipliers) are not checked against its weights. Checkpoints made by the "
                "Granite converter carry it; for another checkpoint, copy the HF config.json of its "
                f"weights there as {HF_CONFIG_COPY} to enable the check"
            )
        return None
    hf_cfg = _read_hf_config_copy(copy)
    values = dict(vars(args))
    # vocab_size is not compared here: the tokenizer gives the padded vocab, as at
    # conversion time (both use make_vocab_size_divisible_by), and loading the checkpoint
    # checks the embedding shape.
    values.pop("vocab_size", None)
    check_preset(
        hf_cfg if hf_cfg.get("vocab_size") is not None else {**hf_cfg, "vocab_size": 1},
        values,
        where=f"the Granite model args (checked against the config of {how}: {copy})",
    )
    log(f"[Primus:Granite] model args match the HF config {copy} ({how})")
    return hf_cfg


# The Granite fields (from derive_megatron_fields) set on the Llama Megatron args.
GRANITE_EXTRA_FIELDS = (
    "use_mup",
    "mup_embedding_mult",
    "mup_output_mult",
    "mup_base_head_dim",
    "mup_attn_scale_power",
    "residual_multiplier",
    "mup_base_hidden_size",
)


def build_margs(cfg_fields, args):
    """Llama Megatron args (``llama.build_margs``) plus the Granite multipliers and
    ``model_type: granite``.

    The Granite fields are set after Llama's ``validate_args``: none of them is read by
    ``validate_args``. They are read when ``GraniteTransformerConfig`` is built from the
    args (the MuP fields by upstream ``TransformerConfig``, ``residual_multiplier`` by the
    Granite config class, which checks it).
    """
    m = llama.build_margs(cfg_fields, args)
    for key in GRANITE_EXTRA_FIELDS:
        setattr(m, key, cfg_fields[key])
    m.model_type = "granite"
    return m


def build_model(margs):
    """Build the Granite ``GPTModel`` with the Granite builder that training uses
    (``get_model_provider("granite")``), so the converted parameter names and the config
    class match SFT."""
    return common.build_mcore_gpt_model(margs, model_type="granite")


# ---------------------------------------------------------------------------
# HF -> mcore weight mapping (identical to Llama; no weight is rescaled)
# ---------------------------------------------------------------------------
# The weight mapping is Llama's (derive_megatron_fields refuses biased checkpoints).
build_mcore_state_dict = llama.build_mcore_state_dict


# ---------------------------------------------------------------------------
# In-process entrypoint (the steps of native_loader_llama.convert, Granite's functions)
# ---------------------------------------------------------------------------
def _write_json_atomically(obj, target):
    """Write ``obj`` as JSON to ``target`` atomically (a temporary file per process, then
    ``os.replace``), so a killed process leaves either no file or a complete one, and two
    processes writing the same path never mix their files."""
    target = Path(target)
    tmp = target.with_name(f"{target.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
        os.replace(tmp, target)
    finally:
        # A failed write or replace (ENOSPC, permissions) leaves no stray temporary file.
        if tmp.exists():
            tmp.unlink()


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
    """Convert a Granite HF checkpoint to a legacy Megatron torch checkpoint.

    Single process, CPU-only, same options as ``native_loader_llama.convert``. Assumes the
    ``phase="convert"`` patches are applied in this process (the conversion hook does so).
    """
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

    log(f"reading HF config from {hf_dir}")
    hf_cfg = common.read_hf_config(hf_dir)
    cfg_fields = derive_megatron_fields(hf_cfg)
    log(
        f"arch: L={cfg_fields['num_layers']} h={cfg_fields['hidden_size']} "
        f"heads={cfg_fields['num_attention_heads']} kv_groups={cfg_fields['num_query_groups']} "
        f"head_dim={cfg_fields['kv_channels']} ffn={cfg_fields['ffn_hidden_size']} "
        f"rope_theta={cfg_fields['rotary_base']} emb_mult={cfg_fields['mup_embedding_mult']} "
        f"output_mult={cfg_fields['mup_output_mult']} residual_mult={cfg_fields['residual_multiplier']}"
    )

    margs = build_margs(cfg_fields, args)
    margs = common.init_megatron_single_process(margs, master_port, log=log)
    common.assert_bridge_free("post-init")

    log("building Granite GPTModel on CPU ...")
    model = build_model(margs)

    store = common.SafetensorsStore(hf_dir)
    log("mapping HF safetensors -> mcore parameters ...")
    sd, consumed = build_mcore_state_dict(model, store, margs, cfg_fields, log)

    # load_validate_save raises before Megatron's save_checkpoint when the mapping is not exact:
    # then no checkpoint and no copy are written.
    common.load_validate_save(model, sd, consumed, store, margs, save_dir, log)
    # The fine-tuning run checks its model args against this copy (check_args_against_checkpoint).
    # Megatron writes its tracker file last inside save_checkpoint, so the copy follows it. If
    # writing the copy fails, the tracker is removed: the conversion hook skips a directory whose
    # tracker exists, so without it the next run converts again instead of loading unchecked.
    it_dir = Path(save_dir) / _iter_dir_name(margs.iteration)
    try:
        if not it_dir.is_dir():
            raise RuntimeError(f"no checkpoint for iteration {margs.iteration} in {save_dir} after the save")
        _write_json_atomically(hf_cfg, it_dir / HF_CONFIG_COPY)
    except Exception:
        tracker = Path(save_dir) / TRACKER_FILE
        if tracker.exists():
            tracker.unlink()
        raise
    log(f"saved the HF config with the checkpoint: {it_dir / HF_CONFIG_COPY}")
    common.assert_bridge_free("post-save")
    return save_dir
