###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Native, bridge-free HuggingFace -> Megatron-Core converter for dense Mistral
checkpoints with full causal attention (Mistral-7B-v0.2 / v0.3).

The weight layout is Llama's, so this is ``native_loader_llama`` behind the
Mistral config checks of ``native_config_mistral`` (which lists them).
``derive_megatron_fields`` is the checks plus Llama's translation of the config,
with the one checked RoPE base passed on as ``rope_theta`` (a base given only under
``rope_parameters``, as transformers v5 writes it, would otherwise be read as
10000). ``convert`` is Llama's ``convert`` with this translation. The other two
native-loader functions, ``build_margs`` and ``build_mcore_state_dict``, take the
translated fields, so they are Llama's as they are (callers that build and load the model step by
step, such as parity tests, use the native-loader interface).

The conversion hook, ``01_convert_checkpoints.py``, converts the ``mistral``
family (``MistralForCausalLM``) with ``convert``. Weights are read with
``native_convert_common.SafetensorsStore`` (HF ``model*.safetensors`` or
``pytorch_model.bin``), never Mistral's ``consolidated.safetensors``.
"""

from primus.backends.megatron.checkpoint import native_loader_llama as _llama
from primus.backends.megatron.checkpoint.native_config_mistral import check_config


def derive_megatron_fields(cfg):
    """Native-loader interface: check the Mistral config, then Llama's translation of
    it with the checked RoPE base as ``rope_theta`` (the same fields ``convert`` uses).
    For ``head_dim: null`` Llama's translation gives ``kv_channels`` None, which
    Megatron's ``validate_args`` sets to hidden_size // num_attention_heads, as HF
    Mistral does."""
    return _llama.derive_megatron_fields(dict(cfg, rope_theta=check_config(cfg)))


# Native-loader interface: Llama's, as they are (Mistral's weight layout is Llama's).
build_margs = _llama.build_margs
build_mcore_state_dict = _llama.build_mcore_state_dict


def _log(*a):
    print("[mistral-convert]", *a, flush=True)


def _derive_fields_for_convert(cfg):
    """``derive_megatron_fields`` with the hint to the Megatron-Bridge path on a failed check."""
    try:
        return derive_megatron_fields(cfg)
    except ValueError as e:
        raise ValueError(f"{e}. Set native_ckpt_convert: false to use the Megatron-Bridge path.") from e


def convert(hf_dir, save_dir, **kwargs):
    """Convert a Mistral HF checkpoint to a legacy Megatron torch checkpoint:
    ``native_loader_llama.convert`` with this module's translation. Its config checks
    run once, when Llama's convert translates config.json, before Megatron is set up
    or any weight is read; a failed check raises ``ValueError``."""
    kwargs.setdefault("log", _log)
    return _llama.convert(hf_dir, save_dir, derive_fields=_derive_fields_for_convert, **kwargs)
