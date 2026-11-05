###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared values of the Granite unit tests: the HF config.json of ibm-granite/granite-4.1-3b
and the shipped model preset."""

GRANITE_HF = {
    "architectures": ["GraniteForCausalLM"],
    "attention_bias": False,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12.0,
    "hidden_size": 2560,
    "intermediate_size": 8192,
    "logits_scaling": 10.0,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "granite",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "residual_multiplier": 0.22,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "tie_word_embeddings": True,
    "vocab_size": 100352,
}


def preset():
    """The Granite model preset resolved by Primus's own preset loader (extends chain,
    deep merge and env replacement, as at runtime)."""
    from primus.core.config.preset_loader import PresetLoader

    return PresetLoader.load("granite_4_1_3b", framework="megatron")


def shipped_preset():
    """The shipped Granite model preset as a dict."""
    return dict(preset())
