###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU tests: Mistral routing to the native Llama converter and its HF config translation."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
HOOK = REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"

# config.json of mistralai/Mistral-7B-v0.3 @ caa1feb0e54d415e2df31207e5f4e273e33509b1
MISTRAL_V03 = {
    "architectures": ["MistralForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 32768,
}


def _hook():
    spec = importlib.util.spec_from_file_location("convert_checkpoints_hook", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _loader():
    pytest.importorskip("torch")
    from primus.backends.megatron.checkpoint import native_loader_llama

    return native_loader_llama


def test_mistral_v03_routes_to_native_llama(tmp_path):
    hook = _hook()
    (tmp_path / "config.json").write_text(json.dumps(MISTRAL_V03))
    assert hook._detect_native_family(str(tmp_path)) == "llama"
    assert hook._native_family_from_config({"model_type": "mistral", "sliding_window": None}) == "llama"


@pytest.mark.parametrize(
    "cfg",
    [
        dict(MISTRAL_V03, sliding_window=4096),  # Mistral-7B-v0.1
        dict(MISTRAL_V03, model_type="Mistral", sliding_window=4096),
        {"architectures": ["MistralForCausalLM"], "sliding_window": 4096},
        {"model_type": "mistral", "sliding_window": 4096},
    ],
)
def test_windowed_mistral_falls_back_to_bridge(cfg):
    assert _hook()._native_family_from_config(cfg) is None


def test_other_families_unchanged():
    hook = _hook()
    assert hook._native_family_from_config({"architectures": ["LlamaForCausalLM"]}) == "llama"
    assert hook._native_family_from_config({"model_type": "qwen3"}) == "qwen3"
    assert hook._native_family_from_config({"architectures": ["DeepseekV3ForCausalLM"]}) == "deepseek"
    assert hook._native_family_from_config({"architectures": ["MixtralForCausalLM"]}) is None


def test_derive_fields_mistral_v03():
    f = _loader().derive_megatron_fields(MISTRAL_V03)
    assert f == dict(
        hidden_size=4096,
        num_attention_heads=32,
        num_layers=32,
        max_position_embeddings=32768,
        norm_epsilon=1e-05,
        rotary_base=1000000.0,
        untie_embeddings_and_output_weights=True,
        ffn_hidden_size=14336,
        num_query_groups=8,
        kv_channels=128,
        add_qkv_bias=False,
    )


@pytest.mark.parametrize("model_type", ["mistral", "Mistral", None])
def test_derive_fields_rejects_sliding_window(model_type):
    cfg = dict(MISTRAL_V03, sliding_window=4096)
    if model_type is None:
        cfg.pop("model_type")
    else:
        cfg["model_type"] = model_type
    with pytest.raises(ValueError, match="sliding_window"):
        _loader().derive_megatron_fields(cfg)


def test_derive_fields_head_dim_null_and_rope_parameters():
    cfg = dict(MISTRAL_V03, head_dim=None)
    cfg.pop("rope_theta")
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000.0}
    f = _loader().derive_megatron_fields(cfg)
    assert f["kv_channels"] == 128
    assert f["rotary_base"] == 1000000.0
