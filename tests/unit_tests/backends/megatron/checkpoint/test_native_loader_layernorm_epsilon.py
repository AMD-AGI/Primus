###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The native converters' build_margs (llama, qwen3, deepseek) set ``layernorm_epsilon`` (the dest of
--norm-epsilon, the TransformerConfig field the model reads) from config.json's
rms_norm_eps. Before, they set only ``norm_epsilon``, which Megatron does not read, so the
model was built with Megatron's default 1e-5."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from tests.unit_tests.backends.megatron.checkpoint.native_loader_test_utils import (  # noqa: E402
    margs_args,
    require_megatron_training,
)

LLAMA_CFG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
}


# Not 1e-5: that is Megatron's default and would pass without the converter setting it.
@pytest.mark.parametrize("eps", [1e-6, 3e-6])
def test_llama_build_margs_sets_layernorm_epsilon(tmp_path, eps):
    require_megatron_training()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    fields = llama.derive_megatron_fields({**LLAMA_CFG, "rms_norm_eps": eps})
    m = llama.build_margs(fields, margs_args())
    assert m.layernorm_epsilon == eps


@pytest.mark.parametrize("eps", [1e-6, 3e-6])
def test_qwen3_build_margs_sets_layernorm_epsilon(tmp_path, eps):
    require_megatron_training()
    from primus.backends.megatron.checkpoint import native_loader_qwen3 as qwen3

    cfg = {**LLAMA_CFG, "rms_norm_eps": eps, "head_dim": 16, "tie_word_embeddings": True}
    fields = qwen3.derive_megatron_fields(cfg)
    m = qwen3.build_margs(fields, margs_args())
    assert m.layernorm_epsilon == eps


@pytest.mark.parametrize("eps", [1e-6, 3e-6])
def test_deepseek_build_margs_sets_layernorm_epsilon(tmp_path, monkeypatch, eps):
    require_megatron_training()
    import torch

    # validate_args asks the GPU its capability for MoE settings; on a CPU host give a value.
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 4))
    from primus.backends.megatron.checkpoint import native_loader_deepseek as deepseek

    cfg = {
        **LLAMA_CFG,
        "rms_norm_eps": eps,
        "kv_lora_rank": 16,
        "q_lora_rank": None,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "first_k_dense_replace": 1,
        "vocab_size": 256,
    }
    fields = deepseek.derive_megatron_fields(cfg)
    m = deepseek.build_margs(fields, margs_args())
    assert m.layernorm_epsilon == eps
