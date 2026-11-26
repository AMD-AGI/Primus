###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LFM2 decoder layer specs build and run on Megatron-core 0.19."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.models.gpt import gpt_layer_specs
from primus.backends.megatron.core.transformer.attention import Lfm2ShortConv

LAYER_TYPES = ["conv", "full_attention", "conv", "full_attention"]


def _config():
    return TransformerConfig(
        num_layers=len(LAYER_TYPES),
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_layer_freq=[0, 0, 1, 1],
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )


def _args():
    return SimpleNamespace(
        lfm_layer_types=LAYER_TYPES,
        num_layers=len(LAYER_TYPES),
        conv_L_cache=3,
        conv_bias=False,
        enable_turbo_attention_float8=False,
        data_parallel_replicate_degree=1,
    )


def test_decoder_layer_specs_place_short_conv(monkeypatch):
    import megatron.training

    monkeypatch.setattr(megatron.training, "get_args", _args)
    specs = gpt_layer_specs.get_gpt_decoder_layer_specs(_config(), use_transformer_engine=True)

    assert [spec.submodules.self_attention.module is Lfm2ShortConv for spec in specs] == [
        layer_type == "conv" for layer_type in LAYER_TYPES
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE linears need a GPU")
def test_short_conv_layer_builds_and_runs(init_parallel_state):
    """TransformerLayer passes name= to its self_attention from Megatron-core 0.19 on."""
    from megatron.training.global_vars import set_args

    set_args(_args())
    config = _config()
    spec = gpt_layer_specs.get_lfm_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, use_te_activation_func=config.use_te_activation_func
    )
    layer = build_module(spec, config=config, layer_number=1).cuda()

    assert isinstance(layer.self_attention, Lfm2ShortConv)
    hidden_states = torch.randn(16, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    output, _ = layer(hidden_states, attention_mask=None)
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
