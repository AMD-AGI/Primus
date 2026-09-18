###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from enum import Enum
from types import ModuleType, SimpleNamespace

import torch

from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as fused


def test_te_attention_router_guard_is_narrow(monkeypatch):
    monkeypatch.setattr(fused, "_can_fuse", lambda _layer: True)

    class Scope(Enum):
        # Match MCore's integer-valued CudaGraphScope.
        attn = 2
        moe_router = 5

    layer = SimpleNamespace(
        config=SimpleNamespace(
            cuda_graph_impl="transformer_engine",
            cuda_graph_scope=[Scope.attn, Scope.moe_router],
        ),
        is_moe_layer=True,
        _v2_carry=None,
    )
    assert fused._can_fuse_te_attention_router(layer)

    layer.config.cuda_graph_scope = [Scope.attn]
    assert not fused._can_fuse_te_attention_router(layer)
    layer.config.cuda_graph_scope = [Scope.attn, Scope.moe_router]
    layer._v2_carry = (torch.ones(1), torch.ones(1))
    assert not fused._can_fuse_te_attention_router(layer)


def test_te_capture_fuses_attention_residual_norm_and_keeps_router_contract(monkeypatch):
    utils = ModuleType("megatron.core.utils")
    utils.deprecate_inference_params = lambda context, _params: context
    utils.nvtx_range_push = lambda **_kwargs: None
    utils.nvtx_range_pop = lambda **_kwargs: None
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.utils", utils)

    calls = {}

    def input_norm(value):
        return value + 1

    def attention(value, **kwargs):
        calls["attention_kwargs"] = kwargs
        return value * 2, None

    def pre_mlp_norm(value, residual=None):
        combined = value + residual
        return combined / 2, combined

    def router(value, **kwargs):
        calls["router_kwargs"] = kwargs
        return value * 3, value * 4

    layer = SimpleNamespace(
        input_layernorm=input_norm,
        self_attention=attention,
        pre_mlp_layernorm=pre_mlp_norm,
        mlp=router,
    )
    hidden = torch.ones(2)
    mask = torch.ones(1, dtype=torch.bool)
    outputs = fused._do_fused_te_attention_router_capture(
        layer,
        hidden_states=hidden,
        attention_mask=mask,
        rotary_pos_emb=torch.ones(1),
        padding_mask=torch.zeros(1, dtype=torch.bool),
    )

    residual_post_attn = torch.full((2,), 5.0)
    normalized = residual_post_attn / 2
    assert len(outputs) == 3
    assert torch.equal(outputs[0], normalized * 3)
    assert torch.equal(outputs[1], normalized * 4)
    assert torch.equal(outputs[2], residual_post_attn)
    assert calls["attention_kwargs"]["attention_mask"] is mask
    assert calls["router_kwargs"]["padding_mask"].dtype == torch.bool
