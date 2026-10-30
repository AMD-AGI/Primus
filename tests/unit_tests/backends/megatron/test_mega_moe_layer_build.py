# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Building ``PrimusTurboMegaMoELayer`` the way Megatron builds an MoE layer, without a GPU.

``tests/unit_tests/megatron/moe/test_mega_moe.py`` checks the layer's numerics, but it needs 8 gfx950
GPUs and so never runs in CI. This covers what the constructor and the router call do with
Megatron-core 0.19's MoE spec and with the extra arguments Megatron-LM main passes. The Turbo kernels
are stubbed out: nothing here reaches them.
"""

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import megatron.training
import pytest
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter

# Megatron's own class: the router patch rebinds the name ``TopKRouter`` inside Megatron's modules.
TopKRouter = PrimusTopKRouter.__bases__[0]

_MEGA_MOE = Path(__file__).resolve().parents[4] / "primus/backends/megatron/core/extensions/mega_moe.py"


@pytest.fixture(scope="module")
def mega_moe():
    """``mega_moe.py`` loaded under a private name, with the two Turbo op modules it imports stubbed."""
    with pytest.MonkeyPatch.context() as mp:
        for name in (
            "primus_turbo",
            "primus_turbo.pytorch",
            "primus_turbo.pytorch.ops",
            "primus_turbo.pytorch.ops.moe",
        ):
            if name not in sys.modules:
                package = ModuleType(name)
                package.__path__ = []
                mp.setitem(sys.modules, name, package)
        ops = ModuleType("primus_turbo.pytorch.ops.moe.fused_mega_moe")
        ops.GLUActivation = object
        ops.fused_mega_moe_stage1 = ops.fused_mega_moe_stage2 = None
        fp8_ops = ModuleType("primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8")
        fp8_ops.fused_mega_moe_fp8_stage1 = fp8_ops.fused_mega_moe_fp8_stage2 = None
        mp.setitem(sys.modules, ops.__name__, ops)
        mp.setitem(sys.modules, fp8_ops.__name__, fp8_ops)

        spec = importlib.util.spec_from_file_location("mega_moe_under_test", _MEGA_MOE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(monkeypatch):
    """The Megatron args MegaMoE reads; the layer imports ``get_args`` when it needs it."""
    args = SimpleNamespace(turbo_mega_moe_precision="bf16", disable_primus_topk_router=False)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    return args


class _RouterCalled(Exception):
    """Raised by the stub router so ``forward`` stops before the (stubbed) kernels."""


def _stub_router(record):
    class StubRouter(torch.nn.Module):
        topk = 2

        def __init__(self, **kwargs):
            super().__init__()
            record["router"] = kwargs

        def forward(self, hidden_states, padding_mask=None, **kwargs):
            record["route"] = (padding_mask, kwargs)
            raise _RouterCalled

    return StubRouter


def _stub_shared_experts(record):
    def build(**kwargs):
        record["shared_experts"] = kwargs
        return torch.nn.Identity()

    return build


def _config(shared_expert=True):
    return TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_ffn_hidden_size=8,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=8 if shared_expert else None,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
    )


class _EPGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


_PG_COLLECTION = SimpleNamespace(ep=_EPGroup())


def _build(mega_moe, submodules, config=None, **kwargs):
    # how Megatron's TransformerLayer builds its MLP: a builder called with keywords only
    builder = functools.partial(mega_moe.PrimusTurboMegaMoELayer, submodules=submodules)
    return builder(config=config or _config(), pg_collection=_PG_COLLECTION, is_mtp_layer=False, **kwargs)


@pytest.mark.parametrize("disable", [False, True], ids=["primus_router", "disable_primus_topk_router"])
def test_router_follows_the_router_patch(mega_moe, args, disable):
    args.disable_primus_topk_router = disable
    # Megatron-core 0.19's GPT MoE spec is a functools.partial whose router is left at the default
    submodules = get_submodules(get_gpt_layer_local_spec(num_experts=4).submodules.mlp)
    for router in (TopKRouter, submodules.router):
        assert mega_moe.mega_moe_router(router) is (router if disable else PrimusTopKRouter)
    for router in (PrimusTopKRouter, InferenceTopKRouter, functools.partial(TopKRouter)):
        assert mega_moe.mega_moe_router(router) is router


@pytest.mark.parametrize("name", ["decoder.layers.3.mlp", None])
def test_build_with_core_v019_layer_arguments(mega_moe, args, name):
    record = {}
    submodules = MoESubmodules(
        experts=None, shared_experts=_stub_shared_experts(record), router=_stub_router(record)
    )
    config = _config()
    layer = _build(mega_moe, submodules, config, name=name)

    assert record["router"] == dict(config=config, pg_collection=_PG_COLLECTION, is_mtp_layer=False)
    expected_name = None if name is None else name + ".shared_experts"
    assert record["shared_experts"] == dict(
        config=config, pg_collection=_PG_COLLECTION, gate=False, name=expected_name
    )
    assert layer.experts.fc1_weight.weight.shape == (4, 2 * 8, 32)  # [experts, 2 * ffn, hidden]


def test_module_spec_shared_experts_are_built_without_a_name(mega_moe, args):
    record = {}

    class SharedExperts(torch.nn.Identity):
        def __init__(self, **kwargs):
            super().__init__()
            record["shared_experts"] = kwargs

    submodules = MoESubmodules(
        experts=None, shared_experts=ModuleSpec(module=SharedExperts), router=_stub_router(record)
    )
    _build(mega_moe, submodules, name="decoder.layers.3.mlp")
    assert "name" not in record["shared_experts"]


def test_shared_experts_builder_required(mega_moe, args):
    submodules = MoESubmodules(experts=None, router=_stub_router({}))
    with pytest.raises(AssertionError, match="Shared experts builder"):
        _build(mega_moe, submodules)
    _build(mega_moe, submodules, config=_config(shared_expert=False))


@pytest.mark.parametrize("threshold", [None, 2])
def test_megatron_main_hash_moe_layer_threshold(mega_moe, args, threshold):
    # Megatron-LM main's TransformerLayer passes it to every MLP builder, None unless hash routing is on
    record = {}
    submodules = MoESubmodules(
        experts=None, shared_experts=_stub_shared_experts(record), router=_stub_router(record)
    )
    _build(mega_moe, submodules, hash_moe_layer_threshold=threshold)
    assert record["router"].get("hash_moe_layer_threshold") == threshold
    assert ("hash_moe_layer_threshold" in record["router"]) == (threshold is not None)


def test_router_gets_only_the_routing_inputs_it_is_given(mega_moe, args):
    record = {}
    submodules = MoESubmodules(
        experts=None, shared_experts=_stub_shared_experts(record), router=_stub_router(record)
    )
    layer = _build(mega_moe, submodules)
    hidden_states = torch.zeros(5, 3, 32, dtype=torch.bfloat16)  # [seq, batch, hidden]
    padding_mask = torch.zeros(3, 5, dtype=torch.bool)  # [batch, seq]

    # Megatron-core 0.19 passes only padding_mask, which the layer hands over sequence-first
    with pytest.raises(_RouterCalled):
        layer(hidden_states, padding_mask=padding_mask)
    routed_mask, router_kwargs = record["route"]
    assert routed_mask.shape == (5, 3) and router_kwargs == {}

    # Megatron-LM main adds the token ids (hash routing) and the packed-sequence params
    input_ids, packed_seq_params = torch.zeros(3, 5, dtype=torch.long), object()
    with pytest.raises(_RouterCalled):
        layer(
            hidden_states, padding_mask=padding_mask, input_ids=input_ids, packed_seq_params=packed_seq_params
        )
    router_kwargs = record["route"][1]
    assert set(router_kwargs) == {"input_ids", "packed_seq_params"}
    assert router_kwargs["input_ids"] is input_ids
    assert router_kwargs["packed_seq_params"] is packed_seq_params
