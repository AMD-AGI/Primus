###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from contextlib import contextmanager
from enum import Enum
from types import ModuleType, SimpleNamespace

from primus.backends.megatron.patches.turbo.mxfp4_attention_cudagraph_patches import (
    _cache_static_replay_kwargs,
    _is_mxfp4_nonexpert_graph,
    _is_turbo_mxfp4_nonexpert_graph,
    _replace_none_attention_mask_with_static_zero,
    _same_tensor_tree_signature,
    patch_mxfp4_attention_cudagraph,
)


def _config(**overrides):
    values = {
        "cuda_graph_impl": "transformer_engine",
        "cuda_graph_scope": ["attn"],
        "fp4": "e2m1",
        "fp4_recipe": "mxfp4",
        "enable_primus_turbo": True,
        "use_turbo_attention": True,
        "use_turbo_gemm": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_fake_transformer_layer(monkeypatch):
    class FakeTransformerLayer:
        def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
            return args, kwargs.copy()

    transformer_layer = ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer.TransformerLayer = FakeTransformerLayer
    monkeypatch.setitem(
        __import__("sys").modules,
        "megatron.core.transformer.transformer_layer",
        transformer_layer,
    )
    return FakeTransformerLayer


def test_guard_accepts_enum_attention_scope():
    class Scope(Enum):
        attn = "attn"

    assert _is_turbo_mxfp4_nonexpert_graph(_config(cuda_graph_scope=[Scope.attn]))


def test_guard_accepts_attention_and_router_scopes():
    assert _is_turbo_mxfp4_nonexpert_graph(
        _config(cuda_graph_scope=["attn", "moe_router", "moe_preprocess"])
    )


def test_guard_rejects_graph_that_includes_moe():
    assert not _is_turbo_mxfp4_nonexpert_graph(
        cuda_config := _config(cuda_graph_scope=["attn", "moe"])
    )
    assert cuda_config.fp4 == "e2m1"


def test_runtime_config_cannot_reconstruct_registration_guard():
    config = _config()
    del config.fp4_recipe
    del config.enable_primus_turbo
    del config.use_turbo_attention
    del config.use_turbo_gemm

    assert not _is_mxfp4_nonexpert_graph(config)
    assert not _is_turbo_mxfp4_nonexpert_graph(config)


def test_static_attention_inputs_are_cached_per_layer_and_microbatch():
    import torch

    layers = [SimpleNamespace(), SimpleNamespace()]
    masks = [torch.ones(1, dtype=torch.bool) for _ in range(4)]
    ropes = [torch.zeros(2) for _ in range(4)]
    helper = SimpleNamespace(
        callables_per_chunk=[layers],
        config=SimpleNamespace(overlap_moe_expert_parallel_comm=False),
        num_microbatches=2,
    )
    kwargs = {
        "sample_kwargs": [
            {"attention_mask": masks[0], "rotary_pos_emb": ropes[0]},
            {"attention_mask": masks[1], "rotary_pos_emb": ropes[1]},
            {"attention_mask": masks[2], "rotary_pos_emb": ropes[2]},
            {"attention_mask": masks[3], "rotary_pos_emb": ropes[3]},
        ]
    }

    _cache_static_replay_kwargs(helper, kwargs)

    assert layers[0]._primus_te_static_replay_kwargs == [
        {"rotary_pos_emb": ropes[0]},
        {"rotary_pos_emb": ropes[2]},
    ]
    assert layers[1]._primus_te_static_replay_kwargs == [
        {"rotary_pos_emb": ropes[1]},
        {"rotary_pos_emb": ropes[3]},
    ]
    assert layers[0]._primus_te_static_attention_masks == [masks[0], masks[2]]
    assert layers[0]._primus_te_static_attention_masks_initialized == [False, False]

    layers[0].current_microbatch = 1
    replay_kwargs = _replace_none_attention_mask_with_static_zero(
        layers[0], {"attention_mask": None}
    )
    assert replay_kwargs["attention_mask"] is masks[2]
    assert not masks[2].any()
    assert layers[0]._primus_te_static_attention_masks_initialized == [False, True]

    actual_mask = torch.ones(1, dtype=torch.bool)
    replay_kwargs = _replace_none_attention_mask_with_static_zero(
        layers[0], {"attention_mask": actual_mask}
    )
    assert replay_kwargs["attention_mask"] is actual_mask
    assert _same_tensor_tree_signature(ropes[0], ropes[1])
    assert not _same_tensor_tree_signature(ropes[0], torch.zeros(3))


def test_patch_bypasses_only_te_recipe_lookup(monkeypatch):
    _install_fake_transformer_layer(monkeypatch)
    observed = []
    recipe_calls = []

    fp4_utils = ModuleType("megatron.core.fp4_utils")

    def reject_mxfp4(config):
        recipe_calls.append(config.fp4)
        raise ValueError("TE does not support MXFP4")

    fp4_utils.get_fp4_recipe = reject_mxfp4
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.fp4_utils", fp4_utils)

    class FakeTECudaGraphHelper:
        def __init__(self, config):
            self.config = config

        def _get_cuda_graph_input_data(self):
            observed.append(self.config.fp4)
            from megatron.core.fp4_utils import get_fp4_recipe

            return "sample_args", {
                "fp8_enabled": bool(self.config.fp4),
                "fp8_recipe": get_fp4_recipe(self.config),
                "fp8_weight_caching": True,
                "fp8_group": "group",
            }

        def create_cudagraphs(self):
            return "graphs"

    cuda_graphs = ModuleType("megatron.core.transformer.cuda_graphs")
    cuda_graphs.TECudaGraphHelper = FakeTECudaGraphHelper
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.cuda_graphs", cuda_graphs)

    patch_mxfp4_attention_cudagraph(None)

    config = _config()
    del config.fp4_recipe
    del config.enable_primus_turbo
    del config.use_turbo_attention
    del config.use_turbo_gemm
    result = FakeTECudaGraphHelper(config)._get_cuda_graph_input_data()
    assert observed == ["e2m1"]
    assert recipe_calls == []
    assert result == ("sample_args", {"fp8_enabled": False})
    assert config.fp4 == "e2m1"
    assert fp4_utils.get_fp4_recipe is reject_mxfp4


def test_patch_uses_registration_decision_at_runtime(monkeypatch):
    _install_fake_transformer_layer(monkeypatch)
    observed = []

    fp4_utils = ModuleType("megatron.core.fp4_utils")
    fp4_utils.get_fp4_recipe = lambda _config: "unused"
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.fp4_utils", fp4_utils)

    class FakeTECudaGraphHelper:
        def __init__(self, config):
            self.config = config

        def _get_cuda_graph_input_data(self):
            observed.append(self.config.fp4)
            return "sample_args", {"fp8_enabled": True, "fp8_recipe": "mxfp4"}

        def create_cudagraphs(self):
            return "graphs"

    cuda_graphs = ModuleType("megatron.core.transformer.cuda_graphs")
    cuda_graphs.TECudaGraphHelper = FakeTECudaGraphHelper
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.cuda_graphs", cuda_graphs)

    patch_mxfp4_attention_cudagraph(None)

    config = SimpleNamespace(fp4="e2m1")
    result = FakeTECudaGraphHelper(config)._get_cuda_graph_input_data()
    assert observed == ["e2m1"]
    assert result == ("sample_args", {"fp8_enabled": False})
    assert config.fp4 == "e2m1"


def test_patch_keeps_turbo_fp4_context_active_during_capture(monkeypatch):
    _install_fake_transformer_layer(monkeypatch)
    context_events = []
    capture_observed = []

    fp4_utils = ModuleType("megatron.core.fp4_utils")
    fp4_utils.get_fp4_recipe = lambda _config: "unused"
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.fp4_utils", fp4_utils)

    primus_fp4_utils = ModuleType("primus.backends.megatron.core.fp4_utils")

    @contextmanager
    def get_fp4_context(config):
        context_events.append(("enter", config.fp4))
        try:
            yield
        finally:
            context_events.append(("exit", config.fp4))

    primus_fp4_utils.get_fp4_context = get_fp4_context
    monkeypatch.setitem(
        __import__("sys").modules,
        "primus.backends.megatron.core.fp4_utils",
        primus_fp4_utils,
    )

    class FakeTECudaGraphHelper:
        def __init__(self, config):
            self.config = config

        def _get_cuda_graph_input_data(self):
            return "sample_args", {"fp8_enabled": True, "fp8_recipe": "mxfp4"}

        def create_cudagraphs(self):
            capture_observed.append(list(context_events))
            return "graphs"

    cuda_graphs = ModuleType("megatron.core.transformer.cuda_graphs")
    cuda_graphs.TECudaGraphHelper = FakeTECudaGraphHelper
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.cuda_graphs", cuda_graphs)

    patch_mxfp4_attention_cudagraph(None)

    config = SimpleNamespace(fp4="e2m1")
    assert FakeTECudaGraphHelper(config).create_cudagraphs() == "graphs"
    assert capture_observed == [[("enter", "e2m1")]]
    assert context_events == [("enter", "e2m1"), ("exit", "e2m1")]
