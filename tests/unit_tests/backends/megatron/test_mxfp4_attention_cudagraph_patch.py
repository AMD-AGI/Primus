###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from contextlib import contextmanager
from enum import Enum
from types import ModuleType, SimpleNamespace

from primus.backends.megatron.patches.turbo.mxfp4_attention_cudagraph_patches import (
    _is_mxfp4_nonexpert_graph,
    _is_turbo_mxfp4_nonexpert_graph,
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


def test_patch_bypasses_only_te_recipe_lookup(monkeypatch):
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
