###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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


def test_runtime_guard_does_not_require_primus_only_flags():
    config = _config()
    del config.enable_primus_turbo
    del config.use_turbo_attention
    del config.use_turbo_gemm

    assert _is_mxfp4_nonexpert_graph(config)
    assert not _is_turbo_mxfp4_nonexpert_graph(config)


def test_patch_hides_fp4_only_during_input_preparation(monkeypatch):
    observed = []

    class FakeTECudaGraphHelper:
        def __init__(self, config):
            self.config = config

        def _get_cuda_graph_input_data(self):
            observed.append(self.config.fp4)
            return "sample_args", {"fp8_enabled": bool(self.config.fp4)}

    cuda_graphs = ModuleType("megatron.core.transformer.cuda_graphs")
    cuda_graphs.TECudaGraphHelper = FakeTECudaGraphHelper
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.cuda_graphs", cuda_graphs)

    patch_mxfp4_attention_cudagraph(None)

    config = _config()
    del config.enable_primus_turbo
    del config.use_turbo_attention
    del config.use_turbo_gemm
    result = FakeTECudaGraphHelper(config)._get_cuda_graph_input_data()
    assert observed == [None]
    assert result == ("sample_args", {"fp8_enabled": False})
    assert config.fp4 == "e2m1"


def test_patch_leaves_other_graph_scopes_unchanged(monkeypatch):
    observed = []

    class FakeTECudaGraphHelper:
        def __init__(self, config):
            self.config = config

        def _get_cuda_graph_input_data(self):
            observed.append(self.config.fp4)

    cuda_graphs = ModuleType("megatron.core.transformer.cuda_graphs")
    cuda_graphs.TECudaGraphHelper = FakeTECudaGraphHelper
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.cuda_graphs", cuda_graphs)

    patch_mxfp4_attention_cudagraph(None)

    config = _config(cuda_graph_scope=["attn", "moe"])
    FakeTECudaGraphHelper(config)._get_cuda_graph_input_data()
    assert observed == ["e2m1"]
    assert config.fp4 == "e2m1"
