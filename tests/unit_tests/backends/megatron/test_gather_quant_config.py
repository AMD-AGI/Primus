# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU coverage for configuration and the explicit Turbo consumer scope."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/moe_patches/gather_quant_config.py"
)
_SPEC = importlib.util.spec_from_file_location("gather_config_under_test", _PATH)
gather = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gather)


def config(**overrides):
    args = dict(
        moe_permute_quant_fusion=True,
        moe_backward_permute_quant_fusion=True,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        enable_primus_turbo=True,
        use_turbo_grouped_gemm=True,
        turbo_fused_grouped_gemm=True,
        fp4="e2m1",
        fp4_recipe="mxfp4",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_defaults_ignore_legacy_environment(monkeypatch):
    for name in ("GPTOSS_FUSED_PERMUTE_QUANT", "GPTOSS_FUSED_BACKWARD_PERMUTE_QUANT"):
        monkeypatch.setenv(name, "1")
    assert gather.validate_gather_quant_config(SimpleNamespace()) == (False, False)
    original = object()
    assert gather.configure_gather_quant_fusion(SimpleNamespace(), original) is original


@pytest.mark.parametrize(
    "key,value",
    [
        ("tensor_model_parallel_size", 2),
        ("expert_model_parallel_size", 2),
        ("expert_tensor_parallel_size", 2),
        ("moe_skip_identity_sort", False),
        ("moe_token_dispatcher_type", "allgather"),
        ("moe_permute_fusion", False),
        ("enable_primus_turbo", False),
        ("use_turbo_grouped_gemm", False),
        ("turbo_fused_grouped_gemm", False),
        ("fp4", None),
        ("fp4_recipe", "nvfp4"),
        ("moe_pad_expert_input_to_capacity", True),
        ("moe_apply_probs_on_input", True),
        ("moe_router_padding_for_quantization", True),
    ],
)
def test_incompatible_consumer_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        gather.validate_gather_quant_config(config(**{key: value}))


def test_backward_requires_forward():
    with pytest.raises(ValueError, match="requires moe_permute_quant_fusion"):
        gather.validate_gather_quant_config(config(moe_permute_quant_fusion=False))


@pytest.mark.parametrize("backward", [False, True])
def test_only_dispatcher_scope_opts_in_and_restores_after_error(monkeypatch, backward):
    def permute(*args, **kwargs):
        if args == ("raise",):
            raise RuntimeError("test failure")
        return kwargs

    class Dispatcher:
        def dispatch_preprocess(self, *args):
            return wrapped(*args)

    module = ModuleType("megatron.core.transformer.moe.token_dispatcher")
    module.MoEAlltoAllTokenDispatcher = Dispatcher
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(gather, "_uses_mxfp4", lambda: True)
    wrapped = gather.configure_gather_quant_fusion(
        config(moe_backward_permute_quant_fusion=backward), permute
    )
    expected_eager = dict(fuse_permute_quant=False, fuse_backward_permute_quant=False)
    assert wrapped() == expected_eager
    assert Dispatcher().dispatch_preprocess() == dict(
        fuse_permute_quant=True, fuse_backward_permute_quant=backward
    )
    with pytest.raises(RuntimeError, match="test failure"):
        Dispatcher().dispatch_preprocess("raise")
    assert wrapped() == expected_eager
    monkeypatch.setattr(gather, "_uses_mxfp4", lambda: False)
    assert Dispatcher().dispatch_preprocess() == expected_eager
