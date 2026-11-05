###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for the TorchTitan v0.2.2 DeepSeek memory patch.

DeepSeek uses whole-block compilation with an explicit graph break at
GroupedExperts.forward; EP dispatch/combine hooks stay in the compiled block.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

import primus.backends.torchtitan.patches.dsv3_v022_perf_patches as dsv3_patch
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

COMBINE_PATCH_ID = "torchtitan.dsv3.moe_bf16_combine"
COMPILE_PATCH_ID = "torchtitan.dsv3.whole_block_compile"


def _ctx(model_name, compile_enable=True):
    params = SimpleNamespace(compile=SimpleNamespace(enable=compile_enable))
    module_config = SimpleNamespace(params=params)
    return PatchContext(
        backend="torchtitan",
        phase="setup",
        model_name=model_name,
        extra={"module_config": module_config},
    )


class TestDsv3V022PatchRegistration:
    def test_safe_whole_block_compile_is_registered(self):
        patch = PatchRegistry.get(COMPILE_PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"

    def test_bf16_combine_remains_registered(self):
        patch = PatchRegistry.get(COMBINE_PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"


class TestDsv3V022PatchCondition:
    def test_compile_patch_requires_deepseek_and_compile(self):
        patch = PatchRegistry.get(COMPILE_PATCH_ID)
        assert patch is not None
        assert patch.condition(_ctx("deepseek_v3", True)) is True
        assert patch.condition(_ctx("deepseek_v3", False)) is False
        assert patch.condition(_ctx("llama3", True)) is False

    def test_bf16_combine_applies_to_deepseek(self):
        patch = PatchRegistry.get(COMBINE_PATCH_ID)
        assert patch is not None
        assert patch.condition(_ctx("deepseek_v3")) is True
        assert patch.condition(_ctx(SimpleNamespace(name="DeepSeek-V3"))) is True

    def test_bf16_combine_does_not_apply_to_other_models(self):
        patch = PatchRegistry.get(COMBINE_PATCH_ID)
        assert patch is not None
        assert patch.condition(_ctx("llama3")) is False
        assert patch.condition(_ctx(None)) is False


class _Leaf(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class _MoE(nn.Module):
    def __init__(self, *, optional_modules=True):
        super().__init__()
        self.experts = _Leaf("experts")
        self.router = _Leaf("router")
        self.reorderer = _Leaf("reorderer") if optional_modules else None
        self.shared_experts = _Leaf("shared_experts") if optional_modules else None


class _Block(nn.Module):
    def __init__(self, moe_enabled):
        super().__init__()
        self.moe_enabled = moe_enabled
        self.name = "moe_block" if moe_enabled else "dense_block"
        self.attention = _Leaf("attention")
        self.attention_norm = _Leaf("attention_norm")
        self.ffn_norm = _Leaf("ffn_norm")
        if moe_enabled:
            self.moe = _MoE()
        else:
            self.feed_forward = _Leaf("feed_forward")


class _Compiled(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "dense": _Block(moe_enabled=False),
                "moe": _Block(moe_enabled=True),
            }
        )


def test_unified_compile_breaks_at_grouped_experts_once(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    first_model = _Model()
    second_model = _Model()
    compile_calls = []
    disable_calls = []

    def fake_compile(module, *, backend, fullgraph):
        compile_calls.append((module.name, backend, fullgraph))
        return _Compiled(module)

    def fake_disable(fn, *, recursive):
        disable_calls.append((fn, recursive))

        def eager_boundary(*args, **kwargs):
            return fn(*args, **kwargs)

        return eager_boundary

    monkeypatch.setattr(dsv3_patch.torch, "compile", fake_compile)
    monkeypatch.setattr(dsv3_patch.torch.compiler, "disable", fake_disable)
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", False)
    original_forward = moe_module.GroupedExperts.forward
    with patch.object(moe_module.GroupedExperts, "forward", original_forward):
        dsv3_patch._apply_unified_compile(
            first_model,
            SimpleNamespace(backend="inductor"),
            ep_enabled=True,
        )
        installed = moe_module.GroupedExperts.forward
        assert installed is not original_forward
        assert installed._primus_graph_break is True

        dsv3_patch._apply_unified_compile(
            second_model,
            SimpleNamespace(backend="inductor"),
            ep_enabled=True,
        )
        assert moe_module.GroupedExperts.forward is installed

    assert moe_module.GroupedExperts.forward is original_forward

    assert compile_calls == [
        ("dense_block", "inductor", False),
        ("moe_block", "inductor", False),
        ("dense_block", "inductor", False),
        ("moe_block", "inductor", False),
    ]
    assert torch._dynamo.config.capture_scalar_outputs is True
    assert disable_calls == [(original_forward, False)]


def _tiny_deepseek(seed=0):
    from torchtitan.models.deepseek_v3 import DeepSeekV3ModelArgs
    from torchtitan.models.deepseek_v3.model.model import DeepSeekV3Model
    from torchtitan.models.moe import MoEArgs

    model_args = DeepSeekV3ModelArgs(
        vocab_size=128,
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_layers=2,
        n_dense_layers=1,
        n_heads=2,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        max_seq_len=64,
    )
    model_args.moe_impl = "standard"
    torch.manual_seed(seed)
    device = torch.device("cuda")
    with device:
        model = DeepSeekV3Model(model_args)
    with torch.no_grad():
        model.init_weights(buffer_device=device)
    return model, model_args


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE router histc requires a GPU")
def test_unified_compile_compiles_checkpoint_wrapped_moe_block(monkeypatch):
    import torchtitan.models.moe.moe as moe_module
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )
    from torchtitan.config.job_config import ActivationCheckpoint
    from torchtitan.distributed.activation_checkpoint import apply_ac

    captured_ops = []

    def counting_backend(gm, example_inputs):
        captured_ops.append({getattr(n.target, "__name__", str(n.target)) for n in gm.graph.nodes})
        return gm.forward

    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", False)
    with patch.object(moe_module.GroupedExperts, "forward", moe_module.GroupedExperts.forward):
        reference, model_args = _tiny_deepseek()
        model, _ = _tiny_deepseek()
        tokens = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        expected = reference(tokens)

        apply_ac(model, ActivationCheckpoint(mode="full"), model_compile_enabled=True)
        torch._dynamo.reset()
        try:
            dsv3_patch._apply_unified_compile(
                model,
                SimpleNamespace(backend=counting_backend),
                ep_enabled=False,
            )
            out = model(tokens)
            out.float().pow(2).mean().backward()
        finally:
            torch._dynamo.reset()

    assert any(
        "topk" in ops for ops in captured_ops
    ), "MoE block fell back to eager under activation checkpointing"
    torch.testing.assert_close(out, expected)
    moe_block = model.layers["1"]
    assert isinstance(moe_block, CheckpointWrapper)
    assert isinstance(moe_block._checkpoint_wrapped_module, torch._dynamo.eval_frame.OptimizedModule)


def test_compile_patch_rebinds_source_and_deepseek_alias(monkeypatch):
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)
    with (
        patch.object(llama4_parallelize, "apply_compile", object()),
        patch.object(deepseek_parallelize, "apply_compile", object()),
    ):
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        assert llama4_parallelize.apply_compile is dsv3_patch._apply_unified_compile
        assert deepseek_parallelize.apply_compile is dsv3_patch._apply_unified_compile


def test_compile_patch_repeated_install_is_idempotent(monkeypatch):
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)
    with (
        patch.object(llama4_parallelize, "apply_compile", object()),
        patch.object(deepseek_parallelize, "apply_compile", object()),
    ):
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        dsv3_patch.patch_whole_block_compile(_ctx("deepseek_v3"))
        assert llama4_parallelize.apply_compile is dsv3_patch._apply_unified_compile
        assert deepseek_parallelize.apply_compile is dsv3_patch._apply_unified_compile


def test_moe_forward_is_installed_directly(monkeypatch):
    import torchtitan.models.moe.moe as moe_module

    monkeypatch.setattr(dsv3_patch, "log_rank_0", lambda *args, **kwargs: None)

    with patch.object(moe_module.MoE, "forward", moe_module.MoE.forward):
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is dsv3_patch._moe_forward_bf16_combine
        dsv3_patch.patch_moe_bf16_combine(_ctx("deepseek_v3"))
        assert moe_module.MoE.forward is dsv3_patch._moe_forward_bf16_combine
