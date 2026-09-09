###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the GPT-OSS Primus-Turbo sink-attention integration.

Two groups:
    * Registration / condition (CPU-only, no torchtitan needed): the setup patch
      is registered and only applies for gpt_oss with turbo attention enabled.
    * Mirror interface (needs torchtitan; auto-skipped otherwise): the mirror
      Attention keeps the learnable per-head sinks and accepts ``positions``,
      and the mirror TransformerBlock selects the sliding window on even layers.
"""

import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Importing the patch module registers the patch (top-level import pulls in only
# primus.core.*, torchtitan is imported lazily inside the handler).
import primus.backends.torchtitan.patches.turbo.gptoss_sink_attention_patches as gptoss_patch
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.primus_turbo.gptoss_sink_attention"


def _ctx(model_name, enable_turbo, use_turbo_attention, backend="torchtitan", phase="setup"):
    params = SimpleNamespace(
        model=SimpleNamespace(name=model_name),
        primus_turbo=SimpleNamespace(
            enable_primus_turbo=enable_turbo,
            use_turbo_attention=use_turbo_attention,
        ),
    )
    module_config = SimpleNamespace(params=params)
    return PatchContext(backend=backend, phase=phase, extra={"module_config": module_config})


class TestSinkAttentionPatchRegistration:
    def test_patch_registered(self):
        assert PATCH_ID in PatchRegistry.list_ids()
        patch = PatchRegistry.get(PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"

    def test_condition_enabled_for_gpt_oss(self):
        assert gptoss_patch._gptoss_turbo_enabled(_ctx("gpt_oss", True, True)) is True

    def test_condition_disabled_for_other_models(self):
        for name in ("llama3", "llama4", "qwen3", "deepseek_v3"):
            assert gptoss_patch._gptoss_turbo_enabled(_ctx(name, True, True)) is False

    def test_condition_disabled_when_turbo_off(self):
        assert gptoss_patch._gptoss_turbo_enabled(_ctx("gpt_oss", False, True)) is False
        assert gptoss_patch._gptoss_turbo_enabled(_ctx("gpt_oss", True, False)) is False

    def test_applies_to_enabled_for_gpt_oss(self):
        # End-to-end wiring (backend + phase + condition) resolves to True for a
        # gpt_oss turbo run. Negative cases are covered by the condition tests
        # above (applies_to logs on skip, which needs the Primus logger set up).
        patch = PatchRegistry.get(PATCH_ID)
        assert patch.applies_to(_ctx("gpt_oss", True, True)) is True


@pytest.fixture
def gpt_oss_args():
    pytest.importorskip("torchtitan")
    from torchtitan.models.gpt_oss import GptOssModelArgs
    from torchtitan.models.moe import MoEArgs

    return GptOssModelArgs(
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        sliding_window_size=64,
        vocab_size=512,
        moe_inter_dim=256,
        n_layers=2,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=0,
            top_k=2,
            use_grouped_mm=True,
            score_func="softmax",
            route_norm=True,
            gate_bias=True,
        ),
    )


class TestSinkAttentionMirror:
    def test_attention_keeps_sinks_and_accepts_positions(self, gpt_oss_args):
        from primus.backends.torchtitan.models.gpt_oss.model.model import Attention

        attn = Attention(gpt_oss_args)
        # Learnable per-head sinks preserved from the upstream module.
        assert tuple(attn.sinks.shape) == (gpt_oss_args.n_heads,)
        # Full (no) window by default; per-layer window set by the block.
        assert attn._turbo_window == (-1, -1)
        assert attn.sliding_window_size == gpt_oss_args.sliding_window_size
        # GPT-OSS attention forward matches the upstream signature (rope_cache +
        # attention_masks; GPT-OSS does not take the llama/qwen positions arg).
        params = inspect.signature(attn.forward).parameters
        assert "rope_cache" in params
        assert "attention_masks" in params

    def test_forward_calls_turbo_flash_attn_with_sink_and_window(self, gpt_oss_args, monkeypatch):
        """Attention.forward routes through primus_turbo's flash_attn_func, passing
        the per-head sinks and the currently-selected window, and reshapes the
        kernel output back through the output projection."""
        import torch
        from torchtitan.models.gpt_oss.model.model import precompute_rope_cache

        from primus.backends.torchtitan.models.gpt_oss.model.model import (
            Attention,
            apply_rotary_emb,
        )

        attn = Attention(gpt_oss_args)
        attn._turbo_window = (gpt_oss_args.sliding_window_size, 0)

        bsz, seqlen = 2, 4
        x = torch.randn(bsz, seqlen, gpt_oss_args.dim)
        rope_cache = precompute_rope_cache(gpt_oss_args.head_dim, seqlen)

        # Recreate the reference q/k/v (post-RoPE) so we can assert the kernel is
        # called with exactly the tensors the forward pass computes.
        hidden_shape = (bsz, seqlen, -1, gpt_oss_args.head_dim)
        q_ref = attn.wq(x).view(hidden_shape)
        k_ref = attn.wk(x).view(hidden_shape)
        v_ref = attn.wv(x).view(hidden_shape)
        q_ref, k_ref = apply_rotary_emb(q_ref, k_ref, rope_cache)

        kernel_output = torch.randn_like(q_ref)
        fake_flash_attn_func = MagicMock(return_value=kernel_output)
        fake_turbo_pkg = types.ModuleType("primus_turbo")
        fake_turbo_pytorch = types.ModuleType("primus_turbo.pytorch")
        fake_turbo_pytorch.ops = SimpleNamespace(flash_attn_func=fake_flash_attn_func)
        fake_turbo_pkg.pytorch = fake_turbo_pytorch
        monkeypatch.setitem(sys.modules, "primus_turbo", fake_turbo_pkg)
        monkeypatch.setitem(sys.modules, "primus_turbo.pytorch", fake_turbo_pytorch)

        out = attn.forward(x, rope_cache, attention_masks=None)

        fake_flash_attn_func.assert_called_once()
        call_args, call_kwargs = fake_flash_attn_func.call_args
        q, k, v = call_args
        assert torch.allclose(q, q_ref)
        assert torch.allclose(k, k_ref)
        assert torch.allclose(v, v_ref)
        assert call_kwargs["causal"] is True
        assert call_kwargs["window_size"] == (gpt_oss_args.sliding_window_size, 0)
        assert call_kwargs["softmax_scale"] == attn.softmax_scale
        assert torch.allclose(call_kwargs["sink"], attn.sinks.to(q.dtype))

        expected = attn.wo(kernel_output.reshape(bsz, seqlen, -1).contiguous())
        assert out.shape == (bsz, seqlen, gpt_oss_args.dim)
        assert torch.allclose(out, expected)

    def test_block_selects_sliding_window_on_even_layers(self, gpt_oss_args):
        import torch
        import torch.nn as nn

        from primus.backends.torchtitan.models.gpt_oss.model.model import (
            TransformerBlock,
        )

        captured = {}

        class AttnStub(nn.Module):
            def __init__(self, sliding_window_size):
                super().__init__()
                self.sliding_window_size = sliding_window_size
                self._turbo_window = (-1, -1)

            def forward(self, x, rope_cache, attention_masks=None):
                captured["window"] = self._turbo_window
                return torch.zeros_like(x)

        class MoeStub(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        for layer_id, expected in [
            (0, (gpt_oss_args.sliding_window_size, 0)),  # even -> sliding
            (1, (-1, -1)),  # odd -> full
        ]:
            block = TransformerBlock(layer_id, gpt_oss_args)
            assert block.use_sliding_attention == (layer_id % 2 == 0)
            # Replace the heavy attention/moe with CPU stubs to test the block's
            # window-selection logic without invoking the Triton flash kernel.
            block.attention = AttnStub(gpt_oss_args.sliding_window_size)
            block.moe = MoeStub()
            x = torch.zeros(1, 4, gpt_oss_args.dim)
            block(x, None, None)
            assert captured["window"] == expected
