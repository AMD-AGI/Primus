###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the keep-in-FP32 contract on ``ape`` and ``attn_sink``.

Both are FP32 in the released DeepSeek-V4 checkpoint, so this mechanism can hold
them at FP32 through the framework's blanket ``module.bfloat16()``. Covers the
mechanism in isolation, the two V4 parameters that use it, and the property that
actually matters when it is on: the FP32 values survive *bit-exactly* rather than
being round-tripped through BF16.

The mechanism is **off by default** (a second parameter dtype breaks the
distributed optimizer's single-dtype assumption -- see the module docstring), so
everything here opts in explicitly via the fixture below. The default-off
behaviour has its own tests at the bottom.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

mla_module = pytest.importorskip(
    "megatron.core.transformer.multi_latent_attention",
    reason="MLA base module not importable in this environment",
)

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: E402
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.compressor import Compressor  # noqa: E402
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (  # noqa: E402
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE  # noqa: E402
from primus.backends.megatron.core.transformer.keep_in_fp32 import (  # noqa: E402
    ENABLE_ENV_VAR,
    KeepInFp32Mixin,
    is_enabled,
    is_marked_keep_in_fp32,
    mark_keep_in_fp32,
    unmark_keep_in_fp32,
)


@pytest.fixture(autouse=True)
def _enable_keep_in_fp32(monkeypatch):
    """Opt in for this module: the mechanism ships disabled."""
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------


class _Pinned(KeepInFp32Mixin, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pinned = nn.Parameter(torch.randn(4, dtype=torch.float32))
        self.floating = nn.Parameter(torch.randn(4, dtype=torch.float32))
        mark_keep_in_fp32(self.pinned)


def test_bfloat16_skips_the_pinned_parameter():
    module = _Pinned()
    module.bfloat16()

    assert module.pinned.dtype == torch.float32
    assert module.floating.dtype == torch.bfloat16


def test_pinned_values_are_not_round_tripped_through_bf16():
    """The whole point is precision, so restoring must not go via BF16.

    A cast-down-then-up would leave the values quantised to BF16 steps; the
    saved FP32 copy has to come back bit-exact.
    """
    module = _Pinned()
    # Values with mantissa bits BF16 cannot hold, so a round trip is visible.
    with torch.no_grad():
        module.pinned.copy_(torch.tensor([1.0000001, 3.1415927, -2.7182818, 1e-8]))
    before = module.pinned.detach().clone()

    module.bfloat16()

    assert not torch.equal(before.bfloat16().float(), before), "pick values BF16 actually loses"
    torch.testing.assert_close(module.pinned, before, rtol=0, atol=0)


def test_mark_survives_repeated_conversions():
    """``_apply`` may replace the Parameter object, dropping the attribute."""
    module = _Pinned()
    module.bfloat16()
    assert is_marked_keep_in_fp32(module.pinned)

    # A second conversion must still be protected.
    module.half()
    assert module.pinned.dtype == torch.float32
    assert module.floating.dtype == torch.float16


def test_unmark_lets_the_parameter_follow_the_model():
    module = _Pinned()
    unmark_keep_in_fp32(module.pinned)
    module.bfloat16()
    assert module.pinned.dtype == torch.bfloat16


def test_float_conversion_of_an_unpinned_module_is_unaffected():
    """Sanity: the mixin is inert when nothing is marked."""

    class _Plain(KeepInFp32Mixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(3))

    module = _Plain()
    module.bfloat16()
    assert module.w.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Compressor.ape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_ape_stays_fp32(ratio: int):
    comp = Compressor(hidden_size=32, head_dim=16, ratio=ratio)
    assert comp.ape.dtype == torch.float32

    comp.bfloat16()
    assert comp.ape.dtype == torch.float32, "ape must not follow the model dtype"
    # ...and the projections did convert, so the test is not vacuous.
    other = comp.wkv_gate.weight if hasattr(comp, "wkv_gate") else comp.wkv.weight
    assert other.dtype == torch.bfloat16


def test_compressor_forward_accepts_bf16_activations_with_fp32_ape():
    """Mixed dtypes must not break the pooling path.

    The eager body promotes via ``score + ape``; the Triton kernel loads ape
    with an explicit ``.to(tl.float32)``. Only the eager path is reachable on
    CPU, which is the one that relies on PyTorch's promotion rules.
    """
    comp = Compressor(hidden_size=32, head_dim=16, ratio=4)
    comp.bfloat16()
    hidden = torch.randn(1, 8, 32, dtype=torch.bfloat16)

    pooled = comp(hidden)

    assert pooled.shape == (1, 2, 16)
    assert torch.isfinite(pooled.float()).all()


def test_compressor_state_dict_carries_fp32_ape():
    """Checkpoint parity: the saved tensor dtype has to match the reference."""
    comp = Compressor(hidden_size=32, head_dim=16, ratio=4)
    comp.bfloat16()
    assert comp.state_dict()["ape"].dtype == torch.float32


# ---------------------------------------------------------------------------
# DeepseekV4Attention.attn_sink
# ---------------------------------------------------------------------------


def _make_attention(compress_ratio: int) -> DeepseekV4Attention:
    config = DeepSeekV4TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=16,
        qk_pos_emb_head_dim=8,
        qk_head_dim=8,
        v_head_dim=16,
        kv_lora_rank=16,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sliding_window=0,
        attn_sink=True,
        compress_ratios=None,
        compress_rope_theta=40000.0,
        use_v4_attention_backend="eager",
        use_v4_csa_attention_backend="eager",
        layernorm_epsilon=1e-6,
        norm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    config.index_topk = 2
    config.index_head_dim = 16
    config.index_n_heads = 2

    rope = DualRoPE(
        rotary_dim=config.qk_pos_emb_head_dim,
        rope_theta=config.rotary_base,
        compress_rope_theta=config.compress_rope_theta,
        yarn_factor=1.0,
        original_max_position_embeddings=config.original_max_position_embeddings,
    )
    return DeepseekV4Attention(config, rope=rope, compress_ratio=compress_ratio, submodules=None)


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_attn_sink_stays_fp32(compress_ratio: int):
    attn = _make_attention(compress_ratio)
    assert attn.attn_sink is not None
    assert attn.attn_sink.dtype == torch.float32

    attn.bfloat16()
    assert attn.attn_sink.dtype == torch.float32, "attn_sink must not follow the model dtype"
    # ...and an ordinary weight did convert, so the assertion is not vacuous.
    assert attn.linear_q_up_proj.weight.dtype == torch.bfloat16


def test_attn_sink_fp32_matches_the_gluon_kernel_contract():
    """The gluon / flydsl_v1 kernels assert contiguous FP32 sink of shape [H].

    Those kernels are satisfied by the promotion their callers already do, so
    this only pins that *when* the mechanism is on, the parameter arrives in the
    shape and layout they expect rather than needing a second cast.
    """
    attn = _make_attention(4)
    attn.bfloat16()

    sink = attn.attn_sink
    assert sink.dtype == torch.float32
    assert sink.is_contiguous()
    assert sink.shape == (attn.num_heads,)


def test_nested_compressor_ape_survives_the_parent_conversion():
    """``_apply`` recurses, so the CSA compressor / indexer must be covered too."""
    attn = _make_attention(4)
    attn.bfloat16()

    assert attn.compressor is not None
    assert attn.compressor.ape.dtype == torch.float32
    assert attn.indexer is not None
    assert attn.indexer.indexer_compressor.ape.dtype == torch.float32


def test_attention_state_dict_dtypes_match_the_reference_checkpoint():
    """``ape`` / ``attn_sink`` save as FP32; the weight matrices do not."""
    attn = _make_attention(4)
    attn.bfloat16()
    state = attn.state_dict()

    pinned = {k for k in state if k == "attn_sink" or k.endswith(".ape")}
    assert pinned, "expected attn_sink and at least one compressor ape"
    for key in pinned:
        assert state[key].dtype == torch.float32, f"{key} should be saved as FP32"

    weights = {k for k in state if k.endswith(".weight")}
    assert weights, "expected some weight matrices to compare against"
    for key in weights:
        assert state[key].dtype == torch.bfloat16, f"{key} should follow the model dtype"


# ---------------------------------------------------------------------------
# Default-off behaviour
# ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    """Shipping default: a second parameter dtype breaks the distributed
    optimizer (``single dtype supported, for now``), and on 4N/PP4 it aborted
    every GPU with a memory access fault. So the model must come up uniform.
    """
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert not is_enabled()


@pytest.mark.parametrize("value", ["0", "", "false", "True", "yes"])
def test_only_exactly_one_enables(monkeypatch, value: str):
    """Opt-in is ``"1"`` and nothing else -- a stray value must not half-enable
    the mechanism, since the failure it causes is a GPU abort rather than an
    exception.
    """
    monkeypatch.setenv(ENABLE_ENV_VAR, value)
    assert not is_enabled()


def test_parameters_follow_the_model_dtype_when_disabled(monkeypatch):
    """The whole model is one dtype when off -- this is what makes the
    distributed optimizer's single-dtype assumption hold.
    """
    monkeypatch.setenv(ENABLE_ENV_VAR, "0")

    comp = Compressor(hidden_size=32, head_dim=16, ratio=4)
    comp.bfloat16()
    assert comp.ape.dtype == torch.bfloat16

    attn = _make_attention(4)
    attn.bfloat16()
    assert attn.attn_sink.dtype == torch.bfloat16
    assert attn.compressor.ape.dtype == torch.bfloat16
    assert attn.indexer.indexer_compressor.ape.dtype == torch.bfloat16

    dtypes = {p.dtype for p in attn.parameters()}
    assert dtypes == {torch.bfloat16}, f"expected a single parameter dtype, got {dtypes}"


def test_disabled_pooling_forward_still_runs(monkeypatch):
    """Guard the path that actually matters: ``score.float()`` before the
    softmax means a BF16 ``ape`` does not change the pooling result's finiteness.
    """
    monkeypatch.setenv(ENABLE_ENV_VAR, "0")

    comp = Compressor(hidden_size=32, head_dim=16, ratio=4)
    comp.bfloat16()
    pooled = comp(torch.randn(1, 8, 32, dtype=torch.bfloat16))

    assert pooled.shape == (1, 2, 16)
    assert torch.isfinite(pooled.float()).all()
