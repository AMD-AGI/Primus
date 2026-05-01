###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the plan-2 V4-faithful :class:`DeepseekV4Attention`.

These tests run on CPU (fp32) and verify the dense (``compress_ratio == 0``)
attention math against an *inline* reference implementation that mirrors
the released ``DeepSeek-V4-Flash/inference/model.py`` semantics:

* Single-latent KV: ``K = V = wkv(hidden)`` broadcast to all query heads.
* Per-head ``q_rms``: parameter-less RMS on ``head_dim`` after ``wq_b``.
* Learnable per-head ``attn_sink``: an extra "virtual key" with zero value
  joined into the softmax (then dropped before the value-weighted sum).
* Grouped low-rank O: einsum-based ``wo_a`` per group + ``wo_b``.
* Partial **interleaved** RoPE on the last ``qk_pos_emb_head_dim`` channels.

The reference uses the *same* interleaved RoPE convention as Primus's
:mod:`dual_rope` (per the techblog correction over the original HF PR
which used rotate-half), so the two implementations should agree to
machine precision when given identical weights and positions.

Plan-2 P17 will replace this inline reference with the actual HF
``DeepseekV4Attention.forward`` from the released checkpoint.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

# Importing the module under test pulls Megatron / TE through the
# Primus extensions; skip the whole module when those are unavailable
# (e.g. a CPU-only smoke environment).
mla_module = pytest.importorskip(
    "megatron.core.transformer.multi_latent_attention",
    reason="Megatron MLA not importable in this environment",
)

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: E402
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (  # noqa: E402
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE  # noqa: E402

# ---------------------------------------------------------------------------
# Inline reference V4 attention forward (single-latent KV, attn_sink, grouped O)
# ---------------------------------------------------------------------------


def _rms_norm_per_head(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Parameter-less per-head RMS (matches the released checkpoint)."""
    in_dtype = x.dtype
    x32 = x.float()
    rms = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return (x32 * rms).to(in_dtype)


def _reference_v4_attention_forward(
    *,
    hidden: torch.Tensor,  # [B, S, D]
    position_ids: torch.Tensor,  # [B, S] or [S]
    rope: DualRoPE,
    wq_a: torch.Tensor,  # [q_lora_rank, D]
    wq_b: torch.Tensor,  # [n_heads * head_dim, q_lora_rank]
    q_norm_w: torch.Tensor,  # [q_lora_rank]
    wkv: torch.Tensor,  # [head_dim, D]
    kv_norm_w: torch.Tensor,  # [head_dim]
    wo_a: torch.Tensor,  # [o_groups * o_lora_rank, n_heads * head_dim / o_groups]
    wo_b: torch.Tensor,  # [D, o_groups * o_lora_rank]
    attn_sink,  # Optional[torch.Tensor] of shape [n_heads]; None disables sink
    n_heads: int,
    head_dim: int,
    rotary_dim: int,
    o_groups: int,
    o_lora_rank: int,
    norm_eps: float,
) -> torch.Tensor:
    """Reference V4 attention forward (CPU, fp32-ish, no parallel linear)."""
    B, S, D = hidden.shape

    # Q branch.
    q_compressed = hidden @ wq_a.t()  # [B, S, q_lora_rank]
    # RMSNorm on q_lora_rank with learnable gamma (== q_layernorm).
    q32 = q_compressed.float()
    q_rms = torch.rsqrt(q32.square().mean(dim=-1, keepdim=True) + norm_eps)
    q_compressed = (q32 * q_rms).to(q_compressed.dtype) * q_norm_w
    q = q_compressed @ wq_b.t()  # [B, S, n_heads * head_dim]
    q = q.view(B, S, n_heads, head_dim)
    q = _rms_norm_per_head(q, norm_eps)

    # KV branch (single-latent).
    kv = hidden @ wkv.t()  # [B, S, head_dim]
    kv32 = kv.float()
    kv_rms = torch.rsqrt(kv32.square().mean(dim=-1, keepdim=True) + norm_eps)
    kv = (kv32 * kv_rms).to(kv.dtype) * kv_norm_w
    kv = kv.view(B, S, 1, head_dim)

    # Partial RoPE (interleaved) on Q and K. K = kv (rope-applied).
    q = rope.apply_rope(q, position_ids=position_ids, compress_ratio=0)
    kv = rope.apply_rope(kv, position_ids=position_ids, compress_ratio=0)
    k = kv  # [B, S, 1, head_dim]
    v = kv  # K = V (single latent)

    # Broadcast K / V across heads.
    k = k.expand(B, S, n_heads, head_dim)
    v = v.expand(B, S, n_heads, head_dim)

    # Causal mask (no SWA in the test).
    q_idx = torch.arange(S, device=hidden.device).unsqueeze(1)
    k_idx = torch.arange(S, device=hidden.device).unsqueeze(0)
    causal = torch.where(q_idx >= k_idx, 0.0, float("-inf")).to(hidden.dtype)

    # Move heads dim before sequence.
    q_bh = q.transpose(1, 2)  # [B, H, S, head_dim]
    k_bh = k.transpose(1, 2)
    v_bh = v.transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    logits = torch.matmul(q_bh.float(), k_bh.float().transpose(-2, -1)) * scale
    logits = logits + causal

    if attn_sink is None:
        # Plain causal softmax (no virtual key column).
        logits = logits - logits.amax(dim=-1, keepdim=True).detach()
        probs = logits.softmax(dim=-1).to(v_bh.dtype)
    else:
        # attn_sink: append a virtual key column with zero value, drop after softmax.
        sink_col = attn_sink.float().view(1, n_heads, 1, 1).expand(B, n_heads, S, 1)
        logits_aug = torch.cat([logits, sink_col], dim=-1)
        logits_aug = logits_aug - logits_aug.amax(dim=-1, keepdim=True).detach()
        probs = logits_aug.softmax(dim=-1)
        probs = probs[..., :-1].to(v_bh.dtype)
    out = torch.matmul(probs, v_bh.float()).to(hidden.dtype)
    out = out.transpose(1, 2).contiguous()  # [B, S, H, head_dim]

    # Grouped low-rank O.
    out_g = out.reshape(B, S, o_groups, (n_heads * head_dim) // o_groups)
    wo_a_w = wo_a.view(o_groups, o_lora_rank, (n_heads * head_dim) // o_groups)
    o = torch.einsum("bsgd,grd->bsgr", out_g, wo_a_w)
    o = o.flatten(2)
    return o @ wo_b.t()


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


_TEST_DTYPE = torch.float32


def _make_v4_config(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_lora_rank: int,
    o_groups: int,
    o_lora_rank: int,
    attn_sink: bool,
    norm_eps: float = 1e-6,
) -> DeepSeekV4TransformerConfig:
    """Minimal V4 config for CPU unit tests.

    Relies on dataclass defaults wherever possible and only sets the
    fields the V4 attention actually reads.
    """
    return DeepSeekV4TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=1,
        kv_channels=head_dim,
        qk_pos_emb_head_dim=rotary_dim,
        qk_head_dim=head_dim - rotary_dim,
        v_head_dim=head_dim,
        kv_lora_rank=head_dim,  # unused — KV branch is overridden by V4
        rope_type="rope",
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        # V4 extras
        q_lora_rank=q_lora_rank,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        attn_sliding_window=0,
        attn_sink=attn_sink,
        compress_ratios=None,
        compress_rope_theta=160000.0,
        # Misc
        layernorm_epsilon=norm_eps,
        norm_epsilon=norm_eps,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


def _make_attention(config: DeepSeekV4TransformerConfig) -> DeepseekV4Attention:
    """Construct V4 attention with default (no-spec) submodules, on CPU.

    With ``submodules=None`` every projection falls back to ``nn.Linear``
    (replicated, no TP), which is exactly what we want for a CPU smoke
    test against the inline reference.
    """
    rope = DualRoPE(
        rotary_dim=config.qk_pos_emb_head_dim,
        rope_theta=config.rotary_base,
        compress_rope_theta=config.compress_rope_theta,
        yarn_factor=1.0,
        original_max_position_embeddings=config.original_max_position_embeddings,
    )
    return DeepseekV4Attention(
        config,
        rope=rope,
        compress_ratio=0,
        submodules=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_state_dict_keys_match_v4_canonical_layout():
    """The new attention exposes V4-canonical state-dict keys.

    These are exactly the keys the P17 state-dict adapter will map from
    the released ``layers.{i}.attn.{wq_a,wq_b,wkv,q_norm,kv_norm,wo_a,wo_b,attn_sink}``
    safetensors layout.
    """
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=True,
    )
    attn = _make_attention(config)
    keys = set(attn.state_dict().keys())

    expected = {
        "linear_q_down_proj.weight",
        "linear_q_up_proj.weight",
        "linear_kv.weight",
        "q_layernorm.weight",
        "kv_layernorm.weight",
        "linear_o_a.weight",
        "linear_o_b.weight",
        "attn_sink",  # direct nn.Parameter, matches released checkpoint key
    }
    missing = expected - keys
    assert not missing, f"missing V4-canonical keys: {missing}"

    legacy = {
        "q_a.weight",
        "q_b.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
    }
    bleed = legacy & keys
    assert not bleed, f"legacy plan-1 keys leaked into V4-faithful attention: {bleed}"


def test_forward_shape_and_finite():
    """Forward pass returns ``[B, S, hidden_size]`` of finite values."""
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=True,
    )
    attn = _make_attention(config).to(_TEST_DTYPE)

    B, S, D = 2, 8, 64
    hidden = torch.randn(B, S, D, dtype=_TEST_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)
    out = attn(hidden, position_ids)
    assert out.shape == (B, S, D)
    assert torch.isfinite(out).all()


def _copy_weights_to_reference(attn: DeepseekV4Attention) -> dict:
    """Extract weights from the new attention into the inline reference's
    parameter dict."""
    return {
        "wq_a": attn.linear_q_down_proj.weight.detach().clone(),
        "wq_b": attn.linear_q_up_proj.weight.detach().clone(),
        "q_norm_w": attn.q_layernorm.weight.detach().clone(),
        "wkv": attn.linear_kv.weight.detach().clone(),
        "kv_norm_w": attn.kv_layernorm.weight.detach().clone(),
        "wo_a": attn.linear_o_a.weight.detach().clone(),
        "wo_b": attn.linear_o_b.weight.detach().clone(),
    }


@pytest.mark.parametrize("attn_sink_enabled", [False, True])
def test_forward_matches_inline_reference(attn_sink_enabled: bool):
    """1-layer V4 attention forward agrees with the inline reference.

    Both implementations apply the same math (single-latent KV, partial
    interleaved RoPE, attn_sink as virtual key column, grouped low-rank
    O), so the agreement is essentially numerical noise (≤1e-5 in fp32).
    """
    torch.manual_seed(0)
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=attn_sink_enabled,
    )
    attn = _make_attention(config).to(_TEST_DTYPE)
    sink_tensor = None
    if attn_sink_enabled:
        # Pull in some non-trivial sink scalars so the test exercises
        # the virtual-key-column path.
        with torch.no_grad():
            attn.attn_sink.copy_(torch.linspace(-0.5, 0.5, attn.num_heads))
        sink_tensor = attn.attn_sink.detach().clone()

    weights = _copy_weights_to_reference(attn)

    B, S = 2, 8
    hidden = torch.randn(B, S, config.hidden_size, dtype=_TEST_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    with torch.no_grad():
        ours = attn(hidden, position_ids)
        ref = _reference_v4_attention_forward(
            hidden=hidden,
            position_ids=position_ids,
            rope=attn.rope,
            attn_sink=sink_tensor,
            n_heads=attn.num_heads,
            head_dim=attn.head_dim,
            rotary_dim=attn.rotary_dim,
            o_groups=attn.o_groups,
            o_lora_rank=attn.o_lora_rank,
            norm_eps=attn.norm_eps,
            **weights,
        )

    diff = (ours - ref).abs().max().item()
    assert diff < 1e-3, f"forward mismatch: max abs diff = {diff:.3e}"


def test_per_head_q_rms_is_parameterless():
    """The released checkpoint stores no separate ``q_rms`` parameter.

    Plan-2 P13 lands per-head q_rms as inline math (parameter-less RMS on
    ``head_dim``), not as an additional ``nn.Module`` with a learnable
    gamma. This regression test guards that contract.
    """
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=False,
    )
    attn = _make_attention(config)
    keys = set(attn.state_dict().keys())
    forbidden = {"q_rms.weight", "q_rms_norm.weight", "q_per_head_rms.weight"}
    leaked = forbidden & keys
    assert not leaked, (
        f"Per-head q_rms must be parameter-less (released checkpoint has no "
        f"such key); leaked params: {leaked}"
    )


def test_o_lora_rank_zero_falls_back_to_flat_proj():
    """Setting ``o_lora_rank == 0`` skips the grouped O path."""
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=0,
        attn_sink=False,
    )
    attn = _make_attention(config)
    assert attn.linear_o_a is None
    assert attn.linear_o_b is None
    assert attn.linear_proj is not None
    keys = set(attn.state_dict().keys())
    assert "linear_proj.weight" in keys
    assert "linear_o_a.weight" not in keys
    assert "linear_o_b.weight" not in keys


def test_compress_ratio_nonzero_rejected():
    """V4-faithful class refuses compress_ratio != 0 (plan-2 P13 first commit)."""
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=False,
    )
    rope = DualRoPE(
        rotary_dim=config.qk_pos_emb_head_dim,
        rope_theta=config.rotary_base,
        compress_rope_theta=config.compress_rope_theta,
    )
    with pytest.raises(ValueError, match="compress_ratio == 0 only"):
        DeepseekV4Attention(config, rope=rope, compress_ratio=4, submodules=None)


def test_q_lora_rank_zero_rejected():
    """V4 always uses Q LoRA (wq_a + wq_b); reject q_lora_rank == 0."""
    config = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=0,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=False,
    )
    rope = DualRoPE(
        rotary_dim=config.qk_pos_emb_head_dim,
        rope_theta=config.rotary_base,
        compress_rope_theta=config.compress_rope_theta,
    )
    with pytest.raises(ValueError, match="q_lora_rank > 0"):
        DeepseekV4Attention(config, rope=rope, compress_ratio=0, submodules=None)
