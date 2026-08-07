###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the indexer's partial RoPE + Hadamard rotation.

The open-source reference prepares both scoring operands before the ReLU'd dot
product: partial RoPE at the compressed-branch base (queries at their token
positions, compressed keys at ``s * compress_ratio``) followed by a normalised
Hadamard rotation. It does this to the indexer queries directly, and inside the
indexer's own compressor, which is built with rotation enabled -- unlike the main
compressed pool, which is not rotated.

Covers the rotation primitive, the two operand paths, and the wiring that keeps
a CSA layer from silently constructing an unrotated indexer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

mla_module = pytest.importorskip(
    "megatron.core.transformer.multi_latent_attention",
    reason="MLA base module not importable in this environment",
)

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: E402
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (  # noqa: E402
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import (  # noqa: E402
    DualRoPE,
    apply_interleaved_partial_rope,
)
from primus.backends.megatron.core.transformer.hadamard_rotation import (  # noqa: E402
    rotate_activation,
)
from primus.backends.megatron.core.transformer.indexer import Indexer  # noqa: E402

_HIDDEN = 64
_HEAD_DIM = 16
_N_HEADS = 4
_TOPK = 4
_RATIO = 4
_ROTARY_DIM = 8
_THETA = 40000.0


def _make_rope() -> DualRoPE:
    return DualRoPE(
        rotary_dim=_ROTARY_DIM,
        rope_theta=10000.0,
        compress_rope_theta=_THETA,
        yarn_factor=1.0,
        original_max_position_embeddings=2048,
    )


def _make_indexer(*, rotated: bool = True) -> Indexer:
    rope = _make_rope().get_rope(compress_ratio=_RATIO) if rotated else None
    return Indexer(
        hidden_size=_HIDDEN,
        index_head_dim=_HEAD_DIM,
        index_n_heads=_N_HEADS,
        index_topk=_TOPK,
        compress_ratio=_RATIO,
        rope=rope,
        rotary_dim=_ROTARY_DIM if rotated else 0,
    )


# ---------------------------------------------------------------------------
# The rotation primitive
# ---------------------------------------------------------------------------


def test_hadamard_rotation_preserves_inner_products():
    """It is orthogonal, so scores are unchanged in exact arithmetic.

    This is why applying it to both operands is safe: the value it adds is
    numerical (spreading channel energy for the low-precision QK), not a change
    to what the indexer computes.
    """
    torch.manual_seed(0)
    a = torch.randn(5, 32, dtype=torch.float64)
    b = torch.randn(7, 32, dtype=torch.float64)

    before = a @ b.T
    after = rotate_activation(a) @ rotate_activation(b).T

    torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-10)


def test_hadamard_rotation_is_an_involution():
    """``H / sqrt(n)`` is symmetric and orthogonal, so applying it twice is identity."""
    torch.manual_seed(0)
    x = torch.randn(3, 16, dtype=torch.float64)
    torch.testing.assert_close(rotate_activation(rotate_activation(x)), x, rtol=1e-10, atol=1e-10)


def test_hadamard_rotation_actually_mixes_channels():
    """Guard the two assertions above from passing on an identity transform."""
    x = torch.zeros(1, 16, dtype=torch.float64)
    x[0, 0] = 1.0
    rotated = rotate_activation(x)
    assert (rotated.abs() > 0).all(), "a basis vector must spread over every channel"


def test_hadamard_rotation_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        rotate_activation(torch.randn(2, 12))


# ---------------------------------------------------------------------------
# Operand preparation
# ---------------------------------------------------------------------------


def test_keys_are_rotated_at_compressed_positions():
    """Compressed key ``s`` must be rotated at original token ``s * ratio``.

    Same convention as the main compressed pool; using the bare block index
    would put the keys in a different coordinate system from the queries.
    """
    torch.manual_seed(0)
    indexer = _make_indexer()
    hidden = torch.randn(1, 16, _HIDDEN, dtype=torch.float32)

    with torch.no_grad():
        pooled = indexer.indexer_compressor(hidden)  # [B, P, Hd]
        actual = indexer._rotate_keys(pooled)

        P = pooled.shape[1]
        positions = torch.arange(P) * _RATIO
        cos, sin = indexer.rope(positions)
        expected = rotate_activation(
            apply_interleaved_partial_rope(
                pooled.unsqueeze(2),
                cos.unsqueeze(0).expand(pooled.shape[0], -1, -1),
                sin.unsqueeze(0).expand(pooled.shape[0], -1, -1),
                rotary_dim=_ROTARY_DIM,
            ).squeeze(2)
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Not a no-op: block indices would give a different answer.
    with torch.no_grad():
        cos_b, sin_b = indexer.rope(torch.arange(P))
        block_indexed = rotate_activation(
            apply_interleaved_partial_rope(
                pooled.unsqueeze(2),
                cos_b.unsqueeze(0).expand(pooled.shape[0], -1, -1),
                sin_b.unsqueeze(0).expand(pooled.shape[0], -1, -1),
                rotary_dim=_ROTARY_DIM,
            ).squeeze(2)
        )
    assert not torch.allclose(actual, block_indexed, rtol=1e-4, atol=1e-5)


def test_queries_are_rotated_at_token_positions():
    torch.manual_seed(0)
    indexer = _make_indexer()
    B, S = 1, 16
    q_i = torch.randn(B, S, _N_HEADS, _HEAD_DIM, dtype=torch.float32)
    position_ids = torch.arange(S)

    with torch.no_grad():
        actual = indexer._rotate_queries(q_i, position_ids)
        cos, sin = indexer.rope(position_ids)
        expected = rotate_activation(
            apply_interleaved_partial_rope(
                q_i,
                cos.expand(B, S, -1),
                sin.expand(B, S, -1),
                rotary_dim=_ROTARY_DIM,
            )
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_query_rotation_defaults_to_arange_positions():
    torch.manual_seed(0)
    indexer = _make_indexer()
    q_i = torch.randn(1, 16, _N_HEADS, _HEAD_DIM, dtype=torch.float32)
    with torch.no_grad():
        implicit = indexer._rotate_queries(q_i, None)
        explicit = indexer._rotate_queries(q_i, torch.arange(16))
    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)


def test_position_ids_change_the_selection_scores():
    """Rotating the queries has to depend on where the token actually is."""
    torch.manual_seed(0)
    indexer = _make_indexer()
    hidden = torch.randn(1, 16, _HIDDEN, dtype=torch.float32)

    with torch.no_grad():
        _, scores_a = indexer(hidden, torch.arange(16))
        _, scores_b = indexer(hidden, torch.arange(16) + 64)

    finite = torch.isfinite(scores_a) & torch.isfinite(scores_b)
    assert finite.any()
    assert not torch.allclose(scores_a[finite], scores_b[finite], rtol=1e-4, atol=1e-5)


def test_rotation_changes_the_scores_versus_the_unrotated_indexer():
    """The whole point: adding RoPE + Hadamard is not a no-op on the scores."""
    torch.manual_seed(0)
    rotated = _make_indexer(rotated=True)
    plain = _make_indexer(rotated=False)
    plain.load_state_dict(rotated.state_dict())

    hidden = torch.randn(1, 16, _HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        _, scores_rotated = rotated(hidden)
        _, scores_plain = plain(hidden)

    finite = torch.isfinite(scores_rotated) & torch.isfinite(scores_plain)
    assert finite.any()
    assert not torch.allclose(scores_rotated[finite], scores_plain[finite], rtol=1e-4, atol=1e-5)


def test_sentinels_and_shapes_are_unchanged_by_the_rotation():
    torch.manual_seed(0)
    indexer = _make_indexer()
    hidden = torch.randn(1, 8, _HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        idxs, scores = indexer(hidden)

    assert idxs.shape == scores.shape == (1, 8, _TOPK)
    invalid = idxs < 0
    assert invalid.any()
    assert torch.isneginf(scores[invalid]).all()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _make_csa_attention() -> DeepseekV4Attention:
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
        compress_rope_theta=_THETA,
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
    return DeepseekV4Attention(config, rope=rope, compress_ratio=4, submodules=None)


def test_csa_layer_always_builds_a_rotated_indexer():
    """Guard against a silently unrotated indexer in production.

    ``Indexer`` tolerates ``rope=None`` for isolated unit tests, so the layer
    that builds it is where the contract has to be enforced.
    """
    attn = _make_csa_attention()
    assert attn.indexer is not None
    assert attn.indexer.rope is not None, "CSA layer built an indexer with no RoPE"
    assert attn.indexer.rotary_dim == attn.rotary_dim > 0


def test_indexer_shares_the_layers_compressed_rope_cache():
    """It must be the same object, so base / YaRN can never diverge."""
    attn = _make_csa_attention()
    assert attn.indexer.rope is attn.rope.get_rope(compress_ratio=4)
    assert attn.indexer.rope is not attn.rope.main_rope


def test_csa_forward_threads_position_ids_into_the_indexer():
    """Shifting the positions must reach the selector, not just the attention."""
    torch.manual_seed(0)
    attn = _make_csa_attention().to(torch.float32)
    attn.eval()

    B, S = 1, 8
    hidden = torch.randn(B, S, attn.config.hidden_size, dtype=torch.float32)

    captured = []
    original = attn.indexer.forward

    def _spy(h, position_ids=None):
        captured.append(position_ids)
        return original(h, position_ids)

    attn.indexer.forward = _spy
    try:
        with torch.no_grad():
            attn(hidden, torch.arange(S).unsqueeze(0).expand(B, S))
    finally:
        attn.indexer.forward = original

    assert captured, "the indexer was never called"
    assert captured[0] is not None, "position_ids did not reach the indexer"
    assert captured[0].shape[-1] == S
