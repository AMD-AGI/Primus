###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the CSA indexer distillation loss.

Covers the two halves separately:

* the loss itself -- KL against a known target, the all-masked-row guard, and
  the fact that it actually produces indexer gradients;
* the wiring -- ``v4_indexer_distill_loss_coeff`` gates both the loss and
  whether the indexer parameters are trainable at all.
"""

from __future__ import annotations

import math

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
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE  # noqa: E402
from primus.backends.megatron.core.transformer.indexer_distill_loss import (  # noqa: E402
    V4IndexerLossAutoScaler,
    compute_indexer_distill_loss,
)

_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# The loss itself
# ---------------------------------------------------------------------------


def test_kl_is_zero_when_indexer_matches_attention():
    """A perfectly-predicting indexer incurs no loss.

    Build the indexer scores so their softmax equals the (single-head)
    attention distribution over the selected entries; KL must vanish.
    """
    torch.manual_seed(0)
    B, H, S, K, Dh, P = 1, 1, 3, 4, 8, 6
    query = torch.randn(B, H, S, Dh, dtype=_DTYPE)
    pool = torch.randn(B, P, Dh, dtype=_DTYPE)
    topk_idxs = torch.arange(K, dtype=torch.long).view(1, 1, K).expand(B, S, K).contiguous()
    scale = 1.0 / math.sqrt(Dh)

    gathered = pool[torch.arange(B).view(B, 1, 1), topk_idxs]  # [B,S,K,Dh]
    attn_logits = torch.einsum("bhsd,bskd->bhsk", query, gathered) * scale
    # One head, so the target distribution is exactly this row's softmax; give
    # the indexer the same logits.
    index_scores = attn_logits[:, 0]

    loss = compute_indexer_distill_loss(
        index_topk_scores=index_scores,
        topk_idxs=topk_idxs,
        query=query,
        pool=pool,
        softmax_scale=scale,
        loss_coeff=1.0,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_and_scales_with_coeff():
    """A mismatched indexer is penalised, linearly in the coefficient."""
    torch.manual_seed(0)
    B, H, S, K, Dh, P = 2, 4, 5, 3, 8, 7
    query = torch.randn(B, H, S, Dh, dtype=_DTYPE)
    pool = torch.randn(B, P, Dh, dtype=_DTYPE)
    topk_idxs = torch.randint(0, P, (B, S, K), dtype=torch.long)
    index_scores = torch.randn(B, S, K, dtype=_DTYPE)
    scale = 1.0 / math.sqrt(Dh)

    kwargs = dict(
        index_topk_scores=index_scores,
        topk_idxs=topk_idxs,
        query=query,
        pool=pool,
        softmax_scale=scale,
    )
    loss_1 = compute_indexer_distill_loss(loss_coeff=1.0, **kwargs)
    loss_2 = compute_indexer_distill_loss(loss_coeff=2.0, **kwargs)

    assert loss_1.item() > 0.0
    assert loss_2.item() == pytest.approx(2.0 * loss_1.item(), rel=1e-5)


def test_fully_masked_rows_do_not_produce_nan():
    """Early queries can have zero legal compressed entries.

    Those rows softmax over all -inf; the loss must neutralise them rather
    than emit NaN (which would poison the whole step).
    """
    B, H, S, K, Dh, P = 1, 2, 4, 3, 8, 5
    query = torch.randn(B, H, S, Dh, dtype=_DTYPE)
    pool = torch.randn(B, P, Dh, dtype=_DTYPE)

    topk_idxs = torch.randint(0, P, (B, S, K), dtype=torch.long)
    index_scores = torch.randn(B, S, K, dtype=_DTYPE)
    # Row 0 has no legal entry at all; row 1 has a single one.
    topk_idxs[0, 0, :] = -1
    index_scores[0, 0, :] = float("-inf")
    topk_idxs[0, 1, 1:] = -1
    index_scores[0, 1, 1:] = float("-inf")

    loss = compute_indexer_distill_loss(
        index_topk_scores=index_scores,
        topk_idxs=topk_idxs,
        query=query,
        pool=pool,
        softmax_scale=1.0 / math.sqrt(Dh),
        loss_coeff=1e-2,
    )
    assert torch.isfinite(loss), f"loss must stay finite, got {loss}"


def test_loss_produces_indexer_gradients():
    """The KL flows back into whatever produced the index scores."""
    torch.manual_seed(0)
    B, H, S, K, Dh, P = 1, 2, 4, 3, 8, 6
    query = torch.randn(B, H, S, Dh, dtype=_DTYPE)
    pool = torch.randn(B, P, Dh, dtype=_DTYPE)
    topk_idxs = torch.randint(0, P, (B, S, K), dtype=torch.long)

    index_scores = torch.randn(B, S, K, dtype=_DTYPE, requires_grad=True)
    loss = compute_indexer_distill_loss(
        index_topk_scores=index_scores,
        topk_idxs=topk_idxs,
        query=query,
        pool=pool,
        softmax_scale=1.0 / math.sqrt(Dh),
        loss_coeff=1e-2,
    )
    loss.backward()

    assert index_scores.grad is not None
    assert torch.isfinite(index_scores.grad).all()
    assert index_scores.grad.abs().sum() > 0.0


def test_auto_scaler_is_transparent_forward_and_seeds_aux_backward():
    """The scaler passes the tensor through and differentiates the aux loss."""
    x = torch.randn(3, 4, requires_grad=True)
    aux_src = torch.randn(2, requires_grad=True)
    aux_loss = aux_src.sum()

    out = V4IndexerLossAutoScaler.apply(x, aux_loss)
    torch.testing.assert_close(out, x, rtol=0, atol=0)

    out.sum().backward()
    # x keeps its own gradient, and the aux subgraph got seeded with ones.
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))
    assert aux_src.grad is not None and torch.allclose(aux_src.grad, torch.ones_like(aux_src))


# ---------------------------------------------------------------------------
# Wiring: the coefficient gates training of the indexer
# ---------------------------------------------------------------------------


def _make_csa_attention(coeff: float) -> DeepseekV4Attention:
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
        v4_indexer_distill_loss_coeff=coeff,
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


def test_indexer_frozen_when_coeff_is_zero(monkeypatch):
    """Default (0.0): no loss, and the indexer stays out of the grad buckets."""
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    attn = _make_csa_attention(coeff=0.0)

    assert attn.indexer_distill_enabled is False
    assert attn.indexer is not None
    assert not any(p.requires_grad for p in attn.indexer.parameters())


def test_indexer_trainable_when_coeff_positive(monkeypatch):
    """A positive coefficient unfreezes the indexer."""
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    attn = _make_csa_attention(coeff=1e-2)

    assert attn.indexer_distill_enabled is True
    assert attn.indexer is not None
    assert all(p.requires_grad for p in attn.indexer.parameters())


def test_forward_backward_reaches_indexer_weights(monkeypatch):
    """End to end: a CSA step with the loss on gives the indexer gradients.

    Without the distillation loss the indexer is unreachable from the output
    (only argTopK indices are consumed), so a non-zero gradient here is proof
    the aux objective is wired into the main backward.
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=1e-2).to(_DTYPE)
    attn.train()

    B, S = 1, 8  # P = S // 4 = 2 pool entries
    hidden = torch.randn(B, S, attn.config.hidden_size, dtype=_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    out = attn(hidden, position_ids)
    out.sum().backward()

    assert attn.last_indexer_distill_loss is not None
    assert torch.isfinite(attn.last_indexer_distill_loss)

    grads = [p.grad for p in attn.indexer.parameters() if p.grad is not None]
    assert grads, "indexer received no gradient at all"
    total = sum(g.abs().sum().item() for g in grads)
    assert math.isfinite(total) and total > 0.0, f"indexer gradient is degenerate: {total}"


def test_no_indexer_loss_in_eval(monkeypatch):
    """Eval must not build the aux graph."""
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=1e-2).to(_DTYPE)
    attn.eval()
    attn.last_indexer_distill_loss = None

    B, S = 1, 8
    hidden = torch.randn(B, S, attn.config.hidden_size, dtype=_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)
    with torch.no_grad():
        attn(hidden, position_ids)

    assert attn.last_indexer_distill_loss is None
