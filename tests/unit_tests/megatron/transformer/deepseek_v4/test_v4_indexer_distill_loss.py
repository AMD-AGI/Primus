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

import copy
import math
import os

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
    INDEXER_DISTILL_LOSS_NAME,
    V4IndexerLossAutoScaler,
    compute_indexer_distill_loss,
    log_indexer_distill_loss,
)

_DTYPE = torch.float32


@pytest.fixture(autouse=True)
def _reset_aux_loss_scale():
    """Keep the class-level scale override from leaking across tests."""
    V4IndexerLossAutoScaler.set_loss_scale(None)
    yield
    V4IndexerLossAutoScaler.set_loss_scale(None)


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


def test_loss_does_not_flow_into_the_target_side():
    """Distillation is one-directional: the target must not be trainable.

    ``query`` and ``pool`` describe the distribution the indexer is supposed to
    imitate. If the KL could reach them, the main attention would be rewarded
    for becoming easier to predict instead of the indexer for predicting
    better.
    """
    torch.manual_seed(0)
    B, H, S, K, Dh, P = 1, 2, 4, 3, 8, 6
    query = torch.randn(B, H, S, Dh, dtype=_DTYPE, requires_grad=True)
    pool = torch.randn(B, P, Dh, dtype=_DTYPE, requires_grad=True)
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

    assert query.grad is None, "the KL target must be detached"
    assert pool.grad is None, "the compressed pool must be detached"
    # ...while the side that is supposed to learn still gets its gradient.
    assert index_scores.grad is not None and index_scores.grad.abs().sum() > 0.0


def test_head_sum_is_all_reduced_when_the_heads_are_sharded():
    """The target distribution needs every head before it is renormalised.

    V4 gathers its Q projection so the sum is already complete, but the loss has
    to reduce when told the heads are sharded -- otherwise dropping the gather
    would quietly turn the target into a partial sum.
    """
    torch.manual_seed(0)
    B, H, S, K, Dh, P = 1, 2, 4, 3, 8, 6
    kwargs = dict(
        index_topk_scores=torch.randn(B, S, K, dtype=_DTYPE),
        topk_idxs=torch.randint(0, P, (B, S, K), dtype=torch.long),
        query=torch.randn(B, H, S, Dh, dtype=_DTYPE),
        pool=torch.randn(B, P, Dh, dtype=_DTYPE),
        softmax_scale=1.0 / math.sqrt(Dh),
        loss_coeff=1.0,
    )

    calls = []

    class _FakeGroup:
        pass

    group = _FakeGroup()

    import primus.backends.megatron.core.transformer.indexer_distill_loss as mod

    real_world_size = torch.distributed.get_world_size
    real_all_reduce = torch.distributed.all_reduce
    torch.distributed.get_world_size = lambda g=None: 2 if g is group else 1
    torch.distributed.all_reduce = lambda t, group=None: calls.append((t.shape, group))
    try:
        loss = mod.compute_indexer_distill_loss(head_reduce_group=group, **kwargs)
    finally:
        torch.distributed.get_world_size = real_world_size
        torch.distributed.all_reduce = real_all_reduce

    assert torch.isfinite(loss)
    assert calls, "the head sum was not reduced"
    assert calls[0][0] == (B, S, K), "must reduce the head-summed target, not the per-head scores"
    assert calls[0][1] is group


def test_head_sum_is_not_reduced_without_a_group():
    """No group means no collective -- the common V4 case must stay comm-free."""
    torch.manual_seed(0)
    B, H, S, K, Dh, P = 1, 2, 4, 3, 8, 6
    calls = []
    real_all_reduce = torch.distributed.all_reduce
    torch.distributed.all_reduce = lambda t, group=None: calls.append(t)
    try:
        compute_indexer_distill_loss(
            index_topk_scores=torch.randn(B, S, K, dtype=_DTYPE),
            topk_idxs=torch.randint(0, P, (B, S, K), dtype=torch.long),
            query=torch.randn(B, H, S, Dh, dtype=_DTYPE),
            pool=torch.randn(B, P, Dh, dtype=_DTYPE),
            softmax_scale=1.0 / math.sqrt(Dh),
            loss_coeff=1.0,
        )
    finally:
        torch.distributed.all_reduce = real_all_reduce
    assert not calls


def test_head_group_is_none_when_q_is_gathered(monkeypatch):
    """V4's column-parallel Q gathers its output, so no group is resolved."""
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    attn = _make_csa_attention(coeff=1e-2)
    assert attn.num_attention_heads_per_partition == int(attn.config.num_attention_heads)
    assert attn._indexer_loss_head_group() is None


def test_auto_scaler_is_transparent_forward_and_seeds_aux_backward():
    """The scaler passes the tensor through and differentiates the aux loss."""
    V4IndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    x = torch.randn(3, 4, requires_grad=True)
    aux_src = torch.randn(2, requires_grad=True)
    aux_loss = aux_src.sum()

    out = V4IndexerLossAutoScaler.apply(x, aux_loss)
    torch.testing.assert_close(out, x, rtol=0, atol=0)

    out.sum().backward()
    # x keeps its own gradient, and the aux subgraph got seeded with ones.
    assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))
    assert aux_src.grad is not None and torch.allclose(aux_src.grad, torch.ones_like(aux_src))


def test_auto_scaler_applies_the_explicit_scale():
    """An explicit override is the gradient seeded into the aux loss."""
    V4IndexerLossAutoScaler.set_loss_scale(torch.tensor(0.25))
    x = torch.randn(2, 2, requires_grad=True)
    aux_src = torch.randn(3, requires_grad=True)

    V4IndexerLossAutoScaler.apply(x, aux_src.sum()).sum().backward()

    assert aux_src.grad is not None
    torch.testing.assert_close(aux_src.grad, torch.full_like(aux_src, 0.25))


def test_auto_scaler_follows_the_moe_aux_loss_scale(monkeypatch):
    """Default behaviour: inherit the per-microbatch MoE aux-loss scale.

    Seeding a bare 1.0 would make the effective coefficient
    ``num_microbatches`` times too large under gradient accumulation, so the
    scaler reads the quantity Megatron's schedule already computes.
    """
    import primus.backends.megatron.core.transformer.indexer_distill_loss as mod

    monkeypatch.setattr(mod, "_moe_aux_loss_scale", lambda: torch.tensor(0.125))

    aux_src = torch.randn(3, requires_grad=True)
    x = torch.randn(2, 2, requires_grad=True)
    V4IndexerLossAutoScaler.apply(x, aux_src.sum()).sum().backward()

    assert aux_src.grad is not None
    torch.testing.assert_close(aux_src.grad, torch.full_like(aux_src, 0.125))


def test_auto_scaler_falls_back_to_one_without_megatron(monkeypatch):
    """No override and no MoE scale available -> neutral seed, never a crash."""
    import primus.backends.megatron.core.transformer.indexer_distill_loss as mod

    monkeypatch.setattr(mod, "_moe_aux_loss_scale", lambda: None)

    aux_src = torch.randn(3, requires_grad=True)
    x = torch.randn(2, 2, requires_grad=True)
    V4IndexerLossAutoScaler.apply(x, aux_src.sum()).sum().backward()

    assert aux_src.grad is not None
    torch.testing.assert_close(aux_src.grad, torch.ones_like(aux_src))


def test_explicit_scale_wins_over_the_moe_scale(monkeypatch):
    """``set_loss_scale`` is an override, not a default."""
    import primus.backends.megatron.core.transformer.indexer_distill_loss as mod

    monkeypatch.setattr(mod, "_moe_aux_loss_scale", lambda: torch.tensor(0.125))
    V4IndexerLossAutoScaler.set_loss_scale(torch.tensor(2.0))

    aux_src = torch.randn(3, requires_grad=True)
    x = torch.randn(2, 2, requires_grad=True)
    V4IndexerLossAutoScaler.apply(x, aux_src.sum()).sum().backward()

    torch.testing.assert_close(aux_src.grad, torch.full_like(aux_src, 2.0))

    # ...and clearing it restores the MoE-following default.
    V4IndexerLossAutoScaler.set_loss_scale(None)
    aux_src2 = torch.randn(3, requires_grad=True)
    V4IndexerLossAutoScaler.apply(x, aux_src2.sum()).sum().backward()
    torch.testing.assert_close(aux_src2.grad, torch.full_like(aux_src2, 0.125))


# ---------------------------------------------------------------------------
# Wiring: the coefficient gates training of the indexer
# ---------------------------------------------------------------------------


def _make_v4_config(coeff: float) -> DeepSeekV4TransformerConfig:
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
    # Only needed by callers that build a whole DeepseekV4HybridLayer; a dense
    # MLP keeps that construction cheap.
    config.num_moe_experts = 0
    config.ffn_hidden_size = 128
    return config


def _make_v4_rope(config: DeepSeekV4TransformerConfig) -> DualRoPE:
    return DualRoPE(
        rotary_dim=config.qk_pos_emb_head_dim,
        rope_theta=config.rotary_base,
        compress_rope_theta=config.compress_rope_theta,
        yarn_factor=1.0,
        original_max_position_embeddings=config.original_max_position_embeddings,
    )


def _make_csa_attention(coeff: float, layer_number: int = 1) -> DeepseekV4Attention:
    config = _make_v4_config(coeff)
    return DeepseekV4Attention(
        config,
        rope=_make_v4_rope(config),
        compress_ratio=4,
        submodules=None,
        layer_number=layer_number,
    )


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


def test_kl_leaves_the_backbone_gradients_untouched(monkeypatch):
    """Turning the loss on must not change a single backbone gradient.

    The indexer is fed a detached hidden state and the KL target is detached,
    so the auxiliary objective is entirely confined to the indexer's own
    parameters. Running the same weights and the same input with the loss off
    and with an absurdly large coefficient therefore has to produce identical
    gradients everywhere outside ``indexer.*`` -- including on the input, which
    stands in for every layer below.
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    torch.manual_seed(0)
    off = _make_csa_attention(coeff=0.0).to(_DTYPE)
    on = _make_csa_attention(coeff=1e3).to(_DTYPE)
    on.load_state_dict(copy.deepcopy(off.state_dict()))
    off.train()
    on.train()

    B, S = 1, 8
    torch.manual_seed(1)
    base = torch.randn(B, S, off.config.hidden_size, dtype=_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    grads = {}
    for tag, module in (("off", off), ("on", on)):
        hidden = base.clone().requires_grad_(True)
        module(hidden, position_ids).sum().backward()
        grads[tag] = {
            "__input__": hidden.grad.clone(),
            **{
                name: p.grad.clone()
                for name, p in module.named_parameters()
                if p.grad is not None and not name.startswith("indexer.")
            },
        }

    assert grads["off"].keys() == grads["on"].keys()
    for name in grads["off"]:
        torch.testing.assert_close(
            grads["on"][name],
            grads["off"][name],
            rtol=0,
            atol=0,
            msg=lambda s, n=name: f"indexer KL perturbed backbone gradient {n}: {s}",
        )

    # Guard against a vacuous pass: the loss really was active and large.
    assert on.last_indexer_distill_loss is not None
    assert on.last_indexer_distill_loss.abs().item() > 0.0
    indexer_grad = sum(p.grad.abs().sum().item() for p in on.indexer.parameters() if p.grad is not None)
    assert indexer_grad > 0.0, "the indexer itself received no gradient"


def test_loss_is_reported_to_the_aux_loss_tracker(monkeypatch):
    """Without this the loss is unobservable once enabled."""
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    recorded = []

    import megatron.core.transformer.moe.moe_utils as moe_utils

    monkeypatch.setattr(
        moe_utils,
        "save_to_aux_losses_tracker",
        lambda name, loss, layer_number, num_layers, **kw: recorded.append(
            (name, float(loss), layer_number, num_layers)
        ),
    )

    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=1e-2).to(_DTYPE)
    attn.train()

    B, S = 1, 8
    hidden = torch.randn(B, S, attn.config.hidden_size, dtype=_DTYPE)
    attn(hidden, torch.arange(S).unsqueeze(0).expand(B, S)).sum().backward()

    assert recorded, "nothing reached the aux-loss tracker"
    name, value, layer_number, num_layers = recorded[0]
    assert name == INDEXER_DISTILL_LOSS_NAME
    assert value == pytest.approx(attn.last_indexer_distill_loss.item(), rel=1e-6)
    assert layer_number == attn.layer_number
    assert num_layers >= attn.config.num_layers


def test_nothing_is_reported_when_the_loss_is_off(monkeypatch):
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    recorded = []

    import megatron.core.transformer.moe.moe_utils as moe_utils

    monkeypatch.setattr(
        moe_utils,
        "save_to_aux_losses_tracker",
        lambda *a, **kw: recorded.append(a),
    )

    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=0.0).to(_DTYPE)
    attn.train()

    B, S = 1, 8
    hidden = torch.randn(B, S, attn.config.hidden_size, dtype=_DTYPE)
    attn(hidden, torch.arange(S).unsqueeze(0).expand(B, S)).sum().backward()

    assert not recorded, "the tracker must stay untouched at coeff 0"


def test_layers_without_an_indexer_still_report_a_zero(monkeypatch):
    """Every V4 layer must report, or the cross-PP reduction diverges.

    ``reduce_aux_losses_tracker_across_ranks`` all-reduces over whatever keys a
    rank holds, so a key present only on the ranks owning a CSA layer would hang
    the collective. Non-CSA layers therefore contribute an explicit zero.
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    recorded = []

    import megatron.core.transformer.moe.moe_utils as moe_utils

    monkeypatch.setattr(
        moe_utils,
        "save_to_aux_losses_tracker",
        lambda name, loss, layer_number, num_layers, **kw: recorded.append((name, float(loss))),
    )

    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=1e-2).to(_DTYPE)
    # An HCA layer owns no indexer, but shares the coefficient.
    attn.compress_ratio = 128
    attn.indexer = None
    attn.last_indexer_distill_loss = None
    attn.train()

    attn._log_indexer_distill_loss(torch.device("cpu"))

    assert recorded == [(INDEXER_DISTILL_LOSS_NAME, 0.0)]


@pytest.mark.parametrize("layer_number", [None, 0])
def test_unnumbered_layers_are_not_reported(monkeypatch, layer_number):
    """``layer_number`` indexes the tracker, so the sentinel must be rejected.

    ``0`` is the default on a standalone attention module and would otherwise
    land on ``values[-1]``, corrupting the last layer's entry.
    """
    recorded = []
    import megatron.core.transformer.moe.moe_utils as moe_utils

    monkeypatch.setattr(moe_utils, "save_to_aux_losses_tracker", lambda *a, **kw: recorded.append(a))

    log_indexer_distill_loss(
        torch.tensor(1.0),
        layer_number=layer_number,
        num_layers=8,
        device=torch.device("cpu"),
    )
    assert not recorded


def test_eval_does_not_report_a_stale_loss(monkeypatch):
    """A value from an earlier training step must not leak into eval logs."""
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    recorded = []

    import megatron.core.transformer.moe.moe_utils as moe_utils

    monkeypatch.setattr(moe_utils, "save_to_aux_losses_tracker", lambda *a, **kw: recorded.append(a))

    attn = _make_csa_attention(coeff=1e-2).to(_DTYPE)
    attn.last_indexer_distill_loss = torch.tensor(7.0)
    attn.eval()

    attn._log_indexer_distill_loss(torch.device("cpu"))

    assert not recorded


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


# ---------------------------------------------------------------------------
# The target is a conditional of the layer's joint softmax
# ---------------------------------------------------------------------------
#
# A CSA layer takes one softmax over [sliding window, sparse compressed, sink].
# The distribution it places on the compressed entries is therefore a
# conditional of that joint softmax, and the share of a head's mass that gets
# there is below one and differs per head. These tests pin the target to the
# joint softmax the reference op (``eager_v4_csa_attention``) actually computes,
# rather than to a softmax taken over the compressed entries on their own.


def _joint_softmax_target(*, query, k_local, pool, topk_idxs, sink, swa_window, scale):
    """Reference target: one joint softmax, keep the sparse block, sum heads.

    Deliberately written the long way -- materialise the whole
    ``[B, H, S, S + K + 1]`` logit tensor and softmax it once -- so it mirrors
    ``eager_v4_csa_attention`` step for step and shares no code with the
    implementation under test.
    """
    B, H, S, _ = query.shape
    K = topk_idxs.shape[-1]
    window = swa_window if swa_window > 0 else S

    local_logits = torch.matmul(query, k_local.transpose(-1, -2)).float() * scale
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    dist = i - j
    local_logits = local_logits.masked_fill(~((dist >= 0) & (dist < window)), float("-inf"))

    gathered = pool[torch.arange(B).view(B, 1, 1), topk_idxs.clamp_min(0)]
    sparse_logits = torch.einsum("bhsd,bskd->bhsk", query, gathered).float() * scale
    sparse_logits = sparse_logits.masked_fill(~(topk_idxs >= 0).unsqueeze(1), float("-inf"))

    parts = [local_logits, sparse_logits]
    if sink is not None:
        parts.append(sink.float().view(1, H, 1, 1).expand(B, H, S, 1))
    probs = torch.softmax(torch.cat(parts, dim=-1), dim=-1)

    target = probs[..., S : S + K].sum(dim=1)  # head-sum -> [B, S, K]
    return target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def _distill_target(*, query, pool, topk_idxs, scale, nc_lse):
    """``_target_distribution`` with the masks the loss entry point derives."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        _target_distribution,
    )

    valid = topk_idxs >= 0
    return _target_distribution(
        query=query,
        pool=pool,
        topk_idxs=topk_idxs,
        valid=valid,
        row_valid=valid.any(dim=-1),
        softmax_scale=scale,
        normalize=True,
        nc_lse=nc_lse,
    )


def _target_case(*, S=12, K=4, H=4, P=6, window=5, with_sink=True, seed=0):
    torch.manual_seed(seed)
    B, Dh = 1, 8
    query = torch.randn(B, H, S, Dh, dtype=_DTYPE)
    k_local = torch.randn(B, H, S, Dh, dtype=_DTYPE)
    pool = torch.randn(B, P, Dh, dtype=_DTYPE)
    topk_idxs = torch.randint(0, P, (B, S, K), dtype=torch.long)
    sink = torch.randn(H, dtype=_DTYPE) if with_sink else None
    return {
        "query": query,
        "k_local": k_local,
        "pool": pool,
        "topk_idxs": topk_idxs,
        "sink": sink,
        "swa_window": window,
        "scale": 1.0 / math.sqrt(Dh),
    }


@pytest.mark.parametrize("with_sink", [True, False])
@pytest.mark.parametrize("window", [3, 5, 12])
def test_target_matches_the_joint_softmax(with_sink, window):
    """The target must equal the joint softmax's conditional on the sparse block."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _target_case(window=window, with_sink=with_sink)
    expected = _joint_softmax_target(**case)

    nc_lse = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"],
        sink=case["sink"],
        swa_window=case["swa_window"],
        softmax_scale=case["scale"],
    )
    got = _distill_target(
        query=case["query"],
        pool=case["pool"],
        topk_idxs=case["topk_idxs"],
        scale=case["scale"],
        nc_lse=nc_lse,
    )

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_per_branch_softmax_disagrees_with_the_joint_one():
    """Guard against the fix being a no-op.

    Renormalising each head over the compressed entries alone gives every head
    the same weight in the head sum; the joint softmax weights them by how much
    of their attention actually reaches those entries. The two must differ, or
    the ``nc_lse`` argument is not doing anything.
    """
    case = _target_case()
    joint = _joint_softmax_target(**case)
    per_branch = _distill_target(
        query=case["query"],
        pool=case["pool"],
        topk_idxs=case["topk_idxs"],
        scale=case["scale"],
        nc_lse=None,
    )

    assert not torch.allclose(per_branch, joint, rtol=1e-2, atol=1e-3)


def test_noncompressed_lse_matches_an_explicit_mask():
    """``noncompressed_lse`` is chunked; check it against the unchunked form."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _target_case(S=20, window=7)
    query, k_local, sink = case["query"], case["k_local"], case["sink"]
    scale, window = case["scale"], case["swa_window"]
    S = query.shape[2]

    logits = torch.matmul(query, k_local.transpose(-1, -2)).float() * scale
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    dist = i - j
    logits = logits.masked_fill(~((dist >= 0) & (dist < window)), float("-inf"))
    expected = torch.logsumexp(logits, dim=-1)
    expected = torch.logaddexp(expected, sink.float().view(1, -1, 1))

    for chunk in ("3", "7", "64"):
        os.environ["PRIMUS_V4_DISTILL_WINDOW_CHUNK"] = chunk
        try:
            got = noncompressed_lse(
                query=query,
                k_local=k_local,
                sink=sink,
                swa_window=window,
                softmax_scale=scale,
            )
        finally:
            del os.environ["PRIMUS_V4_DISTILL_WINDOW_CHUNK"]
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_joint_target_survives_fully_masked_rows():
    """Rows with no legal compressed entry stay zero instead of going NaN.

    With the joint denominator such a row's compressed log-sum-exp is ``-inf``,
    and only the window and sink keep the denominator finite -- the arithmetic
    that has to not divide ``-inf`` by ``-inf``.
    """
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _target_case(S=8, K=3)
    topk_idxs = case["topk_idxs"].clone()
    topk_idxs[:, :2] = -1  # first two queries select nothing

    nc_lse = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"],
        sink=case["sink"],
        swa_window=case["swa_window"],
        softmax_scale=case["scale"],
    )
    got = _distill_target(
        query=case["query"],
        pool=case["pool"],
        topk_idxs=topk_idxs,
        scale=case["scale"],
        nc_lse=nc_lse,
    )

    assert torch.isfinite(got).all()
    assert (got[:, :2] == 0).all()
    torch.testing.assert_close(got[:, 2:].sum(dim=-1), torch.ones_like(got[:, 2:, 0]), rtol=1e-5, atol=1e-6)


def test_loss_entry_point_uses_the_joint_denominator(monkeypatch):
    """Passing ``k_local`` changes the loss; ``PRIMUS_V4_DISTILL_NONCOMP_LSE=0`` reverts it."""
    case = _target_case()
    B, _, S, _ = case["query"].shape
    K = case["topk_idxs"].shape[-1]
    torch.manual_seed(1)
    index_scores = torch.randn(B, S, K, dtype=_DTYPE)

    common = dict(
        index_topk_scores=index_scores,
        topk_idxs=case["topk_idxs"],
        query=case["query"],
        pool=case["pool"],
        softmax_scale=case["scale"],
        loss_coeff=1.0,
    )
    joint_kwargs = dict(k_local=case["k_local"], sink=case["sink"], swa_window=case["swa_window"])

    monkeypatch.delenv("PRIMUS_V4_DISTILL_NONCOMP_LSE", raising=False)
    per_branch = compute_indexer_distill_loss(**common)
    joint = compute_indexer_distill_loss(**common, **joint_kwargs)
    assert joint.item() != pytest.approx(per_branch.item(), rel=1e-3)

    monkeypatch.setenv("PRIMUS_V4_DISTILL_NONCOMP_LSE", "0")
    reverted = compute_indexer_distill_loss(**common, **joint_kwargs)
    assert reverted.item() == pytest.approx(per_branch.item(), rel=1e-6)


# ---------------------------------------------------------------------------
# The fused kernel takes the same joint denominator
# ---------------------------------------------------------------------------
#
# The tests above run on CPU tensors, which `can_use_triton_target` declines, so
# they only exercise the eager body. These pin the kernel to it.


def _gpu_target_case(*, S=64, K=16, H=16, P=32, Dh=64, window=8, with_sink=True, seed=0):
    """Shapes both fused kernels accept: ``K`` a power of two >= 16, ``H``
    divisible by a legal head block, and ``Dh`` divisible by the target
    kernel's 32-wide and the window kernel's 64-wide feature tiles."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    B = 1
    query = torch.randn(B, H, S, Dh, device=dev, dtype=torch.bfloat16)
    k_local = torch.randn(B, H, S, Dh, device=dev, dtype=torch.bfloat16)
    pool = torch.randn(B, P, Dh, device=dev, dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, P, (B, S, K), device=dev, dtype=torch.long)
    sink = torch.randn(H, device=dev, dtype=torch.float32) if with_sink else None
    return {
        "query": query,
        "k_local": k_local,
        "pool": pool,
        "topk_idxs": topk_idxs,
        "sink": sink,
        "swa_window": window,
        "scale": 1.0 / math.sqrt(Dh),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
@pytest.mark.parametrize("with_sink", [True, False])
def test_fused_target_matches_eager_with_joint_denominator(monkeypatch, with_sink):
    """Kernel and eager body must agree once the joint denominator is in play."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _gpu_target_case(with_sink=with_sink)
    nc_lse = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"],
        sink=case["sink"],
        swa_window=case["swa_window"],
        softmax_scale=case["scale"],
    )
    shared = dict(
        query=case["query"],
        pool=case["pool"],
        topk_idxs=case["topk_idxs"],
        scale=case["scale"],
        nc_lse=nc_lse,
    )

    monkeypatch.setenv("PRIMUS_V4_DISTILL_TARGET_TRITON", "0")
    eager = _distill_target(**shared)
    monkeypatch.setenv("PRIMUS_V4_DISTILL_TARGET_TRITON", "1")
    fused = _distill_target(**shared)

    assert torch.isfinite(fused).all()
    # bf16 inputs, so the tolerance is set by the inputs rather than the kernel;
    # tl.dot accumulates in fp32 and is if anything the more accurate of the two.
    torch.testing.assert_close(fused, eager, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
def test_fused_target_survives_fully_masked_rows():
    """``exp(nclse - m)`` must not turn an empty row into NaN in the kernel."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _gpu_target_case()
    topk_idxs = case["topk_idxs"].clone()
    topk_idxs[:, :4] = -1

    nc_lse = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"],
        sink=case["sink"],
        swa_window=case["swa_window"],
        softmax_scale=case["scale"],
    )
    fused = _distill_target(
        query=case["query"],
        pool=case["pool"],
        topk_idxs=topk_idxs,
        scale=case["scale"],
        nc_lse=nc_lse,
    )

    assert torch.isfinite(fused).all()
    assert (fused[:, :4] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
@pytest.mark.parametrize("with_sink", [True, False])
@pytest.mark.parametrize("window", [8, 16, 64])
def test_fused_window_lse_matches_eager(monkeypatch, with_sink, window):
    """The window log-sum-exp kernel must agree with the chunked eager body.

    This is the piece the whole fix costs, so it gets its own comparison rather
    than only being checked through the target it feeds.
    """
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_distill_window_lse import (
        can_use_triton_window_lse,
    )

    case = _gpu_target_case(window=window, with_sink=with_sink)
    args = dict(
        query=case["query"],
        k_local=case["k_local"],
        sink=case["sink"],
        swa_window=window,
        softmax_scale=case["scale"],
    )
    assert can_use_triton_window_lse(
        query=case["query"], k_local=case["k_local"], swa_window=window
    ), "the kernel should cover this shape, otherwise the test proves nothing"

    monkeypatch.setenv("PRIMUS_V4_DISTILL_WINDOW_TRITON", "0")
    eager = noncompressed_lse(**args)
    monkeypatch.setenv("PRIMUS_V4_DISTILL_WINDOW_TRITON", "1")
    fused = noncompressed_lse(**args)

    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused, eager, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
def test_fused_window_lse_declines_full_causal():
    """A non-positive window means full causal, which the band cannot cover."""
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_distill_window_lse import (
        can_use_triton_window_lse,
    )

    case = _gpu_target_case()
    assert not can_use_triton_window_lse(query=case["query"], k_local=case["k_local"], swa_window=0)


def test_layer_number_reaches_the_attention_module():
    """The tracker is indexed by ``layer_number``, and 0 is its "unnumbered"
    sentinel.

    ``DeepseekV4HybridLayer`` knows its 1-based layer number but used not to
    pass it to ``build_module`` for the attention, so every attention module
    came up as layer 0 and ``log_indexer_distill_loss`` dropped the value on the
    floor -- the loss was trained but never reported. Nothing else in the
    forward depends on it, so only a test keeps it wired.
    """
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
        DeepseekV4HybridLayer,
    )

    config = _make_v4_config(coeff=1e-2)
    rope = _make_v4_rope(config)

    for layer_idx in (0, 3, 7):
        layer = DeepseekV4HybridLayer(
            config=config,
            layer_idx=layer_idx,
            compress_ratio=4,
            rope=rope,
        )
        assert layer.layer_number == layer_idx + 1
        assert layer.self_attention.layer_number == layer.layer_number, (
            "attention did not receive the layer number; the indexer loss will "
            "be silently dropped by the tracker"
        )


def test_disabled_coeff_never_reaches_the_new_code(monkeypatch):
    """With the loss off, none of this machinery may run.

    ``v4_indexer_distill_loss_coeff`` defaults to 0, which is what every model
    other than a deliberately-configured V4-Flash run uses. Poison the two
    entry points the fix added so that touching them fails loudly, then drive a
    full CSA forward.
    """
    monkeypatch.delenv("PRIMUS_V4_INDEXER_TRAINABLE", raising=False)
    import primus.backends.megatron.core.transformer.indexer_distill_loss as idl

    def _poisoned(*_a, **_kw):
        raise AssertionError("the distillation path ran with the loss disabled")

    monkeypatch.setattr(idl, "noncompressed_lse", _poisoned)
    monkeypatch.setattr(idl, "compute_indexer_distill_loss", _poisoned)

    torch.manual_seed(0)
    attn = _make_csa_attention(coeff=0.0).to(_DTYPE)
    attn.train()

    B, S = 1, 8
    hidden = torch.randn(B, S, attn.config.hidden_size, dtype=_DTYPE)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)
    attn(hidden, position_ids)  # must not raise

    assert attn.last_indexer_distill_loss is None


def test_noncompressed_lse_full_causal_window():
    """``swa_window <= 0`` means full causal, matching ``_local_mask``."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _target_case(S=10, window=0, with_sink=True)
    query, k_local, sink, scale = (
        case["query"],
        case["k_local"],
        case["sink"],
        case["scale"],
    )
    S = query.shape[2]

    logits = torch.matmul(query, k_local.transpose(-1, -2)).float() * scale
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    logits = logits.masked_fill(i < j, float("-inf"))  # plain causal
    expected = torch.logaddexp(torch.logsumexp(logits, dim=-1), sink.float().view(1, -1, 1))

    got = noncompressed_lse(query=query, k_local=k_local, sink=sink, swa_window=0, softmax_scale=scale)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_window_wider_than_the_sequence_is_still_causal():
    """A window past the end of the sequence must not reach beyond the diagonal."""
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _target_case(S=6, window=100, with_sink=False)
    wide = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"],
        sink=None,
        swa_window=100,
        softmax_scale=case["scale"],
    )
    causal = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"],
        sink=None,
        swa_window=0,
        softmax_scale=case["scale"],
    )
    torch.testing.assert_close(wide, causal, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
def test_fused_kernels_reject_mismatched_inputs():
    """Bad shapes must raise rather than read out of bounds."""
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_distill_target import (
        target_distribution_triton,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_distill_window_lse import (
        window_lse_triton,
    )

    case = _gpu_target_case()
    B, H, S, _ = case["query"].shape

    with pytest.raises(ValueError, match="noncompressed_lse must be"):
        target_distribution_triton(
            query=case["query"],
            pool=case["pool"],
            topk_idxs=case["topk_idxs"],
            softmax_scale=case["scale"],
            noncompressed_lse=torch.zeros(B, H, S + 1, device="cuda"),
        )

    with pytest.raises(ValueError, match="sink must hold"):
        window_lse_triton(
            query=case["query"],
            k_local=case["k_local"],
            sink=torch.zeros(H + 1, device="cuda"),
            swa_window=case["swa_window"],
            softmax_scale=case["scale"],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
def test_fused_window_lse_declines_shapes_it_cannot_tile():
    """A head_dim the feature tile does not divide falls back to eager."""
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_distill_window_lse import (
        can_use_triton_window_lse,
    )

    odd = _gpu_target_case(Dh=48)  # 48 % 64 != 0
    assert not can_use_triton_window_lse(
        query=odd["query"], k_local=odd["k_local"], swa_window=odd["swa_window"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
def test_fused_target_handles_a_dominant_local_branch():
    """A head that barely looks at the compressed entries must not produce NaN.

    Scaling the window keys up drives ``nclse`` far above the compressed
    logits, which is exactly the case the kernel's clamp exists for: the head's
    compressed share underflows to zero, and zero is the right answer.
    """
    from primus.backends.megatron.core.transformer.indexer_distill_loss import (
        noncompressed_lse,
    )

    case = _gpu_target_case()
    nc_lse = noncompressed_lse(
        query=case["query"],
        k_local=case["k_local"] * 100.0,
        sink=case["sink"],
        swa_window=case["swa_window"],
        softmax_scale=case["scale"],
    )
    fused = _distill_target(
        query=case["query"],
        pool=case["pool"],
        topk_idxs=case["topk_idxs"],
        scale=case["scale"],
        nc_lse=nc_lse,
    )

    assert torch.isfinite(fused).all()
