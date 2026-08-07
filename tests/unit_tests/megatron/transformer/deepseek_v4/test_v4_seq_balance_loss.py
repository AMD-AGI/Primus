###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the MoE sequence-wise load-balancing loss.

The V4 yaml declares ``moe_router_load_balancing_type: seq_aux_loss`` with
``moe_aux_loss_coeff: 0.001``, matching the paper's "slight sequence-wise
balance loss". These tests pin the formula (whose normalisation makes a
perfectly balanced sequence score exactly ``alpha``), the fact that it is
per-sequence rather than per-batch, and the sharding contract.

The loss function itself is plain torch and runs on CPU; the router that drives
it is GPU-only (``_compute_route`` asserts CUDA), so the wiring assertions are
made against the router's constructor and the config plumbing rather than a
forward pass.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.transformer.moe.v4_seq_balance_loss import (  # noqa: E402
    SEQ_BALANCE_LOSS_NAME,
    log_seq_balance_loss,
    normalise_affinities,
    sequence_balance_loss,
)


def _balanced(batch: int, seq: int, experts: int, topk: int):
    """Perfectly balanced routing with uniform affinities."""
    scores = torch.full((batch, seq, experts), 1.0 / experts, dtype=torch.float32)
    routing_map = torch.zeros(batch, seq, experts, dtype=torch.bool)
    for b in range(batch):
        for t in range(seq):
            picks = [(t * topk + k) % experts for k in range(topk)]
            routing_map[b, t, picks] = True
    return scores, routing_map


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def test_balanced_sequence_scores_exactly_the_coefficient():
    """``f`` sums to E and ``P`` to 1, so balanced routing gives ``alpha``.

    This is what makes the logged number readable: anything above the
    coefficient is measurable imbalance.
    """
    scores, routing_map = _balanced(batch=2, seq=8, experts=4, topk=2)
    loss = sequence_balance_loss(scores=scores, routing_map=routing_map, topk=2, coeff=1e-3)
    assert loss.item() == pytest.approx(1e-3, rel=1e-5)


def test_collapsed_sequence_is_penalised():
    """Routing every token to the same experts must cost more than balanced."""
    batch, seq, experts, topk = 1, 8, 4, 2
    balanced_scores, balanced_map = _balanced(batch, seq, experts, topk)

    collapsed_map = torch.zeros(batch, seq, experts, dtype=torch.bool)
    collapsed_map[..., :topk] = True
    collapsed_scores = torch.zeros(batch, seq, experts, dtype=torch.float32)
    collapsed_scores[..., :topk] = 0.5

    balanced = sequence_balance_loss(scores=balanced_scores, routing_map=balanced_map, topk=topk, coeff=1.0)
    collapsed = sequence_balance_loss(
        scores=collapsed_scores, routing_map=collapsed_map, topk=topk, coeff=1.0
    )
    assert collapsed.item() > balanced.item()
    # Fully collapsed onto topk of E experts costs E / topk.
    assert collapsed.item() == pytest.approx(experts / topk, rel=1e-5)


def test_loss_is_linear_in_the_coefficient():
    scores, routing_map = _balanced(1, 8, 4, 2)
    scores = scores + torch.rand_like(scores) * 0.1
    scores = normalise_affinities(scores)
    a = sequence_balance_loss(scores=scores, routing_map=routing_map, topk=2, coeff=1.0)
    b = sequence_balance_loss(scores=scores, routing_map=routing_map, topk=2, coeff=3.0)
    assert b.item() == pytest.approx(3.0 * a.item(), rel=1e-6)


def test_loss_is_per_sequence_not_per_batch():
    """A batch of two collapsed-but-opposite sequences must still be penalised.

    Pooled over the batch the two sequences look balanced; per-sequence they are
    each fully collapsed. Catching this is the entire reason the loss exists.
    """
    seq, experts, topk = 8, 4, 2
    routing_map = torch.zeros(2, seq, experts, dtype=torch.bool)
    routing_map[0, :, :topk] = True
    routing_map[1, :, topk:] = True

    scores = torch.zeros(2, seq, experts, dtype=torch.float32)
    scores[0, :, :topk] = 0.5
    scores[1, :, topk:] = 0.5

    loss = sequence_balance_loss(scores=scores, routing_map=routing_map, topk=topk, coeff=1.0)
    # Each sequence is collapsed onto topk of E, so the mean is E / topk.
    assert loss.item() == pytest.approx(experts / topk, rel=1e-5)

    # Whereas the batch-pooled view of the same routing looks perfectly balanced.
    pooled_map = routing_map.reshape(1, 2 * seq, experts)
    pooled_scores = scores.reshape(1, 2 * seq, experts)
    pooled = sequence_balance_loss(scores=pooled_scores, routing_map=pooled_map, topk=topk, coeff=1.0)
    assert pooled.item() < loss.item()


def test_loss_is_differentiable_through_the_affinities():
    scores, routing_map = _balanced(1, 8, 4, 2)
    scores = scores.clone().requires_grad_(True)
    sequence_balance_loss(scores=scores, routing_map=routing_map, topk=2, coeff=1.0).backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0.0


def test_rejects_flat_inputs():
    """The loss is per-sequence, so a collapsed [N, E] view is a bug."""
    with pytest.raises(ValueError):
        sequence_balance_loss(
            scores=torch.rand(8, 4),
            routing_map=torch.zeros(8, 4, dtype=torch.bool),
            topk=2,
            coeff=1.0,
        )


def test_normalise_affinities_produces_a_distribution():
    raw = torch.rand(2, 5, 7) + 0.1
    normed = normalise_affinities(raw)
    torch.testing.assert_close(normed.sum(dim=-1), torch.ones(2, 5), rtol=1e-5, atol=1e-6)


def test_normalise_affinities_is_safe_on_all_zero_rows():
    normed = normalise_affinities(torch.zeros(1, 2, 4))
    assert torch.isfinite(normed).all()


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def test_expert_counts_are_reduced_across_the_sequence_shards():
    """With the sequence split, ``f`` must be global while ``P`` stays local."""
    batch, seq, experts, topk = 1, 8, 4, 2
    scores, routing_map = _balanced(batch, seq, experts, topk)

    class _Group:
        pass

    group = _Group()
    calls = []
    real_world_size = torch.distributed.get_world_size
    real_all_reduce = torch.distributed.all_reduce
    torch.distributed.get_world_size = lambda g=None: 2 if g is group else 1
    torch.distributed.all_reduce = lambda t, group=None: calls.append((tuple(t.shape), group))
    try:
        sharded = sequence_balance_loss(
            scores=scores, routing_map=routing_map, topk=topk, coeff=1.0, reduce_group=group
        )
    finally:
        torch.distributed.get_world_size = real_world_size
        torch.distributed.all_reduce = real_all_reduce

    assert calls == [((batch, experts), group)], "must reduce the [B, E] expert counts"
    # The fake all-reduce is a no-op, so with T doubled and counts unchanged the
    # local share is a quarter of the unsharded value (T appears squared).
    local = sequence_balance_loss(scores=scores, routing_map=routing_map, topk=topk, coeff=1.0)
    assert sharded.item() == pytest.approx(local.item() / 4.0, rel=1e-5)


def test_no_group_means_no_collective():
    batch, seq, experts, topk = 1, 8, 4, 2
    scores, routing_map = _balanced(batch, seq, experts, topk)
    calls = []
    real_all_reduce = torch.distributed.all_reduce
    torch.distributed.all_reduce = lambda t, group=None: calls.append(t)
    try:
        sequence_balance_loss(scores=scores, routing_map=routing_map, topk=topk, coeff=1.0)
    finally:
        torch.distributed.all_reduce = real_all_reduce
    assert not calls


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_loss_is_reported_under_the_expected_name(monkeypatch):
    pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    import megatron.core.transformer.moe.moe_utils as moe_utils

    recorded = []
    monkeypatch.setattr(
        moe_utils,
        "save_to_aux_losses_tracker",
        lambda name, loss, layer_number, num_layers, **kw: recorded.append(
            (name, float(loss), layer_number, num_layers, kw.get("reduce_group"))
        ),
    )

    log_seq_balance_loss(torch.tensor(0.5), layer_number=3, num_layers=44, reduce_group=None)

    assert recorded == [(SEQ_BALANCE_LOSS_NAME, 0.5, 3, 44, None)]


@pytest.mark.parametrize("layer_number", [None, 0])
def test_unnumbered_layers_are_not_reported(monkeypatch, layer_number):
    pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    import megatron.core.transformer.moe.moe_utils as moe_utils

    recorded = []
    monkeypatch.setattr(moe_utils, "save_to_aux_losses_tracker", lambda *a, **kw: recorded.append(a))
    log_seq_balance_loss(torch.tensor(1.0), layer_number=layer_number, num_layers=44)
    assert not recorded


# ---------------------------------------------------------------------------
# Router / config wiring
# ---------------------------------------------------------------------------


def test_router_defaults_to_the_loss_being_off():
    from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
        DeepseekV4LearnedRouter,
    )

    router = DeepseekV4LearnedRouter(hidden_size=16, num_experts=8, topk=2)
    assert router.seq_balance_loss_coeff == 0.0
    assert router.seq_balance_loss_enabled is False


def test_router_enables_the_loss_from_its_coefficient():
    from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
        DeepseekV4LearnedRouter,
    )

    router = DeepseekV4LearnedRouter(hidden_size=16, num_experts=8, topk=2, seq_balance_loss_coeff=1e-3)
    assert router.seq_balance_loss_enabled is True


def test_v4_yaml_declares_a_balancing_type_the_router_implements():
    """The yaml said ``seq_aux_loss`` long before anything read it.

    Pin the two together so the declaration cannot drift back into being
    decorative, and so a switch to a balancing type this router does not
    implement is caught here.
    """
    from pathlib import Path

    from primus.core.config.yaml_loader import parse_yaml

    yaml_dir = Path(__file__).resolve().parents[5] / "primus" / "configs" / "models" / "megatron"
    parsed = parse_yaml(str(yaml_dir / "deepseek_v4_base.yaml"))

    assert parsed["moe_router_load_balancing_type"] == "seq_aux_loss"
    assert float(parsed["moe_aux_loss_coeff"]) > 0.0
