###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for V4 routers (G4, plan-2 §04).

Pins both routers to the HF reference at
``DeepSeek-V4-Flash/inference/model.py:Gate.forward``. Two routers
share the same scoring path; they only differ in *selection* (top-K
argmax vs ``tid2eid`` lookup).

Pass criteria (G4):
* Identical weights -> identical sparse ``(probs, routing_map)`` to the
  HF reference (max-abs <= 1e-6 fp32).
* Routing weight gradient flows to ``weight`` (the learned gate).
* For the hash router, ``tid2eid`` is a parameter with
  ``requires_grad=False`` (frozen) so checkpoint round-trips preserve
  it without polluting the optimizer state.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.moe.v4_hash_router import (
    DeepseekV4HashRouter,
    HashRouter,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
    DeepseekV4LearnedRouter,
    V4TopKRouter,
    v4_score_fn,
)

# ---------------------------------------------------------------------------
# HF reference inline
# ---------------------------------------------------------------------------


def _hf_gate_forward(
    *,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    score_function: str,
    topk: int,
    route_scale: float,
    tid2eid: Optional[torch.Tensor],
    token_ids: Optional[torch.Tensor],
):
    """Mirrors HF reference ``Gate.forward`` exactly.

    Returns dense ``(weights, indices)`` with shapes ``[N, K]``.
    """
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    scores = F.linear(flat_hidden.float(), weight.float())
    if score_function == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_function == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    original_scores = scores
    if bias is not None:
        scores = scores + bias.float()
    if tid2eid is not None:
        assert token_ids is not None
        flat_ids = token_ids.reshape(-1).long()
        indices = tid2eid[flat_ids].long()
    else:
        indices = scores.topk(topk, dim=-1).indices
    weights = original_scores.gather(1, indices)
    if score_function != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * route_scale
    return weights, indices


def _sparse_from_dense(
    *,
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
):
    """Pack dense ``(weights, indices)`` into the sparse ``(probs,
    routing_map)`` format our routers return."""
    N = weights.shape[0]
    probs = torch.zeros(N, num_experts, dtype=weights.dtype, device=weights.device)
    probs.scatter_(1, indices, weights)
    routing_map = torch.zeros(N, num_experts, dtype=torch.bool, device=weights.device)
    routing_map.scatter_(1, indices, True)
    return probs, routing_map


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_v4_score_fn_matches_inline_reference(score_function: str) -> None:
    torch.manual_seed(123)
    logits = torch.randn(8, 16, dtype=torch.float32)

    out = v4_score_fn(logits, score_function=score_function)
    if score_function == "softmax":
        ref = F.softmax(logits, dim=-1)
    elif score_function == "sigmoid":
        ref = torch.sigmoid(logits)
    else:
        ref = F.softplus(logits).sqrt()
    assert (out - ref).abs().max().item() <= 1.0e-6


# ---------------------------------------------------------------------------
# Learned router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score_function", ["sqrtsoftplus", "sigmoid", "softmax"])
@pytest.mark.parametrize("enable_expert_bias", [False, True])
def test_learned_router_matches_hf_reference(score_function: str, enable_expert_bias: bool) -> None:
    """G4 learned-router gate: matches HF reference exactly (fp32)."""
    torch.manual_seed(2025)
    H, E, K = 12, 8, 2
    route_scale = 1.5

    router = DeepseekV4LearnedRouter(
        hidden_size=H,
        num_experts=E,
        topk=K,
        score_function=score_function,
        enable_expert_bias=enable_expert_bias,
        topk_scaling_factor=route_scale,
    )
    if enable_expert_bias:
        with torch.no_grad():
            router.expert_bias.copy_(torch.randn(E, dtype=torch.float32))

    hidden = torch.randn(2, 5, H, dtype=torch.float32)

    probs, routing_map = router(hidden)

    weights_hf, indices_hf = _hf_gate_forward(
        hidden=hidden,
        weight=router.weight,
        bias=router.expert_bias,
        score_function=score_function,
        topk=K,
        route_scale=route_scale,
        tid2eid=None,
        token_ids=None,
    )
    probs_hf, routing_map_hf = _sparse_from_dense(weights=weights_hf, indices=indices_hf, num_experts=E)

    assert probs.shape == probs_hf.shape
    assert routing_map.shape == routing_map_hf.shape
    assert torch.equal(routing_map, routing_map_hf), "routing_map differs"
    max_abs = (probs - probs_hf).abs().max().item()
    assert max_abs <= 1.0e-6, f"probs max-abs vs HF reference = {max_abs}"


def test_learned_router_back_compat_alias() -> None:
    """``V4TopKRouter`` is an alias for the renamed router."""
    assert V4TopKRouter is DeepseekV4LearnedRouter


def test_learned_router_grad_flows_to_gate_weight() -> None:
    """Probs gradient propagates back into ``weight`` (the gate)."""
    torch.manual_seed(11)
    H, E, K = 6, 4, 2
    router = DeepseekV4LearnedRouter(hidden_size=H, num_experts=E, topk=K, score_function="sqrtsoftplus")
    hidden = torch.randn(3, 4, H, dtype=torch.float32, requires_grad=False)
    probs, _ = router(hidden)
    loss = probs.sum()
    loss.backward()
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
    assert router.weight.grad.abs().sum().item() > 0.0


def test_learned_router_expert_bias_does_not_contribute_to_probs_grad() -> None:
    """``expert_bias`` is selection-only: probs use un-biased scores.

    With ``enable_expert_bias=True`` the bias enters the *selection*
    path (top-K) but not the gathered weights. The gradient on the bias
    can therefore be exactly zero when the chosen indices stay the same
    if you only differentiate the routing weights at the selected
    experts. We assert ``expert_bias.grad is None`` (no graph) **after**
    a forward + backward, since the bias never enters the autograd
    chain in our implementation.
    """
    torch.manual_seed(42)
    H, E, K = 6, 4, 2
    router = DeepseekV4LearnedRouter(
        hidden_size=H,
        num_experts=E,
        topk=K,
        score_function="sqrtsoftplus",
        enable_expert_bias=True,
    )
    hidden = torch.randn(3, 4, H, dtype=torch.float32)
    probs, _ = router(hidden)
    probs.sum().backward()
    # Bias is detached from the probs graph — grad stays None (or zero).
    assert (router.expert_bias.grad is None) or (router.expert_bias.grad.abs().max().item() == 0.0)


def test_learned_router_softmax_skips_renormalization() -> None:
    """Softmax probs already sum to 1, so the post-topK renorm is skipped.

    We assert by setting ``route_scale != 1`` and confirming the gathered
    weights match HF (i.e. they are *not* re-normalized to 1 before the
    scale).
    """
    torch.manual_seed(0)
    H, E, K = 6, 4, 2
    router = DeepseekV4LearnedRouter(
        hidden_size=H,
        num_experts=E,
        topk=K,
        score_function="softmax",
        topk_scaling_factor=2.0,
    )
    hidden = torch.randn(2, 3, H, dtype=torch.float32)
    probs, routing_map = router(hidden)

    weights_hf, indices_hf = _hf_gate_forward(
        hidden=hidden,
        weight=router.weight,
        bias=None,
        score_function="softmax",
        topk=K,
        route_scale=2.0,
        tid2eid=None,
        token_ids=None,
    )
    probs_hf, routing_map_hf = _sparse_from_dense(weights=weights_hf, indices=indices_hf, num_experts=E)
    assert torch.equal(routing_map, routing_map_hf)
    assert (probs - probs_hf).abs().max().item() <= 1.0e-6


# ---------------------------------------------------------------------------
# Hash router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score_function", ["sqrtsoftplus", "sigmoid", "softmax"])
def test_hash_router_matches_hf_reference(score_function: str) -> None:
    """G4 hash-router: matches HF reference exactly (fp32)."""
    torch.manual_seed(7)
    H, E, K, V = 12, 8, 2, 32

    router = DeepseekV4HashRouter(
        hidden_size=H,
        num_experts=E,
        topk=K,
        vocab_size=V,
        seed=17,
        score_function=score_function,
        topk_scaling_factor=1.0,
    )
    hidden = torch.randn(2, 5, H, dtype=torch.float32)
    token_ids = torch.randint(0, V, (2, 5), dtype=torch.long)

    probs, routing_map = router(hidden, token_ids)

    weights_hf, indices_hf = _hf_gate_forward(
        hidden=hidden,
        weight=router.weight,
        bias=None,
        score_function=score_function,
        topk=K,
        route_scale=1.0,
        tid2eid=router.tid2eid,
        token_ids=token_ids,
    )
    probs_hf, routing_map_hf = _sparse_from_dense(weights=weights_hf, indices=indices_hf, num_experts=E)
    assert torch.equal(routing_map, routing_map_hf), "routing_map differs"
    max_abs = (probs - probs_hf).abs().max().item()
    assert max_abs <= 1.0e-6, f"probs max-abs vs HF reference = {max_abs}"


def test_hash_router_back_compat_alias() -> None:
    """``HashRouter`` is an alias for the renamed router."""
    assert HashRouter is DeepseekV4HashRouter


def test_hash_router_tid2eid_is_frozen_parameter() -> None:
    """``tid2eid`` is a Parameter with requires_grad=False.

    Matches the HF reference layout (released checkpoint stores it as
    a parameter) so a state-dict round-trip preserves it without
    pulling it into the optimizer state.
    """
    router = DeepseekV4HashRouter(
        hidden_size=8,
        num_experts=4,
        topk=2,
        vocab_size=16,
        seed=0,
    )
    assert "tid2eid" in dict(router.named_parameters())
    assert router.tid2eid.requires_grad is False
    assert router.tid2eid.dtype == torch.int32
    assert tuple(router.tid2eid.shape) == (16, 2)


def test_hash_router_state_dict_keys() -> None:
    """State-dict exposes ``weight`` and ``tid2eid`` (matches HF gate keys)."""
    router = DeepseekV4HashRouter(hidden_size=8, num_experts=4, topk=2, vocab_size=16, seed=0)
    keys = set(router.state_dict().keys())
    assert "weight" in keys
    assert "tid2eid" in keys


def test_hash_router_grad_flows_to_gate_weight() -> None:
    """Even with static expert ids, the routing weights' gradient flows
    back into the learned gate ``weight``."""
    torch.manual_seed(13)
    H, E, K, V = 6, 4, 2, 16
    router = DeepseekV4HashRouter(
        hidden_size=H,
        num_experts=E,
        topk=K,
        vocab_size=V,
        seed=0,
        score_function="sqrtsoftplus",
    )
    hidden = torch.randn(3, 4, H, dtype=torch.float32)
    token_ids = torch.randint(0, V, (3, 4), dtype=torch.long)
    probs, _ = router(hidden, token_ids)
    probs.sum().backward()
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
    assert router.weight.grad.abs().sum().item() > 0.0
    # tid2eid should never accumulate gradient.
    assert router.tid2eid.grad is None


def test_hash_router_deterministic_table_across_seeds() -> None:
    """Same seed -> identical table; different seeds -> different table."""
    a = DeepseekV4HashRouter(hidden_size=4, num_experts=8, topk=2, vocab_size=16, seed=42)
    b = DeepseekV4HashRouter(hidden_size=4, num_experts=8, topk=2, vocab_size=16, seed=42)
    c = DeepseekV4HashRouter(hidden_size=4, num_experts=8, topk=2, vocab_size=16, seed=43)
    assert torch.equal(a.tid2eid, b.tid2eid)
    assert not torch.equal(a.tid2eid, c.tid2eid)


def test_hash_router_rejects_oob_token_ids() -> None:
    """Out-of-vocab token ids surface a clear error."""
    router = DeepseekV4HashRouter(hidden_size=4, num_experts=4, topk=2, vocab_size=8, seed=0)
    hidden = torch.randn(1, 1, 4, dtype=torch.float32)
    token_ids = torch.tensor([[8]], dtype=torch.long)  # >= vocab_size
    with pytest.raises(ValueError, match="vocab_size"):
        router(hidden, token_ids)


def test_hash_router_rejects_shape_mismatch() -> None:
    """``hidden`` and ``token_ids`` must flatten to the same length."""
    router = DeepseekV4HashRouter(hidden_size=4, num_experts=4, topk=2, vocab_size=8, seed=0)
    hidden = torch.randn(2, 3, 4, dtype=torch.float32)  # N=6
    token_ids = torch.zeros(1, 5, dtype=torch.long)  # N=5
    with pytest.raises(ValueError, match="flatten to the same length"):
        router(hidden, token_ids)
