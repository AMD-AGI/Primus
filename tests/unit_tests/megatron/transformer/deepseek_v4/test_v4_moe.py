###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for V4 MoE forward (G5, plan-2 §04).

Pins :class:`DeepseekV4MoE` against the HF reference at
``DeepSeek-V4-Flash/inference/model.py:MoE.forward``. The test runs on
CPU, fp32, with ``pg_collection=None`` so the V4 MoE uses its
local-experts path (per-expert dispatch loop, no Megatron dispatcher).

Pass criteria (G5):
* 1L MoE forward agrees with the inline HF reference to <= 1e-3 max-abs
  for both routing modes (learned + hash).
* Output dtype / shape match input.
* Gradient flows from MoE output back into the gate ``weight`` (router)
  AND the expert ``w1`` / ``w2`` / ``w3`` Linears.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.clamped_swiglu import ClampedSwiGLUMLP
from primus.backends.megatron.core.transformer.moe.v4_hash_router import (
    DeepseekV4HashRouter,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import (
    DeepseekV4MoE,
    DeepseekV4MoESubmodules,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
    DeepseekV4LearnedRouter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_moe_config(
    *,
    hidden_size: int,
    moe_intermediate_size: int,
    shared_expert_intermediate_size: Optional[int],
    num_experts: int,
    topk: int,
    swiglu_limit: float,
    score_function: str,
    num_hash_layers: int,
    vocab_size: int,
    enable_expert_bias: bool = False,
    topk_scaling_factor: float = 1.0,
) -> DeepSeekV4TransformerConfig:
    """Minimal V4 config for the MoE CPU smoke."""
    return DeepSeekV4TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=32,
        ffn_hidden_size=hidden_size * 4,  # only consumed by attention path
        moe_ffn_hidden_size=moe_intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        moe_shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_moe_experts=num_experts,
        moe_router_topk=topk,
        moe_router_score_function=score_function,
        moe_router_enable_expert_bias=enable_expert_bias,
        moe_router_topk_scaling_factor=topk_scaling_factor,
        swiglu_limit=swiglu_limit,
        num_hash_layers=num_hash_layers,
        hash_routing_seed=11,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        layernorm_epsilon=1.0e-6,
        norm_epsilon=1.0e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        # MLATransformerConfig requirements (not exercised by MoE-only test)
        qk_pos_emb_head_dim=8,
        qk_head_dim=24,
        v_head_dim=32,
        kv_lora_rank=32,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
    )


def _make_moe(
    config: DeepSeekV4TransformerConfig,
    *,
    layer_idx: int,
) -> DeepseekV4MoE:
    """Build a CPU-friendly V4 MoE (no pg_collection -> local-experts path)."""
    return DeepseekV4MoE(
        config=config,
        layer_idx=layer_idx,
        pg_collection=None,
        submodules=DeepseekV4MoESubmodules(),
    )


# ---------------------------------------------------------------------------
# Inline HF reference
# ---------------------------------------------------------------------------


def _hf_gate_forward(
    *,
    hidden_flat: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    score_function: str,
    topk: int,
    route_scale: float,
    tid2eid: Optional[torch.Tensor],
    token_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirrors ``Gate.forward`` exactly. Returns dense ``(weights, indices)``."""
    scores = F.linear(hidden_flat.float(), weight.float())
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
        indices = tid2eid[token_ids.reshape(-1).long()].long()
    else:
        indices = scores.topk(topk, dim=-1).indices
    weights = original_scores.gather(1, indices)
    if score_function != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * route_scale
    return weights, indices


def _hf_expert_forward(
    expert: ClampedSwiGLUMLP,
    x: torch.Tensor,
    weights: Optional[torch.Tensor],
    *,
    swiglu_limit: float,
) -> torch.Tensor:
    """Mirrors ``Expert.forward`` (pre-mul clamp + post-w1/w3 weight scaling)."""
    gate = expert.w1(x).float()
    up = expert.w3(x).float()
    if swiglu_limit > 0.0:
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = F.silu(gate) * up
    if weights is not None:
        h = weights * h
    return expert.w2(h.to(x.dtype))


def _hf_moe_forward(
    *,
    moe: DeepseekV4MoE,
    hidden: torch.Tensor,
    token_ids: Optional[torch.Tensor],
    score_function: str,
    swiglu_limit: float,
    route_scale: float,
) -> torch.Tensor:
    """Inline reference for the MoE forward, sharing weights with ``moe``.

    Builds the (weights, indices) dense form via ``_hf_gate_forward`` and
    runs the per-expert dispatch loop the way the released reference
    does, then adds the shared-expert contribution. All tensors share
    storage with the live :class:`DeepseekV4MoE` so a single set of
    weights drives both forward paths.
    """
    shape = hidden.shape
    flat_hidden = hidden.reshape(-1, moe.hidden_size)

    # Gate
    if moe.use_hash_router:
        router = moe.router
        assert isinstance(router, DeepseekV4HashRouter)
        weights, indices = _hf_gate_forward(
            hidden_flat=flat_hidden,
            weight=router.weight,
            bias=None,
            score_function=score_function,
            topk=moe.moe_router_topk,
            route_scale=route_scale,
            tid2eid=router.tid2eid,
            token_ids=token_ids,
        )
    else:
        router = moe.learned_router
        assert isinstance(router, DeepseekV4LearnedRouter)
        weights, indices = _hf_gate_forward(
            hidden_flat=flat_hidden,
            weight=router.weight,
            bias=router.expert_bias,
            score_function=score_function,
            topk=moe.moe_router_topk,
            route_scale=route_scale,
            tid2eid=None,
            token_ids=None,
        )

    y = torch.zeros_like(flat_hidden, dtype=torch.float32)
    n_routed = moe.num_routed_experts
    counts = torch.bincount(indices.flatten(), minlength=n_routed).tolist()
    assert moe.local_experts is not None
    for local_i, global_i in enumerate(moe.local_expert_indices):
        if counts[global_i] == 0:
            continue
        expert = moe.local_experts[local_i]
        idx, top = torch.where(indices == global_i)
        y[idx] += _hf_expert_forward(
            expert,
            flat_hidden[idx],
            weights[idx, top, None],
            swiglu_limit=swiglu_limit,
        ).float()

    if moe.shared_expert is not None:
        assert isinstance(moe.shared_expert, ClampedSwiGLUMLP)
        y = (
            y
            + _hf_expert_forward(
                moe.shared_expert,
                flat_hidden,
                None,
                swiglu_limit=swiglu_limit,
            ).float()
        )

    return y.type_as(flat_hidden).view(*shape)


# ---------------------------------------------------------------------------
# Construction sanity
# ---------------------------------------------------------------------------


def test_v4_moe_subclasses_megatron_module() -> None:
    """``DeepseekV4MoE`` is a ``MegatronModule`` (config plumbing parity)."""
    from megatron.core.transformer.module import MegatronModule

    assert issubclass(DeepseekV4MoE, MegatronModule)


def test_v4_moe_cpu_path_builds_local_experts_and_shared_expert() -> None:
    """When ``pg_collection`` is None the MoE skips the dispatcher and
    builds local :class:`ClampedSwiGLUMLP` experts plus a single
    :class:`ClampedSwiGLUMLP` shared expert (matches HF reference)."""
    config = _make_moe_config(
        hidden_size=12,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function="sqrtsoftplus",
        num_hash_layers=0,
        vocab_size=16,
    )
    moe = _make_moe(config, layer_idx=0)

    assert moe.token_dispatcher is None
    assert moe.grouped_experts is None
    assert moe.local_experts is not None
    assert len(moe.local_experts) == 4
    assert all(isinstance(e, ClampedSwiGLUMLP) for e in moe.local_experts)
    assert isinstance(moe.shared_expert, ClampedSwiGLUMLP)
    assert moe.local_expert_indices == [0, 1, 2, 3]


def test_v4_moe_set_layer_number_updates_attribute() -> None:
    """``set_layer_number`` mirrors :class:`BaseMoELayer`."""
    config = _make_moe_config(
        hidden_size=12,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=None,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function="sqrtsoftplus",
        num_hash_layers=0,
        vocab_size=16,
    )
    moe = _make_moe(config, layer_idx=0)
    moe.set_layer_number(7)
    assert moe.layer_number == 7


# ---------------------------------------------------------------------------
# G5: numerical alignment vs HF reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score_function", ["sqrtsoftplus", "sigmoid", "softmax"])
@pytest.mark.parametrize("with_shared_expert", [True, False])
def test_v4_moe_learned_layer_matches_hf_reference(score_function: str, with_shared_expert: bool) -> None:
    """G5: learned router + clamped-SwiGLU MoE forward agrees with
    the inline HF reference on a 1L toy (CPU fp32)."""
    torch.manual_seed(2025)
    H = 16
    I = 24
    config = _make_moe_config(
        hidden_size=H,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=I if with_shared_expert else None,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function=score_function,
        num_hash_layers=0,  # learned router for layer_idx >= 0
        vocab_size=32,
    )
    moe = _make_moe(config, layer_idx=0)

    hidden = torch.randn(2, 5, H, dtype=torch.float32) * 0.5
    out = moe(hidden, token_ids=None)
    ref = _hf_moe_forward(
        moe=moe,
        hidden=hidden,
        token_ids=None,
        score_function=score_function,
        swiglu_limit=7.0,
        route_scale=1.0,
    )

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 1.0e-3, f"max-abs vs HF reference = {max_abs}"


@pytest.mark.parametrize("score_function", ["sqrtsoftplus", "sigmoid", "softmax"])
def test_v4_moe_hash_layer_matches_hf_reference(score_function: str) -> None:
    """G5: hash router + clamped-SwiGLU MoE forward agrees with
    the inline HF reference on a 1L toy (CPU fp32)."""
    torch.manual_seed(7)
    H = 16
    I = 24
    V = 32
    config = _make_moe_config(
        hidden_size=H,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=I,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function=score_function,
        num_hash_layers=1,  # hash router for layer_idx == 0
        vocab_size=V,
    )
    moe = _make_moe(config, layer_idx=0)

    hidden = torch.randn(2, 5, H, dtype=torch.float32) * 0.5
    token_ids = torch.randint(0, V, (2, 5), dtype=torch.long)

    out = moe(hidden, token_ids=token_ids)
    ref = _hf_moe_forward(
        moe=moe,
        hidden=hidden,
        token_ids=token_ids,
        score_function=score_function,
        swiglu_limit=7.0,
        route_scale=1.0,
    )

    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
    max_abs = (out - ref).abs().max().item()
    assert max_abs <= 1.0e-3, f"max-abs vs HF reference = {max_abs}"


def test_v4_moe_route_scale_propagates_to_output() -> None:
    """``moe_router_topk_scaling_factor`` (HF ``route_scale``) is honored."""
    torch.manual_seed(3)
    H = 12
    config = _make_moe_config(
        hidden_size=H,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=None,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function="sqrtsoftplus",
        num_hash_layers=0,
        vocab_size=16,
        topk_scaling_factor=2.5,
    )
    moe = _make_moe(config, layer_idx=0)
    hidden = torch.randn(1, 3, H, dtype=torch.float32) * 0.5

    out = moe(hidden, token_ids=None)
    ref = _hf_moe_forward(
        moe=moe,
        hidden=hidden,
        token_ids=None,
        score_function="sqrtsoftplus",
        swiglu_limit=7.0,
        route_scale=2.5,
    )
    assert (out - ref).abs().max().item() <= 1.0e-3


def test_v4_moe_gradient_flows_to_router_and_experts() -> None:
    """G5 follow-up: backward pass populates grads on ``router.weight``
    and on at least one expert's ``w1`` / ``w2`` / ``w3`` weight."""
    torch.manual_seed(13)
    H = 12
    config = _make_moe_config(
        hidden_size=H,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function="sqrtsoftplus",
        num_hash_layers=0,
        vocab_size=16,
    )
    moe = _make_moe(config, layer_idx=0)
    hidden = torch.randn(2, 4, H, dtype=torch.float32, requires_grad=False) * 0.5

    out = moe(hidden, token_ids=None)
    out.sum().backward()

    assert moe.learned_router is not None
    assert moe.learned_router.weight.grad is not None
    assert torch.isfinite(moe.learned_router.weight.grad).all()

    # At least one routed expert must have received gradient. Some experts
    # may be unselected in a small toy run; assert >= 1 has a non-zero grad.
    any_expert_grad = False
    assert moe.local_experts is not None
    for expert in moe.local_experts:
        if (
            expert.w1.weight.grad is not None
            and expert.w1.weight.grad.abs().sum().item() > 0.0
            and expert.w2.weight.grad is not None
            and expert.w3.weight.grad is not None
        ):
            any_expert_grad = True
            break
    assert any_expert_grad, "expected at least one routed expert to receive gradient"

    # Shared expert is always-on so it always sees gradient.
    assert moe.shared_expert is not None
    assert moe.shared_expert.w1.weight.grad is not None
    assert moe.shared_expert.w1.weight.grad.abs().sum().item() > 0.0


def test_v4_moe_hash_layer_requires_token_ids() -> None:
    """Hash-routed layers raise a clear error if ``token_ids`` is missing."""
    config = _make_moe_config(
        hidden_size=12,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=None,
        num_experts=4,
        topk=2,
        swiglu_limit=7.0,
        score_function="sqrtsoftplus",
        num_hash_layers=1,
        vocab_size=16,
    )
    moe = _make_moe(config, layer_idx=0)
    hidden = torch.randn(1, 3, 12, dtype=torch.float32)
    with pytest.raises(ValueError, match="token_ids is required"):
        moe(hidden, token_ids=None)
