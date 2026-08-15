###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for shared-expert / all-to-all overlap wiring.

``moe_shared_expert_overlap: true`` in the model yaml asks for the shared
expert's GEMMs to be interleaved with the dispatch and combine all-to-alls.
The token dispatcher owns the hooks, and once it owns the shared expert it also
**adds the shared output into the combine result** -- so the risky part of this
wiring is not the overlap itself but the double-count it would cause if the MoE
layer kept adding the shared expert as well.

These tests drive ``_enable_shared_expert_overlap`` and the forward's add-or-not
decision against fakes, so they need no GPU or process group.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.transformer.moe.v4_moe import (  # noqa: E402
    DeepseekV4MoE,
)


class _FakeConfig:
    def __init__(self, overlap: bool):
        self.moe_shared_expert_overlap = overlap


class _OverlapCapableSharedExpert:
    """Stands in for ``SharedExpertMLP``: has the overlap protocol."""

    def pre_forward_comm(self, x):  # pragma: no cover - never called here
        raise AssertionError


class _PlainSharedExpert:
    """Stands in for the CPU ``ClampedSwiGLUMLP``: no overlap protocol."""


class _Dispatcher:
    def __init__(self):
        self.received = None

    def set_shared_experts(self, shared_experts):
        self.received = shared_experts


class _FlexDispatcher:
    def set_shared_experts(self, shared_experts):
        raise NotImplementedError("Shared expert overlap is not supported")


def _layer(*, overlap: bool, shared_expert, dispatcher) -> DeepseekV4MoE:
    """A bare object with only what ``_enable_shared_expert_overlap`` reads."""
    layer = DeepseekV4MoE.__new__(DeepseekV4MoE)
    layer.config = _FakeConfig(overlap)
    layer.shared_expert = shared_expert
    layer.token_dispatcher = dispatcher
    layer.layer_idx = 3
    return layer


def test_overlap_hands_the_shared_expert_to_the_dispatcher():
    dispatcher = _Dispatcher()
    shared = _OverlapCapableSharedExpert()
    layer = _layer(overlap=True, shared_expert=shared, dispatcher=dispatcher)

    assert layer._enable_shared_expert_overlap() is True
    assert dispatcher.received is shared


def test_overlap_off_leaves_the_dispatcher_alone():
    dispatcher = _Dispatcher()
    layer = _layer(overlap=False, shared_expert=_OverlapCapableSharedExpert(), dispatcher=dispatcher)

    assert layer._enable_shared_expert_overlap() is False
    assert dispatcher.received is None


def test_unsupported_dispatcher_falls_back_to_serial():
    """The flex dispatcher rejects overlap; that must not be fatal."""
    layer = _layer(
        overlap=True,
        shared_expert=_OverlapCapableSharedExpert(),
        dispatcher=_FlexDispatcher(),
    )
    assert layer._enable_shared_expert_overlap() is False


def test_shared_expert_without_the_protocol_is_not_handed_over():
    """The CPU shared expert has no ``pre_forward_comm``."""
    dispatcher = _Dispatcher()
    layer = _layer(overlap=True, shared_expert=_PlainSharedExpert(), dispatcher=dispatcher)

    assert layer._enable_shared_expert_overlap() is False
    assert dispatcher.received is None


def test_no_shared_expert_means_no_overlap():
    layer = _layer(overlap=True, shared_expert=None, dispatcher=_Dispatcher())
    assert layer._enable_shared_expert_overlap() is False


# ---------------------------------------------------------------------------
# The add-or-not decision in forward
# ---------------------------------------------------------------------------


def _forward_layer(*, overlap: bool):
    """A layer stub exercising only the production branch of ``forward``."""
    layer = DeepseekV4MoE.__new__(DeepseekV4MoE)
    layer.local_experts = None
    layer.hidden_size = 4
    layer.shared_expert_overlap = overlap
    layer.shared_expert = lambda x: torch.ones_like(x)
    layer._dispatcher_forward = lambda hidden, probs, routing_map: torch.zeros_like(hidden)
    layer._route = lambda hidden, token_ids: (None, None)
    return layer


def test_forward_does_not_add_the_shared_expert_under_overlap():
    """The dispatcher already folded it in; adding again would double-count."""
    layer = _forward_layer(overlap=True)
    hidden = torch.zeros(2, 3, 4)
    out = DeepseekV4MoE.forward(layer, hidden)
    torch.testing.assert_close(out, torch.zeros_like(hidden))


def test_forward_adds_the_shared_expert_without_overlap():
    layer = _forward_layer(overlap=False)
    hidden = torch.zeros(2, 3, 4)
    out = DeepseekV4MoE.forward(layer, hidden)
    torch.testing.assert_close(out, torch.ones_like(hidden))


def test_yaml_requests_overlap():
    """Pin the declaration the wiring above now honours."""
    from pathlib import Path

    from primus.core.config.yaml_loader import parse_yaml

    yaml_dir = Path(__file__).resolve().parents[5] / "primus" / "configs" / "models" / "megatron"
    parsed = parse_yaml(str(yaml_dir / "deepseek_v4_base.yaml"))
    assert parsed["moe_shared_expert_overlap"] is True
