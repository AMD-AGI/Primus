###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for Per-Head Muon (Kimi K3 report §2.5).

Five claims need pinning, and they are deliberately layered so that the ones that
are pure arithmetic do not depend on an optional package:

1. **Blocking is exactly a loop.** ``orthogonalize_per_head`` on a synthetic
   ``[H*D, N]`` momentum equals looping over heads and calling the underlying
   orthogonalization on each ``[D, N]`` block — checked first with a stub
   orthogonalizer (pure torch, always runs) and then against the real
   ``newton_schulz`` (skipped without ``emerging_optimizers``).
2. **Each output block is orthogonalized.** Singular values of every per-head
   block, after dividing out the Muon scale factor, sit near 1 — a spectral
   check, not a shape check.
3. **It differs from whole-matrix orthogonalization**, in exactly the way the
   report claims: with deliberately unequal per-head scales, whole-matrix
   orthogonalization leaves the small heads under-normalised and lets one head's
   scale change every other head's update, while per-head does neither.
4. **Parameter selection is the documented rule.** The K3 Q/K/V projections are
   picked up; the latent down-projections, the gates, the output projections,
   ``b_proj``, the embeddings and the experts are not. Asserted both against a
   synthetic name list and against a **real** Kimi K3 decoder block.
5. **The option defaults to OFF**, so nothing else in the repo changes behaviour.

``emerging_optimizers`` is not in the ROCm image; it comes from the pinned
submodule ``third_party/Emerging-Optimizers`` (or the
``01_install_emerging_optimizers.sh`` hook). Tests needing it use
``pytest.importorskip``, and the pure-torch layer above covers the blocking logic
without it.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest

# transformer_engine SIGABRTs unless torch is imported first (node/README.md:125-136).
import torch

from primus.backends.megatron.core.optimizer.per_head_muon import (
    MIN_BLOCK_ROWS,
    PER_HEAD_SPEC_ATTR,
    HeadBlockSpec,
    PerHeadMuonConfig,
    batched_newton_schulz,
    head_block_spec_for,
    make_per_head_orthogonalize,
    orthogonalize_per_head,
    propagate_specs_to_master_weights,
    tag_per_head_params,
)

# ---------------------------------------------------------------------------
# The kimi_k3_debug.yaml geometry, which is also this work package's A/B shape.
# MLA:  q_head_dim = qk_head_dim + qk_pos_emb_head_dim = 32 + 16 = 48
# KDA:  linear_key_head_dim == linear_value_head_dim == 32
# ---------------------------------------------------------------------------
NUM_HEADS = 8
HIDDEN = 1024
Q_LORA_RANK = 256
KV_LORA_RANK = 128
QK_HEAD_DIM = 32
QK_POS_EMB_HEAD_DIM = 16
V_HEAD_DIM = 32
Q_HEAD_DIM = QK_HEAD_DIM + QK_POS_EMB_HEAD_DIM
KDA_HEAD_DIM = 32


class _StubModelConfig:
    """The head-dim fields :func:`head_block_spec_for` reads, and nothing else.

    A stub rather than a real ``TransformerConfig`` so the selection rule is tested
    in isolation; :func:`test_selection_on_a_real_k3_decoder_block` covers the real
    config and the real modules.
    """

    def __init__(self, **overrides):
        self.qk_head_dim = QK_HEAD_DIM
        self.qk_pos_emb_head_dim = QK_POS_EMB_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.linear_key_head_dim = KDA_HEAD_DIM
        self.linear_value_head_dim = KDA_HEAD_DIM
        self.num_attention_heads = NUM_HEADS
        self.linear_num_key_heads = NUM_HEADS
        self.linear_num_value_heads = NUM_HEADS
        self.hidden_size = HIDDEN
        self.__dict__.update(overrides)


def _prefix(layer: int = 3) -> str:
    """The parameter-name prefix a real run produces.

    Copied from ``validate/logs/c1_k3_200_constlr_seed1234.log:2179-2500``, the
    8-layer debug run: ``module.decoder.layers.<i>.self_attention.<leaf>.weight``.
    """
    return f"module.decoder.layers.{layer}.self_attention."


# ===========================================================================
# 1. Blocking, without emerging_optimizers (pure torch)
# ===========================================================================


def _qr_orthogonalize(block: torch.Tensor, _tp_group=None, _partition_dim=None) -> torch.Tensor:
    """A stub ``scaled_orthogonalize_fn``: exact orthogonalization via QR.

    Signature matches ``TensorParallelMuon``'s closure
    (``muon.py:67-71``: ``(grad, tp_group, partition_dim)``). QR is used rather
    than Newton-Schulz so this layer needs no optional dependency and so the
    "each block is orthogonalized" property is exact instead of approximate.
    """
    rows, cols = block.shape
    if rows <= cols:
        q, _ = torch.linalg.qr(block.mT)
        return q.mT.contiguous()
    q, _ = torch.linalg.qr(block)
    return q.contiguous()


def _row_index_marker(block: torch.Tensor, _tp_group=None, _partition_dim=None) -> torch.Tensor:
    """A stub that returns the block's first element broadcast over the block.

    Makes the *provenance* of every output element checkable: if reassembly puts a
    block back in the wrong place, the marker lands in the wrong rows.
    """
    return torch.full_like(block, float(block.reshape(-1)[0]))


def _spec(rows: Tuple[int, ...], head_axis: int = 0) -> HeadBlockSpec:
    return HeadBlockSpec(rows=rows, head_axis=head_axis, rule="test")


def test_blocking_equals_an_explicit_loop_over_heads():
    """Claim 1, with a stub orthogonalizer: the reference loop, spelled out."""
    torch.manual_seed(0)
    grad = torch.randn(NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK, dtype=torch.float64)
    spec = _spec((Q_HEAD_DIM,))

    got = orthogonalize_per_head(grad, spec, _qr_orthogonalize)

    expected_blocks = []
    for head in range(NUM_HEADS):
        block = grad[head * Q_HEAD_DIM : (head + 1) * Q_HEAD_DIM, :]
        expected_blocks.append(_qr_orthogonalize(block))
    expected = torch.cat(expected_blocks, dim=0)

    assert got.shape == grad.shape
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_blocking_round_trips_under_an_identity_orthogonalizer():
    """Split + reassemble is lossless and order-preserving for every row structure."""
    torch.manual_seed(1)
    for rows in [(Q_HEAD_DIM,), (QK_HEAD_DIM, V_HEAD_DIM), (QK_HEAD_DIM + V_HEAD_DIM,)]:
        grad = torch.randn(NUM_HEADS * sum(rows), KV_LORA_RANK, dtype=torch.float64)
        out = orthogonalize_per_head(grad, _spec(rows), lambda b, *_: b)
        torch.testing.assert_close(out, grad, rtol=0, atol=0)


def test_split_kv_row_order_matches_the_mla_forward_pass():
    """Inside a head, rows are ``[K | V]`` — verified against the forward pass.

    ``MultiLatentAttention`` views ``linear_kv_up_proj``'s output as
    ``[..., num_heads, qk_head_dim + v_head_dim]`` (``multi_latent_attention.py:737-741``)
    and then splits ``[qk_head_dim, v_head_dim]`` off the last dim (``:876``). So the
    K rows of head ``h`` are ``h*(qk+v) .. h*(qk+v)+qk`` and the V rows follow.
    A marker orthogonalizer makes that placement observable.
    """
    rows_per_head = QK_HEAD_DIM + V_HEAD_DIM
    grad = torch.arange(
        NUM_HEADS * rows_per_head * 4, dtype=torch.float64
    ).reshape(NUM_HEADS * rows_per_head, 4)

    out = orthogonalize_per_head(grad, _spec((QK_HEAD_DIM, V_HEAD_DIM)), _row_index_marker)

    for head in range(NUM_HEADS):
        base = head * rows_per_head
        k_rows = out[base : base + QK_HEAD_DIM]
        v_rows = out[base + QK_HEAD_DIM : base + rows_per_head]
        # Each block's marker is its own top-left input element.
        assert torch.all(k_rows == grad[base, 0]), head
        assert torch.all(v_rows == grad[base + QK_HEAD_DIM, 0]), head
        # K and V of the same head are genuinely separate blocks.
        assert k_rows[0, 0] != v_rows[0, 0]


def test_unsplit_kv_keeps_one_block_per_head():
    """With ``split_kv=False`` the head is a single ``[qk+v, r]`` block."""
    rows_per_head = QK_HEAD_DIM + V_HEAD_DIM
    grad = torch.arange(
        NUM_HEADS * rows_per_head * 4, dtype=torch.float64
    ).reshape(NUM_HEADS * rows_per_head, 4)

    out = orthogonalize_per_head(grad, _spec((rows_per_head,)), _row_index_marker)

    for head in range(NUM_HEADS):
        base = head * rows_per_head
        block = out[base : base + rows_per_head]
        assert torch.all(block == grad[base, 0]), head


def test_head_axis_1_blocks_along_the_input_dim():
    """The opt-in output-projection layout: ``[hidden, H*v_head_dim]``.

    ``linear_proj`` is ``[hidden_size, num_attention_heads * v_head_dim]``
    (``multi_latent_attention.py:171-183``), so its heads live on dim 1.
    """
    grad = torch.arange(
        6 * NUM_HEADS * V_HEAD_DIM, dtype=torch.float64
    ).reshape(6, NUM_HEADS * V_HEAD_DIM)
    spec = _spec((V_HEAD_DIM,), head_axis=1)

    assert spec.num_heads(grad.shape) == NUM_HEADS
    out = orthogonalize_per_head(grad, spec, _row_index_marker)
    assert out.shape == grad.shape
    for head in range(NUM_HEADS):
        block = out[:, head * V_HEAD_DIM : (head + 1) * V_HEAD_DIM]
        assert torch.all(block == grad[0, head * V_HEAD_DIM])

    # And it round-trips.
    identity = orthogonalize_per_head(grad, spec, lambda b, *_: b)
    torch.testing.assert_close(identity, grad, rtol=0, atol=0)


def test_every_stub_block_is_exactly_orthogonal():
    """Claim 2 in its exact form: with QR blocks, ``B @ B.T == I`` per head."""
    torch.manual_seed(2)
    grad = torch.randn(NUM_HEADS * KDA_HEAD_DIM, HIDDEN, dtype=torch.float64)
    out = orthogonalize_per_head(grad, _spec((KDA_HEAD_DIM,)), _qr_orthogonalize)

    eye = torch.eye(KDA_HEAD_DIM, dtype=torch.float64)
    for head in range(NUM_HEADS):
        block = out[head * KDA_HEAD_DIM : (head + 1) * KDA_HEAD_DIM]
        torch.testing.assert_close(block @ block.mT, eye, rtol=1e-10, atol=1e-10)


def test_num_heads_is_derived_from_the_local_shape_not_the_config():
    """The tensor-parallel story, as a property of :class:`HeadBlockSpec`.

    Head-axis-0 parameters are column-parallel over heads, so a TP rank holds a
    whole number of *complete* head blocks. Deriving the head count from the local
    shape is what makes per-head blocking correct under TP without communication.
    """
    spec = _spec((Q_HEAD_DIM,))
    for tp_size in (1, 2, 4, 8):
        local_rows = (NUM_HEADS // tp_size) * Q_HEAD_DIM
        assert spec.matches((local_rows, Q_LORA_RANK))
        assert spec.num_heads((local_rows, Q_LORA_RANK)) == NUM_HEADS // tp_size

    # TP == num_heads leaves one head per rank, and that must still take the
    # per-head path: the whole-matrix path would all-gather and couple the heads.
    assert spec.matches((Q_HEAD_DIM, Q_LORA_RANK))
    assert spec.num_heads((Q_HEAD_DIM, Q_LORA_RANK)) == 1


def test_a_shape_that_contradicts_the_spec_is_rejected():
    """A non-multiple extent must fall back, not mis-block."""
    spec = _spec((Q_HEAD_DIM,))
    assert not spec.matches((NUM_HEADS * Q_HEAD_DIM + 1, Q_LORA_RANK))
    assert not spec.matches((Q_HEAD_DIM - 1, Q_LORA_RANK))
    assert not spec.matches((NUM_HEADS * Q_HEAD_DIM,))


def test_one_row_blocks_are_refused():
    """A ``[1, N]`` block is a normalisation, not an orthogonalization.

    KDA's ``b_proj`` is ``[num_heads, hidden]`` (``kimi_delta_attention.py:355``) and
    is already excluded by name; this guards the degenerate geometry generally.
    """
    assert MIN_BLOCK_ROWS >= 2
    assert not _spec((1,)).matches((NUM_HEADS, HIDDEN))
    assert not _spec((1, 1)).matches((2 * NUM_HEADS, HIDDEN))


# ===========================================================================
# 2. Parameter selection
# ===========================================================================

_MLA_SHAPES = {
    "linear_q_down_proj": (Q_LORA_RANK, HIDDEN),
    "linear_q_up_proj": (NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK),
    "linear_kv_down_proj": (KV_LORA_RANK + QK_POS_EMB_HEAD_DIM, HIDDEN),
    "linear_kv_up_proj": (NUM_HEADS * (QK_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
    "linear_proj": (HIDDEN, NUM_HEADS * V_HEAD_DIM),
    "linear_o_gate": (NUM_HEADS * V_HEAD_DIM, HIDDEN),
}

_KDA_SHAPES = {
    "q_proj": (NUM_HEADS * KDA_HEAD_DIM, HIDDEN),
    "k_proj": (NUM_HEADS * KDA_HEAD_DIM, HIDDEN),
    "v_proj": (NUM_HEADS * KDA_HEAD_DIM, HIDDEN),
    "f_a_proj": (KDA_HEAD_DIM, HIDDEN),
    "f_b_proj": (NUM_HEADS * KDA_HEAD_DIM, KDA_HEAD_DIM),
    "b_proj": (NUM_HEADS, HIDDEN),
    "g_proj": (NUM_HEADS * KDA_HEAD_DIM, HIDDEN),
    "o_proj": (HIDDEN, NUM_HEADS * KDA_HEAD_DIM),
}

_NON_ATTENTION_SHAPES = {
    "module.embedding.word_embeddings.weight": (163840, HIDDEN),
    "module.output_layer.weight": (163840, HIDDEN),
    "module.decoder.layers.1.mlp.experts.linear_fc1.weight": (2 * 512, HIDDEN),
    "module.decoder.layers.1.mlp.experts.linear_fc2.weight": (HIDDEN, 512),
    # With moe_grouped_gemm the expert weights are named weight0..weightN, which is
    # what a real run shows (validate/logs/c1_...:2179 "linear_fc2.weight5").
    "module.decoder.layers.1.mlp.experts.linear_fc1.weight5": (2 * 512, HIDDEN),
    "module.decoder.layers.1.mlp.router.weight": (8, HIDDEN),
    "module.decoder.layers.0.mlp.linear_fc1.weight": (2 * 2048, HIDDEN),
    "module.decoder.layers.1.mlp.fc2_latent_proj.weight": (HIDDEN, 512),
}


def _names_and_shapes() -> List[Tuple[str, Tuple[int, int]]]:
    items = [(_prefix(3) + leaf + ".weight", shape) for leaf, shape in _MLA_SHAPES.items()]
    items += [(_prefix(0) + leaf + ".weight", shape) for leaf, shape in _KDA_SHAPES.items()]
    items += list(_NON_ATTENTION_SHAPES.items())
    return items


def _selected(config: PerHeadMuonConfig) -> dict:
    model_config = _StubModelConfig()
    out = {}
    for name, shape in _names_and_shapes():
        spec = head_block_spec_for(name, shape, model_config, config)
        if spec is not None:
            out[name] = spec
    return out


def test_tagging_is_gated_on_enabled():
    """``enabled`` is the single gate, enforced in the only function that mutates.

    ``head_block_spec_for`` is the pure rule and answers regardless; nothing is ever
    tagged unless the switch is on.
    """
    params = {name: torch.nn.Parameter(torch.zeros(*shape)) for name, shape in _names_and_shapes()}
    off = tag_per_head_params(params.items(), _StubModelConfig(), PerHeadMuonConfig())
    assert off.num_selected == 0
    assert off.skipped_head_structured == []
    assert all(getattr(p, PER_HEAD_SPEC_ATTR, None) is None for p in params.values())

    on = tag_per_head_params(params.items(), _StubModelConfig(), PerHeadMuonConfig(enabled=True))
    assert on.num_selected == 5


def test_default_selection_is_exactly_the_k3_qkv_projections():
    """Claim 4. The whole rule, as one assertion on the selected set."""
    selected = _selected(PerHeadMuonConfig(enabled=True))
    assert {name.rsplit(".", 2)[-2] for name in selected} == {
        "linear_q_up_proj",
        "linear_kv_up_proj",
        "q_proj",
        "k_proj",
        "v_proj",
    }
    assert selected[_prefix(3) + "linear_q_up_proj.weight"].rows == (Q_HEAD_DIM,)
    assert selected[_prefix(3) + "linear_kv_up_proj.weight"].rows == (QK_HEAD_DIM, V_HEAD_DIM)
    assert selected[_prefix(0) + "q_proj.weight"].rows == (KDA_HEAD_DIM,)
    assert all(spec.head_axis == 0 for spec in selected.values())


@pytest.mark.parametrize(
    "leaf",
    [
        # Latent / shared projections: no head axis at all.
        "linear_q_down_proj",
        "linear_kv_down_proj",
        "f_a_proj",
        # Head-structured but not Q/K/V, and opt-in only.
        "linear_proj",
        "o_proj",
        "linear_o_gate",
        "g_proj",
        "f_b_proj",
        # One row per head.
        "b_proj",
    ],
)
def test_default_selection_rejects_everything_else_in_attention(leaf: str):
    shape = _MLA_SHAPES.get(leaf) or _KDA_SHAPES[leaf]
    name = _prefix(3 if leaf in _MLA_SHAPES else 0) + leaf + ".weight"
    assert head_block_spec_for(name, shape, _StubModelConfig(), PerHeadMuonConfig(enabled=True)) is None


@pytest.mark.parametrize("name,shape", sorted(_NON_ATTENTION_SHAPES.items()))
def test_embeddings_experts_router_and_mlp_are_never_selected(name: str, shape):
    for config in (
        PerHeadMuonConfig(enabled=True),
        PerHeadMuonConfig(enabled=True, include_output_proj=True, include_gates=True),
    ):
        assert head_block_spec_for(name, shape, _StubModelConfig(), config) is None


def test_leaf_matching_is_not_substring_matching():
    """``q_proj`` is a substring of ``linear_q_proj``; ``b_proj`` ends in ``_proj``.

    Selection matches the dotted **leaf module** name, so ``linear_q_proj``
    (MLA without q-LoRA) must be picked up as MLA and not confused with KDA's
    ``q_proj``, which has a different head dim.
    """
    config = PerHeadMuonConfig(enabled=True)
    mla_no_lora = head_block_spec_for(
        _prefix(3) + "linear_q_proj.weight",
        (NUM_HEADS * Q_HEAD_DIM, HIDDEN),
        _StubModelConfig(),
        config,
    )
    assert mla_no_lora is not None
    assert mla_no_lora.rows == (Q_HEAD_DIM,)
    assert mla_no_lora.rule == "mla.linear_q_proj"

    kda = head_block_spec_for(
        _prefix(0) + "q_proj.weight", (NUM_HEADS * KDA_HEAD_DIM, HIDDEN), _StubModelConfig(), config
    )
    assert kda is not None and kda.rows == (KDA_HEAD_DIM,) and kda.rule == "kda.q_proj"


def test_split_kv_false_fuses_k_and_v_within_a_head():
    config = PerHeadMuonConfig(enabled=True, split_kv=False)
    spec = head_block_spec_for(
        _prefix(3) + "linear_kv_up_proj.weight",
        _MLA_SHAPES["linear_kv_up_proj"],
        _StubModelConfig(),
        config,
    )
    assert spec is not None
    assert spec.rows == (QK_HEAD_DIM + V_HEAD_DIM,)
    assert spec.rule.endswith("fused_kv")


def test_opt_ins_add_exactly_what_they_advertise():
    base = set(_selected(PerHeadMuonConfig(enabled=True)))

    gates = set(_selected(PerHeadMuonConfig(enabled=True, include_gates=True)))
    assert {name.rsplit(".", 2)[-2] for name in gates - base} == {"linear_o_gate", "g_proj"}
    assert all(
        _selected(PerHeadMuonConfig(enabled=True, include_gates=True))[name].head_axis == 0
        for name in gates - base
    )

    outs = set(_selected(PerHeadMuonConfig(enabled=True, include_output_proj=True)))
    assert {name.rsplit(".", 2)[-2] for name in outs - base} == {"linear_proj", "o_proj"}
    selected_outs = _selected(PerHeadMuonConfig(enabled=True, include_output_proj=True))
    # Heads live on dim 1 for the output projections.
    assert all(selected_outs[name].head_axis == 1 for name in outs - base)


def test_non_weight_tensors_are_ignored():
    config = PerHeadMuonConfig(enabled=True)
    assert (
        head_block_spec_for(
            _prefix(0) + "q_proj.bias", (NUM_HEADS * KDA_HEAD_DIM,), _StubModelConfig(), config
        )
        is None
    )
    assert (
        head_block_spec_for(
            _prefix(0) + "q_proj._extra_state", (4, 4), _StubModelConfig(), config
        )
        is None
    )


def test_missing_head_dims_select_nothing():
    """A model whose config has no MLA/KDA head dims must not be blocked.

    This is what keeps the patch inert for GPT / Llama / DeepSeek-V4 shapes.
    """
    config = PerHeadMuonConfig(enabled=True)
    bare = _StubModelConfig(
        qk_head_dim=None,
        v_head_dim=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
    )
    for name, shape in _names_and_shapes():
        assert head_block_spec_for(name, shape, bare, config) is None


def test_tagging_sets_the_attribute_and_reports_a_summary():
    params = {name: torch.nn.Parameter(torch.zeros(*shape)) for name, shape in _names_and_shapes()}
    summary = tag_per_head_params(params.items(), _StubModelConfig(), PerHeadMuonConfig(enabled=True))

    assert summary.num_selected == 5
    assert summary.by_rule() == {
        "mla.linear_q_up_proj": 1,
        "mla.linear_kv_up_proj.split_kv": 1,
        "kda.q_proj": 1,
        "kda.k_proj": 1,
        "kda.v_proj": 1,
    }
    assert {n.rsplit(".", 2)[-2] for n in summary.skipped_head_structured} == {
        "linear_proj",
        "o_proj",
        "linear_o_gate",
        "g_proj",
        "f_b_proj",
    }
    for name, param in params.items():
        tagged = getattr(param, PER_HEAD_SPEC_ATTR, None) is not None
        assert tagged == (name in summary.selected), name


def test_frozen_parameters_are_skipped():
    """``get_megatron_muon_optimizer`` skips ``requires_grad=False`` (``muon.py:240-241``)."""
    param = torch.nn.Parameter(torch.zeros(NUM_HEADS * KDA_HEAD_DIM, HIDDEN))
    param.requires_grad = False
    summary = tag_per_head_params(
        [(_prefix(0) + "q_proj.weight", param)], _StubModelConfig(), PerHeadMuonConfig(enabled=True)
    )
    assert summary.num_selected == 0


def test_the_spec_reaches_the_fp32_master_weight():
    """The trap: ``main_param`` is a fresh clone that keeps only five attributes.

    ``Float16OptimizerWithFloat16Params`` does
    ``main_param = param.detach().clone().float()`` and then copies only
    ``shared`` plus ``_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS``
    (``optimizer.py:675-684``, ``tensor_parallel/layers.py:60-66``). Under bf16 —
    what ``get_megatron_muon_optimizer`` builds (``muon.py:292-303``) —
    ``orthogonalize`` sees that clone, so the spec has to be copied across
    explicitly.
    """
    from megatron.core.tensor_parallel import copy_tensor_model_parallel_attributes

    param = torch.nn.Parameter(
        torch.zeros(NUM_HEADS * KDA_HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
    )
    tag_per_head_params(
        [(_prefix(0) + "q_proj.weight", param)], _StubModelConfig(), PerHeadMuonConfig(enabled=True)
    )
    assert getattr(param, PER_HEAD_SPEC_ATTR, None) is not None

    main_param = param.detach().clone().float()
    copy_tensor_model_parallel_attributes(main_param, param)
    # Upstream's copy really does drop it -- this is why the extra step exists.
    assert getattr(main_param, PER_HEAD_SPEC_ATTR, None) is None

    param.main_param = main_param
    assert propagate_specs_to_master_weights([("q_proj.weight", param)]) == 1
    assert getattr(main_param, PER_HEAD_SPEC_ATTR, None) is not None
    # Idempotent.
    assert propagate_specs_to_master_weights([("q_proj.weight", param)]) == 0


# ===========================================================================
# 3. The option defaults to OFF
# ===========================================================================


def test_defaults_are_off_and_documented():
    """Claim 5."""
    config = PerHeadMuonConfig()
    assert config.enabled is False
    assert config.impl == "loop"
    # The resolved ambiguities, as asserted defaults.
    assert config.split_kv is True
    assert config.include_output_proj is False
    assert config.include_gates is False
    assert config.strict is True


def test_from_args_on_an_empty_namespace_is_the_default():
    class _Empty:
        pass

    assert PerHeadMuonConfig.from_args(_Empty()) == PerHeadMuonConfig()


def test_from_args_reads_every_field():
    class _Args:
        muon_per_head = True
        muon_per_head_split_kv = False
        muon_per_head_include_output_proj = True
        muon_per_head_include_gates = True
        muon_per_head_impl = "batched"
        muon_per_head_strict = False

    assert PerHeadMuonConfig.from_args(_Args()) == PerHeadMuonConfig(
        enabled=True,
        split_kv=False,
        include_output_proj=True,
        include_gates=True,
        impl="batched",
        strict=False,
    )


def test_an_unknown_impl_is_rejected_at_construction():
    with pytest.raises(ValueError, match="muon_per_head_impl"):
        PerHeadMuonConfig(impl="magic")


@pytest.mark.parametrize(
    "raw,expected",
    [(True, True), (False, False), ("true", True), ("false", False), ("0", False), (None, False)],
)
def test_a_string_flag_is_not_silently_true(raw, expected):
    """``bool("false")`` is ``True``; the flag must not fall into that.

    Primus's loader normally coerces ``${VAR:false}`` to a real bool
    (``yaml_loader.py:15, 111-112``), but a CLI override can still deliver a string.
    """

    class _Args:
        muon_per_head = raw

    assert PerHeadMuonConfig.from_args(_Args()).enabled is expected


def test_an_uninterpretable_flag_raises_rather_than_guessing():
    class _Args:
        muon_per_head = "maybe"

    with pytest.raises(ValueError, match="boolean"):
        PerHeadMuonConfig.from_args(_Args())


def _patch_ctx(**arg_attrs):
    """A ``PatchContext`` shaped the way ``get_args`` expects.

    ``get_args(ctx)`` reads ``ctx.extra["module_config"].params``
    (``primus/core/patches/context.py:106-110``) — the merged Primus namespace, which
    is also why a Muon flag can be set from any Primus YAML even though it is absent
    from Megatron's argparse (``train_runtime.py:442-443``).
    """
    from types import SimpleNamespace

    from primus.core.patches import PatchContext

    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**arg_attrs))},
    )


def test_the_patch_is_registered_once_and_gated_off_by_default():
    """The registered patch must not fire for a job that did not ask for it."""
    import primus.backends.megatron.patches.per_head_muon_patches  # noqa: F401
    from primus.core.patches import PatchRegistry

    patch = PatchRegistry.get("megatron.optimizer.per_head_muon")
    assert patch is not None, "the patch must be registered"
    assert patch.phase == "before_train" and patch.backend == "megatron"
    assert patch.condition is not None

    # Muon selected but the flag unset -> must not fire.
    assert patch.condition(_patch_ctx(optimizer="muon")) is False
    # The flag on but the optimizer is AdamW -> must not fire.
    assert patch.condition(_patch_ctx(optimizer="adam", muon_per_head=True)) is False
    # Neither -> must not fire.
    assert patch.condition(_patch_ctx(optimizer="adam")) is False
    # An explicit false, including as a string, must not fire: the predicate resolves
    # the flag through PerHeadMuonConfig so bool("false") cannot leak through.
    assert patch.condition(_patch_ctx(optimizer="muon", muon_per_head=False)) is False
    assert patch.condition(_patch_ctx(optimizer="muon", muon_per_head="false")) is False
    # Both -> fires, for plain muon and for dist_muon (arguments.py:1422 tests substring).
    assert patch.condition(_patch_ctx(optimizer="muon", muon_per_head=True)) is True
    assert patch.condition(_patch_ctx(optimizer="dist_muon", muon_per_head=True)) is True
    assert patch.condition(_patch_ctx(optimizer="muon", muon_per_head="true")) is True


def test_the_wrapper_falls_through_for_untagged_parameters():
    """A parameter with no spec must reach the original method unchanged."""
    calls = []

    def original(self, p, grad, **kwargs):
        calls.append((tuple(grad.shape), kwargs))
        return grad * 2.0

    wrapped = make_per_head_orthogonalize(original, PerHeadMuonConfig(enabled=True))

    class _Opt:
        pg_collection = None
        mode = "blockwise"
        scaled_orthogonalize_fn = staticmethod(_qr_orthogonalize)

    grad = torch.randn(NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK)
    untagged = torch.nn.Parameter(torch.zeros_like(grad))
    out = wrapped(_Opt(), untagged, grad, lr=1.0)
    torch.testing.assert_close(out, grad * 2.0)
    assert calls == [((NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK), {"lr": 1.0})]

    # Tagged -> per-head, original never called again.
    tagged = torch.nn.Parameter(torch.zeros_like(grad))
    setattr(tagged, PER_HEAD_SPEC_ATTR, _spec((Q_HEAD_DIM,)))
    out = wrapped(_Opt(), tagged, grad, lr=1.0)
    assert len(calls) == 1
    torch.testing.assert_close(out, orthogonalize_per_head(grad, _spec((Q_HEAD_DIM,)), _qr_orthogonalize))


# ===========================================================================
# 4. Against the real Newton-Schulz (needs emerging_optimizers)
# ===========================================================================


@pytest.fixture()
def high_matmul_precision():
    """Force fp32 Newton-Schulz.

    ``newton_schulz`` runs the whole iteration in **bf16** when
    ``torch.get_float32_matmul_precision() == "medium"`` (``muon_utils.py:140-150``),
    which is Megatron's default (``muon_fp32_matmul_prec: "medium"``). The spectral
    assertions below are about the algorithm, not about bf16 rounding.
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _real_scaled_orthogonalize(steps: int = 5, scale_mode: str = "spectral"):
    """``TensorParallelMuon``'s closure, rebuilt (``muon.py:67-90``)."""
    from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

    def fn(grad, _tp_group=None, partition_dim=None):
        assert partition_dim is None, "the TP path is not exercised by these tests"
        orth = newton_schulz(grad, steps=steps, coefficient_type="quintic")
        return orth * get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=scale_mode)

    return fn


def test_per_head_matches_a_manual_newton_schulz_loop(high_matmul_precision):
    """Claim 1 against the real orthogonalization, not a stub."""
    pytest.importorskip("emerging_optimizers")
    torch.manual_seed(3)
    grad = torch.randn(NUM_HEADS * Q_HEAD_DIM, Q_LORA_RANK)
    fn = _real_scaled_orthogonalize()

    got = orthogonalize_per_head(grad, _spec((Q_HEAD_DIM,)), fn)
    expected = torch.cat(
        [fn(grad[h * Q_HEAD_DIM : (h + 1) * Q_HEAD_DIM]) for h in range(NUM_HEADS)], dim=0
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_each_head_block_is_approximately_orthogonalized(high_matmul_precision):
    """Claim 2: singular values, not shapes.

    Newton-Schulz with the tuned quintic coefficients returns ``U S' V^T`` with
    ``S'`` "noisy values around 1" rather than exactly ``U V^T``
    (``muon_utils.py:83-85``), so this asserts a band around 1 plus a large
    improvement in conditioning over the input.
    """
    pytest.importorskip("emerging_optimizers")
    torch.manual_seed(4)
    grad = torch.randn(NUM_HEADS * KDA_HEAD_DIM, HIDDEN)
    fn = _real_scaled_orthogonalize(steps=10)
    out = orthogonalize_per_head(grad, _spec((KDA_HEAD_DIM,)), fn)

    scale = float(max(KDA_HEAD_DIM, HIDDEN)) ** 0.5
    for head in range(NUM_HEADS):
        block_out = out[head * KDA_HEAD_DIM : (head + 1) * KDA_HEAD_DIM] / scale
        sv_out = torch.linalg.svdvals(block_out.double())
        assert sv_out.min() > 0.5, (head, sv_out.min().item())
        assert sv_out.max() < 1.5, (head, sv_out.max().item())
        assert (sv_out.max() / sv_out.min()).item() < 2.0, (head, sv_out)


def _block_with_spectrum(rows: int, cols: int, singular_values: torch.Tensor) -> torch.Tensor:
    """A ``[rows, cols]`` matrix with exactly ``singular_values`` as its spectrum."""
    left, _ = torch.linalg.qr(torch.randn(rows, rows))
    right, _ = torch.linalg.qr(torch.randn(cols, rows))
    return left @ torch.diag(singular_values) @ right.mT


def test_per_head_orthogonalization_whitens_an_ill_conditioned_block(high_matmul_precision):
    """Claim 2, restated where it has teeth: a deliberately skewed spectrum.

    A random Gaussian ``[32, 1024]`` block already has a condition number near 1.4
    (Marchenko-Pastur), so "the output is well conditioned" says little about it.
    Feeding each head a spectrum spanning two orders of magnitude makes whitening the
    thing under test.
    """
    pytest.importorskip("emerging_optimizers")
    torch.manual_seed(9)
    spectrum = torch.logspace(0, -2, KDA_HEAD_DIM)
    blocks = [
        _block_with_spectrum(KDA_HEAD_DIM, HIDDEN, spectrum) for _ in range(NUM_HEADS)
    ]
    grad = torch.cat(blocks, dim=0)

    # 25 steps = 5 full cycles of the quintic schedule (muon_utils.py:135-136).
    out = orthogonalize_per_head(
        grad, _spec((KDA_HEAD_DIM,)), _real_scaled_orthogonalize(steps=25)
    )

    scale = float(max(KDA_HEAD_DIM, HIDDEN)) ** 0.5
    cond_in = (spectrum.max() / spectrum.min()).item()
    assert cond_in > 50.0, cond_in
    for head in range(NUM_HEADS):
        block_out = out[head * KDA_HEAD_DIM : (head + 1) * KDA_HEAD_DIM] / scale
        sv_out = torch.linalg.svdvals(block_out.double())
        cond_out = (sv_out.max() / sv_out.min()).item()
        assert cond_out < 2.0, (head, cond_out)
        assert cond_out < cond_in / 10.0, (head, cond_in, cond_out)


def test_per_head_differs_from_whole_matrix_under_unequal_head_scales(high_matmul_precision):
    """Claim 3, the report's actual claim, stated two ways.

    §2.5: "full-matrix orthogonalization treats all heads as a single coupled
    block, so heads with larger gradient or momentum scales dominate the shared
    update direction, while smaller-scale heads receive insufficiently normalized
    updates; per-head orthogonalization equalizes the update scale across heads."

    Newton-Schulz normalises its input by the **Frobenius norm of the whole
    matrix** (``muon_utils.py:121-124``), so under whole-matrix orthogonalization a
    head whose momentum is 1000x smaller enters the iteration with tiny singular
    values and comes out under-normalised.
    """
    pytest.importorskip("emerging_optimizers")
    torch.manual_seed(5)
    spec = _spec((KDA_HEAD_DIM,))
    grad = torch.randn(NUM_HEADS * KDA_HEAD_DIM, HIDDEN)
    # Head h gets scale 10**h: four orders of magnitude across 8 heads.
    for head in range(NUM_HEADS):
        grad[head * KDA_HEAD_DIM : (head + 1) * KDA_HEAD_DIM] *= 10.0 ** (head - NUM_HEADS // 2)

    fn = _real_scaled_orthogonalize(steps=10)
    per_head = orthogonalize_per_head(grad, spec, fn)
    whole = fn(grad)

    def block_norms(matrix):
        return torch.tensor(
            [
                matrix[h * KDA_HEAD_DIM : (h + 1) * KDA_HEAD_DIM].norm().item()
                for h in range(NUM_HEADS)
            ]
        )

    ph, wm = block_norms(per_head), block_norms(whole)
    # Per-head equalises: every head's update has essentially the same norm.
    assert (ph.max() / ph.min()).item() < 1.2, ph
    # Whole-matrix does not: the spread survives by orders of magnitude.
    assert (wm.max() / wm.min()).item() > 50.0, wm
    # And the two results are simply different.
    assert not torch.allclose(per_head, whole, rtol=1e-2, atol=1e-2)


def test_whole_matrix_couples_heads_and_per_head_does_not(high_matmul_precision):
    """The coupling claim, isolated: perturb head 0, look at head 1.

    Under per-head blocking head 1's output is a function of head 1's input alone,
    so it is **bit-identical**. Under whole-matrix orthogonalization it is not.
    """
    pytest.importorskip("emerging_optimizers")
    torch.manual_seed(6)
    spec = _spec((KDA_HEAD_DIM,))
    grad = torch.randn(NUM_HEADS * KDA_HEAD_DIM, HIDDEN)
    perturbed = grad.clone()
    perturbed[:KDA_HEAD_DIM] *= 100.0

    fn = _real_scaled_orthogonalize(steps=10)
    other = slice(KDA_HEAD_DIM, 2 * KDA_HEAD_DIM)

    ph_a = orthogonalize_per_head(grad, spec, fn)[other]
    ph_b = orthogonalize_per_head(perturbed, spec, fn)[other]
    torch.testing.assert_close(ph_a, ph_b, rtol=0, atol=0)

    wm_a, wm_b = fn(grad)[other], fn(perturbed)[other]
    assert not torch.allclose(wm_a, wm_b, rtol=1e-3, atol=1e-3)


def test_batched_implementation_matches_the_loop(high_matmul_precision):
    """The fast path must be the same algorithm, to fp32 round-off."""
    pytest.importorskip("emerging_optimizers")
    torch.manual_seed(7)
    fn = _real_scaled_orthogonalize(steps=5)
    ns_kwargs = {
        "steps": 5,
        "coefficient_type": "quintic",
        "scale_mode": "spectral",
        "extra_scale_factor": 1.0,
    }
    for rows, cols in [
        ((Q_HEAD_DIM,), Q_LORA_RANK),
        ((QK_HEAD_DIM, V_HEAD_DIM), KV_LORA_RANK),
        ((KDA_HEAD_DIM,), HIDDEN),
    ]:
        spec = _spec(rows)
        grad = torch.randn(NUM_HEADS * sum(rows), cols)
        loop = orthogonalize_per_head(grad, spec, fn, impl="loop")
        batched = orthogonalize_per_head(
            grad, spec, fn, impl="batched", batched_kwargs=ns_kwargs
        )
        torch.testing.assert_close(batched, loop, rtol=2e-4, atol=2e-4)


def test_batched_newton_schulz_matches_the_scalar_one(high_matmul_precision):
    """The batched kernel against upstream's ``newton_schulz``, head by head."""
    pytest.importorskip("emerging_optimizers")
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

    torch.manual_seed(8)
    for shape in [(NUM_HEADS, Q_HEAD_DIM, Q_LORA_RANK), (NUM_HEADS, HIDDEN, KDA_HEAD_DIM)]:
        stack = torch.randn(*shape)
        got = batched_newton_schulz(stack, steps=5, coefficient_type="quintic")
        for head in range(shape[0]):
            torch.testing.assert_close(
                got[head],
                newton_schulz(stack[head].contiguous(), steps=5, coefficient_type="quintic"),
                rtol=2e-4,
                atol=2e-4,
            )


def test_batched_newton_schulz_rejects_bad_input():
    pytest.importorskip("emerging_optimizers")
    with pytest.raises(ValueError, match="3-D stack"):
        batched_newton_schulz(torch.randn(4, 4), steps=5)
    with pytest.raises(ValueError, match="float32"):
        batched_newton_schulz(torch.randn(2, 4, 4, dtype=torch.float64), steps=5)
    with pytest.raises(ValueError, match="multiple"):
        batched_newton_schulz(torch.randn(2, 4, 4), steps=3, coefficient_type="quintic")


# ===========================================================================
# 5. Against a real Kimi K3 decoder block
# ===========================================================================


@pytest.fixture()
def mpu_tp1():
    """A 1-rank process group plus Megatron model-parallel state.

    Same fixture shape as ``test_kimi_k3_block.py:114-148``: ``KimiDeltaAttention``
    asserts it was given a ``pg_collection`` and every projection needs a TP group.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", world_size=1, rank=0
        )
        created = True
    try:
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
        if torch.cuda.is_available():
            model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if created:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def _real_k3_config():
    """The ``kimi_k3_debug.yaml`` geometry, narrowed the way the block tests narrow it."""
    import torch.nn.functional as F

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    hidden = 256
    return KimiK3TransformerConfig(
        num_layers=2,
        hidden_size=hidden,
        num_attention_heads=NUM_HEADS,
        ffn_hidden_size=512,
        kv_channels=V_HEAD_DIM,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        qk_pos_emb_head_dim=QK_POS_EMB_HEAD_DIM,
        rope_type="rope",
        mscale=1.0,
        mscale_all_dim=0.0,
        apply_rope_fusion=False,
        multi_latent_attention=False,
        linear_num_key_heads=NUM_HEADS,
        linear_num_value_heads=NUM_HEADS,
        linear_key_head_dim=KDA_HEAD_DIM,
        linear_value_head_dim=KDA_HEAD_DIM,
        linear_conv_kernel_dim=4,
        # One KDA layer and one full-attention layer: both parameter families.
        linear_attention_freq=[1, 0],
        kda_backend="eager",
        kda_chunk_size=64,
        attn_res_block_size=None,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        routed_expert_hidden_size=hidden // 2,
        latent_moe_use_norm=True,
        moe_layer_freq=[0, 1],
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-3,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_shared_expert_overlap=False,
        moe_permute_fusion=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        use_te_activation_func=True,
        bias_activation_fusion=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        params_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bf16=torch.cuda.is_available(),
        init_method_std=0.02,
        use_cpu_initialization=not torch.cuda.is_available(),
        perform_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the K3 decoder block builds TE modules and its causal mask on CUDA",
)
def test_selection_on_a_real_k3_decoder_block(mpu_tp1):
    """Claim 4 against real modules: the rule and every shape, from the source of truth.

    This is what makes the shape table in ``per_head_muon.py``'s docstring a checked
    claim rather than a transcription: the tensors come from
    :class:`KimiK3MLASelfAttention` and :class:`KimiDeltaAttention` themselves.
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )

    config = _real_k3_config()
    block = build_module(
        get_kimi_k3_runtime_decoder_spec(config),
        config=config,
        pre_process=True,
        post_process=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )

    named = list(block.named_parameters())
    assert named, "the block must expose parameters"
    summary = tag_per_head_params(named, config, PerHeadMuonConfig(enabled=True))

    # layer 0 is KDA (q/k/v_proj), layer 1 is MLA (linear_q_up_proj, linear_kv_up_proj).
    assert summary.by_rule() == {
        "kda.q_proj": 1,
        "kda.k_proj": 1,
        "kda.v_proj": 1,
        "mla.linear_q_up_proj": 1,
        "mla.linear_kv_up_proj.split_kv": 1,
    }, summary.by_rule()

    shapes = {name: tuple(p.shape) for name, p in named}
    q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim

    for name, spec in summary.selected.items():
        shape = shapes[name]
        assert spec.num_heads(shape) == NUM_HEADS, (name, shape)
        assert spec.head_axis == 0
        assert shape[0] == NUM_HEADS * spec.rows_per_head

    def only(leaf):
        matches = [n for n in shapes if n.split(".")[-2] == leaf]
        assert len(matches) == 1, (leaf, matches)
        return matches[0]

    # Every shape in the module docstring's table, checked against the real tensors.
    assert shapes[only("linear_q_up_proj")] == (NUM_HEADS * q_head_dim, config.q_lora_rank)
    assert shapes[only("linear_kv_up_proj")] == (
        NUM_HEADS * (config.qk_head_dim + config.v_head_dim),
        config.kv_lora_rank,
    )
    assert shapes[only("linear_q_down_proj")] == (config.q_lora_rank, config.hidden_size)
    assert shapes[only("linear_kv_down_proj")] == (
        config.kv_lora_rank + config.qk_pos_emb_head_dim,
        config.hidden_size,
    )
    assert shapes[only("linear_proj")] == (
        config.hidden_size,
        NUM_HEADS * config.v_head_dim,
    )
    assert shapes[only("linear_o_gate")] == (NUM_HEADS * config.v_head_dim, config.hidden_size)
    assert shapes[only("q_proj")] == (NUM_HEADS * config.linear_key_head_dim, config.hidden_size)
    assert shapes[only("v_proj")] == (NUM_HEADS * config.linear_value_head_dim, config.hidden_size)
    assert shapes[only("f_a_proj")] == (config.linear_key_head_dim, config.hidden_size)
    assert shapes[only("b_proj")] == (NUM_HEADS, config.hidden_size)
    assert shapes[only("o_proj")] == (config.hidden_size, NUM_HEADS * config.linear_value_head_dim)

    # Nothing outside attention is selected, and no expert/router/MLP tensor is.
    for name in summary.selected:
        assert ".self_attention." in name, name
    assert not [n for n in summary.selected if "experts" in n or "router" in n or ".mlp." in n]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the K3 decoder block builds TE modules and its causal mask on CUDA",
)
def test_with_the_option_off_a_real_block_selects_nothing(mpu_tp1):
    """Claim 5 against real modules.

    The switch is checked by the patch condition rather than inside the rule, so
    this asserts the operative guarantee: with the *default* config nothing is
    tagged, whatever the rule would say.
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )

    config = _real_k3_config()
    block = build_module(
        get_kimi_k3_runtime_decoder_spec(config),
        config=config,
        pre_process=True,
        post_process=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    named = list(block.named_parameters())
    assert named

    assert PerHeadMuonConfig().enabled is False
    summary = tag_per_head_params(named, config, PerHeadMuonConfig())
    assert summary.num_selected == 0
    for name, param in named:
        assert getattr(param, PER_HEAD_SPEC_ATTR, None) is None, name

    # And the same block with the switch on does select, so the assertion above is
    # about the default rather than about the rule failing to match.
    assert tag_per_head_params(named, config, PerHeadMuonConfig(enabled=True)).num_selected == 5
