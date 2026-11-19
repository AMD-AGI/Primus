###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Op models: registry, coverage (gap detection), blending into the profilers, validator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.core.projection.op_models import base, coverage, integration, registry
from primus.core.projection.op_models.base import (
    SLOT_ATTENTION,
    SLOT_ATTENTION_EXTRA,
    OpContext,
    OpCost,
    OpModel,
)
from primus.core.projection.op_models.validate import Workload, validate_op_model

_ROOT = Path(__file__).resolve().parents[4]
_QWEN35 = _ROOT / "examples/megatron/configs/MI355X/qwen3_5_35B_A3B-BF16-pretrain.yaml"


def _config(num_layers=8, tp=1, cp=1, **extra):
    mc = SimpleNamespace(
        hidden_size=1024,
        num_attention_heads=8,
        kv_channels=128,
        num_layers=num_layers,
        fp8=None,
        extra_args=dict(extra),
    )
    mp = SimpleNamespace(tensor_model_parallel_size=tp, context_model_parallel_size=cp)
    return SimpleNamespace(model_config=mc, model_parallel_config=mp)


class _Toy(OpModel):
    """A GEMM plus an elementwise stream; FLOPs and bytes counted exactly."""

    name = "toy_op"
    slot = SLOT_ATTENTION
    claims = {"toy_variant": ("toy",)}
    consumes = ("toy_width",)
    references = ("unit test",)
    frac = 0.5

    def layer_fraction(self, model_config):
        return self.frac

    def cost(self, ctx: OpContext) -> OpCost:
        w = int(ctx.value("toy_width", 512))
        f, b = ctx.linear(ctx.tokens, w, ctx.mc.hidden_size)
        nbytes = 3 * ctx.tokens * w * 2
        parts = {"proj": (f, b), "elem": (ctx.stream_ms(nbytes), ctx.stream_ms(2 * nbytes))}
        flops = 2.0 * ctx.tokens * w * ctx.mc.hidden_size
        return OpCost(
            fwd_ms=f + parts["elem"][0],
            bwd_ms=b + parts["elem"][1],
            fwd_flops=flops,
            bwd_flops=2 * flops,
            fwd_bytes=nbytes,
            bwd_bytes=2 * nbytes,
            num_params=ctx.mc.hidden_size * w,
            breakdown=parts,
        )

    def reference_flops(self, ctx):
        f = 2.0 * ctx.tokens * int(ctx.value("toy_width", 512)) * ctx.mc.hidden_size
        return f, 2 * f


@pytest.fixture
def toy():
    registry.register_op_model(_Toy)
    yield _Toy
    registry.unregister_op_model(_Toy.name)


# ---------------------------------------------------------------- registry


def test_register_rejects_bad_name_and_slot():
    class Bad(OpModel):
        name = "Bad-Name"

    with pytest.raises(ValueError):
        registry.register_op_model(Bad)

    class BadSlot(OpModel):
        name = "bad_slot"
        slot = "nowhere"

    with pytest.raises(ValueError):
        registry.register_op_model(BadSlot)


def test_active_op_models_matches_claims_and_respects_kill_switch(toy, monkeypatch):
    mc = _config(toy_variant="toy").model_config
    assert [o.name for o in registry.active_op_models(mc)] == ["toy_op"]
    assert registry.active_op_models(_config(toy_variant="other").model_config) == []
    monkeypatch.setenv("PRIMUS_OP_MODELS", "0")
    assert registry.active_op_models(mc) == []


def test_shipped_generated_ops_discover_cleanly():
    names = {c.name for c in registry.registered_op_models()}
    assert {"gated_delta_net", "attention_output_gate"} <= names
    assert not {k: v for k, v in registry.discovery_errors().items() if "generated" in k}


def test_interleave_fraction_uses_full_layer_count():
    mc = _config(num_layers=1, projection_full_num_layers=40).model_config
    assert base.interleave_fraction(mc, 4) == pytest.approx(30 / 40)


# ---------------------------------------------------------------- coverage


def test_unknown_selector_value_is_a_gap_and_absorbs_its_params():
    rep = coverage.detect_op_gaps(
        {"experimental_attention_variant": "brand_new_attn", "linear_key_head_dim": 64}, baseline={}
    )
    (gap,) = rep.blocking
    assert gap.op_id == "brand_new_attn" and gap.slot == SLOT_ATTENTION
    assert set(gap.keys) == {"experimental_attention_variant", "linear_key_head_dim"}


def test_registered_op_covers_its_keys(toy):
    cfg = {"toy_variant": "toy", "toy_width": 256}
    assert coverage.detect_op_gaps(cfg, baseline={}).blocking == []
    assert "toy_op" in coverage.detect_op_gaps(cfg, baseline={}).covered_by


def test_op_switch_without_reader_is_a_gap_and_inherited_defaults_are_not():
    rep = coverage.detect_op_gaps({"fancy_router_gate": True}, baseline={})
    assert [g.op_id for g in rep.blocking] == ["fancy"]
    assert (
        coverage.detect_op_gaps({"fancy_router_gate": True}, baseline={"fancy_router_gate": True}).blocking
        == []
    )


def test_same_slot_families_merge_into_one_gap():
    rep = coverage.detect_op_gaps(
        {"kda_use_full_rank_gate": True, "kda_chunk_size": 64, "linear_conv_kernel_dim": 4}, baseline={}
    )
    attn = [g for g in rep.gaps if g.slot == SLOT_ATTENTION]
    assert len(attn) == 1 and {"kda_chunk_size", "linear_conv_kernel_dim"} <= set(attn[0].keys)


def test_hybrid_pattern_symbols_name_the_op():
    rep = coverage.detect_op_gaps(
        {"hybrid_override_pattern": "*-M-M-*-", "is_hybrid_model": True}, baseline={}
    )
    assert [g.op_id for g in rep.blocking] == ["hybrid_mamba"]


def test_qwen35_gaps_are_closed_by_shipped_ops(monkeypatch):
    cfg = {
        "experimental_attention_variant": "gated_delta_net",
        "linear_attention_freq": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "attention_output_gate": True,
        "moe_shared_expert_gate": True,
    }
    assert coverage.detect_op_gaps(cfg, baseline={}).blocking == []
    monkeypatch.setenv("PRIMUS_OP_MODELS", "0")
    ids = {g.op_id for g in coverage.detect_op_gaps(cfg, baseline={}).blocking}
    assert {"gated_delta_net", "attention_output"} <= ids


def test_gap_markers_round_trip():
    rep = coverage.detect_op_gaps({"experimental_attention_variant": "brand_new_attn"}, baseline={})
    text = "noise\n" + "\n".join(g.marker_line() for g in rep.blocking) + "\nmore noise"
    (g,) = coverage.parse_gap_markers(text)
    assert g.op_id == "brand_new_attn" and g.keys == {"experimental_attention_variant": "brand_new_attn"}


# ---------------------------------------------------------------- integration


def test_attention_blend_replaces_fraction_and_adds_extras(toy):
    class Extra(_Toy):
        name = "toy_extra"
        slot = SLOT_ATTENTION_EXTRA
        frac = 0.25

    cfg = _config(toy_variant="toy")
    ctx = OpContext(cfg, 2, 1024)
    c = _Toy().cost(ctx)
    fwd, bwd = integration.simulate_attention_with_op_models(
        [_Toy(), Extra()], cfg, 2, 1024, None, None, built_in=lambda: (10.0, 20.0)
    )
    assert fwd == pytest.approx(0.5 * 10.0 + 0.5 * c.fwd_ms + 0.25 * c.fwd_ms)
    assert bwd == pytest.approx(0.5 * 20.0 + 0.5 * c.bwd_ms + 0.25 * c.bwd_ms)


def test_full_replacement_skips_the_built_in(toy):
    class Full(_Toy):
        name = "toy_full"
        frac = 1.0

    called = []
    integration.simulate_attention_with_op_models(
        [Full()],
        _config(toy_variant="toy"),
        1,
        128,
        None,
        None,
        built_in=lambda: called.append(1) or (1.0, 1.0),
    )
    assert called == []


def test_memory_terms_blend(toy):
    cfg = _config(toy_variant="toy")
    params = _Toy().cost(OpContext(cfg, 1, 1)).num_params
    got = integration.blended_memory_term([_Toy()], cfg, 1, 1, 1000.0, "num_params")
    assert got == pytest.approx(1000.0 + 0.5 * (params - 1000.0))


# ---------------------------------------------------------------- validator


def _wl(**extra):
    return Workload(_config(**extra), None, None, batch_size=2, seq_len=1024)


def test_validator_accepts_a_sound_op():
    rep = validate_op_model(_Toy, _wl(toy_variant="toy"), gap_keys=["toy_variant", "toy_width"])
    assert rep.ok, rep.feedback()


def test_validator_rejects_uncovered_keys_absent_op_and_beating_the_roofline():
    rep = validate_op_model(_Toy, _wl(toy_variant="toy"), gap_keys=["toy_variant", "toy_mystery_key"])
    assert {c.name for c in rep.failures()} == {"gap_keys_covered"}

    assert "matches" in {c.name for c in validate_op_model(_Toy, _wl()).failures()}

    class TooFast(_Toy):
        def cost(self, ctx):
            c = super().cost(ctx)
            c.fwd_ms, c.breakdown = c.fwd_ms / 10, {}
            return c

    names = {c.name for c in validate_op_model(TooFast, _wl(toy_variant="toy")).failures()}
    assert "roofline[workload]" in names


def test_validator_cross_checks_closed_form_flops():
    class Miscounted(_Toy):
        def reference_flops(self, ctx):
            f, b = super().reference_flops(ctx)
            return 2 * f, 2 * b

    names = {c.name for c in validate_op_model(Miscounted, _wl(toy_variant="toy")).failures()}
    assert "reference_flops[workload]" in names


@pytest.mark.skipif(not _QWEN35.is_file(), reason="Qwen3.5 workload not in this checkout")
@pytest.mark.parametrize("name", ["gated_delta_net", "attention_output_gate"])
def test_shipped_ops_validate_on_qwen35(name, monkeypatch):
    from primus.core.projection.op_models.validate import load_workload

    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("PRIMUS_OP_COVERAGE", "0")
    ops = {c.name: c for c in registry.registered_op_models()}
    wl = load_workload(str(_QWEN35), simulate_backends=False)
    rep = validate_op_model(ops[name], wl)
    assert rep.ok, rep.feedback()
