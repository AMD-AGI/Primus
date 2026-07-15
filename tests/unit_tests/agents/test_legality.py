###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for the tuning-agent legality knowledge base (``legality.py``).

``legality.py`` is the deterministic core the LLM is constrained by: it
derives per-axis legal value sets for a (model, cluster) pair and validates
candidate :class:`TrialConfig`s before they reach the projection tool.  These
tests cover the helper math, the ``TrialConfig`` (de)serialisation contract,
``derive_legality`` for dense + MoE, ``derived_dp``, and the many ``validate``
rejection rules.
"""

from __future__ import annotations

import pytest

pytest.importorskip("primus.agents.tuning_agent.legality")

from primus.agents.tuning_agent.config import TargetCluster  # noqa: E402
from primus.agents.tuning_agent.legality import (  # noqa: E402
    TrialConfig,
    _divisors,
    _powers_of_two,
    derive_legality,
    derived_dp,
    fill_defaults_from_baseline,
    validate,
)
from primus.agents.tuning_agent.workload import ArchitectureRecord  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / builders
# ─────────────────────────────────────────────────────────────────────────────


def _dense_arch(**kw) -> ArchitectureRecord:
    base = dict(
        model_name="dense",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        seq_length=4096,
        is_moe=False,
        global_batch_size=128,
        micro_batch_size=2,
    )
    base.update(kw)
    return ArchitectureRecord(**base)


def _moe_arch(**kw) -> ArchitectureRecord:
    base = dict(
        model_name="moe",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        seq_length=4096,
        is_moe=True,
        num_experts=8,
        moe_router_topk=2,
        global_batch_size=128,
        micro_batch_size=2,
    )
    base.update(kw)
    return ArchitectureRecord(**base)


def _cluster(num_nodes=1, gpus_per_node=8) -> TargetCluster:
    return TargetCluster(num_nodes=num_nodes, gpus_per_node=gpus_per_node, gpu_arch="mi355x")


# ─────────────────────────────────────────────────────────────────────────────
# Helper math
# ─────────────────────────────────────────────────────────────────────────────


def test_divisors():
    assert _divisors(8) == [1, 2, 4, 8]
    assert _divisors(12, max_val=4) == [1, 2, 3, 4]
    # Degenerate input falls back to [1].
    assert _divisors(0) == [1]
    assert _divisors(-5) == [1]


def test_powers_of_two():
    assert _powers_of_two(16) == [1, 2, 4, 8, 16]
    assert _powers_of_two(1) == [1]
    # Non-power ceiling stops at the largest power <= max.
    assert _powers_of_two(10) == [1, 2, 4, 8]


# ─────────────────────────────────────────────────────────────────────────────
# TrialConfig (de)serialisation
# ─────────────────────────────────────────────────────────────────────────────


def test_trialconfig_from_dict_coerces_null_strings():
    cfg = TrialConfig.from_dict(
        {
            "tp": "2",
            "pp": 4,
            "ep": 8,
            "vpp": "null",
            "enable_zero_bubble": "true",
            "use_torch_fsdp2": "false",
            "fp8": "null",
            "sync_free_stage": "",
        }
    )
    assert cfg.tp == 2 and cfg.pp == 4 and cfg.ep == 8
    assert cfg.vpp is None  # "null" → None
    assert cfg.enable_zero_bubble is True
    assert cfg.use_torch_fsdp2 is False
    assert cfg.fp8 is None  # "null" → None
    assert cfg.sync_free_stage is None  # "" → None


def test_trialconfig_as_dict_roundtrip():
    cfg = TrialConfig(tp=2, pp=4, ep=8, cp=1, mbs=2, gbs=128, vpp=2, fp8="hybrid")
    again = TrialConfig.from_dict(cfg.as_dict())
    assert again.as_dict() == cfg.as_dict()


def test_trialconfig_signature_is_order_independent():
    cfg = TrialConfig(tp=1, pp=2, ep=4)
    # Signature is a sorted-key string; rebuilding from its dict yields the
    # same signature regardless of insertion order.
    d = cfg.as_dict()
    shuffled = {k: d[k] for k in reversed(list(d))}
    assert TrialConfig.from_dict(shuffled).signature() == cfg.signature()


# ─────────────────────────────────────────────────────────────────────────────
# derive_legality
# ─────────────────────────────────────────────────────────────────────────────


def test_derive_legality_dense_axes():
    arch = _dense_arch()
    leg = derive_legality(arch, _cluster(gpus_per_node=8))
    # TP divides both heads(32) and hidden(4096), capped at gpus_per_node=8.
    assert leg.tp == [1, 2, 4, 8]
    # PP divides num_layers=32 and <= world=8.
    assert set(leg.pp) <= {1, 2, 4, 8}
    assert 8 in leg.pp
    # Dense → EP is trivially [1].
    assert leg.ep == [1]
    assert "auto" in leg.pp_schedules_by_vpp[1]
    assert "none" in leg.recompute_granularity


def test_derive_legality_moe_ep_and_cp_folding():
    arch = _moe_arch(num_experts=8)
    leg = derive_legality(arch, _cluster(num_nodes=1, gpus_per_node=8))
    # EP divides num_experts=8, <= world=8.
    assert leg.ep == [1, 2, 4, 8]
    # MoE CP is folded into EP (CP <= EP).
    assert set(leg.cp) <= set(leg.ep)


# ─────────────────────────────────────────────────────────────────────────────
# derived_dp
# ─────────────────────────────────────────────────────────────────────────────


def test_derived_dp_moe_uses_ep():
    cfg = TrialConfig(tp=1, pp=2, ep=4, cp=1)
    arch = _moe_arch()
    cluster = _cluster(num_nodes=4, gpus_per_node=8)  # world=32
    # MoE denom = tp*pp*ep = 8 → dp = 32//8 = 4.
    assert derived_dp(cfg, arch, cluster) == 4


def test_derived_dp_dense_uses_cp():
    cfg = TrialConfig(tp=2, pp=2, ep=1, cp=2)
    arch = _dense_arch()
    cluster = _cluster(num_nodes=2, gpus_per_node=8)  # world=16
    # Dense denom = tp*pp*cp = 8 → dp = 16//8 = 2.
    assert derived_dp(cfg, arch, cluster) == 2


# ─────────────────────────────────────────────────────────────────────────────
# validate — happy path + rejection rules
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_accepts_legal_dense_config():
    arch = _dense_arch()
    cluster = _cluster(gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    cfg = TrialConfig(tp=2, pp=2, ep=1, cp=1, mbs=2, gbs=128)
    ok, reason = validate(cfg, arch, cluster, leg)
    assert ok, reason


def test_validate_rejects_axis_out_of_legal_set():
    arch = _dense_arch()
    cluster = _cluster(gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    cfg = TrialConfig(tp=3, pp=2, mbs=2, gbs=128)  # 3 not a legal TP
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "TP=3" in reason


def test_validate_rejects_gbs_not_divisible():
    arch = _dense_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    # tp*pp*cp = 1 → dp = 8.  gbs=100 not divisible by mbs(2)*dp(8)=16.
    cfg = TrialConfig(tp=1, pp=1, ep=1, cp=1, mbs=2, gbs=100)
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "divisible" in reason


def test_validate_rejects_too_many_gpus_per_replica():
    arch = _dense_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)  # world=8
    leg = derive_legality(arch, cluster)
    # Force a legal-set TP/PP but tp*pp*cp = 8*... exceed world via pp.
    cfg = TrialConfig(tp=8, pp=8, ep=1, cp=1, mbs=1, gbs=8)
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    # Either the per-replica GPU check or DP<=0 fires; both indicate the
    # config cannot be placed on the cluster.
    assert "GPUs per replica" in reason or "DP" in reason


def test_validate_moe_folding_rules():
    arch = _moe_arch(num_experts=8)
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    # CP must divide EP and be <= EP.  ep=2, cp=4 violates both.
    cfg = TrialConfig(tp=1, pp=1, ep=2, cp=4, mbs=1, gbs=8)
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "folding" in reason


def test_validate_fsdp2_and_distopt_mutually_exclusive():
    arch = _dense_arch()
    cluster = _cluster(gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    cfg = TrialConfig(
        tp=1,
        pp=1,
        ep=1,
        cp=1,
        mbs=2,
        gbs=16,
        use_torch_fsdp2=True,
        use_distributed_optimizer=True,
    )
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "mutually" in reason


def test_validate_syncfree_stage_requires_deepep_consistency():
    arch = _moe_arch(num_experts=8)
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    # stage>=2 auto-enables DeepEP; explicitly False is contradictory.
    cfg = TrialConfig(
        tp=1,
        pp=1,
        ep=8,
        cp=1,
        mbs=2,
        gbs=16,
        sync_free_stage=2,
        use_turbo_deepep=False,
    )
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "DeepEP" in reason


def test_validate_deepep_rejected_for_dense():
    arch = _dense_arch()
    cluster = _cluster(gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    cfg = TrialConfig(tp=1, pp=1, ep=1, cp=1, mbs=2, gbs=16, use_turbo_deepep=True)
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "MoE" in reason


def test_validate_rejects_bad_fp8_value():
    arch = _dense_arch()
    cluster = _cluster(gpus_per_node=8)
    leg = derive_legality(arch, cluster)
    cfg = TrialConfig(tp=1, pp=1, ep=1, cp=1, mbs=2, gbs=16, fp8="bogus")
    ok, reason = validate(cfg, arch, cluster, leg)
    assert not ok
    assert "fp8" in reason


def test_validate_target_ep_size_positive_and_moe_only():
    cluster = _cluster(gpus_per_node=8)
    # Non-positive on MoE.
    moe = _moe_arch()
    leg_moe = derive_legality(moe, cluster)
    cfg = TrialConfig(tp=1, pp=1, ep=8, cp=1, mbs=2, gbs=16, target_ep_size=0)
    ok, reason = validate(cfg, moe, cluster, leg_moe)
    assert not ok and "target_ep_size must be positive" in reason

    # Positive but on a dense model.
    dense = _dense_arch()
    leg_dense = derive_legality(dense, cluster)
    cfg2 = TrialConfig(tp=1, pp=1, ep=1, cp=1, mbs=2, gbs=16, target_ep_size=4)
    ok2, reason2 = validate(cfg2, dense, cluster, leg_dense)
    assert not ok2 and "MoE" in reason2


# ─────────────────────────────────────────────────────────────────────────────
# fill_defaults_from_baseline
# ─────────────────────────────────────────────────────────────────────────────


def test_fill_defaults_from_baseline():
    arch = _dense_arch(global_batch_size=256, micro_batch_size=4)
    cfg = TrialConfig(gbs=1, mbs=0)  # both unset / sentinel
    out = fill_defaults_from_baseline(cfg, arch)
    assert out.gbs == 256
    assert out.mbs == 4
