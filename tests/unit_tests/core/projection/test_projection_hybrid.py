###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Simulated hybrid (Mamba / GDN / KDA) stacks: mixer cost, layer mix and fused CE head."""

from types import SimpleNamespace

import pytest

from primus.core.projection.module_profilers import hybrid_mixer, language_model
from primus.core.projection.module_profilers.hybrid_mixer import (
    mixer_activation_bytes,
    mixer_kind,
    mixer_time_ms,
)
from primus.core.projection.module_profilers.language_model import LanguageModelProfiler
from primus.core.projection.module_profilers.loss import LossProfiler

SEQ = 8192


class _FlopsBackend:
    """GEMMs at a flat 500 TFLOP/s; 5.3 TB/s HBM; MI325X vector peak."""

    hbm_bandwidth_gbps = 5300.0

    def simulate_gemm(self, m, n, k, dtype):
        return SimpleNamespace(forward_time_ms=2.0 * m * n * k / 500e12 * 1e3)

    def vector_flops(self):
        return 304 * 2.1e9 * 256


@pytest.mark.parametrize(
    "spec, kind",
    [
        ("primus.backends.megatron.core.models.hybrid.hybrid_block_specs.gdn_hybrid_stack_spec", "gdn"),
        ("kda_hybrid_stack_spec", "kda"),
        (["primus.x", "hybrid_stack_spec"], "mamba"),
        (None, "mamba"),
    ],
)
def test_mixer_kind_from_spec(spec, kind):
    assert mixer_kind(spec) == kind


def _gdn():
    return SimpleNamespace(
        hybrid_mixer="gdn",
        hidden_size=2048,
        linear_num_key_heads=8,
        linear_key_head_dim=64,
        linear_num_value_heads=16,
        linear_value_head_dim=64,
    )


def test_mixer_time_scales_with_tokens_and_splits_with_tp():
    be = _FlopsBackend()
    f1, b1 = mixer_time_ms(_gdn(), SEQ, be)
    f2, b2 = mixer_time_ms(_gdn(), 2 * SEQ, be)
    assert f2 == pytest.approx(2 * f1, rel=1e-6) and b2 == pytest.approx(2 * b1, rel=1e-6)
    ft, bt = mixer_time_ms(_gdn(), SEQ, be, tp=2)
    assert ft == pytest.approx(f1 / 2, rel=1e-6) and bt == pytest.approx(b1 / 2, rel=1e-6)
    assert b1 > 2 * f1  # the delta-rule backward dominates
    assert mixer_activation_bytes(_gdn(), SEQ, tp=2) < mixer_activation_bytes(_gdn(), SEQ)


def test_mixer_defaults_follow_megatron():
    be = _FlopsBackend()
    explicit = SimpleNamespace(
        hybrid_mixer="kda",
        hidden_size=2048,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    implicit = SimpleNamespace(hybrid_mixer="kda", hidden_size=2048, linear_num_key_heads=None)
    assert mixer_time_ms(explicit, SEQ, be) == mixer_time_ms(implicit, SEQ, be)
    mamba = SimpleNamespace(hybrid_mixer="mamba", hidden_size=4096)
    assert hybrid_mixer._shapes(mamba, 1)[3] == "ssd"


# Origami's MI325X bf16 GEMM times (ms) for every (m, n, k) the benchmark cases
# issue. Replayed rather than simulated live: GEMM predictions shift between
# origami builds, and the benchmark margins are tighter than that drift.
_MI325X_GEMM_MS = {
    (128, 4096, 65536): 0.1117,
    (1024, 2048, 114688): 0.6621,
    (2048, 3104, 114688): 2.9603,
    (2048, 8480, 65536): 2.5167,
    (2048, 9280, 65536): 2.5079,
    (3072, 14384, 32768): 3.1360,
    (4096, 2048, 65536): 1.6768,
    (4096, 18496, 16384): 2.2821,
    (6144, 3072, 32768): 1.2517,
    (8192, 4096, 16384): 1.1331,
    (16384, 4096, 8192): 1.1076,
    (16384, 4096, 18496): 2.4750,
    (16384, 8192, 4096): 1.0125,
    (16384, 18496, 4096): 2.3288,
    (32768, 3072, 6144): 1.2888,
    (32768, 3072, 14384): 2.9875,
    (32768, 6144, 3072): 1.2032,
    (32768, 14384, 3072): 2.6516,
    (65536, 128, 4096): 0.1173,
    (65536, 2048, 4096): 1.0125,
    (65536, 2048, 8480): 2.0723,
    (65536, 2048, 9280): 2.2535,
    (65536, 4096, 128): 0.1222,
    (65536, 4096, 2048): 1.0399,
    (65536, 8480, 2048): 2.1953,
    (65536, 9280, 2048): 2.4113,
    (114688, 1024, 2048): 0.4484,
    (114688, 2048, 1024): 0.4718,
    (114688, 2048, 3104): 1.3437,
    (114688, 3104, 2048): 1.5594,
}


class _MI325XBackend(_FlopsBackend):
    """Recorded MI325X GEMM times; 6 TB/s HBM; MI325X vector peak."""

    hbm_bandwidth_gbps = 6000.0

    def simulate_gemm(self, m, n, k, dtype):
        assert dtype == "bf16"
        return SimpleNamespace(forward_time_ms=_MI325X_GEMM_MS[(m, n, k)])


# Per-layer MI325X benchmark times (fwd, bwd ms) of the ``M`` layers, which
# include half of the layer's norm + residual traffic (6 fwd / 7 bwd passes).
_MEASURED = [
    (_gdn(), 14, 8.43, 18.67),
    (SimpleNamespace(hybrid_mixer="kda", hidden_size=2048), 8, 20.34, 55.81),
    (
        SimpleNamespace(hybrid_mixer="mamba", hidden_size=2048, mamba_state_dim=64, mamba_head_dim=64),
        8,
        6.06,
        18.44,
    ),
    (
        SimpleNamespace(hybrid_mixer="mamba", hidden_size=3072, mamba_state_dim=128, mamba_head_dim=128),
        4,
        5.78,
        17.37,
    ),
    (
        SimpleNamespace(hybrid_mixer="mamba", hidden_size=4096, mamba_state_dim=128, mamba_head_dim=128),
        2,
        4.26,
        12.39,
    ),
]


@pytest.mark.parametrize("mc, batch, fwd_ms, bwd_ms", _MEASURED)
def test_mixer_matches_mi325x_benchmarks(mc, batch, fwd_ms, bwd_ms):
    from primus.core.projection.module_profilers.moe_mlp import _ACTIVATION_BW_FRACTION

    be = _MI325XBackend()
    tokens = batch * SEQ
    fwd, bwd = mixer_time_ms(mc, tokens, be)
    pass_ms = tokens * mc.hidden_size * 2 / (be.hbm_bandwidth_gbps * _ACTIVATION_BW_FRACTION * 1e6)
    assert fwd + 6 * pass_ms == pytest.approx(fwd_ms, rel=0.12)
    assert bwd + 7 * pass_ms == pytest.approx(bwd_ms, rel=0.12)


class _Sub:
    def __init__(self, fwd, bwd, mem=0):
        self.fwd, self.bwd, self.mem = fwd, bwd, mem

    def measured_forward_time(self, b, s):
        return self.fwd

    def measured_backward_time(self, b, s):
        return self.bwd

    def measured_activation_memory(self, b, s):
        return self.mem

    estimated_activation_memory = measured_activation_memory


class _Layer:
    def __init__(self, **subs):
        self.subs = subs

    def get_sub_profiler(self, name):
        return self.subs[name]


def _hybrid_profiler(monkeypatch, recompute=None):
    monkeypatch.setattr(language_model, "_layer_ln_res_ms", lambda *a: (2.0, 4.0))
    monkeypatch.setattr(language_model, "_estimate_tp_allreduce_time_ms", lambda *a: 0.0)
    monkeypatch.setattr(language_model, "mixer_time_ms", lambda mc, tokens, be, tp=1: (3.0, 9.0))
    monkeypatch.setattr(language_model, "mixer_activation_bytes", lambda mc, tokens, tp=1: 100)
    prof = LanguageModelProfiler.__new__(LanguageModelProfiler)
    prof.config = SimpleNamespace(
        model_config=SimpleNamespace(hybrid_mixer="gdn", hidden_size=2048),
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=1, context_parallel_size=1, recompute_granularity=recompute
        ),
    )
    prof.sub_profilers = {
        "dense_transformer_layer": _Layer(
            self_attention=_Sub(5.0, 10.0, 10), mlp=_Sub(7.0, 14.0, 20), layer_norm=_Sub(0, 0, 1)
        )
    }
    prof.layers = [0, 1, 2]
    prof._gemm_backend = None
    return prof


def test_hybrid_simulation_weights_single_purpose_layers(monkeypatch):
    prof = _hybrid_profiler(monkeypatch)
    res = prof._run_hybrid_layer_simulation(["M", "M", "-", "-", "*", "M", "-", "-"], 2, SEQ)
    assert sorted(res) == [0, 1, 2]
    kinds = res[0]["hybrid_kinds"]
    # Each kind: its own sublayer plus half the layer's norm/residual cost.
    assert kinds["M"] == {"forward_time_ms": 4.0, "backward_time_ms": 11.0, "count": 3}
    assert kinds["-"] == {"forward_time_ms": 8.0, "backward_time_ms": 16.0, "count": 4}
    assert kinds["*"] == {"forward_time_ms": 6.0, "backward_time_ms": 12.0, "count": 1}
    assert res[0]["forward_time_ms"] == pytest.approx((3 * 4 + 4 * 8 + 6) / 8)
    assert res[0]["backward_time_ms"] == pytest.approx((3 * 11 + 4 * 16 + 12) / 8)
    assert res[0]["activation_memory_bytes"] == (3 * 101 + 4 * 21 + 11) // 8


def test_hybrid_simulation_selective_recompute_reruns_attention(monkeypatch):
    prof = _hybrid_profiler(monkeypatch, recompute="selective")
    res = prof._run_hybrid_layer_simulation(["*", "-"], 2, SEQ)
    assert res[0]["hybrid_kinds"]["*"]["backward_time_ms"] == 10.0 + 5.0 + 2.0
    assert res[0]["hybrid_kinds"]["-"]["backward_time_ms"] == 14.0 + 2.0


def test_hybrid_simulation_rejects_unknown_kind(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported hybrid layer kind"):
        _hybrid_profiler(monkeypatch)._run_hybrid_layer_simulation(["M", "X"], 2, SEQ)


def _loss(**model):
    mc = SimpleNamespace(padded_vocab_size=128256, hidden_size=2048, **model)
    pc = SimpleNamespace(tensor_model_parallel_size=1, context_model_parallel_size=1)
    return LossProfiler(SimpleNamespace(model_config=mc, model_parallel_config=pc))


def test_fla_fused_ce_prices_one_forward_pass_and_no_backward():
    dense = _loss(hybrid_layer_types=None)
    fused = _loss(hybrid_layer_types=["M", "-"])
    off = _loss(hybrid_layer_types=["M", "-"], fused_ce_mode=0)
    assert not dense.fla_fused() and fused.fla_fused() and not off.fla_fused()
    f_dense, b_dense = dense._loss_time_ms(2, SEQ)
    assert fused._loss_time_ms(2, SEQ) == (f_dense, 0.0)
    assert off._loss_time_ms(2, SEQ) == (f_dense, b_dense)
    assert fused.estimated_activation_memory(2, SEQ) == 0
    assert off.estimated_activation_memory(2, SEQ) > 0


def test_mixer_kind_honours_and_validates_an_explicit_mixer():
    assert mixer_kind("gdn_hybrid_stack_spec", "KDA") == "kda"
    with pytest.raises(ValueError, match="hybrid_mixer must be one of"):
        mixer_kind(None, "rwkv")
    with pytest.warns(UserWarning, match="names no known mixer"):
        assert mixer_kind("my_custom_stack_spec") == "mamba"


@pytest.mark.parametrize(
    "mc",
    [
        _gdn(),
        SimpleNamespace(hybrid_mixer="mamba", hidden_size=2048, mamba_state_dim=64, mamba_head_dim=64),
    ],
)
def test_mixer_keeps_one_head_per_rank_when_tp_exceeds_heads(mc):
    # GDN has 8 key heads and Mamba 64 heads: TP 128 leaves each rank one head.
    at_heads = mixer_time_ms(mc, 4096, None, tp=64)
    beyond = mixer_time_ms(mc, 4096, None, tp=128)
    assert all(t > 0 for t in (*at_heads, *beyond))
    assert mixer_activation_bytes(mc, 4096, tp=128) > 0


def test_mixer_elementwise_passes_drop_at_the_fusion_ceiling():
    from primus.core.projection.ceiling import ceiling_levers

    expected = mixer_time_ms(_gdn(), 8192, None)
    with ceiling_levers(("fusion",)):
        fused = mixer_time_ms(_gdn(), 8192, None)
    assert fused[0] < expected[0] and fused[1] < expected[1]


def _hybrid_module(pattern):
    return SimpleNamespace(hybrid_layer_pattern=pattern, num_layers=len(pattern), moe_layer_freq=1)


def test_max_layers_cap_never_drops_a_layer_kind(monkeypatch):
    from primus.core.projection.performance_projection.projection import (
        _limit_hybrid_layers_for_projection,
    )

    monkeypatch.setenv("PRIMUS_PROJ_MAX_LAYERS", "1")
    mc = _hybrid_module("M-M-*-M-")
    assert _limit_hybrid_layers_for_projection(mc)
    assert mc.hybrid_layer_pattern == "M-M-*" and mc.num_layers == 5
    monkeypatch.setenv("PRIMUS_PROJ_MAX_LAYERS", "7")
    mc = _hybrid_module("M-M-*-M-")
    _limit_hybrid_layers_for_projection(mc)
    assert mc.num_layers == 7


def test_hybrid_stack_without_layers_is_a_clear_error():
    from primus.core.projection.performance_projection.projection import (
        _limit_hybrid_layers_for_projection,
    )

    mc = SimpleNamespace(is_hybrid_model=True, num_layers=1, hybrid_attention_ratio=0.0)
    with pytest.raises(ValueError, match="no decoder layers"):
        _limit_hybrid_layers_for_projection(mc)
