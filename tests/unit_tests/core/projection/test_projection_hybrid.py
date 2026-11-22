###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Simulated hybrid (Mamba / GDN / KDA) stacks: mixer cost, layer mix and fused CE head."""

from types import SimpleNamespace

import pytest

from primus.core.projection.module_profilers import hybrid_mixer, transformer_layer
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


def _mi325x_backend():
    try:
        from primus.core.projection.simulation_backends.factory import (
            get_gemm_simulation_backend,
        )

        return get_gemm_simulation_backend(gpu_arch="mi325x")
    except Exception as exc:  # pragma: no cover - origami not installed
        pytest.skip(f"MI325X GEMM backend unavailable: {exc}")


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

    be = _mi325x_backend()
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
    monkeypatch.setattr(transformer_layer, "_layer_ln_res_ms", lambda *a: (2.0, 4.0))
    monkeypatch.setattr(transformer_layer, "_estimate_tp_allreduce_time_ms", lambda *a: 0.0)
    monkeypatch.setattr(hybrid_mixer, "mixer_time_ms", lambda mc, tokens, be, tp=1: (3.0, 9.0))
    monkeypatch.setattr(hybrid_mixer, "mixer_activation_bytes", lambda mc, tokens, tp=1: 100)
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
