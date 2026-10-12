###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the projection's JAX/MaxText benchmark path.

The measurement core is exercised against real JAX on CPU -- timing, the vjp
backward, and XLA's activation accounting all work there, so the parts most
likely to be wrong are actually run rather than mocked.

The MaxText runner's own logic -- picking the dense and MoE blocks apart,
apportioning a layer total, shaping the artifact -- is tested directly, since
building a MaxText layer needs the MaxText checkout and an accelerator.
"""

from types import SimpleNamespace

import pytest

from primus.core.projection.bench_harness import (
    available_bench_frameworks,
    get_bench_runner,
    register_bench_runner,
)
from primus.core.projection.bench_harness.jax_timing import JaxMeasurement, _median
from primus.core.projection.bench_harness.maxtext import MaxTextLayerBench
from primus.core.projection.frameworks.jax import maxtext_apply_bench_overrides

jax = pytest.importorskip("jax", reason="JAX bench path needs jax installed")
jnp = pytest.importorskip("jax.numpy")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_maxtext_and_jax_are_benchmarkable():
    frameworks = available_bench_frameworks()
    assert "maxtext" in frameworks
    assert "jax" in frameworks


def test_maxtext_uses_a_self_contained_runner():
    """Its Flax layers have no autograd to hook, so it measures itself."""
    assert isinstance(get_bench_runner("maxtext"), MaxTextLayerBench)
    assert isinstance(get_bench_runner("jax"), MaxTextLayerBench)


def test_torch_backends_have_no_runner():
    """Megatron and TorchTitan go through the shared torch harness instead."""
    assert get_bench_runner("megatron") is None
    assert get_bench_runner("torchtitan") is None


def test_runner_registry_requires_a_run_method():
    with pytest.raises(TypeError):
        register_bench_runner("bogus", object())


# ---------------------------------------------------------------------------
# Measurement core, against real JAX
# ---------------------------------------------------------------------------


def _mlp_forward(params, x):
    return jnp.tanh(x @ params["w1"]) @ params["w2"]


def _mlp_params(hidden=64, ffn=128):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (hidden, ffn), jnp.float32),
        "w2": jax.random.normal(k2, (ffn, hidden), jnp.float32),
    }


def _activations(batch=2, seq=16, hidden=64):
    return jax.random.normal(jax.random.key(1), (batch, seq, hidden), jnp.float32)


@pytest.fixture
def measurement():
    from primus.core.projection.bench_harness.jax_timing import benchmark_jax_module

    return benchmark_jax_module(_mlp_forward, _mlp_params(), [_activations()], num_warmup=1, num_iters=3)


def test_forward_and_backward_are_both_timed(measurement):
    assert measurement.forward_ms > 0
    assert measurement.backward_ms > 0


def test_backward_is_the_difference_of_two_executables_and_never_negative(measurement):
    """Forward and backward cannot be split inside one XLA executable.

    Backward is reported as the forward+backward time minus the forward's, so
    the one thing to guarantee is that host-side jitter on a fast kernel can
    never turn it into a negative number.
    """
    assert measurement.backward_ms >= 0.0


def test_timing_grows_with_the_workload():
    from primus.core.projection.bench_harness.jax_timing import benchmark_jax_module

    def forward_ms(seq):
        return benchmark_jax_module(
            _mlp_forward, _mlp_params(), [_activations(seq=seq)], num_warmup=2, num_iters=5
        ).forward_ms

    assert forward_ms(512) > forward_ms(16)


def test_activation_bytes_match_xlas_own_accounting():
    """XLA reports scratch plus output; for this MLP that is the tanh and the result."""
    from primus.core.projection.bench_harness.jax_timing import benchmark_jax_module

    batch, seq, hidden, ffn = 2, 16, 64, 128
    measured = benchmark_jax_module(
        _mlp_forward,
        _mlp_params(hidden, ffn),
        [_activations(batch, seq, hidden)],
        num_warmup=1,
        num_iters=2,
    )
    # The output alone is batch*seq*hidden float32; anything less means the
    # analysis was not read at all.
    assert measured.activation_bytes >= batch * seq * hidden * 4


def test_activation_bytes_grow_with_the_workload():
    from primus.core.projection.bench_harness.jax_timing import benchmark_jax_module

    def measure(seq):
        return benchmark_jax_module(
            _mlp_forward, _mlp_params(), [_activations(seq=seq)], num_warmup=1, num_iters=2
        ).activation_bytes

    assert measure(64) > measure(16)


def test_a_fixed_cotangent_is_used_not_a_summed_loss():
    """An all-ones cotangent lets XLA fold away work a real step would do."""
    from primus.core.projection.bench_harness.jax_timing import benchmark_jax_module

    cotangent = jnp.ones((2, 16, 64), jnp.float32)
    measured = benchmark_jax_module(
        _mlp_forward,
        _mlp_params(),
        [_activations()],
        cotangent=cotangent,
        num_warmup=1,
        num_iters=2,
    )
    assert measured.backward_ms > 0


def test_device_memory_snapshot_has_the_torch_harness_shape():
    """The memory extrapolator reads one shape; JAX must not need a special case."""
    from primus.core.projection.bench_harness.jax_timing import device_memory_snapshot

    snapshot = device_memory_snapshot("post_setup")
    assert snapshot["label"] == "post_setup"
    for field in (
        "allocated_bytes",
        "reserved_bytes",
        "max_allocated_bytes",
        "max_reserved_bytes",
        "free_bytes",
        "total_bytes",
    ):
        assert isinstance(snapshot[field], int)


@pytest.mark.parametrize(
    "values,expected",
    [([], 0.0), ([5.0], 5.0), ([3.0, 1.0, 2.0], 2.0), ([4.0, 1.0, 2.0, 3.0], 2.5)],
)
def test_median_ignores_ordering_and_handles_even_counts(values, expected):
    assert _median(values) == expected


def test_iteration_counts_are_overridable(monkeypatch):
    from primus.core.projection.bench_harness.jax_timing import _iter_counts

    monkeypatch.setenv("PRIMUS_BENCH_JAX_WARMUP", "2")
    monkeypatch.setenv("PRIMUS_BENCH_JAX_ITERS", "7")
    assert _iter_counts() == (2, 7)


# ---------------------------------------------------------------------------
# MaxText runner logic
# ---------------------------------------------------------------------------


def test_a_single_decoder_class_serves_dense_and_moe_layers():
    """Most MaxText architectures use one block for the whole stack."""
    dense, moe = MaxTextLayerBench()._classify_layer_classes(["OnlyBlock"])
    assert dense == "OnlyBlock"
    assert moe == "OnlyBlock"


def test_two_decoder_classes_are_dense_then_moe():
    """DeepSeek returns its dense block first, then its MoE block."""
    dense, moe = MaxTextLayerBench()._classify_layer_classes(["DenseBlock", "MoEBlock"])
    assert dense == "DenseBlock"
    assert moe == "MoEBlock"


def test_no_decoder_classes_yields_nothing_to_benchmark():
    assert MaxTextLayerBench()._classify_layer_classes([]) == (None, None)


def _profiler_with_split(attention_ms, mlp_ms):
    layer = SimpleNamespace(
        get_sub_profiler=lambda name: SimpleNamespace(
            estimated_forward_time=lambda b, s: attention_ms if name == "self_attention" else mlp_ms
        )
    )
    return SimpleNamespace(sub_profilers={"dense_transformer_layer": layer})


def test_layer_halves_are_apportioned_by_the_analytical_ratio():
    """One traced graph has no sub-module to time, so the total is divided."""
    bench = MaxTextLayerBench()
    measurement = JaxMeasurement(forward_ms=10.0, backward_ms=20.0, activation_bytes=1000)

    entry = bench._layer_entry(measurement, "dense", _profiler_with_split(3.0, 1.0), 1, 128)

    assert entry["forward_time_ms"] == 10.0
    assert entry["attention"]["forward_time_ms"] == pytest.approx(7.5)
    assert entry["mlp"]["forward_time_ms"] == pytest.approx(2.5)
    assert entry["attention"]["backward_time_ms"] == pytest.approx(15.0)


def test_apportioned_halves_are_marked_as_estimates():
    """Nothing downstream should read them as separate measurements."""
    bench = MaxTextLayerBench()
    entry = bench._layer_entry(JaxMeasurement(1.0, 2.0, 16), "dense", _profiler_with_split(1.0, 1.0), 1, 128)
    assert entry["estimated_split"] is True


def test_halves_sum_to_the_measured_total():
    bench = MaxTextLayerBench()
    measurement = JaxMeasurement(forward_ms=9.0, backward_ms=18.0, activation_bytes=900)
    entry = bench._layer_entry(measurement, "moe", _profiler_with_split(2.0, 1.0), 1, 128)

    halves = entry["attention"]["forward_time_ms"] + entry["mlp"]["forward_time_ms"]
    assert halves == pytest.approx(entry["forward_time_ms"])


def test_split_falls_back_to_even_without_a_profiler():
    """A missing estimate degrades the breakdown, never the measurement."""
    bench = MaxTextLayerBench()
    entry = bench._layer_entry(JaxMeasurement(10.0, 10.0, 100), "dense", None, 1, 128)
    assert entry["attention"]["forward_time_ms"] == pytest.approx(5.0)
    assert entry["forward_time_ms"] == 10.0


def test_expert_all_to_all_is_left_to_the_analytical_model():
    """MaxText expresses EP as sharding; there is no call to time."""
    bench = MaxTextLayerBench()
    entry = bench._layer_entry(JaxMeasurement(1.0, 1.0, 8), "moe", None, 1, 128)
    assert entry["mlp"]["a2a_forward_time_ms"] == 0.0
    assert entry["mlp"]["a2a_backward_time_ms"] == 0.0


def test_memory_payload_matches_the_torch_harness_shape():
    bench = MaxTextLayerBench()
    snapshots = [
        {
            "label": "post_setup",
            "allocated_bytes": 10,
            "reserved_bytes": 20,
            "max_allocated_bytes": 30,
            "max_reserved_bytes": 40,
        },
        {
            "label": "post_layer_benchmark",
            "allocated_bytes": 15,
            "reserved_bytes": 25,
            "max_allocated_bytes": 60,
            "max_reserved_bytes": 80,
        },
    ]

    payload = bench._memory_payload(snapshots)

    assert payload["global_peak_allocated_bytes"] == 60
    assert payload["global_peak_reserved_bytes"] == 80
    assert payload["phase_deltas"] == [
        {
            "from": "post_setup",
            "to": "post_layer_benchmark",
            "allocated_delta_bytes": 5,
            "reserved_delta_bytes": 5,
        }
    ]


def test_artifact_is_readable_by_the_shared_extraction():
    """The whole point of a separate runner is that nothing downstream knows.

    MaxText measures itself, so the contract it has to keep is the artifact --
    the same dict the torch harness produces, read by the same extraction.
    """
    from primus.core.projection.performance_projection.projection import (
        _extract_layer_type_timings,
    )

    bench = MaxTextLayerBench()
    results = {
        0: bench._layer_entry(JaxMeasurement(12.0, 24.0, 5_000_000), "dense", None, 1, 4096),
        1: bench._layer_entry(JaxMeasurement(30.0, 61.0, 9_000_000), "moe", None, 1, 4096),
    }

    timings = _extract_layer_type_timings(results)

    assert timings["dense"]["forward"] == 12.0
    assert timings["dense"]["backward"] == 24.0
    assert timings["moe"]["forward"] == 30.0
    assert timings["moe"]["backward"] == 61.0
    # Activation is reported in GB by the extraction.
    assert timings["moe"]["activation"] == pytest.approx(9_000_000 / 1024**3)


def test_missing_config_is_an_error_not_a_silent_skip():
    with pytest.raises(RuntimeError, match="setup_model_only"):
        MaxTextLayerBench().run(
            trainer=SimpleNamespace(train_config=None),
            training_config=SimpleNamespace(),
            batch_size=1,
            seq_len=128,
        )


# ---------------------------------------------------------------------------
# Benchmark write-back (flat projection fields -> MaxText's own config)
# ---------------------------------------------------------------------------


def _bench_shaped_args(**overrides):
    args = SimpleNamespace(
        framework="maxtext",
        num_layers=2,
        num_experts=8,
        tensor_model_parallel_size=2,
        context_model_parallel_size=1,
        expert_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        micro_batch_size=3,
        seq_length=4096,
        ici_tensor_parallelism=8,
        ici_tensor_sequence_parallelism=2,
        ici_fsdp_parallelism=-1,
        dcn_data_parallelism=-1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_writeback_caps_the_layer_stack():
    args = _bench_shaped_args()
    maxtext_apply_bench_overrides(args)
    assert args.base_num_decoder_layers == 2


def test_writeback_collapses_a_mesh_axis_group_onto_one_axis():
    """The product is what decides shapes; which sibling carries it does not."""
    args = _bench_shaped_args()
    maxtext_apply_bench_overrides(args)

    assert args.ici_tensor_parallelism == 2
    assert args.ici_tensor_sequence_parallelism == 1
    assert args.ici_tensor_transpose_parallelism == 1
    assert args.dcn_tensor_parallelism == 1


def test_writeback_moves_expert_and_pipeline_degrees():
    args = _bench_shaped_args()
    maxtext_apply_bench_overrides(args)

    assert args.ici_expert_parallelism == 4
    assert args.ici_pipeline_parallelism == 1
    assert args.num_experts == 8


def test_writeback_unshards_data_parallelism():
    """An auto-filled FSDP axis would shard the parameters the bench needs whole."""
    args = _bench_shaped_args()
    maxtext_apply_bench_overrides(args)

    assert args.ici_fsdp_parallelism == 1
    assert args.dcn_data_parallelism == 1


def test_writeback_sizes_the_batch_per_device():
    args = _bench_shaped_args()
    maxtext_apply_bench_overrides(args)

    assert args.per_device_batch_size == 3.0
    assert args.max_target_length == 4096
