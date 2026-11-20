###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for benchmark-mode hardware calibration.

CPU tests pin the pieces that need no GPU: workload GEMM extraction through
the profiler tree, GEMM selection, the clock inversion, the memory-clock
arithmetic, sysfs parsing, reporting, and the artifact round trip.  The GPU
tests (skipped without one) run the real GEMM timing, STREAM kernels, and the
end-to-end calibration on device 0.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from primus.core.projection.hardware_calibration import (
    PAYLOAD_KEY,
    SKIP_ENV,
    calibration_skip_reason,
    format_calibration_report,
    gather_rank_summaries,
)
from primus.core.projection.hardware_calibration.compute_clock import (
    back_calculate_compute_clock,
    mfma_bound_clock_mhz,
    solve_clock_mhz,
)
from primus.core.projection.hardware_calibration.device_info import (
    ClockReading,
    ClockSampler,
    SysfsClockReader,
    read_dpm_table,
    resolve_profile_key,
    run_until_settled,
)
from primus.core.projection.hardware_calibration.gemm_shapes import (
    REFERENCE_GEMM,
    GemmShape,
    RecordingGEMMBackend,
    collect_workload_gemms,
    select_calibration_gemms,
)
from primus.core.projection.hardware_calibration.stream_bandwidth import (
    STREAM_DERATE_ENV,
    STREAM_WORKLOAD_DERATE,
    analyze_memory_clock,
    derive_model_hbm_bandwidth,
)
from primus.core.projection.simulation_backends.base import (
    HBM_BANDWIDTH_ENV,
    hbm_bandwidth_override_gbps,
    resolve_hbm_bytes_per_ms,
    set_hbm_bandwidth_override,
)
from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)

HIDDEN = 1024
FFN = 4096
VOCAB = 32000
SEQ = 2048
NUM_EXPERTS = 8


def _training_config(moe: bool = False, tp: int = 1) -> TrainingConfig:
    model = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        padded_vocab_size=VOCAB,
        ffn_hidden_size=FFN,
        num_attention_heads=16,
        kv_channels=64,
        num_query_groups=16,
        group_query_attention=False,
        swiglu=True,
        multi_latent_attention=False,
        use_flash_attn=True,
    )
    if moe:
        model.update(num_experts=NUM_EXPERTS, moe_ffn_hidden_size=FFN, moe_router_topk=2, moe_pattern=[1, 1])
    else:
        model.update(num_experts=None, moe_pattern=[0, 0])
    return TrainingConfig(
        model_config=ModelConfig(**model),
        runtime_config=RuntimeConfig(
            global_batch_size=8, micro_batch_size=1, sequence_length=SEQ, data_parallel_size=1
        ),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=tp),
    )


# ---------------------------------------------------------------------------
# Workload GEMM extraction and selection
# ---------------------------------------------------------------------------


def test_recording_backend_counts_shapes_and_ignores_degenerate():
    rec = RecordingGEMMBackend()
    rec.simulate_gemm(8, 16, 32, "bf16")
    rec.simulate_gemm(8, 16, 32, "fp8")
    rec.simulate_gemm(0, 16, 32)
    result = rec.simulate_gemm(4, 4, 4, batch=3)
    assert rec.shapes == {
        GemmShape(8, 16, 32): 1,
        GemmShape(8, 16, 32, dtype="fp8"): 1,
        GemmShape(4, 4, 4, 3): 1,
    }
    assert rec.dtypes["bf16"] == 2 and rec.dtypes["fp8"] == 1
    assert result.forward_time_ms > 0


def test_fp8_workload_records_fp8_gemms():
    cfg = _training_config()
    cfg.model_config.fp8 = "hybrid"
    shapes, dtypes = collect_workload_gemms(cfg)
    assert "fp8" in dtypes
    assert GemmShape(SEQ, FFN, HIDDEN, dtype="fp8") in shapes


def test_gemm_bytes_follow_dtype():
    assert GemmShape(1024, 1024, 1024).nbytes == 3 * 2 * 1024**2
    # fp8 inputs, bf16 output.
    assert GemmShape(1024, 1024, 1024, dtype="fp8").nbytes == (2 * 1 + 2) * 1024**2


def test_dense_workload_gemms_match_swiglu_mlp_and_output_layer():
    shapes, dtypes = collect_workload_gemms(_training_config())
    tokens = SEQ
    # MLP: gate/up fwd, down fwd, and their wgrads.
    assert GemmShape(tokens, FFN, HIDDEN) in shapes
    assert GemmShape(tokens, HIDDEN, FFN) in shapes
    assert GemmShape(HIDDEN, FFN, tokens) in shapes
    # Vocab projection.
    assert GemmShape(tokens, VOCAB, HIDDEN) in shapes
    assert set(dtypes) == {"bf16"}


def test_workload_gemms_follow_tensor_parallel_split():
    tp1, _ = collect_workload_gemms(_training_config(tp=1))
    tp2, _ = collect_workload_gemms(_training_config(tp=2))
    assert tp1 != tp2


def test_moe_workload_records_expert_gemms():
    dense, _ = collect_workload_gemms(_training_config())
    moe, _ = collect_workload_gemms(_training_config(moe=True))
    assert set(moe) - set(dense), "MoE layers should contribute expert / router GEMMs"


def test_select_ranks_by_total_flops_and_filters():
    big = GemmShape(4096, 4096, 4096)
    frequent = GemmShape(2048, 4096, 4096)
    tiny = GemmShape(64, 64, 64)
    huge = GemmShape(65536, 65536, 1024)
    selected = select_calibration_gemms({big: 1, frequent: 3, tiny: 1000, huge: 1}, max_gemms=4)
    assert [s for s, _ in selected] == [frequent, big]


def test_select_respects_max_gemms():
    counts = {GemmShape(4096 + i, 4096, 4096): 1 for i in range(10)}
    assert len(select_calibration_gemms(counts, max_gemms=3)) == 3


def test_select_falls_back_to_reference_gemm():
    assert select_calibration_gemms({}) == [(REFERENCE_GEMM, 1)]
    assert select_calibration_gemms({GemmShape(32, 32, 32): 5}) == [(REFERENCE_GEMM, 1)]
    fp8_tiny = {GemmShape(32, 32, 32, dtype="fp8"): 5}
    assert select_calibration_gemms(fp8_tiny) == [(GemmShape(8192, 8192, 8192, dtype="fp8"), 1)]


# ---------------------------------------------------------------------------
# Compute-clock back-calculation
# ---------------------------------------------------------------------------


def _cycle_model(cycles_per_flop: float = 1.0 / 2048 / 304):
    """Ideal model: time = flops x cycles_per_flop / clock."""

    def predict(clock_mhz, shape):
        return shape.flops * cycles_per_flop / (clock_mhz * 1e6) * 1e3

    return predict


@pytest.mark.parametrize("true_mhz", [800.0, 1500.0, 2100.0])
def test_solve_clock_recovers_true_clock(true_mhz):
    shape = GemmShape(8192, 8192, 8192)
    predict = _cycle_model()
    mhz, clamped = solve_clock_mhz(lambda f: predict(f, shape), predict(true_mhz, shape))
    assert not clamped
    assert mhz == pytest.approx(true_mhz, abs=1.0)


def test_solve_clock_clamps_outside_search_range():
    predict = lambda f: 1000.0 / f  # noqa: E731
    assert solve_clock_mhz(predict, 1e-6, lo_mhz=100, hi_mhz=4000) == (4000.0, True)
    assert solve_clock_mhz(predict, 1e6, lo_mhz=100, hi_mhz=4000) == (100.0, True)


def test_mfma_bound_matches_mi300x_peak():
    # 1307.4 TF over 1 s on 304 CUs x 2048 FLOP/clk is exactly 2100 MHz.
    flops = 304 * 2048 * 2.1e9
    assert mfma_bound_clock_mhz(flops, 1000.0, "gfx942", 304) == pytest.approx(2100.0)
    assert mfma_bound_clock_mhz(flops, 1000.0, "gfx1100", 304) is None
    # fp8 runs at twice the bf16 rate: 2614.9 TF on MI300X is also 2100 MHz.
    assert mfma_bound_clock_mhz(2 * flops, 1000.0, "gfx942", 304, "fp8") == pytest.approx(2100.0)
    assert mfma_bound_clock_mhz(flops, 1000.0, "gfx942", 304, "mx4") is None
    # MXFP4 on MI355X: 10.1 PF = 256 CU x 2.4 GHz x 16384.
    assert mfma_bound_clock_mhz(256 * 16384 * 2.4e9, 1000.0, "gfx950", 256, "mx4") == pytest.approx(2400.0)


@pytest.mark.parametrize(
    "workload, arch, candidates",
    [
        ("bf16", "gfx942", ["bf16"]),
        ("fp16", "gfx942", ["fp16"]),
        ("fp8", "gfx942", ["fp8", "bf16"]),
        ("mx8", "gfx942", ["fp8", "bf16"]),
        ("mx4", "gfx942", ["fp8", "bf16"]),
        ("mx8", "gfx950", ["mx8", "fp8", "bf16"]),
        ("fp4", "gfx950", ["mx4", "fp8", "bf16"]),
        ("mx4", "gfx1250", ["mx4", "fp8", "bf16"]),
        ("mx6", "gfx950", ["fp8", "bf16"]),
        ("int8", "gfx950", ["bf16"]),
    ],
)
def test_gemm_dtype_candidates(workload, arch, candidates):
    from primus.core.projection.hardware_calibration.compute_clock import (
        gemm_dtype_candidates,
    )

    assert gemm_dtype_candidates(workload, arch) == candidates


def test_aggregate_mfma_bound_mixes_dtypes():
    bf16 = GemmShape(8192, 8192, 8192)
    fp8 = GemmShape(8192, 8192, 8192, dtype="fp8")
    # Both at 100 % MFMA use and 1500 MHz: fp8 takes half the bf16 time.
    bf16_ms = bf16.flops / (304 * 2048 * 1.5e9) * 1e3
    result = back_calculate_compute_clock([(bf16, 1, bf16_ms), (fp8, 1, bf16_ms / 2)], "gfx942", 304, None)
    assert result["mfma_bound_clock_mhz"] == pytest.approx(1500.0)
    assert [row["dtype"] for row in result["gemms"]] == ["bf16", "fp8"]


def test_mx_gemms_skip_origami_and_use_live_sclk():
    predict = _cycle_model(1.0 / 4096 / 256)
    mx4 = GemmShape(8192, 8192, 8192, dtype="mx4")
    bf16 = GemmShape(8192, 8192, 8192)
    only_mx = back_calculate_compute_clock([(mx4, 1, 1.0)], "gfx950", 256, predict, live_sclk_mhz=1850.0)
    assert only_mx["gemms"][0]["origami_clock_mhz"] is None
    assert only_mx["gemms"][0]["mfma_bound_clock_mhz"] is not None
    assert only_mx["method"] == "live-sclk" and only_mx["effective_clock_mhz"] == 1850.0
    # With a modelled GEMM alongside, Origami inverts that one only.
    mixed = back_calculate_compute_clock(
        [(mx4, 5, 1.0), (bf16, 1, predict(1600.0, bf16))], "gfx950", 256, predict, live_sclk_mhz=1850.0
    )
    assert mixed["method"] == "origami-inversion"
    assert mixed["effective_clock_mhz"] == pytest.approx(1600.0, abs=1.0)


def test_live_sclk_used_when_origami_is_unavailable():
    shape = GemmShape(8192, 8192, 8192)
    result = back_calculate_compute_clock([(shape, 1, 2.0)], "gfx1250", 256, None, live_sclk_mhz=1700.0)
    assert result["method"] == "live-sclk" and result["effective_clock_mhz"] == 1700.0


def _fake_timer(monkeypatch, failing):
    from primus.core.projection.hardware_calibration import calibration

    timed = []

    def fake_time(shape, device, **kwargs):
        timed.append(shape.dtype)
        if shape.dtype in failing:
            raise failing[shape.dtype]
        return 1.0, {"settled": True}

    monkeypatch.setattr(calibration, "time_gemm_ms", fake_time)
    return calibration, timed


def test_workload_gemm_falls_back_to_bf16_when_fp8_unavailable(monkeypatch):
    calibration, timed = _fake_timer(monkeypatch, {"fp8": RuntimeError("HIPBLAS_STATUS_NOT_SUPPORTED")})
    measured, ms, _, note = calibration._time_workload_gemm(
        GemmShape(4096, 4096, 4096, dtype="mx8"), None, "gfx942"
    )
    assert timed == ["fp8", "bf16"]
    assert measured.dtype == "bf16" and ms == 1.0
    assert note == "mx8 GEMMs timed as bf16 (fp8 unavailable: RuntimeError)"


def test_workload_gemm_falls_back_from_native_mx_to_fp8(monkeypatch):
    calibration, timed = _fake_timer(monkeypatch, {"mx4": NotImplementedError("only on gfx950")})
    measured, _, _, note = calibration._time_workload_gemm(
        GemmShape(4096, 4096, 4096, dtype="mx4"), None, "gfx1250"
    )
    assert timed == ["mx4", "fp8"]
    assert measured.dtype == "fp8"
    assert note == "mx4 GEMMs timed as fp8 (mx4 unavailable: NotImplementedError)"


def test_workload_gemm_runs_native_mx(monkeypatch):
    calibration, timed = _fake_timer(monkeypatch, {})
    measured, _, _, note = calibration._time_workload_gemm(GemmShape(64, 64, 64, dtype="fp4"), None, "gfx950")
    assert measured.dtype == "mx4" and note is None
    measured, _, _, note = calibration._time_workload_gemm(GemmShape(64, 64, 64, dtype="mx8"), None, "gfx942")
    assert measured.dtype == "fp8" and note == "mx8 GEMMs timed as fp8"


def test_back_calculate_with_origami_style_predictor():
    predict = _cycle_model()
    true_mhz = 1600.0
    shapes = [(GemmShape(8192, 8192, 8192), 2), (GemmShape(4096, 14336, 4096), 3)]
    measurements = [(s, c, predict(true_mhz, s)) for s, c in shapes]
    result = back_calculate_compute_clock(measurements, "gfx942", 304, predict)
    assert result["method"] == "origami-inversion"
    assert result["effective_clock_mhz"] == pytest.approx(true_mhz, abs=1.0)
    for row in result["gemms"]:
        assert row["origami_clock_mhz"] == pytest.approx(true_mhz, abs=1.0)
        # The ideal model runs at 100 % MFMA use, so both estimates agree.
        assert row["mfma_bound_clock_mhz"] == pytest.approx(true_mhz, rel=1e-6)


def test_back_calculate_without_origami_reports_lower_bound():
    shape = GemmShape(8192, 8192, 8192)
    measured_ms = _cycle_model()(1000.0, shape) * 2  # 50 % MFMA use at 2 GHz
    result = back_calculate_compute_clock([(shape, 1, measured_ms)], "gfx942", 304, None)
    assert result["method"] == "mfma-peak-lower-bound"
    assert result["effective_clock_mhz"] == pytest.approx(500.0)
    assert result["gemms"][0]["origami_clock_mhz"] is None


@pytest.mark.skipif(importlib.util.find_spec("origami") is None, reason="origami is not installed")
def test_origami_predictor_is_monotonic_in_clock():
    from primus.core.projection.hardware_calibration.compute_clock import (
        make_origami_predictor,
    )

    predict = make_origami_predictor("mi300x", 304)
    assert predict is not None
    shape = GemmShape(8192, 8192, 8192)
    times = [predict(mhz, shape) for mhz in (1000, 1500, 2000, 2500)]
    assert times == sorted(times, reverse=True)


def test_origami_predictor_unknown_profile_is_none():
    from primus.core.projection.hardware_calibration.compute_clock import (
        make_origami_predictor,
    )

    assert make_origami_predictor("not-a-gpu", 304) is None


# ---------------------------------------------------------------------------
# Device identification and sysfs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch,cus,vram_gib,expected",
    [
        ("gfx942:sramecc+:xnack-", 304, 192, "mi300x"),
        ("gfx942", 304, 256, "mi325x"),
        ("gfx942", 228, 128, "mi300a"),
        ("gfx950", 256, 288, "mi355x"),
        ("gfx1100", 48, 24, None),
    ],
)
def test_resolve_profile_key(arch, cus, vram_gib, expected):
    assert resolve_profile_key(arch, cus, vram_gib * 1024**3) == expected


def test_read_dpm_table_levels_and_current(tmp_path):
    sclk = tmp_path / "pp_dpm_sclk"
    sclk.write_text("S: 1532Mhz *\n1: 500Mhz \n2: 2100Mhz \n")
    assert read_dpm_table(str(sclk)) == ([500, 2100], 1532)

    mclk = tmp_path / "pp_dpm_mclk"
    mclk.write_text("0: 900Mhz \n1: 1300Mhz *\n2: 1500Mhz \n")
    assert read_dpm_table(str(mclk)) == ([900, 1300, 1500], 1300)

    mid_transition = tmp_path / "pp_dpm_mclk_none"
    mid_transition.write_text("0: 900Mhz \n1: 1300Mhz \n")
    assert read_dpm_table(str(mid_transition)) == ([900, 1300], None)

    assert read_dpm_table(str(tmp_path / "missing")) == ([], None)


# ---------------------------------------------------------------------------
# Live clocks: steady-state wait and window-only sampling
# ---------------------------------------------------------------------------


class _ScriptedReader:
    """Replays a list of (sclk, mclk) readings, repeating the last one."""

    source = "fake"

    def __init__(self, values):
        self._values = list(values)
        self.reads = 0

    def read(self):
        sclk, mclk = self._values[min(self.reads, len(self._values) - 1)]
        self.reads += 1
        return ClockReading(sclk_mhz=sclk, mclk_mhz=mclk, xcd_sclks_mhz=[int(sclk), int(sclk) - 40])

    def close(self):
        pass


def _short_step():
    import time

    time.sleep(0.002)


def test_settle_waits_past_boost_until_clock_is_stable():
    # Boost at 2100, a power-management dip, then steady near 1500.
    reader = _ScriptedReader([(2100, 900)] * 3 + [(800, 900)] * 3 + [(1500, 900)] * 50)
    out = run_until_settled(_short_step, reader, "sclk", min_s=0.0, window_s=0.02, max_s=5.0)
    assert out["settled"] is True
    assert out["settled_mhz"] == pytest.approx(1500.0)
    assert reader.reads >= 6 + 5


def test_settle_tracks_mclk_for_memory_bound_work():
    ramp = [(2000, mclk) for mclk in (900, 1100, 1300, 1450)] + [(2000, 1500)] * 50
    out = run_until_settled(_short_step, _ScriptedReader(ramp), "mclk", min_s=0.0, window_s=0.02)
    assert out["settled"] is True
    assert out["settled_mhz"] == pytest.approx(1500.0)


def test_settle_gives_up_and_flags_unstable_clock():
    flapping = [(1000, 900), (2000, 900)] * 100_000
    out = run_until_settled(lambda: None, _ScriptedReader(flapping), "sclk", min_s=0.0, max_s=0.05)
    assert out["settled"] is False


def test_settle_without_reader_runs_for_min_duration():
    calls = []
    out = run_until_settled(lambda: calls.append(1), None, "sclk", min_s=0.02)
    assert out["settled"] is None
    assert out["settle_s"] >= 0.02 and calls


def test_sampler_keeps_only_readings_inside_window():
    import time

    reader = _ScriptedReader([(1500, 1300)])
    with ClockSampler(reader, interval_s=0.001) as sampler:
        time.sleep(0.05)
        assert sampler.readings == []
        with sampler.window():
            time.sleep(0.05)
    summary = sampler.summary()
    assert summary["source"] == "fake"
    assert summary["sclk"]["median_mhz"] == 1500.0
    assert summary["sclk"]["per_xcd_median_mhz"] == [1500.0, 1460.0]
    assert summary["mclk"]["median_mhz"] == 1300.0


def test_sysfs_reader_reports_live_level_not_ceiling(tmp_path):
    (tmp_path / "pp_dpm_sclk").write_text("S: 1532Mhz *\n1: 500Mhz \n2: 2100Mhz \n")
    (tmp_path / "pp_dpm_mclk").write_text("0: 900Mhz \n1: 1300Mhz *\n2: 1500Mhz \n")
    reading = SysfsClockReader(str(tmp_path)).read()
    assert (reading.sclk_mhz, reading.mclk_mhz) == (1532.0, 1300.0)


def test_amdsmi_reader_averages_xcds_and_drops_invalid(monkeypatch):
    from primus.core.projection.hardware_calibration import device_info

    fake = SimpleNamespace(
        amdsmi_init=lambda: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: ["h0", "h1"],
        amdsmi_get_gpu_device_bdf=lambda h: {"h0": "0000:83:00.0", "h1": "0000:93:00.0"}[h],
        amdsmi_get_gpu_metrics_info=lambda h: {
            "current_gfxclks": [1500, 1460, "N/A", 65535],
            "current_uclk": 1500,
        },
    )
    monkeypatch.setattr(device_info, "_import_amdsmi", lambda: fake)
    reading = device_info.AmdSmiClockReader("0000:83:00.0").read()
    assert reading.xcd_sclks_mhz == [1500, 1460]
    assert reading.sclk_mhz == pytest.approx(1480.0)
    assert reading.mclk_mhz == 1500.0
    with pytest.raises(RuntimeError):
        device_info.AmdSmiClockReader("0000:ff:00.0")


# ---------------------------------------------------------------------------
# Memory-clock arithmetic
# ---------------------------------------------------------------------------


def test_analyze_memory_clock_mi325x():
    out = analyze_memory_clock(4800.0, peak_gbps=6000.0, max_mclk_mhz=1500, observed_mclk_mhz=1300.0)
    assert out["efficiency_vs_peak"] == pytest.approx(0.8)
    assert out["bytes_per_mclk_cycle"] == pytest.approx(4000.0)
    assert out["min_mclk_mhz_at_full_efficiency"] == pytest.approx(1200.0)
    assert out["efficiency_at_observed_mclk"] == pytest.approx(4800.0 / 5200.0)


def test_analyze_memory_clock_without_spec():
    out = analyze_memory_clock(4800.0, peak_gbps=None, max_mclk_mhz=None, observed_mclk_mhz=None)
    assert out["efficiency_vs_peak"] is None
    assert out["min_mclk_mhz_at_full_efficiency"] is None


# ---------------------------------------------------------------------------
# STREAM -> model HBM bandwidth
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_hbm_override(monkeypatch):
    monkeypatch.delenv(HBM_BANDWIDTH_ENV, raising=False)
    monkeypatch.delenv(STREAM_DERATE_ENV, raising=False)
    set_hbm_bandwidth_override(None)
    yield
    set_hbm_bandwidth_override(None)


def test_model_bandwidth_reproduces_mi325x_spec_peak(clean_hbm_override):
    # Llama-3-8B layer on MI325X: STREAM 4761 GB/s, memory-bound kernels at 72 % of it.
    out = derive_model_hbm_bandwidth(4761.0)
    assert out["stream_derate"] == STREAM_WORKLOAD_DERATE == 0.72
    assert out["workload_bandwidth_gbps"] == pytest.approx(3427.9, abs=0.1)
    assert out["model_hbm_bandwidth_gbps"] == pytest.approx(6000.0, rel=0.01)


def test_stream_derate_env_override_and_validation(clean_hbm_override, monkeypatch):
    monkeypatch.setenv(STREAM_DERATE_ENV, "0.5")
    assert derive_model_hbm_bandwidth(4000.0)["workload_bandwidth_gbps"] == pytest.approx(2000.0)
    monkeypatch.setenv(STREAM_DERATE_ENV, "1.5")
    with pytest.raises(ValueError):
        derive_model_hbm_bandwidth(4000.0)


def test_calibrated_bandwidth_uses_slowest_healthy_rank():
    from primus.core.projection.hardware_calibration import (
        calibrated_model_hbm_bandwidth_gbps,
    )

    assert calibrated_model_hbm_bandwidth_gbps(None) is None
    assert calibrated_model_hbm_bandwidth_gbps({"error": "OOM"}) is None
    payload = _payload(bw=4900.0)
    own = payload["memory"]["model_hbm_bandwidth_gbps"]
    assert calibrated_model_hbm_bandwidth_gbps(payload) == pytest.approx(own)
    payload["per_rank"] = [
        {"rank": 0, "model_hbm_bandwidth_gbps": own},
        {"rank": 1, "model_hbm_bandwidth_gbps": own - 100.0},
        {"rank": 2, "error": "RuntimeError: boom"},
    ]
    assert calibrated_model_hbm_bandwidth_gbps(payload) == pytest.approx(own - 100.0)


def test_hbm_override_replaces_backend_profile_peak(clean_hbm_override, monkeypatch):
    from primus.core.projection.simulation_backends.origami_backend import (
        OrigamiGEMMBackend,
    )

    backend = OrigamiGEMMBackend(gpu_arch="mi325x")
    assert backend.hbm_bandwidth_gbps == 6000.0
    set_hbm_bandwidth_override(7100.0)
    assert backend.hbm_bandwidth_gbps == 7100.0
    assert resolve_hbm_bytes_per_ms(backend) == pytest.approx(7100.0 * 1e6)
    set_hbm_bandwidth_override(None)
    monkeypatch.setenv(HBM_BANDWIDTH_ENV, "6500")
    assert backend.hbm_bandwidth_gbps == 6500.0
    with pytest.raises(ValueError):
        set_hbm_bandwidth_override(0.0)


def test_hbm_override_reaches_sdpa_backend(clean_hbm_override):
    from primus.core.projection.simulation_backends import sdpa_simulator

    sdpa = sdpa_simulator.SDPASimulator.__new__(sdpa_simulator.SDPASimulator)
    sdpa._hw = sdpa_simulator._get_hardware_spec("mi325x")
    assert sdpa.hbm_bandwidth_gbps == 6000.0
    set_hbm_bandwidth_override(7100.0)
    assert sdpa.hbm_bandwidth_gbps == 7100.0


def test_projection_applies_calibration_unless_overridden(clean_hbm_override):
    from primus.core.projection.performance_projection import projection

    payload = _payload(bw=4900.0)
    calibrated = payload["memory"]["model_hbm_bandwidth_gbps"]
    projection._apply_calibrated_hbm_bandwidth(payload, rank=0)
    assert hbm_bandwidth_override_gbps() == pytest.approx(calibrated)

    set_hbm_bandwidth_override(7100.0)
    projection._apply_calibrated_hbm_bandwidth(payload, rank=0)
    assert hbm_bandwidth_override_gbps() == 7100.0

    set_hbm_bandwidth_override(None)
    projection._apply_calibrated_hbm_bandwidth({"error": "OOM"}, rank=0)
    assert hbm_bandwidth_override_gbps() is None


# ---------------------------------------------------------------------------
# Skip logic, reporting, gathering, artifact
# ---------------------------------------------------------------------------


def test_skip_reason_env_and_args(monkeypatch):
    monkeypatch.setenv(SKIP_ENV, "1")
    assert calibration_skip_reason() == f"{SKIP_ENV}=1"
    monkeypatch.delenv(SKIP_ENV)
    assert calibration_skip_reason(SimpleNamespace(skip_hw_calibration=True)) == "--skip-hw-calibration"
    assert calibration_skip_reason(SimpleNamespace(profile_only=True)) == "--profile-only"


def _payload(clock=1600.0, bw=4900.0):
    gemm = {
        **GemmShape(8192, 14336, 4096).to_dict(),
        "occurrences": 3,
        "measured_ms": 1.18,
        "achieved_tflops": 813.0,
        "mfma_bound_clock_mhz": 1306.0,
        "origami_clock_mhz": clock,
        "origami_clamped": False,
    }
    return {
        "device": {"index": 0, "name": "AMD", "profile_key": "mi325x", "gcn_arch": "gfx942", "num_cus": 304},
        "compute": {
            "gemms": [gemm],
            "effective_clock_mhz": clock,
            "mfma_bound_clock_mhz": 1306.0,
            "clamped": False,
            "method": "origami-inversion",
            "shape_source": "workload",
            "measured_dtype": "bf16",
            "max_sclk_mhz": 2100,
            "clock_source": "amdsmi",
            "all_settled": True,
            "observed_sclk": {
                "per_xcd_median_mhz": [1540.0, 1520.0],
                "median_mhz": 1530.0,
                "mean_mhz": 1500.0,
                "min_mhz": 900.0,
                "max_mhz": 1540.0,
                "num_samples": 50,
            },
        },
        "memory": {
            "stream": {
                "array_mib": 1024,
                "dtype": "fp32",
                "trials": 50,
                "settle": {"signal": "mclk", "settled": True, "settle_s": 0.8, "settled_mhz": 1300.0},
                "kernels": {k: {"best_gbps": bw} for k in ("copy", "scale", "add", "triad")},
            },
            "sustained_bandwidth_gbps": bw,
            "max_mclk_mhz": 1500,
            "clock_source": "amdsmi",
            "observed_mclk": {"median_mhz": 1300.0},
            **derive_model_hbm_bandwidth(bw),
            **analyze_memory_clock(bw, 6000.0, 1500, 1300.0),
        },
    }


def test_report_contains_headline_numbers():
    report = format_calibration_report(_payload())
    assert "Effective compute clock: 1600 MHz (Origami inversion, 76% of 2100 MHz max)" in report
    assert "--gpu-clock-mhz 1600" in report
    assert "Sustained HBM bandwidth: 4900 GB/s (82% of 6000 GB/s peak)" in report
    assert "Workload HBM bandwidth:  3528 GB/s (STREAM x 0.72 derate" in report
    assert "--hbm-bandwidth-gbps 6233" in report
    assert "Observed SCLK (amdsmi, mean of all XCDs, during timed GEMMs): median 1530 MHz" in report
    assert "per-XCD medians: 1540, 1520 MHz" in report
    assert "Origami-inverted clock is +4.6% vs observed" in report
    assert "Observed MCLK (amdsmi, during timed STREAM, settled after 0.8s): median 1300 MHz" in report
    assert "WARNING" not in report


def test_report_shows_gemm_dtype_and_fallback_note():
    payload = _payload()
    payload["compute"]["gemms"][0]["dtype"] = "fp8"
    payload["compute"]["measured_dtype"] = "fp8"
    payload["compute"]["dtype_notes"] = ["mx8 GEMMs timed as fp8"]
    report = format_calibration_report(payload)
    assert "Compute clock from 1 GEMM(s), fp8," in report
    row = next(line.split() for line in report.splitlines() if line.split()[:3] == ["8192", "14336", "4096"])
    assert row[3:6] == ["1", "fp8", "3"]  # batch, dtype, count
    assert "note: mx8 GEMMs timed as fp8" in report


def test_report_warns_when_clocks_never_settled():
    payload = _payload()
    payload["compute"]["all_settled"] = False
    payload["memory"]["stream"]["settle"] = {"signal": "mclk", "settled": False, "settle_s": 5.0}
    report = format_calibration_report(payload)
    assert "SCLK did not settle" in report
    assert "MCLK did not settle" in report


def test_report_lists_every_rank():
    payload = _payload()
    payload["per_rank"] = [
        {
            "rank": 0,
            "effective_clock_mhz": 1600.0,
            "observed_sclk_median_mhz": 1530.0,
            "sustained_bandwidth_gbps": 4900.0,
        },
        {"rank": 1, "error": "RuntimeError: boom"},
    ]
    report = format_calibration_report(payload)
    assert "Per GPU (2 ranks)" in report
    assert "failed: RuntimeError: boom" in report


def test_report_for_failed_rank():
    assert "Calibration failed on this rank: OOM" in format_calibration_report({"error": "OOM"})


def test_gather_is_noop_without_distributed():
    payload = _payload()
    assert gather_rank_summaries(payload) is payload
    assert "per_rank" not in payload


def test_run_hardware_calibration_helper_returns_error_payload(monkeypatch):
    from primus.core.projection.performance_projection import projection

    monkeypatch.setattr(
        "primus.core.projection.hardware_calibration.calibration_skip_reason", lambda args=None: None
    )

    def boom(_config):
        raise RuntimeError("no device")

    monkeypatch.setattr("primus.core.projection.hardware_calibration.run_hardware_calibration", boom)
    out = projection._run_hardware_calibration(_training_config(), "megatron", 0, SimpleNamespace())
    assert out == {"error": "RuntimeError: no device"}


def test_run_hardware_calibration_helper_skips_jax_backends(monkeypatch):
    from primus.core.projection.performance_projection import projection

    monkeypatch.setattr(
        "primus.core.projection.hardware_calibration.calibration_skip_reason", lambda args=None: None
    )
    assert projection._run_hardware_calibration(_training_config(), "maxtext", 0, SimpleNamespace()) is None


def test_artifact_round_trip_keeps_calibration(tmp_path):
    from primus.core.projection.performance_projection.projection import (
        _load_profiling_results,
        _save_profiling_results,
    )

    payload = _payload()
    path = tmp_path / "bench.json"
    _save_profiling_results({0: {"type": "dense"}, PAYLOAD_KEY: payload}, {}, str(path))
    loaded, _ = _load_profiling_results(str(path))
    assert loaded[PAYLOAD_KEY]["compute"]["effective_clock_mhz"] == payload["compute"]["effective_clock_mhz"]
    assert (
        loaded[PAYLOAD_KEY]["memory"]["sustained_bandwidth_gbps"]
        == payload["memory"]["sustained_bandwidth_gbps"]
    )


# ---------------------------------------------------------------------------
# On-GPU
# ---------------------------------------------------------------------------


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="needs a GPU")


@requires_gpu
def test_stream_reports_positive_bandwidth_per_kernel():
    import torch

    from primus.core.projection.hardware_calibration.stream_bandwidth import run_stream

    result = run_stream(torch.device("cuda", 0), array_mib=256, trials=3, min_settle_s=0.0)
    assert set(result["kernels"]) == {"copy", "scale", "add", "triad"}
    for kernel in result["kernels"].values():
        assert kernel["best_gbps"] > 0
        assert kernel["best_gbps"] >= kernel["mean_gbps"]


@requires_gpu
def test_time_gemm_is_positive_and_scales_with_work():
    import torch

    from primus.core.projection.hardware_calibration.compute_clock import time_gemm_ms

    device = torch.device("cuda", 0)
    small, _ = time_gemm_ms(GemmShape(2048, 2048, 2048), device, trials=3, min_settle_s=0.0)
    large, _ = time_gemm_ms(GemmShape(4096, 4096, 4096), device, trials=3, min_settle_s=0.0)
    batched, _ = time_gemm_ms(GemmShape(1024, 1024, 1024, batch=4), device, trials=3, min_settle_s=0.0)
    assert 0 < small < large
    assert batched > 0


@requires_gpu
def test_time_gemm_runs_fp8_faster_than_bf16():
    import torch

    from primus.core.projection.hardware_calibration.compute_clock import time_gemm_ms

    device = torch.device("cuda", 0)
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("fp8 matrix cores need gfx942 / gfx950")
    bf16, _ = time_gemm_ms(GemmShape(8192, 8192, 8192), device, trials=3, min_settle_s=0.0)
    fp8, _ = time_gemm_ms(GemmShape(8192, 8192, 8192, dtype="fp8"), device, trials=3, min_settle_s=0.0)
    batched, _ = time_gemm_ms(
        GemmShape(2048, 2048, 2048, batch=2, dtype="fp8"), device, trials=3, min_settle_s=0.0
    )
    assert 0 < fp8 < bf16
    assert batched > 0


@requires_gpu
def test_mx_gemms_run_natively_or_fall_back_to_fp8():
    import torch

    from primus.core.projection.hardware_calibration import calibration

    device = torch.device("cuda", 0)
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("fp8 matrix cores need gfx942 / gfx950")
    kw = dict(trials=3, min_settle_s=0.0)
    for dtype in ("mx8", "mx4"):
        # Claim MX support so parts without it exercise the real fallback.
        shape = GemmShape(8192, 8192, 8192, batch=1 if dtype == "mx8" else 2, dtype=dtype)
        measured, ms, _, note = calibration._time_workload_gemm(shape, device, "gfx950", **kw)
        assert ms > 0
        if arch == "gfx950":
            assert measured.dtype == dtype and note is None
        else:
            assert measured.dtype == "fp8" and f"{dtype} unavailable" in note


@requires_gpu
def test_end_to_end_calibration_leaves_no_memory_behind():
    import torch

    from primus.core.projection.hardware_calibration import run_hardware_calibration

    torch.cuda.set_device(0)
    before = torch.cuda.memory_allocated(0)
    payload = run_hardware_calibration(_training_config(), device_index=0, max_gemms=2, stream_array_mib=256)
    assert torch.cuda.memory_allocated(0) == before
    assert torch.cuda.max_memory_allocated(0) == before

    compute = payload["compute"]
    assert compute["shape_source"] == "workload"
    assert 1 <= len(compute["gemms"]) <= 2
    assert compute["effective_clock_mhz"] is None or compute["effective_clock_mhz"] > 0
    assert payload["memory"]["sustained_bandwidth_gbps"] > 0
    assert "Sustained HBM bandwidth" in format_calibration_report(payload)
