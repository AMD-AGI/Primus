###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Per-datatype matrix peaks and the clocks GEMMs are simulated at."""

import pytest

from primus.core.projection.simulation_backends import origami_backend as ob
from primus.core.projection.simulation_backends import sdpa_simulator as sdpa
from primus.core.projection.simulation_backends.origami_backend import (
    OrigamiGEMMBackend,
    dtype_clock_mhz,
    dtype_peak_tflops,
)


@pytest.fixture(autouse=True)
def _no_clock_env(monkeypatch):
    monkeypatch.delenv("PRIMUS_GPU_CLOCK_MHZ", raising=False)


@pytest.mark.parametrize(
    "arch, dtype, mhz",
    [
        ("mi325x", "bf16", 1328.3),
        ("mi325x", "fp8", 1309.0),
        ("mi300x", "bf16", 930.0),
        ("mi355x", "bf16", 1645.0),
        ("mi355x", "fp8", 1668.9),
        ("gfx950", "fp8", 1668.9),
    ],
)
def test_clock_reaches_the_dtype_peak(arch, dtype, mhz):
    assert dtype_clock_mhz(arch, dtype) == pytest.approx(mhz, abs=0.1)


def test_explicit_clock_applies_to_every_dtype(monkeypatch):
    assert dtype_clock_mhz("mi325x", "fp8", 1500) == 1500
    assert dtype_peak_tflops("mi325x", "bf16", 1500) == pytest.approx(304 * 2048 * 1500 / 1e6)
    assert dtype_peak_tflops("mi325x", "fp8", 1500) == pytest.approx(304 * 4096 * 1500 / 1e6)
    monkeypatch.setenv("PRIMUS_GPU_CLOCK_MHZ", "1200")
    assert dtype_clock_mhz("mi355x", "bf16") == 1200


def test_sub_byte_types_on_the_fp8_instruction_reach_their_own_peak():
    # gfx942 has no FP4 peak: MX4 runs as BF16, issued on the FP8 instruction.
    assert dtype_peak_tflops("mi325x", "mx4") == dtype_peak_tflops("mi325x", "bf16")
    assert dtype_clock_mhz("mi325x", "mx4") == pytest.approx(827e6 / (304 * 4096))
    # gfx950 MXFP4 beats the FP8 instruction's rate at the FP8 clock.
    assert dtype_peak_tflops("mi355x", "mx4") == 5660
    assert dtype_clock_mhz("mi355x", "mx4") == pytest.approx(5660e6 / (256 * 8192))


def test_arch_without_a_peak_table_keeps_its_profile_clock():
    assert dtype_peak_tflops("mi300a") is None
    assert dtype_clock_mhz("mi300a", "bf16") is None
    assert dtype_clock_mhz("mi300a", "bf16", 1800) == 1800
    assert OrigamiGEMMBackend(gpu_arch="mi300a").peak_flops("bf16") is None


def test_attention_simulator_uses_the_peak_table(monkeypatch):
    spec = sdpa._get_hardware_spec("mi355x")
    assert (spec.peak_tflops_bf16, spec.peak_tflops_fp8, spec.hbm_bandwidth_gbps) == (1725, 3500, 8000.0)
    assert sdpa._get_hardware_spec("mi325x", 1500).peak_tflops_bf16 == pytest.approx(304 * 2048 * 1.5e-3)
    # An external overlay for an in-tree arch keeps its own peaks.
    monkeypatch.setitem(sdpa._HW_PROFILES, "mi355x", sdpa.GPUHardwareSpec(peak_tflops_bf16=1000.0))
    monkeypatch.setattr(sdpa, "_EXTERNAL_PROFILES", {"mi355x"})
    assert sdpa._get_hardware_spec("mi355x").peak_tflops_bf16 == 1000.0


def test_peak_flops_reports_the_dtype_peak():
    be = OrigamiGEMMBackend(gpu_arch="mi355x")
    assert be.peak_flops("fp8") == pytest.approx(3500e12)
    assert be.peak_flops("bf16") == pytest.approx(1725e12)


def test_candidate_tiles_feed_wide_instructions():
    assert ob._candidate_tile_sizes(32) == ob._DEFAULT_TILE_SIZES
    wide = ob._candidate_tile_sizes(128)
    assert any(k == 128 for (_m, _n, k) in wide) and any(k == 256 for (_m, _n, k) in wide)


@pytest.mark.parametrize("arch", ["mi300x", "mi325x", "mi355x"])
def test_fp8_gemms_run_faster_than_bf16_and_below_peak(arch):
    be = OrigamiGEMMBackend(gpu_arch=arch)
    if not be.is_available():
        pytest.skip("origami not installed")
    m, n, k = 16384, 14336, 4096
    bf16 = be.simulate_gemm(m, n, k, "bf16")
    fp8 = be.simulate_gemm(m, n, k, "fp8")
    assert not fp8.metadata["fp8_fallback"]
    assert 1.5 < bf16.forward_time_ms / fp8.forward_time_ms < 2.05
    assert bf16.tflops < dtype_peak_tflops(arch, "bf16")
    assert fp8.tflops < dtype_peak_tflops(arch, "fp8")
    assert bf16.metadata["clock_ghz"] == pytest.approx(dtype_clock_mhz(arch, "bf16") / 1e3, abs=1e-3)
