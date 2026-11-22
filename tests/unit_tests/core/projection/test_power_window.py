###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Load-window power measurement and its use by ``power-fit`` / ``power-sample``."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from primus.core.projection.power import cli as power_cli
from primus.core.projection.power.telemetry import (
    TelemetrySample,
    TelemetrySampler,
    _to_unix_seconds,
    load_telemetry,
    load_telemetry_files,
)
from primus.core.projection.power.window import (
    MeasurementWindow,
    auto_window,
    load_window,
    measure_window,
    samples_in_window,
    window_from_mapping,
)

T0 = 1_700_000_000.0


def _trace(power_fn, device="0", start=0.0, end=10.0, step=0.5, busy_fn=None):
    out, t = [], start
    while t <= end + 1e-9:
        busy = busy_fn(t) if busy_fn else None
        out.append(TelemetrySample(power_w=power_fn(t), timestamp=T0 + t, device=device, busy=busy))
        t += step
    return out


# ---------------------------------------------------------------------------
# measure_window
# ---------------------------------------------------------------------------


def test_constant_power_two_devices():
    samples = _trace(lambda t: 500.0, "0") + _trace(lambda t: 700.0, "1")
    m = measure_window(samples, MeasurementWindow(T0 + 2.25, T0 + 7.75), expected_devices=2)
    assert m.valid and m.reasons == []
    assert m.total_energy_j == pytest.approx((500 + 700) * 5.5)
    assert m.avg_power_w == pytest.approx(600.0)
    assert m.total_avg_power_w == pytest.approx(1200.0)
    assert {d.device: d.avg_power_w for d in m.devices} == pytest.approx({"0": 500.0, "1": 700.0})
    assert "Power Valid: yes" in m.describe()


def test_linear_ramp_is_integrated_exactly_with_edge_interpolation():
    # P(t) = 100 + 50 t; windows that start/end between samples.
    m = measure_window(_trace(lambda t: 100.0 + 50.0 * t), MeasurementWindow(T0 + 1.2, T0 + 6.7))
    exact = 100.0 * 5.5 + 25.0 * (6.7**2 - 1.2**2)
    assert m.valid and m.total_energy_j == pytest.approx(exact)


def test_gap_invalidates_and_withholds_values():
    samples = [s for s in _trace(lambda t: 500.0) if not (3.0 < s.timestamp - T0 < 7.0)]
    m = measure_window(samples, MeasurementWindow(T0 + 1.0, T0 + 9.0))
    assert not m.valid and m.reasons == ["sample_gap_exceeds_max"]
    assert m.avg_power_w is None and m.total_energy_j is None
    assert m.max_gap_s == pytest.approx(4.0)
    assert m.as_dict()["power_valid"] == 0 and m.as_dict()["devices"] == []
    assert measure_window(samples, MeasurementWindow(T0 + 1.0, T0 + 9.0), max_gap_s=5.0).valid


def test_window_not_covered():
    m = measure_window(_trace(lambda t: 500.0, end=5.0), MeasurementWindow(T0 + 1.0, T0 + 9.0))
    assert not m.valid and "window_not_covered" in m.reasons


@pytest.mark.parametrize(
    "samples, window, reason",
    [
        (_trace(lambda t: 500.0), MeasurementWindow(T0 + 5.0, T0 + 5.0), "invalid_window"),
        ([], MeasurementWindow(T0, T0 + 1.0), "no_samples"),
        ([TelemetrySample(power_w=500.0)] * 3, MeasurementWindow(T0, T0 + 1.0), "missing_timestamps"),
    ],
)
def test_structural_reasons(samples, window, reason):
    m = measure_window(samples, window)
    assert not m.valid and m.reasons == [reason]


def test_device_count_mismatch():
    m = measure_window(_trace(lambda t: 500.0), MeasurementWindow(T0 + 1.0, T0 + 2.0), expected_devices=8)
    assert not m.valid and m.reasons == ["device_count_mismatch"]


def test_window_helpers(tmp_path):
    w = MeasurementWindow(T0, T0 + 100.0)
    assert w.trimmed(10.0, 5.0).duration_s == pytest.approx(85.0)
    assert window_from_mapping({"benchmark_start_time_unix": 1, "benchmark_end_time_unix": 3}).duration_s == 2
    assert window_from_mapping({"window": {"start": 1, "end": 4}}).duration_s == 3
    with pytest.raises(ValueError):
        window_from_mapping({"begin": 1})
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"start_unix": T0, "end_unix": T0 + 2}))
    assert load_window(str(path)).duration_s == pytest.approx(2.0)

    samples = _trace(lambda t: 500.0, busy_fn=lambda t: 0.9 if 2.0 <= t <= 8.0 else 0.0)
    auto = auto_window(samples)
    assert (auto.start_unix - T0, auto.end_unix - T0) == pytest.approx((2.0, 8.0))
    assert auto_window(_trace(lambda t: 500.0)) is None
    assert len(samples_in_window(samples, auto)) == 13


# ---------------------------------------------------------------------------
# Timestamps and multi-host logs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (T0, T0),
        (T0 * 1e3, T0),
        (T0 * 1e6, T0),
        (T0 * 1e9, T0),
        ("2023-11-14T22:13:20Z", T0),
        (12.5, 12.5),  # relative seconds stay as-is
        ("not a time", None),
        ("", None),
    ],
)
def test_timestamp_parsing(value, expected):
    got = _to_unix_seconds(value)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_multi_host_logs_get_distinct_devices(tmp_path):
    for host in ("node0", "node1"):
        (tmp_path / f"{host}.csv").write_text("timestamp,gpu,power_w\n" + f"{T0},0,500\n{T0 + 1},1,600\n")
    single = load_telemetry(str(tmp_path / "node0.csv"))
    assert {s.device for s in single} == {"0", "1"}
    merged = load_telemetry_files([str(tmp_path / "node0.csv"), str(tmp_path / "node1.csv")])
    assert {s.device for s in merged} == {"node0:0", "node0:1", "node1:0", "node1:1"}


# ---------------------------------------------------------------------------
# power-fit with a window
# ---------------------------------------------------------------------------


def _fit_args(tmp_path, telemetry, **kw):
    base = dict(
        telemetry=[str(telemetry)],
        gpu_arch="mi300x",
        out=str(tmp_path / "fit.yaml"),
        matrix_utilization=None,
        tbp_w=None,
        peak_clock_mhz=None,
        fit_alpha="no",
        window=None,
        window_file=None,
        auto_window=False,
        trim_s=None,
        max_gap_s=None,
        allow_invalid_window=False,
        json=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _write_log(path, skip=lambda t: False):
    rows = ["timestamp,power_w,gfx_activity"]
    for i in range(41):
        t = i * 0.5
        if skip(t):
            continue
        busy = 90 if 5.0 <= t <= 15.0 else 0
        rows.append(f"{T0 + t},{150 + 4 * busy + (t % 2)},{busy}")
    path.write_text("\n".join(rows))


def test_power_fit_restricts_to_window(tmp_path, capsys):
    log = tmp_path / "t.csv"
    _write_log(log)
    win = tmp_path / "t.window.json"
    win.write_text(json.dumps({"start_unix": T0 + 5.0, "end_unix": T0 + 15.0}))
    assert power_cli.run_power_fit(_fit_args(tmp_path, log, window_file=str(win))) == 0
    out = capsys.readouterr().out
    assert "21/41 samples" in out and "Power Valid: yes" in out and "Fitted 21 samples" in out


def test_power_fit_auto_window_and_trim(tmp_path, capsys):
    log = tmp_path / "t.csv"
    _write_log(log)
    assert power_cli.run_power_fit(_fit_args(tmp_path, log, auto_window=True, trim_s=[1.0, 1.0])) == 0
    assert "17/41 samples" in capsys.readouterr().out


def test_power_fit_rejects_invalid_window_unless_allowed(tmp_path, capsys):
    log = tmp_path / "t.csv"
    _write_log(log, skip=lambda t: 8.0 < t < 12.0)
    args = _fit_args(tmp_path, log, window=[T0 + 5.0, T0 + 15.0])
    with pytest.raises(SystemExit, match="sample_gap_exceeds_max"):
        power_cli.run_power_fit(args)
    args.allow_invalid_window = True
    assert power_cli.run_power_fit(args) == 0
    assert "Power Valid: no (sample_gap_exceeds_max)" in capsys.readouterr().out


def test_power_fit_without_window_warns(tmp_path, capsys):
    log = tmp_path / "t.csv"
    _write_log(log)
    assert power_cli.run_power_fit(_fit_args(tmp_path, log)) == 0
    assert "No load window given" in capsys.readouterr().out


def test_power_fit_json_includes_measurement(tmp_path, capsys):
    log = tmp_path / "t.csv"
    _write_log(log)
    power_cli.run_power_fit(_fit_args(tmp_path, log, window=[T0 + 5.0, T0 + 15.0], json=True))
    out = capsys.readouterr().out
    doc = json.loads(out[out.index("{") :])
    assert doc["measurement"]["power_valid"] == 1
    assert doc["measurement"]["avg_power_w"] > 400


# ---------------------------------------------------------------------------
# power-sample around a command
# ---------------------------------------------------------------------------


class _FakeReader:
    def __init__(self, device, watts):
        self.device, self.watts = str(device), watts

    def read(self):
        import time

        return TelemetrySample(power_w=self.watts, timestamp=time.time(), device=self.device, busy=1.0)


def test_power_sample_records_command_window(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "primus.core.projection.power.telemetry.make_telemetry_sampler",
        lambda d, interval_s: TelemetrySampler(_FakeReader(d, 400.0 + 100 * d), interval_s=interval_s),
    )
    out = tmp_path / "run.jsonl"
    args = SimpleNamespace(
        device=[0, 1],
        command=f'"{sys.executable}" -c "import time; time.sleep(0.3)"',
        duration_s=0.0,
        interval_ms=10.0,
        pre_roll_s=0.05,
        out=str(out),
    )
    assert power_cli.run_power_sample(args) == 0
    doc = json.loads((tmp_path / "run.window.json").read_text())
    assert doc["returncode"] == 0 and doc["duration_s"] >= 0.3
    m = doc["measurement"]
    assert m["power_valid"] == 1 and m["num_devices"] == 2
    assert m["avg_power_w"] == pytest.approx(450.0)
    assert "Power Valid: yes" in capsys.readouterr().out
    # The window file feeds straight back into power-fit.
    assert load_window(str(tmp_path / "run.window.json")).duration_s == pytest.approx(doc["duration_s"])
    assert len(load_telemetry(str(out))) > 10


def test_power_sample_parser_accepts_devices_and_command():
    import argparse

    p = argparse.ArgumentParser()
    power_cli.add_power_sample_args(p)
    args = p.parse_args(["--out", "x.jsonl", "--device", "0", "1", "--command", "python bench.py"])
    assert args.device == [0, 1] and args.command == "python bench.py"
    f = argparse.ArgumentParser()
    power_cli.add_power_fit_args(f)
    with pytest.raises(SystemExit):
        f.parse_args(["--telemetry", "a", "--out", "b", "--window", "1", "2", "--auto-window"])
