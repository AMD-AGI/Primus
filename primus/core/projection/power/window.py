###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Measured power over a load window, with a validity verdict.

Telemetry usually runs longer than the load: it starts before warmup and stops
after teardown.  :func:`measure_window` restricts each GPU's power trace to the
interval the load actually ran, interpolates power at the window edges,
integrates it with the trapezoid rule and reports per-GPU average power and
energy.  The result is valid only when every GPU's trace covers the whole
window without gaps; otherwise the measured values are withheld and the
reasons are listed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from primus.core.projection.power.telemetry import TelemetrySample

# Widest gap between consecutive samples (including the samples bracketing the
# window edges) that still counts as continuous coverage.
MAX_SAMPLE_GAP_S = 3.0

_WINDOW_KEYS = (
    ("start_unix", "end_unix"),
    ("benchmark_start_time_unix", "benchmark_end_time_unix"),
    ("start", "end"),
)


@dataclass(frozen=True)
class MeasurementWindow:
    start_unix: float
    end_unix: float
    source: str = "explicit"

    @property
    def duration_s(self) -> float:
        return self.end_unix - self.start_unix

    def trimmed(self, head_s: float = 0.0, tail_s: float = 0.0) -> "MeasurementWindow":
        """Drop *head_s* after the start (e.g. warmup) and *tail_s* before the end."""
        if not head_s and not tail_s:
            return self
        return MeasurementWindow(self.start_unix + head_s, self.end_unix - tail_s, self.source)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start_unix": self.start_unix,
            "end_unix": self.end_unix,
            "duration_s": self.duration_s,
            "source": self.source,
        }


def window_from_mapping(data: Dict[str, Any], source: str = "file") -> MeasurementWindow:
    """Window from ``start_unix``/``end_unix``, ``benchmark_start_time_unix``/... or ``start``/``end``."""
    for start_key, end_key in _WINDOW_KEYS:
        if start_key in data and end_key in data:
            return MeasurementWindow(float(data[start_key]), float(data[end_key]), source)
    if isinstance(data.get("window"), dict):
        return window_from_mapping(data["window"], source)
    raise ValueError(f"no window bounds among keys {sorted(data)}")


def load_window(path: str) -> MeasurementWindow:
    with open(path, "r", encoding="utf-8") as f:
        return window_from_mapping(json.load(f), source=path)


def auto_window(
    samples: Sequence[TelemetrySample], busy_threshold: float = 0.5
) -> Optional[MeasurementWindow]:
    """First to last timestamp at which any GPU reports activity >= *busy_threshold*."""
    busy = [
        s.timestamp
        for s in samples
        if s.timestamp is not None and s.busy is not None and s.busy >= busy_threshold
    ]
    if len(busy) < 2:
        return None
    return MeasurementWindow(min(busy), max(busy), source=f"auto(busy>={busy_threshold:g})")


@dataclass
class DeviceEnergy:
    device: str
    energy_j: float
    avg_power_w: float
    num_samples: int
    max_gap_s: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "energy_j": self.energy_j,
            "avg_power_w": self.avg_power_w,
            "num_samples": self.num_samples,
            "max_gap_s": self.max_gap_s,
        }


@dataclass
class WindowMeasurement:
    """Measured GPU power over one window; values are ``None`` unless ``valid``."""

    window: MeasurementWindow
    valid: bool
    reasons: List[str] = field(default_factory=list)
    devices: List[DeviceEnergy] = field(default_factory=list)
    num_devices: int = 0
    num_samples: int = 0
    max_gap_s: Optional[float] = None
    avg_power_w: Optional[float] = None  # mean per GPU
    total_avg_power_w: Optional[float] = None  # sum over GPUs
    total_energy_j: Optional[float] = None  # sum over GPUs

    def as_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window.as_dict(),
            "power_valid": 1 if self.valid else 0,
            "reasons": list(self.reasons),
            "num_devices": self.num_devices,
            "num_samples": self.num_samples,
            "max_gap_s": self.max_gap_s,
            "avg_power_w": self.avg_power_w,
            "total_avg_power_w": self.total_avg_power_w,
            "total_energy_j": self.total_energy_j,
            "devices": [d.as_dict() for d in self.devices] if self.valid else [],
            "scope": "GPU board power only; excludes host, network, cooling and facility power",
        }

    def describe(self) -> str:
        if not self.valid:
            return f"Power Valid: no ({', '.join(self.reasons)})"
        return (
            f"Power Valid: yes | {self.num_devices} GPU(s), {self.window.duration_s:.1f} s | "
            f"Avg Power/GPU: {self.avg_power_w:.1f} W | GPU Energy: {self.total_energy_j:.1f} J"
        )


def _append(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _interpolate(trace: List[Tuple[float, float]], t: float) -> float:
    """Power at *t*, linearly interpolated between the bracketing samples."""
    for (t0, p0), (t1, p1) in zip(trace, trace[1:]):
        if t0 <= t <= t1:
            return p0 if t1 == t0 else p0 + (p1 - p0) * (t - t0) / (t1 - t0)
    raise ValueError("timestamp is not bracketed by the trace")


def _device_traces(samples: Iterable[TelemetrySample]) -> Dict[str, List[Tuple[float, float]]]:
    traces: Dict[str, Dict[float, float]] = {}
    for s in samples:
        traces.setdefault(s.device if s.device is not None else "0", {})[float(s.timestamp)] = float(
            s.power_w
        )
    return {d: sorted(points.items()) for d, points in traces.items()}


def measure_window(
    samples: Sequence[TelemetrySample],
    window: MeasurementWindow,
    max_gap_s: float = MAX_SAMPLE_GAP_S,
    expected_devices: Optional[int] = None,
) -> WindowMeasurement:
    """Integrate each GPU's power over *window* and validate the coverage.

    Reasons (any one makes the measurement invalid):

    ``invalid_window``           end is not after start
    ``no_samples``               no power samples at all
    ``missing_timestamps``       some samples carry no timestamp
    ``window_not_covered``       a GPU has no sample at/before the start or at/after the end
    ``sample_gap_exceeds_max``   consecutive samples of a GPU are more than *max_gap_s* apart
    ``device_count_mismatch``    the number of GPUs differs from *expected_devices*
    """
    reasons: List[str] = []
    result = WindowMeasurement(window=window, valid=False, reasons=reasons)
    if not (math.isfinite(window.start_unix) and math.isfinite(window.end_unix)) or window.duration_s <= 0:
        _append(reasons, "invalid_window")
        return result
    powered = [s for s in samples if s.power_w and s.power_w > 0]
    if not powered:
        _append(reasons, "no_samples")
        return result
    if any(s.timestamp is None for s in powered):
        _append(reasons, "missing_timestamps")
        return result

    traces = _device_traces(powered)
    result.num_devices = len(traces)
    if expected_devices is not None and len(traces) != expected_devices:
        _append(reasons, "device_count_mismatch")

    devices: List[DeviceEnergy] = []
    worst_gap = 0.0
    total_samples = 0
    for device, trace in sorted(traces.items()):
        times = [t for t, _ in trace]
        if times[0] > window.start_unix or times[-1] < window.end_unix:
            _append(reasons, "window_not_covered")
            continue
        # Samples bracketing the window plus everything inside it.
        first = max(i for i, t in enumerate(times) if t <= window.start_unix)
        last = min(i for i, t in enumerate(times) if t >= window.end_unix)
        span = trace[first : last + 1]
        gap = max((b[0] - a[0] for a, b in zip(span, span[1:])), default=0.0)
        worst_gap = max(worst_gap, gap)
        if gap > max_gap_s:
            _append(reasons, "sample_gap_exceeds_max")
        clipped = [(window.start_unix, _interpolate(span, window.start_unix))]
        clipped += [(t, p) for t, p in span if window.start_unix < t < window.end_unix]
        clipped.append((window.end_unix, _interpolate(span, window.end_unix)))
        energy = sum((t1 - t0) * (p0 + p1) / 2.0 for (t0, p0), (t1, p1) in zip(clipped, clipped[1:]))
        inside = len(clipped) - 2
        total_samples += inside
        devices.append(DeviceEnergy(device, energy, energy / window.duration_s, inside, gap))

    result.num_samples = total_samples
    result.max_gap_s = worst_gap
    if reasons:
        return result
    result.valid = True
    result.devices = devices
    result.total_energy_j = sum(d.energy_j for d in devices)
    result.total_avg_power_w = result.total_energy_j / window.duration_s
    result.avg_power_w = result.total_avg_power_w / len(devices)
    return result


def samples_in_window(samples: Sequence[TelemetrySample], window: MeasurementWindow) -> List[TelemetrySample]:
    return [
        s for s in samples if s.timestamp is not None and window.start_unix <= s.timestamp <= window.end_unix
    ]
