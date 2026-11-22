###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Power telemetry: read, sample and fit.

* :func:`load_telemetry` reads CSV logs from ``amd-smi metric/monitor --csv``,
  ``rocm-smi --csv`` or JSON-lines written by :class:`TelemetrySampler`, and
  normalizes them into :class:`TelemetrySample` rows (power, clock, activities).
* :class:`TelemetrySampler` polls amdsmi on a background thread while a
  workload runs on the local GPU.
* :func:`fit_power_profile` least-squares fits the parametric surrogate (idle
  power, activity coefficients and the frequency exponent ``alpha``) to the
  samples, starting from a base profile for anything the data cannot identify.

The fitted profile is written as a ``gpu_power_profiles`` YAML that is meant to
stay with whoever owns the measurements (``PRIMUS_POWER_PROFILE_DIR``).
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from primus.core.projection.power.specs import (
    REFERENCE_ACTIVITY,
    GPUPowerProfile,
    PowerCoefficients,
)


@dataclass
class TelemetrySample:
    power_w: float
    sclk_mhz: Optional[float] = None
    busy: Optional[float] = None
    hbm: Optional[float] = None
    matrix: Optional[float] = None
    link: Optional[float] = None
    timestamp: Optional[float] = None
    device: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# Header aliases, matched against headers lower-cased with non-alphanumerics
# collapsed to "_".  First match wins, so more specific names come first.
_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "power_w": (
        "power_w",
        "current_socket_power",
        "socket_power",
        "average_socket_power",
        "power_usage",
        "current_socket_graphics_package_power",
        "average_graphics_package_power",
        "graphics_package_power",
        "power",
    ),
    "sclk_mhz": ("sclk_mhz", "current_gfxclk", "gfx_clk", "gfxclk", "gfx_clock", "sclk", "gfx_0_clk"),
    "busy": ("busy", "gfx_activity", "gfx_busy", "gfx_util", "gpu_use", "gpu_util", "gpu_busy", "gfx"),
    "hbm": (
        "hbm",
        "umc_activity",
        "mem_activity",
        "memory_activity",
        "mem_util",
        "gpu_memory_use",
        "memory_use",
        "mem_busy",
        "mem",
    ),
    "matrix": ("matrix", "matrix_util", "mfma_util", "matrix_utilization", "mfu"),
    "link": ("link", "xgmi_util", "xgmi_activity", "link_util"),
    "timestamp": ("timestamp", "time", "ts"),
    "device": ("device", "gpu", "gpu_id", "card"),
}
_EXCLUDE_TOKENS = ("vram", "allocated", "cap", "limit", "max", "min", "voltage", "temp")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")


def _match_columns(headers: Sequence[str]) -> Dict[str, str]:
    normed = {h: _norm_header(h) for h in headers}
    mapping: Dict[str, str] = {}
    used = set()
    for field_name, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            hit = None
            for raw, n in normed.items():
                if raw in used:
                    continue
                if field_name not in ("device", "timestamp") and any(t in n for t in _EXCLUDE_TOKENS):
                    continue
                if n == alias or (len(alias) > 3 and alias in n):
                    hit = raw
                    break
            if hit is not None:
                mapping[field_name] = hit
                used.add(hit)
                break
    return mapping


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    m = _NUM_RE.search(str(value))
    return float(m.group()) if m else None


def _to_unix_seconds(value: Any) -> Optional[float]:
    """Epoch seconds / ms / us / ns, or an ISO-8601 string, as Unix seconds."""
    if value is None or value == "":
        return None
    try:
        t = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    if not math.isfinite(t):
        return None
    for scale in (1e9, 1e6, 1e3):  # ns, us, ms since the epoch
        if t > 1e11 * scale / 1e3:
            return t / scale
    return t


def _normalize_fractions(samples: List[TelemetrySample]) -> None:
    """Activities given in percent (0-100) are rescaled to [0, 1]."""
    for name in ("busy", "hbm", "matrix", "link"):
        vals = [getattr(s, name) for s in samples if getattr(s, name) is not None]
        if vals and max(vals) > 1.5:
            for s in samples:
                v = getattr(s, name)
                if v is not None:
                    setattr(s, name, v / 100.0)


def parse_csv_rows(rows: Iterable[Dict[str, Any]], headers: Sequence[str]) -> List[TelemetrySample]:
    mapping = _match_columns(headers)
    if "power_w" not in mapping:
        raise ValueError(f"no power column among {list(headers)}")
    samples: List[TelemetrySample] = []
    for row in rows:
        power = _to_float(row.get(mapping["power_w"]))
        if power is None or power <= 0:
            continue
        kw: Dict[str, Any] = {"power_w": power}
        for name in ("sclk_mhz", "busy", "hbm", "matrix", "link"):
            if name in mapping:
                kw[name] = _to_float(row.get(mapping[name]))
        if "timestamp" in mapping:
            kw["timestamp"] = _to_unix_seconds(row.get(mapping["timestamp"]))
        if "device" in mapping:
            kw["device"] = str(row.get(mapping["device"]))
        samples.append(TelemetrySample(**kw))
    _normalize_fractions(samples)
    return samples


def load_telemetry(path: str) -> List[TelemetrySample]:
    """Read one telemetry log (CSV from amd-smi / rocm-smi, or JSON lines)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        if stripped.startswith("["):
            records = json.loads(stripped)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        headers = sorted({k for r in records for k in r})
        return parse_csv_rows(records, headers)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # amd-smi may print banner lines before the CSV header; start at the first
    # line that looks like a header with a power column.
    start = 0
    for i, ln in enumerate(lines):
        if "power" in ln.lower() and "," in ln:
            start = i
            break
    reader = csv.DictReader(lines[start:])
    return parse_csv_rows(list(reader), reader.fieldnames or [])


def load_telemetry_files(paths: Sequence[str]) -> List[TelemetrySample]:
    """Load several logs; with more than one, GPU ids are prefixed by the file name.

    Each host numbers its GPUs from 0, so logs from different hosts would
    otherwise merge into the same device.
    """
    samples: List[TelemetrySample] = []
    for path in paths:
        rows = load_telemetry(path)
        if len(paths) > 1:
            stem = os.path.splitext(os.path.basename(path))[0]
            rows = [replace(s, device=f"{stem}:{s.device if s.device is not None else '0'}") for s in rows]
        samples.extend(rows)
    return samples


# ---------------------------------------------------------------------------
# Live sampling
# ---------------------------------------------------------------------------


class AmdSmiTelemetryReader:
    """Socket power, clock and activities from amdsmi GPU metrics."""

    def __init__(self, device_index: int = 0):
        from primus.core.projection.hardware_calibration.device_info import (
            _import_amdsmi,
        )

        amdsmi = _import_amdsmi()
        if amdsmi is None:
            raise RuntimeError("amdsmi is not importable")
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if device_index >= len(handles):
            amdsmi.amdsmi_shut_down()
            raise RuntimeError(f"amdsmi has no device {device_index}")
        self._amdsmi = amdsmi
        self._handle = handles[device_index]
        self._device = str(device_index)
        self._lock = threading.Lock()

    def read(self) -> Optional[TelemetrySample]:
        with self._lock:
            m = self._amdsmi.amdsmi_get_gpu_metrics_info(self._handle)
        power = None
        for key in ("current_socket_power", "average_socket_power"):
            v = _to_float(m.get(key))
            if v is not None and 0 < v < 0xFFFF:
                power = v
                break
        if power is None:
            return None
        clocks = [
            c for c in (m.get("current_gfxclks") or []) if isinstance(c, (int, float)) and 0 < c < 10000
        ]
        sclk = sum(clocks) / len(clocks) if clocks else _to_float(m.get("current_gfxclk"))

        def pct(key: str) -> Optional[float]:
            v = _to_float(m.get(key))
            return v / 100.0 if v is not None and 0 <= v <= 100 else None

        return TelemetrySample(
            power_w=power,
            sclk_mhz=sclk if sclk and sclk < 10000 else None,
            busy=pct("average_gfx_activity"),
            hbm=pct("average_umc_activity"),
            timestamp=time.time(),
            device=self._device,
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._amdsmi.amdsmi_shut_down()


class TelemetrySampler:
    """Poll a reader on a background thread; use as a context manager."""

    def __init__(self, reader, interval_s: float = 0.02):
        self._reader = reader
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[TelemetrySample] = []

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                s = self._reader.read()
            except Exception:
                s = None
            if s is not None:
                self.samples.append(s)
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "TelemetrySampler":
        if self._reader is not None:
            self._thread = threading.Thread(target=self._poll, name="primus-power-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def write_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s.as_dict()) + "\n")

    def summary(self, busy_threshold: float = 0.5) -> Dict[str, Any]:
        return summarize_samples(self.samples, busy_threshold)


def make_telemetry_sampler(device_index: int = 0, interval_s: float = 0.02) -> Optional[TelemetrySampler]:
    try:
        return TelemetrySampler(AmdSmiTelemetryReader(device_index), interval_s=interval_s)
    except Exception:
        return None


def summarize_samples(samples: List[TelemetrySample], busy_threshold: float = 0.5) -> Dict[str, Any]:
    """Mean power / clock / activity over all samples and over busy samples."""

    def mean(vals: List[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    busy = [s for s in samples if s.busy is None or s.busy >= busy_threshold]
    return {
        "num_samples": len(samples),
        "avg_power_w": mean([s.power_w for s in samples]),
        "max_power_w": max((s.power_w for s in samples), default=None),
        "busy_avg_power_w": mean([s.power_w for s in busy]),
        "busy_avg_sclk_mhz": mean([s.sclk_mhz for s in busy if s.sclk_mhz]),
        "busy_avg_activity": mean([s.busy for s in busy if s.busy is not None]),
        "busy_avg_hbm": mean([s.hbm for s in busy if s.hbm is not None]),
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class FitReport:
    profile: GPUPowerProfile
    rmse_w: float
    mape: float
    num_samples: int
    fitted: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rmse_w": self.rmse_w,
            "mape": self.mape,
            "num_samples": self.num_samples,
            "fitted": self.fitted,
            "kept": self.kept,
            "warnings": self.warnings,
            "profile": self.profile.to_yaml_dict(),
        }


def _lstsq(rows: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Ordinary least squares via the normal equations (small, dense)."""
    n = len(rows[0]) if rows else 0
    if n == 0 or len(rows) < n:
        return None
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    aty = [sum(r[i] * yi for r, yi in zip(rows, y)) for i in range(n)]
    for i in range(n):  # tiny ridge keeps nearly collinear features solvable
        ata[i][i] += 1e-9 * (ata[i][i] or 1.0)
    # Gaussian elimination with partial pivoting.
    m = [row[:] + [b] for row, b in zip(ata, aty)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _nn_lstsq(rows: List[List[float]], y: List[float], names: List[str]) -> Tuple[Dict[str, float], float]:
    """Least squares with non-negative coefficients (drop negatives and refit)."""
    active = list(range(len(names)))
    while active:
        sub = [[r[i] for i in active] for r in rows]
        sol = _lstsq(sub, y)
        if sol is None:
            active = active[:-1]
            continue
        neg = [active[k] for k, v in enumerate(sol) if v < 0]
        if not neg:
            coefs = {names[i]: 0.0 for i in range(len(names))}
            for k, i in enumerate(active):
                coefs[names[i]] = sol[k]
            pred = [sum(coefs[names[i]] * r[i] for i in range(len(names))) for r in rows]
            sse = sum((p - t) ** 2 for p, t in zip(pred, y))
            return coefs, sse
        active = [i for i in active if i not in neg]
    return {n: 0.0 for n in names}, float("inf")


def fit_power_profile(
    samples: List[TelemetrySample],
    base: GPUPowerProfile,
    matrix_utilization: Optional[float] = None,
    fit_alpha: Optional[bool] = None,
    alpha_grid: Optional[Sequence[float]] = None,
) -> FitReport:
    """Fit idle power, activity coefficients and ``alpha`` to *samples*.

    Args:
        samples: telemetry rows; rows without a power reading are ignored.
        base: profile to start from; anything the data cannot identify is kept.
        matrix_utilization: matrix utilization to assume when the log has no
            ``matrix`` column (e.g. the MFU the training log reported).  When
            given, the matrix term is ``busy * matrix_utilization``.
        fit_alpha: fit the frequency exponent (default: only when the clock
            varies by more than 5% across the samples).
    """
    pts = [s for s in samples if s.power_w and s.power_w > 0]
    warnings: List[str] = []
    if len(pts) < 3:
        raise ValueError(f"need at least 3 telemetry samples with power, got {len(pts)}")

    def col(name: str) -> List[Optional[float]]:
        return [getattr(s, name) for s in pts]

    has = {
        n: any(v is not None and v > 0 for v in col(n)) for n in ("busy", "hbm", "matrix", "link", "sclk_mhz")
    }
    clocks = [s.sclk_mhz for s in pts if s.sclk_mhz]
    clock_varies = bool(clocks) and (max(clocks) - min(clocks)) > 0.05 * max(clocks)
    if fit_alpha is None:
        fit_alpha = clock_varies
    if fit_alpha and not clock_varies:
        warnings.append("clock is nearly constant in the log; alpha is poorly identified")
    if not has["busy"]:
        warnings.append("no GFX activity column; assuming the GPU is busy in every sample")

    c0 = base.coefficients
    matrix_from_util = not has["matrix"] and matrix_utilization is not None
    if not has["matrix"] and matrix_utilization is None:
        warnings.append(
            "no matrix-utilization data; busy and matrix terms are fitted as one, split like the base"
        )
    peak = base.peak_clock_mhz

    def features(s: TelemetrySample, alpha: float) -> Dict[str, float]:
        rel = (s.sclk_mhz / peak) if s.sclk_mhz else 1.0
        sa = rel**alpha
        busy = s.busy if s.busy is not None else 1.0
        f = {"idle": 1.0}
        if has["matrix"]:
            f["busy"] = sa * busy
            f["matrix"] = sa * (s.matrix or 0.0)
        elif matrix_from_util:
            f["busy"] = sa * busy
            f["matrix"] = sa * busy * float(matrix_utilization)
        else:
            # One clocked term: watts at busy=1 and the reference matrix utilization.
            f["clocked"] = sa * busy
        if has["hbm"]:
            f["hbm"] = s.hbm or 0.0
        if has["link"]:
            f["link"] = s.link or 0.0
        return f

    grid = list(alpha_grid) if alpha_grid else [round(1.0 + 0.05 * i, 2) for i in range(61)]
    alphas = grid if fit_alpha else [base.alpha]
    best = None
    y = [s.power_w for s in pts]
    for a in alphas:
        feats = [features(s, a) for s in pts]
        names = list(feats[0].keys())
        rows = [[f[n] for n in names] for f in feats]
        coefs, sse = _nn_lstsq(rows, y, names)
        if best is None or sse < best[1]:
            best = (a, sse, coefs, names, rows)
    alpha, sse, coefs, names, rows = best

    fitted, kept = ["idle_w"], []
    busy_w, matrix_w = c0.busy_w, c0.matrix_w
    if "clocked" in coefs:
        ref_w = c0.busy_w + c0.matrix_w * REFERENCE_ACTIVITY["matrix"]
        ratio = coefs["clocked"] / ref_w if ref_w > 0 else 1.0
        busy_w, matrix_w = c0.busy_w * ratio, c0.matrix_w * ratio
        fitted.append("busy_w+matrix_w (joint)")
    else:
        busy_w, matrix_w = coefs.get("busy", c0.busy_w), coefs.get("matrix", c0.matrix_w)
        fitted += ["busy_w", "matrix_w"]
    hbm_w = coefs["hbm"] if "hbm" in coefs else c0.hbm_w
    link_w = coefs["link"] if "link" in coefs else c0.link_w
    (fitted if "hbm" in coefs else kept).append("hbm_w")
    (fitted if "link" in coefs else kept).append("link_w")
    (fitted if fit_alpha else kept).append("alpha")

    profile = base.with_overrides(
        idle_w=coefs.get("idle", base.idle_w),
        alpha=alpha,
        coefficients=PowerCoefficients(busy_w=busy_w, matrix_w=matrix_w, hbm_w=hbm_w, link_w=link_w),
        source="telemetry-fit",
        notes=[f"fitted to {len(pts)} telemetry samples"],
    )
    pred = [sum(coefs[n] * r[i] for i, n in enumerate(names)) for r in rows]
    rmse = math.sqrt(sse / len(pts))
    mape = sum(abs(p - t) / t for p, t in zip(pred, y)) / len(pts)
    return FitReport(
        profile=profile,
        rmse_w=rmse,
        mape=mape,
        num_samples=len(pts),
        fitted=fitted,
        kept=kept,
        warnings=warnings,
    )


def write_profile_yaml(profile: GPUPowerProfile, path: str, arch: Optional[str] = None) -> None:
    import yaml

    doc = {"gpu_power_profiles": {arch or profile.name: profile.to_yaml_dict()}}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
