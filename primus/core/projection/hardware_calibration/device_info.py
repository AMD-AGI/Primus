###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Local-GPU identification and live clock sampling for hardware calibration.

The calibration measures the GPU it runs on, so it identifies that part from
the device itself rather than from ``--gpu-arch`` (which names the *target* of
a simulation and may differ from the bench host).

Live clocks are what the part is running at *while the workload runs*, never
the DPM ceiling.  They come from, in order of preference:

* **amdsmi** GPU metrics: the current clock of every XCD (averaged) and the
  current memory clock;
* the amdgpu sysfs DPM tables (``pp_dpm_sclk`` / ``pp_dpm_mclk``), whose
  starred line is the current clock -- of the first XCD only.

Both are optional: without either, no clocks are sampled and fixed-length
warm-ups replace the settle check.
"""

from __future__ import annotations

import contextlib
import glob
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_DRM_GLOB = "/sys/class/drm/card*/device"
_DPM_LINE = re.compile(r"^\s*(\w+):\s*(\d+)\s*mhz\s*(\*)?", re.IGNORECASE)

# MI325X is the MI300X die with HBM3E; VRAM size is the only thing the device
# reports that tells them apart (192 GiB vs 256 GiB).
_MI325X_MIN_VRAM_BYTES = 224 * 1024**3
_MI300A_NUM_CUS = 228


@dataclass
class DeviceInfo:
    index: int
    name: str
    gcn_arch: str
    num_cus: int
    total_memory_bytes: int
    profile_key: Optional[str]
    peak_hbm_gbps: Optional[float]
    max_sclk_mhz: Optional[int]
    max_mclk_mhz: Optional[int]
    pci_bdf: Optional[str] = None
    sysfs_dir: Optional[str] = None
    sclk_levels_mhz: List[int] = field(default_factory=list)
    mclk_levels_mhz: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_dpm_table(path: str) -> Tuple[List[int], Optional[int]]:
    """Parse an amdgpu ``pp_dpm_*`` file into ``(levels_mhz, current_mhz)``.

    Numbered lines are the DPM levels; the starred line (which may be an
    ``S:`` line between levels) is the current clock.
    """
    levels: List[int] = []
    current: Optional[int] = None
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        return levels, None
    for line in lines:
        match = _DPM_LINE.match(line)
        if not match:
            continue
        label, mhz, star = match.group(1), int(match.group(2)), match.group(3)
        if label.isdigit():
            levels.append(mhz)
        if star:
            current = mhz
    return levels, current


def _pci_bdf(props) -> Optional[str]:
    try:
        return f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    except AttributeError:
        return None


def _sysfs_dir_for_device(bdf: Optional[str]) -> Optional[str]:
    """Find the drm sysfs directory of the device at PCI address *bdf*."""
    if bdf is None:
        return None
    for device_dir in sorted(glob.glob(_DRM_GLOB)):
        try:
            with open(os.path.join(device_dir, "uevent"), "r") as f:
                uevent = f.read().lower()
        except OSError:
            continue
        if f"pci_slot_name={bdf}" in uevent and os.path.exists(os.path.join(device_dir, "pp_dpm_sclk")):
            return device_dir
    return None


def resolve_profile_key(gcn_arch: str, num_cus: int, total_memory_bytes: int) -> Optional[str]:
    """Map the local device onto a key of the Origami hardware-profile table."""
    arch = gcn_arch.split(":")[0].lower()
    if arch == "gfx942":
        if num_cus == _MI300A_NUM_CUS:
            return "mi300a"
        return "mi325x" if total_memory_bytes >= _MI325X_MIN_VRAM_BYTES else "mi300x"
    if arch == "gfx950":
        return "mi355x"
    return None


def _peak_hbm_gbps(profile_key: Optional[str]) -> Optional[float]:
    if profile_key is None:
        return None
    from primus.core.projection.simulation_backends.origami_backend import (
        _KNOWN_PROFILES,
    )

    profile = _KNOWN_PROFILES.get(profile_key)
    return profile.hbm_bandwidth_gbps if profile is not None else None


def describe_device(device_index: int) -> DeviceInfo:
    import torch

    props = torch.cuda.get_device_properties(device_index)
    gcn_arch = getattr(props, "gcnArchName", "") or ""
    num_cus = int(props.multi_processor_count)
    total_memory = int(props.total_memory)
    profile_key = resolve_profile_key(gcn_arch, num_cus, total_memory)

    bdf = _pci_bdf(props)
    sysfs_dir = _sysfs_dir_for_device(bdf)
    sclk_levels: List[int] = []
    mclk_levels: List[int] = []
    if sysfs_dir is not None:
        sclk_levels, _ = read_dpm_table(os.path.join(sysfs_dir, "pp_dpm_sclk"))
        mclk_levels, _ = read_dpm_table(os.path.join(sysfs_dir, "pp_dpm_mclk"))

    # torch reports both clocks in kHz.
    clock_khz = getattr(props, "clock_rate", 0) or 0
    mem_clock_khz = getattr(props, "memory_clock_rate", 0) or 0
    max_sclk = max(sclk_levels) if sclk_levels else (clock_khz // 1000 or None)
    max_mclk = max(mclk_levels) if mclk_levels else (mem_clock_khz // 1000 or None)

    return DeviceInfo(
        index=device_index,
        name=torch.cuda.get_device_name(device_index),
        gcn_arch=gcn_arch.split(":")[0],
        num_cus=num_cus,
        total_memory_bytes=total_memory,
        profile_key=profile_key,
        peak_hbm_gbps=_peak_hbm_gbps(profile_key),
        max_sclk_mhz=max_sclk,
        max_mclk_mhz=max_mclk,
        pci_bdf=bdf,
        sysfs_dir=sysfs_dir,
        sclk_levels_mhz=sclk_levels,
        mclk_levels_mhz=mclk_levels,
    )


# ---------------------------------------------------------------------------
# Live clock readers
# ---------------------------------------------------------------------------

# amdsmi reports absent XCDs / unsupported fields as "N/A" or 0xFFFF.
_MAX_PLAUSIBLE_CLOCK_MHZ = 10_000
# Longest a clock may take to settle before the timed runs start anyway.
SETTLE_TIMEOUT_S = 5.0


@dataclass
class ClockReading:
    sclk_mhz: Optional[float]  # mean over XCDs
    mclk_mhz: Optional[float]
    xcd_sclks_mhz: List[int] = field(default_factory=list)


def _valid_clock(value) -> bool:
    return isinstance(value, (int, float)) and 0 < value < _MAX_PLAUSIBLE_CLOCK_MHZ


def _import_amdsmi():
    """Import amdsmi, falling back to the copy ROCm ships outside site-packages."""
    try:
        import amdsmi  # type: ignore[import-not-found]

        return amdsmi
    except ImportError:
        pass
    path = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "share", "amd_smi")
    if not os.path.isdir(path):
        return None
    if path not in sys.path:
        sys.path.append(path)
    try:
        import amdsmi  # type: ignore[import-not-found]

        return amdsmi
    except Exception:
        return None


class AmdSmiClockReader:
    """Current per-XCD SCLK and MCLK from amdsmi GPU metrics (~0.3 ms per read)."""

    source = "amdsmi"

    def __init__(self, bdf: str):
        amdsmi = _import_amdsmi()
        if amdsmi is None:
            raise RuntimeError("amdsmi is not importable")
        amdsmi.amdsmi_init()
        self._amdsmi = amdsmi
        self._handle = None
        for handle in amdsmi.amdsmi_get_processor_handles():
            if str(amdsmi.amdsmi_get_gpu_device_bdf(handle)).lower() == bdf.lower():
                self._handle = handle
                break
        if self._handle is None:
            amdsmi.amdsmi_shut_down()
            raise RuntimeError(f"amdsmi has no device at {bdf}")
        self._lock = threading.Lock()

    def read(self) -> Optional[ClockReading]:
        with self._lock:
            metrics = self._amdsmi.amdsmi_get_gpu_metrics_info(self._handle)
        xcds = [int(v) for v in (metrics.get("current_gfxclks") or []) if _valid_clock(v)]
        if not xcds and _valid_clock(metrics.get("current_gfxclk")):
            xcds = [int(metrics["current_gfxclk"])]
        mclk = metrics.get("current_uclk")
        if not _valid_clock(mclk):
            # Parts that report UCLK per AID leave current_uclk unset.
            aids = [v for v in (metrics.get("current_uclk_aid") or []) if _valid_clock(v)]
            mclk = sum(aids) / len(aids) if aids else None
        return ClockReading(
            sclk_mhz=sum(xcds) / len(xcds) if xcds else None,
            mclk_mhz=float(mclk) if _valid_clock(mclk) else None,
            xcd_sclks_mhz=xcds,
        )

    def close(self) -> None:
        try:
            self._amdsmi.amdsmi_shut_down()
        except Exception:
            pass  # amdsmi may already be shut down; nothing is left to release.


class SysfsClockReader:
    """Current SCLK (first XCD only) and MCLK from the amdgpu DPM tables."""

    source = "sysfs"

    def __init__(self, sysfs_dir: str):
        self._sclk_path = os.path.join(sysfs_dir, "pp_dpm_sclk")
        self._mclk_path = os.path.join(sysfs_dir, "pp_dpm_mclk")

    def read(self) -> Optional[ClockReading]:
        # A table read mid-transition may have no starred level.
        _, sclk = read_dpm_table(self._sclk_path)
        _, mclk = read_dpm_table(self._mclk_path)
        if sclk is None and mclk is None:
            return None
        return ClockReading(
            sclk_mhz=float(sclk) if sclk is not None else None,
            mclk_mhz=float(mclk) if mclk is not None else None,
            xcd_sclks_mhz=[sclk] if sclk is not None else [],
        )

    def close(self) -> None:
        pass


def make_clock_reader(info: DeviceInfo):
    """Best available live-clock reader for *info*'s device, or *None*."""
    if info.pci_bdf is not None:
        try:
            reader = AmdSmiClockReader(info.pci_bdf)
            if reader.read() is not None:
                return reader
            reader.close()
        except Exception:
            pass  # amdsmi unusable for this device; fall back to sysfs.
    if info.sysfs_dir is not None:
        return SysfsClockReader(info.sysfs_dir)
    return None


# ---------------------------------------------------------------------------
# Sampling during timed kernels, and waiting for steady state
# ---------------------------------------------------------------------------


def _summarize(samples: List[float]) -> Optional[Dict[str, float]]:
    if not samples:
        return None
    ordered = sorted(samples)
    return {
        "median_mhz": float(ordered[len(ordered) // 2]),
        "mean_mhz": sum(samples) / len(samples),
        "min_mhz": float(ordered[0]),
        "max_mhz": float(ordered[-1]),
        "num_samples": len(samples),
    }


class ClockSampler:
    """Poll the live SCLK / MCLK of one device on a background thread.

    Use as a context manager for the sampler's lifetime and wrap the timed
    kernels in :meth:`window`; only samples taken inside a window are kept, so
    warm-up, tensor setup and idle gaps do not leak into the result.
    ``summary()`` then reports what the part actually ran at while the timed
    kernels executed.
    """

    def __init__(self, reader, interval_s: float = 0.002):
        self._reader = reader
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._active = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.readings: List[ClockReading] = []

    @contextlib.contextmanager
    def window(self):
        self._active.set()
        try:
            yield
        finally:
            self._active.clear()

    def _poll(self) -> None:
        while not self._stop.is_set():
            if not self._active.wait(timeout=0.05):
                continue
            try:
                reading = self._reader.read()
            except Exception:
                reading = None
            if reading is not None and self._active.is_set():
                self.readings.append(reading)
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "ClockSampler":
        if self._reader is not None:
            self._thread = threading.Thread(target=self._poll, name="primus-clock-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> Dict[str, Any]:
        sclk = [r.sclk_mhz for r in self.readings if r.sclk_mhz is not None]
        mclk = [r.mclk_mhz for r in self.readings if r.mclk_mhz is not None]
        out: Dict[str, Any] = {
            "source": getattr(self._reader, "source", None),
            "sclk": _summarize(sclk),
            "mclk": _summarize(mclk),
        }
        per_xcd = [r.xcd_sclks_mhz for r in self.readings if r.xcd_sclks_mhz]
        if out["sclk"] is not None and per_xcd and len({len(x) for x in per_xcd}) == 1:
            columns = list(zip(*per_xcd))
            out["sclk"]["per_xcd_median_mhz"] = [float(sorted(c)[len(c) // 2]) for c in columns]
        return out


def run_until_settled(
    step: Callable[[], None],
    reader,
    signal: str,
    min_s: float,
    max_s: float = SETTLE_TIMEOUT_S,
    window_s: float = 0.3,
    rel_tol: float = 0.03,
) -> Dict[str, Any]:
    """Keep running the workload until the clock it depends on stops moving.

    *step* runs a short burst of the workload and blocks until it finishes.
    *signal* is ``"sclk"`` (compute-bound work) or ``"mclk"`` (memory-bound).
    Settled means: at least *min_s* elapsed, and every reading in the last
    *window_s* is within *rel_tol* of their mean.  Clocks right after launch
    are not representative -- SCLK boosts to its ceiling and then drops under
    power management; MCLK ramps up over a few hundred ms -- so timing
    without this would measure a transient.  Without a reader, runs for
    *min_s*.
    """
    attr = f"{signal}_mhz"
    history: List[Tuple[float, float]] = []
    start = time.monotonic()
    while True:
        step()
        now = time.monotonic() - start
        reading = None
        if reader is not None:
            try:
                reading = reader.read()
            except Exception:
                reading = None
        value = getattr(reading, attr, None) if reading is not None else None
        if value is not None:
            history.append((now, value))
        if now < min_s:
            continue
        if reader is None:
            return {"signal": signal, "settled": None, "settle_s": now, "settled_mhz": None}
        recent = [v for t, v in history if t >= now - window_s]
        if len(recent) >= 5:
            mean = sum(recent) / len(recent)
            if max(recent) - min(recent) <= rel_tol * mean:
                return {"signal": signal, "settled": True, "settle_s": now, "settled_mhz": mean}
        if now >= max_s:
            last = history[-1][1] if history else None
            return {"signal": signal, "settled": False, "settle_s": now, "settled_mhz": last}
