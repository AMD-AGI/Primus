###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stateless utility helpers shared across collectors and orchestration.

Every function here is pure (no I/O effects beyond reading sysfs / running
a tiny subprocess) and never raises -- callers expect best-effort
defaults so a missing tool / file degrades gracefully.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional


def _which(prog: str) -> Optional[str]:
    """Tiny shutil.which() replacement that doesn't pull in shutil at import."""
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        p = os.path.join(d, prog)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _read_text(path: str, default: str = "") -> str:
    """Best-effort read of a small sysfs/proc text file. Never raises."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return default


def _parse_os_release_pretty() -> Optional[str]:
    """Return PRETTY_NAME from /etc/os-release, or None."""
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return v
    except Exception:
        pass
    return None


def _resolve_gpu_bdf(props: Any) -> Optional[str]:
    """Return the PCIe BDF (e.g. ``"0000:75:00.0"``) for a torch device.

    ``torch.cuda.get_device_properties(i).pci_bus_id`` is annoyingly
    polymorphic across PyTorch + ROCm versions: sometimes a string in the
    canonical ``domain:bus:device.function`` form (lowercase or uppercase),
    sometimes an int (just the bus byte). We coerce all variants into the
    canonical lowercase form and verify the sysfs directory actually
    exists before returning -- so the caller can read PCIe link info
    without an extra existence check.

    Returns None if the BDF cannot be resolved (caller should still
    capture HBM via ``mem_get_info`` and skip PCIe sysfs reads).
    """
    raw = getattr(props, "pci_bus_id", None)
    if raw is None:
        return None
    # 1) String form. Could be "0000:05:00.0", "05:00.0", or uppercase.
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return None
        candidates = [s, f"0000:{s}" if not s.startswith("0000:") else s]
        for c in candidates:
            if os.path.isdir(f"/sys/bus/pci/devices/{c}"):
                return c
        return None
    # 2) Int form (just the bus byte). Standard layout for AMD GPUs is
    # 0000:<bus>:00.0; verify with sysfs and fall back to a glob if the
    # device.function differs from 00.0 on this host.
    if isinstance(raw, int):
        bus_hex = f"{raw:02x}"
        primary = f"0000:{bus_hex}:00.0"
        if os.path.isdir(f"/sys/bus/pci/devices/{primary}"):
            return primary
        import glob

        matches = sorted(glob.glob(f"/sys/bus/pci/devices/0000:{bus_hex}:*"))
        if matches:
            return os.path.basename(matches[0])
        return None
    return None


def _systemctl_is_active(unit: str) -> Optional[str]:
    """Return ``systemctl is-active <unit>`` ('active'/'inactive'/'failed'/...)
    or None if systemctl is missing / errors. Always best-effort."""
    if _which("systemctl") is None:
        return None
    try:
        cp = subprocess.run(
            ["systemctl", "is-active", unit],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=False,
        )
        # systemctl returns non-zero for inactive/failed -- that's fine,
        # we just want the textual state.
        return (cp.stdout or "").strip() or "unknown"
    except Exception:
        return None


def _parse_size_with_unit(s: str) -> Optional[int]:
    """Parse a size string into bytes.

    Accepts (case-insensitive)::

        "12345"        -> 12345
        "256 MB"       -> 268435456
        "256MB"        -> 268435456     (no space)
        "12.5 GiB"     -> 13421772800
        "12.5GiB"      -> 13421772800
        "  -1  "       -> -1            (sentinel for "unlimited")

    Returns ``None`` for empty input, an unrecognised unit (e.g.
    ``"500 MHz"``), or any input the regex cannot fully match (e.g.
    ``"12 GB extra"``). This is deliberate: a silent fallthrough
    to ``num * 1`` would let frequency / count strings masquerade
    as byte counts and corrupt downstream comparisons.
    """
    if not s:
        return None
    import re

    m = re.match(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\.?\s*$",
        s,
    )
    if not m:
        return None
    units = {
        "b": 1,
        "k": 1 << 10,
        "kb": 1 << 10,
        "kib": 1 << 10,
        "m": 1 << 20,
        "mb": 1 << 20,
        "mib": 1 << 20,
        "g": 1 << 30,
        "gb": 1 << 30,
        "gib": 1 << 30,
        "t": 1 << 40,
        "tb": 1 << 40,
        "tib": 1 << 40,
    }
    unit = (m.group(2) or "b").lower()
    if unit not in units:
        return None
    try:
        return int(float(m.group(1)) * units[unit])
    except ValueError:
        return None


def _findings_to_dicts(findings: List[Any]) -> List[Dict[str, Any]]:
    """Normalize Finding dataclasses (different modules each define their own)
    into plain dicts."""
    return [
        {
            "level": getattr(f, "level", "info"),
            "message": getattr(f, "message", str(f)),
            "details": getattr(f, "details", {}),
        }
        for f in findings
    ]
