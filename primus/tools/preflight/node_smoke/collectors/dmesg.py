###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Recent dmesg error scan.

Greps the last ``window_minutes`` of dmesg for known-bad patterns. Best
effort: missing dmesg / failed read becomes ``ok=False`` rather than
raising.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List

# Regex (NOT substring) patterns matched case-insensitively against each
# dmesg line. They MUST be valid Python regex -- if you only want a literal
# substring (e.g. ``mce: ``), it's still a valid regex with no specials.
# Why regex: real amdgpu failure lines look like
#   ``amdgpu 0000:05:00.0: amdgpu_device_resume failed: -19``
#   ``amdgpu: [drm] *ERROR* ring sdma0 timeout``
# so we need ``amdgpu.*(error|fail|timeout)`` -- a substring match against
# the literal pattern ``amdgpu.*error`` would essentially never fire.
_DMESG_PATTERNS = (
    r"\bxid\b",
    r"hardware error",
    r"gpu reset",
    r"hung_task",
    r"hung task",
    r"page allocation failure",
    r"soft lockup",
    r"amdgpu.*(error|fail|timeout)",
    r"\*error\*",  # [drm] *ERROR*
    r"mce: ",
)


def _collect_dmesg_errors(window_minutes: int = 15) -> Dict[str, Any]:
    """Best-effort grep of recent dmesg lines for known-bad patterns.

    Returns a dict with ``ok`` (bool), ``matches`` (list of matched lines, capped),
    and ``error`` (str) when dmesg cannot be read.
    """
    out: Dict[str, Any] = {"ok": True, "matches": [], "error": None}
    try:
        # ``--since`` requires recent util-linux; fall back to the last 2000 lines.
        try:
            cp = subprocess.run(
                ["dmesg", "--since", f"-{window_minutes}min"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            if cp.returncode != 0:
                raise RuntimeError(cp.stderr.strip() or f"rc={cp.returncode}")
            text = cp.stdout
        except Exception:
            cp = subprocess.run(
                ["dmesg"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            if cp.returncode != 0:
                out["ok"] = False
                out["error"] = (cp.stderr or "").strip() or f"rc={cp.returncode}"
                return out
            text = "\n".join(cp.stdout.splitlines()[-2000:])

        import re

        matches: List[str] = []
        # Pre-compile each pattern with re.IGNORECASE; a malformed regex is
        # logged into ``out['pattern_errors']`` but never aborts the scan.
        compiled: List[Any] = []
        for p in _DMESG_PATTERNS:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                out.setdefault("pattern_errors", []).append(f"{p!r}: {e}")
        for line in text.splitlines():
            if any(pat.search(line) for pat in compiled):
                matches.append(line)
                if len(matches) >= 50:
                    break
        out["matches"] = matches
        return out
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
        return out
