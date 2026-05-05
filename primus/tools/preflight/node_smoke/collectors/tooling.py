###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- tooling availability inventory.

Several Tier 1 checks (ECC via amd-smi metric, XGMI via amd-smi topology,
foreign-process enumeration via amd-smi process / lsof, wedged-driver
canary via rocm-smi --version) are best-effort: each collector returns
ok=False and the FAIL rules in _node_status_from then iterate over empty
data, silently no-op'ing. That's fine on a node where the tools are
legitimately absent (containers stripped down for size), but DANGEROUS
in a production cluster -- a node with ECC errors, broken XGMI, leaked
ranks, or a stale ROCm install can PASS smoke just because amd-smi is
missing.

This collector runs ONCE per node, captures which tools were resolvable
in PATH, and feeds three downstream consumers:

  1. A loud `_warn` at run-time so missing tools are visible in the
     srun log right next to "start node-smoke".
  2. An always-on "Tooling availability" section in the aggregator
     report -- a per-node table that's loud even when nothing else is.
  3. The optional `--require-tools` flag, which promotes a missing
     required tool to a hard FAIL via _node_status_from.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..shell_utils import _which

_TRACKED_TOOLS = ("amd-smi", "rocm-smi", "lsof")


def _collect_tooling_inventory() -> Dict[str, Any]:
    """Resolve each tracked tool in PATH and compute per-check coverage.

    Output shape::

        {
            "ok": True,                          # always True; collector itself can't fail
            "tools": {
                "amd-smi":  {"present": True,  "path": "/usr/bin/amd-smi"},
                "rocm-smi": {"present": True,  "path": "/usr/bin/rocm-smi"},
                "lsof":     {"present": True,  "path": "/usr/bin/lsof"},
            },
            "missing": ["rocm-smi"],             # convenience list of absent tools
            "coverage": {
                "ECC":                                  True,
                "XGMI":                                 True,
                "foreign-process":                      True,
                "GPU activity warn":                    True,
                "amd-smi/torch GPU-count cross-check":  True,
                "wedged-driver canary":                 True,
            },
            "uncovered": [],                     # checks with no working tool
        }

    Coverage rules (each check is "covered" if ANY of the listed tools
    is present):

      * ECC                                 -> amd-smi  OR  rocm-smi (--showrasinfo)
      * XGMI link matrix                    -> amd-smi  OR  rocm-smi (--showtopotype)
      * foreign-process enumeration         -> amd-smi  OR  rocm-smi  OR  lsof
      * GPU activity warn (gfx_activity_pct)-> amd-smi  OR  rocm-smi (--showuse)
      * amd-smi/torch GPU-count cross-check -> amd-smi only (rocm-smi enumerates
                                               by `cardN` not by torch's index)
      * wedged-driver canary (rocm-smi --version) -> rocm-smi only
    """
    tools: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for name in _TRACKED_TOOLS:
        path = _which(name)
        tools[name] = {"present": path is not None, "path": path}
        if path is None:
            missing.append(name)

    has_amd = tools["amd-smi"]["present"]
    has_rocm = tools["rocm-smi"]["present"]
    has_lsof = tools["lsof"]["present"]
    coverage = {
        "ECC": has_amd or has_rocm,
        "XGMI": has_amd or has_rocm,
        "foreign-process": has_amd or has_rocm or has_lsof,
        "GPU activity warn": has_amd or has_rocm,
        "amd-smi/torch GPU-count cross-check": has_amd,
        "wedged-driver canary": has_rocm,
    }
    uncovered = [name for name, ok in coverage.items() if not ok]
    return {
        "ok": True,
        "tools": tools,
        "missing": missing,
        "coverage": coverage,
        "uncovered": uncovered,
    }
