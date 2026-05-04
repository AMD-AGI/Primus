###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- D-2: XGMI topology matrix via ``amd-smi topology`` (text parser),
with rocm-smi --showtopotype as a cross-tool fallback."""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from ..shell_utils import _which
from .rocm_smi import _rocm_smi_topotype_json


def _collect_xgmi_topology() -> Dict[str, Any]:
    """Try amd-smi topology first; fall back to rocm-smi --showtopotype.

    Both tools produce slightly different per-row labels (amd-smi uses
    PCIe BDFs, rocm-smi uses DRM device indices), but the link_types
    matrix and non_xgmi_pairs computation is identical, so downstream
    consumers in _node_status_from / the aggregator don't need to know
    which tool produced the data.
    """
    out = _collect_xgmi_topology_amd_smi()
    if out.get("ok"):
        return out
    rocm = _rocm_smi_topotype_json()
    if rocm.get("ok"):
        return {
            "ok": True,
            "tool": rocm.get("tool") or "rocm-smi --showtopotype --json",
            "bdfs": [],  # rocm-smi reports DRM indices, not PCIe BDFs
            "matrix": rocm.get("link_types") or [],
            "n_gpus": rocm.get("n_gpus") or 0,
            "non_xgmi_pairs": rocm.get("non_xgmi_pairs") or [],
            "amd_smi_error": out.get("error"),
        }
    # Both failed -- preserve amd-smi's error for the operator and
    # surface rocm-smi's separately so they can debug both paths.
    out["rocm_smi_error"] = rocm.get("error")
    return out


def _collect_xgmi_topology_amd_smi() -> Dict[str, Any]:
    """Parse ``amd-smi topology`` and return a square link-type matrix.

    ``amd-smi topology`` emits several BDF-labelled sub-tables (ACCESS,
    WEIGHT, HOPS, LINK TYPE, NUMA BW, ...). We pick the ``LINK TYPE TABLE``
    sub-section, which contains values like ``SELF`` (diagonal) and
    ``XGMI`` / ``PCIE`` / ``PIX`` / ``SOC`` etc. Off-diagonal cells that
    aren't ``XGMI`` are recorded as ``non_xgmi_pairs`` and treated as a
    hard fail by ``_node_status_from`` -- the moment a single GPU pair
    falls back to PCIe inside a node, intra-node collectives lose 5-10x
    of the bandwidth NCCL/RCCL expects.

    The on-disk shape:

        {
            "ok": bool,
            "tool": "amd-smi topology" | None,
            "bdfs": ["0000:05:00.0", ...],
            "matrix": [["SELF","XGMI",...], ["XGMI","SELF",...], ...],
            "n_gpus": int,
            "non_xgmi_pairs": [(i, j, link_type), ...],
            "error": "..."
        }
    """
    out: Dict[str, Any] = {
        "ok": False, "tool": None, "bdfs": [], "matrix": [],
        "non_xgmi_pairs": [],
    }
    if _which("amd-smi") is None:
        out["error"] = "amd-smi not found in PATH"
        return out
    try:
        cp = subprocess.run(
            ["amd-smi", "topology"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15, check=False,
        )
    except subprocess.TimeoutExpired:
        out["error"] = "amd-smi topology timed out"
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
    if cp.returncode != 0:
        out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
        return out

    text = cp.stdout

    # Parse: find the `LINK TYPE TABLE:` section, then the BDF header row,
    # then the per-BDF data rows. Stop at the next section header (any all-
    # caps label ending in `TABLE:`) or end of text.
    import re
    bdf_re = re.compile(r"\b([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)\b")
    section_header_re = re.compile(r"^\s*[A-Z][A-Z0-9 -]+TABLE:\s*$")

    lines = text.splitlines()
    try:
        idx = next(
            i for i, l in enumerate(lines)
            if l.strip().upper() == "LINK TYPE TABLE:"
        )
    except StopIteration:
        out["error"] = "no `LINK TYPE TABLE:` section in `amd-smi topology` output"
        out["raw"] = text[:4000]
        return out

    # Header row is the next non-empty line after the label, and contains
    # no leading BDF -- only column BDFs.
    header_bdfs: List[str] = []
    data_start = None
    for j in range(idx + 1, len(lines)):
        l = lines[j].rstrip()
        if not l.strip():
            continue
        if section_header_re.match(l):
            break
        toks = bdf_re.findall(l)
        if not toks:
            continue
        # The header line has only column BDFs (no leading row label), and
        # the first non-whitespace char position lines up with the columns.
        # Heuristic: header has BDFs but no other tokens that look like
        # link-type values (XGMI/PCIE/SELF/...). Data rows always have
        # exactly one leading BDF followed by N value tokens.
        non_bdf_toks = [
            t for t in l.split() if not bdf_re.fullmatch(t)
        ]
        if not non_bdf_toks:
            header_bdfs = toks
            data_start = j + 1
            break

    if not header_bdfs or data_start is None:
        out["error"] = "could not find header row inside LINK TYPE TABLE"
        out["raw"] = text[:4000]
        return out

    n = len(header_bdfs)
    bdf_to_idx = {b: i for i, b in enumerate(header_bdfs)}
    matrix: List[List[str]] = [[""] * n for _ in range(n)]
    seen_rows = 0
    for j in range(data_start, len(lines)):
        l = lines[j].rstrip()
        if not l.strip():
            continue
        if section_header_re.match(l):
            break
        toks = l.split()
        # First token must be a BDF, the remaining N tokens are the row.
        if not bdf_re.fullmatch(toks[0]):
            continue
        row_bdf = toks[0]
        cells = toks[1:]
        if row_bdf not in bdf_to_idx:
            continue
        row_idx = bdf_to_idx[row_bdf]
        for k, cell in enumerate(cells[:n]):
            matrix[row_idx][k] = cell
        seen_rows += 1

    if seen_rows == 0:
        out["error"] = "no BDF-labelled rows found inside LINK TYPE TABLE"
        out["raw"] = text[:4000]
        return out

    healthy_diag = {"SELF", "X", "-", "0"}
    healthy_link = {"XGMI"}
    non_xgmi: List[Any] = []
    for i, row in enumerate(matrix):
        for j_idx, cell in enumerate(row):
            cu = cell.strip().upper()
            if i == j_idx:
                # Diagonal: must be SELF (or empty if the row was missing).
                if cu and cu not in healthy_diag and cu not in healthy_link:
                    non_xgmi.append((i, j_idx, cell))
                continue
            if not cu:
                # Missing cell -> can't certify XGMI -> flag.
                non_xgmi.append((i, j_idx, "<missing>"))
                continue
            if cu in healthy_link or cu in healthy_diag:
                continue
            non_xgmi.append((i, j_idx, cell))

    out["ok"] = True
    out["tool"] = "amd-smi topology"
    out["bdfs"] = header_bdfs
    out["matrix"] = matrix
    out["n_gpus"] = n
    out["non_xgmi_pairs"] = non_xgmi
    return out
