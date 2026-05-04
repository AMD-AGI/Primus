###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- F: rocm-smi self-latency + cross-tool fallbacks for amd-smi checks.

rocm-smi is "usually available" even on nodes that lack amd-smi (older
ROCm installs, stripped-down containers). Every amd-smi check we
silently no-op when amd-smi is missing has a rocm-smi equivalent, so
we can keep ECC / XGMI / foreign-process / activity coverage even
without amd-smi. Each helper produces the same record shape the
upstream amd-smi parser already emits, so _node_status_from and the
aggregator helpers don't need any per-tool conditionals.

Output schemas (cross-tool stable):

  ECC      -> per-GPU {gpu, ecc_correctable_total, ecc_uncorrectable_total}
  XGMI     -> {ok, tool, n_gpus, link_types: [[...]], non_xgmi_pairs}
  procs    -> per-GPU {gpu, processes: [annotated PID dicts]}
  activity -> per-GPU {gpu, gfx_activity_pct}

The self-latency canary (``_collect_rocm_smi_self_latency``) is its own
Tier 1 F check: a wedged amdgpu driver makes ``rocm-smi`` calls take
30-60 s before failing outright -- usually minutes before the GPU
itself stops responding. Hitting the timeout is treated as a hard fail
in ``_node_status_from``.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..shell_utils import _which


def _collect_rocm_smi_self_latency(*, timeout_sec: float) -> Dict[str, Any]:
    """Time a single ``rocm-smi --version`` call against ``timeout_sec``.

    A wedged amdgpu driver makes ``rocm-smi`` calls take 30-60 s before
    failing outright -- usually minutes before the GPU itself stops
    responding. Catching this in preflight gives operators a chance to
    drain the node before a real training job starts hanging on it.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "tool": None,
        "latency_sec": None,
        "timeout_sec": float(timeout_sec),
    }
    binpath = _which("rocm-smi")
    if binpath is None:
        out["error"] = "rocm-smi not found in PATH"
        return out
    out["tool"] = binpath
    t0 = time.monotonic()
    try:
        cp = subprocess.run(
            [binpath, "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_sec, check=False,
        )
        out["latency_sec"] = round(time.monotonic() - t0, 3)
        out["rc"] = cp.returncode
        out["ok"] = (cp.returncode == 0)
        if cp.returncode != 0:
            out["error"] = (cp.stderr or cp.stdout or "").strip()[:200]
    except subprocess.TimeoutExpired:
        out["latency_sec"] = round(time.monotonic() - t0, 3)
        out["timed_out"] = True
        out["error"] = (
            f"rocm-smi --version did not finish in {timeout_sec}s -- driver may be wedging"
        )
    except Exception as e:
        out["error"] = str(e)
    return out


def _rocm_smi_ras_info_text(timeout_sec: float = 15.0) -> Dict[str, Any]:
    """ECC counts via ``rocm-smi --showrasinfo`` (TEXT only).

    The ``--json`` form returns "WARNING: No JSON data to report" so we
    parse the text. Format is one block per GPU::

        GPU[0]:         RAS INFO
                Block       Status    Correctable Error  Uncorrectable Error
                  UMC        ENABLED                  0                    0
                 SDMA        ENABLED                  0                    0
                  GFX        ENABLED                  0                    0
                ...

    Some blocks (ATHUB, PCIE_BIF, HDP, ...) report only Status. We sum
    every numeric Correctable/Uncorrectable cell per GPU so a hardware
    error in any block surfaces in ``ecc_uncorrectable_total``.
    """
    out: Dict[str, Any] = {"ok": False, "tool": None, "per_gpu": []}
    if _which("rocm-smi") is None:
        out["error"] = "rocm-smi not found in PATH"
        return out
    try:
        cp = subprocess.run(
            ["rocm-smi", "--showrasinfo"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_sec, check=False,
        )
        if cp.returncode != 0:
            out["error"] = (cp.stderr or cp.stdout or "").strip()[:200] or f"rc={cp.returncode}"
            return out
        out["per_gpu"] = _parse_rocm_smi_ras_info_text(cp.stdout)
        out["tool"] = "rocm-smi --showrasinfo"
        out["ok"] = True
    except subprocess.TimeoutExpired:
        out["error"] = "rocm-smi --showrasinfo timed out"
    except Exception as e:
        out["error"] = str(e)
    return out


def _parse_rocm_smi_ras_info_text(text: str) -> List[Dict[str, Any]]:
    """Parse the per-GPU ``GPU[N]: RAS INFO`` blocks from rocm-smi text."""
    import re
    out: List[Dict[str, Any]] = []
    cur_gpu: Optional[int] = None
    cur_corr = 0
    cur_uncorr = 0
    in_block = False
    gpu_hdr = re.compile(r"^GPU\[(\d+)\]:\s*RAS INFO", re.IGNORECASE)
    for raw in text.splitlines():
        line = raw.strip()
        m = gpu_hdr.match(line)
        if m:
            # Flush previous GPU
            if cur_gpu is not None:
                out.append({
                    "gpu": cur_gpu,
                    "ecc_correctable_total": cur_corr,
                    "ecc_uncorrectable_total": cur_uncorr,
                })
            cur_gpu = int(m.group(1))
            cur_corr = 0
            cur_uncorr = 0
            in_block = True
            continue
        if not in_block or cur_gpu is None:
            continue
        # End-of-block separator
        if line.startswith("__") or line.startswith("=="):
            continue
        # Per-block row: "BLOCK STATUS CORR UNCORR" -- last two are ints
        # when present. Header row has the words "Correctable Error" so
        # we filter out non-numeric rows naturally.
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            corr = int(toks[-2])
            uncorr = int(toks[-1])
        except ValueError:
            continue
        cur_corr += corr
        cur_uncorr += uncorr
    if cur_gpu is not None:
        out.append({
            "gpu": cur_gpu,
            "ecc_correctable_total": cur_corr,
            "ecc_uncorrectable_total": cur_uncorr,
        })
    return out


def _rocm_smi_topotype_json(timeout_sec: float = 15.0) -> Dict[str, Any]:
    """XGMI link-type matrix via ``rocm-smi --showtopotype --json``.

    Output shape is keyed by pair-string::

        {"system": {
            "(Topology) Link type between DRM devices 0 and 1": "XGMI",
            "(Topology) Link type between DRM devices 0 and 2": "XGMI",
            ...   # upper triangle only
        }}

    We parse the indices out of each key with a regex, build a symmetric
    NxN matrix, and emit it in the same shape ``_collect_xgmi_topology``
    already produces (so downstream consumers don't care which tool
    populated it). ``non_xgmi_pairs`` contains the (i, j, link_type)
    triples for any off-diagonal pair whose link_type is not XGMI.
    """
    import re
    out: Dict[str, Any] = {"ok": False, "tool": None, "n_gpus": 0,
                           "link_types": [], "non_xgmi_pairs": []}
    if _which("rocm-smi") is None:
        out["error"] = "rocm-smi not found in PATH"
        return out
    try:
        cp = subprocess.run(
            ["rocm-smi", "--showtopotype", "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_sec, check=False,
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
            return out
        try:
            doc = json.loads(cp.stdout)
        except Exception as e:
            out["error"] = f"json parse failed: {e}"
            return out
        sys_block = doc.get("system") if isinstance(doc, dict) else None
        if not isinstance(sys_block, dict):
            out["error"] = "no `system` key in rocm-smi --showtopotype output"
            return out
        pat = re.compile(
            r"DRM\s+devices?\s+(\d+)\s+and\s+(\d+)", re.IGNORECASE
        )
        pairs: Dict[tuple, str] = {}
        max_idx = -1
        for k, v in sys_block.items():
            m = pat.search(str(k))
            if not m:
                continue
            i = int(m.group(1))
            j = int(m.group(2))
            pairs[(i, j)] = str(v)
            max_idx = max(max_idx, i, j)
        if max_idx < 0:
            out["error"] = "no parseable DRM-device pair keys"
            return out
        n = max_idx + 1
        mat = [["?" for _ in range(n)] for _ in range(n)]
        for (i, j), t in pairs.items():
            mat[i][j] = t
            mat[j][i] = t
        for i in range(n):
            mat[i][i] = "SELF"
        non_xgmi: List[Any] = []
        for i in range(n):
            for j in range(i + 1, n):
                t = mat[i][j]
                if t and t.upper() != "XGMI":
                    non_xgmi.append((i, j, t))
        out["n_gpus"] = n
        out["link_types"] = mat
        out["non_xgmi_pairs"] = non_xgmi
        out["tool"] = "rocm-smi --showtopotype --json"
        out["ok"] = True
    except subprocess.TimeoutExpired:
        out["error"] = "rocm-smi --showtopotype timed out"
    except Exception as e:
        out["error"] = str(e)
    return out


def _rocm_smi_processes(
    annotate: Any,
    timeout_sec: float = 15.0,
) -> Dict[str, Any]:
    """Foreign-process enumeration via ``rocm-smi --showpids --json``.

    Output shape (verified against rocm-smi on a busy MI300X)::

        {"system": {
            "PID2683309": "python3.11, 1, 24556904448, 0, 0",
            "PID29324":   "gpuagent, 0, 0, 0, 0"
        }}

    Comma-separated fields are: ``name, num_gpus_used, vram_bytes,
    sdma_bytes, cu_occupancy``. We extract ``name`` (field 0) and
    ``vram_bytes`` (field 2) -- enough to surface leaked training PIDs
    holding gigabytes of HBM. ``--showpidgpus --json`` returns
    "No JSON data to report", so per-GPU mapping is not available --
    all PIDs go into the gpu=-1 bucket (same convention as lsof).
    """
    out: Dict[str, Any] = {"ok": False, "tool": None, "per_gpu": []}
    if _which("rocm-smi") is None:
        out["error"] = "rocm-smi not found in PATH"
        return out
    try:
        cp = subprocess.run(
            ["rocm-smi", "--showpids", "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_sec, check=False,
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
            return out
        try:
            doc = json.loads(cp.stdout)
        except Exception as e:
            out["error"] = f"json parse failed: {e}"
            return out
        sys_block = doc.get("system") if isinstance(doc, dict) else None
        if not isinstance(sys_block, dict):
            # Empty or unexpected -- treat as "no PIDs"
            out["tool"] = "rocm-smi --showpids --json"
            out["ok"] = True
            return out
        procs: List[Dict[str, Any]] = []
        for k, v in sys_block.items():
            try:
                pid = int(str(k).lstrip("PID").lstrip("pid"))
            except ValueError:
                continue
            name = ""
            hbm: Optional[int] = None
            if isinstance(v, str):
                parts = [s.strip() for s in v.split(",")]
                if parts:
                    name = parts[0]
                # field 2 = VRAM bytes (rocm-smi --showpids documented format)
                if len(parts) > 2:
                    try:
                        hbm = int(parts[2])
                    except ValueError:
                        hbm = None
            elif isinstance(v, dict):
                name = str(v.get("name") or v.get("process_name") or "")
                vram = v.get("vram") or v.get("vram_bytes")
                if isinstance(vram, (int, float)):
                    hbm = int(vram)
            elif isinstance(v, list) and v:
                name = str(v[0])
            procs.append(annotate(pid, name, hbm))
        if procs:
            out["per_gpu"] = [{"gpu": -1, "processes": procs}]
        out["tool"] = "rocm-smi --showpids --json"
        out["ok"] = True
    except subprocess.TimeoutExpired:
        out["error"] = "rocm-smi --showpids timed out"
    except Exception as e:
        out["error"] = str(e)
    return out


def _rocm_smi_use_json(timeout_sec: float = 15.0) -> Dict[str, Any]:
    """GPU compute-activity % via ``rocm-smi --showuse --json``.

    Output shape::

        {"card0": {"GPU use (%)": "0", "GFX Activity": "465307149"}, ...}

    "GPU use (%)" is the percentage we want for ``gfx_activity_pct``
    (matching the amd-smi field name). "GFX Activity" is a cumulative
    cycle counter, not a percentage -- ignored.
    """
    out: Dict[str, Any] = {"ok": False, "tool": None, "per_gpu": []}
    if _which("rocm-smi") is None:
        out["error"] = "rocm-smi not found in PATH"
        return out
    try:
        cp = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_sec, check=False,
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
            return out
        try:
            doc = json.loads(cp.stdout)
        except Exception as e:
            out["error"] = f"json parse failed: {e}"
            return out
        if not isinstance(doc, dict):
            out["error"] = "unexpected top-level shape"
            return out
        per_gpu: List[Dict[str, Any]] = []
        for k, v in doc.items():
            if not isinstance(v, dict):
                continue
            if not str(k).startswith("card"):
                continue
            try:
                gpu = int(str(k)[len("card"):])
            except ValueError:
                continue
            pct_raw = v.get("GPU use (%)") or v.get("GPU use")
            try:
                pct = float(str(pct_raw).strip())
            except (TypeError, ValueError):
                continue
            per_gpu.append({"gpu": gpu, "gfx_activity_pct": pct})
        out["per_gpu"] = per_gpu
        out["tool"] = "rocm-smi --showuse --json"
        out["ok"] = True
    except subprocess.TimeoutExpired:
        out["error"] = "rocm-smi --showuse timed out"
    except Exception as e:
        out["error"] = str(e)
    return out
