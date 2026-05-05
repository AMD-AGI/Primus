###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- D-1 heavy: per-GPU low-level via amd-smi (ECC, throttle, clocks,
power cap). Runs ONCE per node (not per per-GPU subprocess) so the smoke
step doesn't pay an amd-smi startup tax 8x. Best-effort: missing amd-smi
or unparseable output degrades to {"ok": False, ...} without raising.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

from ..shell_utils import _which
from .rocm_smi import _rocm_smi_ras_info_text, _rocm_smi_use_json


def _collect_amd_smi_metrics() -> Dict[str, Any]:
    """Best-effort capture of per-GPU low-level metrics via ``amd-smi``.

    We try ``amd-smi metric --json`` first (newer builds emit valid JSON);
    if that fails we fall back to text output and surface the raw text under
    ``raw`` so an operator can still grep it. The on-disk shape is:

        {
            "ok": bool,
            "tool": "amd-smi metric --json" | "amd-smi metric" | None,
            "per_gpu": [ {gpu, gfx_clock_mhz, hbm_used_bytes,
                          power_avg_w, power_cap_w, temp_edge_c,
                          ecc_uncorrectable_total, ecc_correctable_total,
                          throttle_status_raw, ...}, ... ],
            "error": "..."  (only when ok is False)
        }

    Hard-fail semantics live in ``_node_status_from``: any non-zero
    uncorrectable ECC count becomes a node FAIL. Throttle status is
    captured under ``throttle_status_raw`` for operator inspection but
    is NOT failed-on -- the amd-smi throttle schema varies too much
    across releases to make a robust default rule.
    """
    out: Dict[str, Any] = {"ok": False, "tool": None, "per_gpu": []}

    if _which("amd-smi") is not None:
        # Try JSON first.
        try:
            cp = subprocess.run(
                ["amd-smi", "metric", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                check=False,
            )
            if cp.returncode == 0 and cp.stdout.strip():
                try:
                    doc = json.loads(cp.stdout)
                    out["ok"] = True
                    out["tool"] = "amd-smi metric --json"
                    out["per_gpu"] = _flatten_amd_smi_metric_json(doc)
                except Exception as e:
                    out["json_parse_error"] = str(e)
            else:
                out["json_rc"] = cp.returncode
                out["json_stderr"] = (cp.stderr or "").strip()[:200]
        except subprocess.TimeoutExpired:
            out["json_error"] = "amd-smi metric --json timed out"
        except Exception as e:
            out["json_error"] = str(e)

        # If JSON didn't work, capture the raw text output so the operator
        # can still grep it. We don't try to parse the human-readable text
        # -- per-GPU outliers will still show up via the sysfs/torch-side
        # details we capture in _per_gpu_body, and the rocm-smi fallback
        # below will fill in ECC + activity in the per_gpu records.
        if not out["ok"]:
            try:
                cp = subprocess.run(
                    ["amd-smi", "metric"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if cp.returncode == 0:
                    out["ok"] = True
                    out["tool"] = "amd-smi metric"
                    out["raw"] = cp.stdout[:8000]  # cap to keep JSON small
                else:
                    out["error"] = (cp.stderr or "").strip()[:200] or f"rc={cp.returncode}"
            except subprocess.TimeoutExpired:
                out["error"] = "amd-smi metric timed out"
            except Exception as e:
                out["error"] = str(e)
    else:
        out["error"] = "amd-smi not found in PATH"

    # rocm-smi fallback / fill-in. We always run it -- even when amd-smi
    # produced JSON, the schema may have missed ECC or activity fields,
    # and rocm-smi gives us a second independent source. Per-GPU records
    # are merged by GPU index so amd-smi-provided power/clocks/temp stay
    # alongside rocm-smi-provided ECC/activity.
    if _which("rocm-smi") is not None:
        # Build an index of existing records so we can merge in place.
        by_gpu: Dict[int, Dict[str, Any]] = {}
        for rec in out.get("per_gpu") or []:
            g = rec.get("gpu")
            if isinstance(g, int):
                by_gpu[g] = rec

        # ECC fill-in (only if not already populated by amd-smi)
        ras = _rocm_smi_ras_info_text()
        ecc_filled = 0
        if ras.get("ok"):
            for rec in ras.get("per_gpu") or []:
                g = rec.get("gpu")
                if not isinstance(g, int):
                    continue
                tgt = by_gpu.setdefault(g, {"gpu": g})
                if tgt.get("ecc_uncorrectable_total") is None:
                    tgt["ecc_uncorrectable_total"] = rec.get("ecc_uncorrectable_total")
                    tgt["ecc_correctable_total"] = rec.get("ecc_correctable_total")
                    tgt["ecc_source"] = "rocm-smi --showrasinfo"
                    ecc_filled += 1
            if ecc_filled:
                out.setdefault("fallback_tools", []).append(
                    f"rocm-smi --showrasinfo (ECC for {ecc_filled} GPU(s))"
                )

        # Activity fill-in (only if not already populated by amd-smi)
        use = _rocm_smi_use_json()
        act_filled = 0
        if use.get("ok"):
            for rec in use.get("per_gpu") or []:
                g = rec.get("gpu")
                if not isinstance(g, int):
                    continue
                tgt = by_gpu.setdefault(g, {"gpu": g})
                if tgt.get("gfx_activity_pct") is None:
                    tgt["gfx_activity_pct"] = rec.get("gfx_activity_pct")
                    tgt["activity_source"] = "rocm-smi --showuse"
                    act_filled += 1
            if act_filled:
                out.setdefault("fallback_tools", []).append(
                    f"rocm-smi --showuse (activity for {act_filled} GPU(s))"
                )

        # Rebuild per_gpu in stable index order if rocm-smi added entries
        if ecc_filled or act_filled:
            out["per_gpu"] = [by_gpu[g] for g in sorted(by_gpu.keys())]
            # If amd-smi produced nothing at all, this is now a successful
            # collection, just sourced entirely from rocm-smi.
            if not out["ok"] and out["per_gpu"]:
                out["ok"] = True
                out["tool"] = "rocm-smi (ECC/activity fallback)"

    return out


def _flatten_amd_smi_metric_json(doc: Any) -> List[Dict[str, Any]]:
    """Pull the fields we care about out of `amd-smi metric --json` output.

    The exact schema varies between amd-smi releases. We touch only the
    most-stable nesting -- a top-level list of per-GPU dicts, each with
    sub-blocks like ``power``, ``clock``, ``temperature``, ``ecc``,
    ``throttle_status`` -- and tolerate missing fields silently.
    """
    out: List[Dict[str, Any]] = []
    items = doc if isinstance(doc, list) else (doc.get("gpus", []) if isinstance(doc, dict) else [])
    for i, g in enumerate(items):
        if not isinstance(g, dict):
            continue
        rec: Dict[str, Any] = {"gpu": i}
        # gpu id may be in g["gpu"] or g["device_id"] depending on schema
        if isinstance(g.get("gpu"), int):
            rec["gpu"] = g["gpu"]
        # power
        power = g.get("power") or {}
        if isinstance(power, dict):
            for k_src, k_dst in (
                ("average_socket_power", "power_avg_w"),
                ("current_socket_power", "power_avg_w"),
                ("socket_power", "power_avg_w"),
                ("power_cap", "power_cap_w"),
                ("power_limit", "power_cap_w"),
            ):
                v = power.get(k_src)
                if isinstance(v, (int, float)) and rec.get(k_dst) is None:
                    rec[k_dst] = v
        # clocks (gfx clock most useful)
        clk = g.get("clock") or g.get("clocks") or {}
        if isinstance(clk, dict):
            gfx = clk.get("gfx") or clk.get("gfx_0") or clk.get("gfx_clock") or {}
            if isinstance(gfx, dict):
                for k in ("clk", "current", "value", "frequency"):
                    if isinstance(gfx.get(k), (int, float)):
                        rec["gfx_clock_mhz"] = gfx[k]
                        break
            elif isinstance(gfx, (int, float)):
                rec["gfx_clock_mhz"] = gfx
        # temperature
        temp = g.get("temperature") or {}
        if isinstance(temp, dict):
            for k in ("edge", "current", "value"):
                v = temp.get(k)
                if isinstance(v, (int, float)):
                    rec["temp_edge_c"] = v
                    break
        # ECC
        ecc = g.get("ecc") or g.get("ecc_count") or {}
        if isinstance(ecc, dict):
            ue = ecc.get("uncorrectable") or ecc.get("uncorrectable_total") or ecc.get("ue") or 0
            ce = ecc.get("correctable") or ecc.get("correctable_total") or ecc.get("ce") or 0
            try:
                rec["ecc_uncorrectable_total"] = int(ue)
                rec["ecc_correctable_total"] = int(ce)
            except Exception:
                pass
        # Throttle
        thr = g.get("throttle_status") or g.get("throttle") or {}
        if isinstance(thr, dict):
            rec["throttle_status_raw"] = thr
        elif isinstance(thr, (str, list)):
            rec["throttle_status_raw"] = thr
        # GPU compute activity %. Used by the aggregator to surface GPUs
        # that are running someone else's compute right now (warn-only;
        # short bursts are normal but a sustained pegged-100% across
        # multiple GPUs is a strong signal of a leaked rank from a
        # previous job).
        usage = g.get("usage") or g.get("activity") or g.get("utilization") or {}
        if isinstance(usage, dict):
            for k in ("gfx_activity", "gfx", "gfx_busy_percent", "gpu_busy_percent"):
                v = usage.get(k)
                if isinstance(v, (int, float)):
                    rec["gfx_activity_pct"] = float(v)
                    break
            if rec.get("gfx_activity_pct") is None:
                v = usage.get("activity") or usage.get("value")
                if isinstance(v, (int, float)):
                    rec["gfx_activity_pct"] = float(v)
        elif isinstance(usage, (int, float)):
            rec["gfx_activity_pct"] = float(usage)
        out.append(rec)
    return out
